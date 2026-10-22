"""PriorTR: pinned-source scoring oracle, physical KV drop, sparse positions."""

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Model
from transformers.cache_utils import DynamicCache

from test_fastv_attention import tiny_config
from llava.model.language_model.llava_qwen import LlavaQwenModel, LlavaQwenForCausalLM
from llava.model.language_model.priortr import priortr_scores
from llava.constants import IMAGE_TOKEN_INDEX
from llava_pruning.methods import load_method


PROJECT = Path(__file__).resolve().parents[1]


def original_strategy():
    name = "_priortr_reference_strategy"
    package = types.ModuleType(name)
    package.__path__ = [str(PROJECT / "third_party/priortr")]
    registry = types.ModuleType(name + ".registry")
    registry.register_strategy = lambda key: lambda cls: cls
    with patch.dict(sys.modules, {name: package, name + ".registry": registry}):
        spec = importlib.util.spec_from_file_location(name + ".priortr", PROJECT / "third_party/priortr/priortr.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.PriorTRStrategy()


def causal_bias(hidden, past=0):
    n = hidden.shape[1]
    visible = torch.arange(past + n)[None] <= torch.arange(past, past + n)[:, None]
    return torch.zeros(1, 1, n, past + n, dtype=hidden.dtype).masked_fill(~visible[None, None], torch.finfo(hidden.dtype).min)


def configure(core, capture=True):
    core.configure_priortr(enabled=True, layer=2, keep_ratio=.4,
                          capture_visualization=capture, capture_attention=capture)
    core.set_priortr_image_spans([[(2, 9)]], 12)
    return core


class PriorTRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_independent_scores_match_pinned_source_on_real_eager_attention(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(674)
            model = Qwen2Model(tiny_config()).to(dtype).eval()
            hidden = torch.randn(1, 16, 32).to(dtype)
            layer = model.layers[1]
            ids = torch.arange(16)[None]
            rotary = model.rotary_emb(hidden, ids)
            with torch.no_grad():
                _, full, _ = layer.self_attn(layer.input_layernorm(hidden), position_ids=ids,
                                            position_embeddings=rotary, attention_mask=causal_bias(hidden),
                                            output_attentions=True, use_cache=False)
            oracle = original_strategy()
            for query in ("last", "question"):
                for head in ("mean", "max"):
                    with self.subTest(dtype=dtype, query=query, head=head):
                        config = types.SimpleNamespace(query_aggregation=query, head_aggregation=head)
                        expected = oracle.compute_scores(full, (2, 9), config)
                        actual = priortr_scores(layer, hidden, rotary, (2, 9), query_aggregation=query,
                                                head_aggregation=head, chunk_size=2)
                        torch.testing.assert_close(actual, expected, rtol=.02 if dtype == torch.bfloat16 else 1e-4,
                                                   atol=1e-3 if dtype == torch.bfloat16 else 1e-7)

    def test_prior_probe_cannot_see_instruction_and_question_mode_uses_all_rows(self):
        torch.manual_seed(675)
        model = Qwen2Model(tiny_config()).eval()
        hidden = torch.randn(1, 14, 32)
        rotary = model.rotary_emb(hidden, torch.arange(14)[None])
        layer = model.layers[0]
        before = hidden.clone()
        # Last-query mode ignores the intermediate instruction queries at this block.
        altered = hidden.clone()
        altered[:, 10:13] += 8
        score = priortr_scores(layer, hidden, rotary, (2, 9), query_aggregation="question", chunk_size=1)
        changed = priortr_scores(layer, altered, rotary, (2, 9), query_aggregation="question", chunk_size=1)
        self.assertFalse(torch.allclose(score, changed))
        self.assertTrue(torch.equal(hidden, before))
        # Explicit prior-only oracle remains unchanged when future tokens change.
        def prior(x):
            with torch.no_grad():
                _, full, _ = layer.self_attn(layer.input_layernorm(x), attention_mask=causal_bias(x),
                                            position_embeddings=rotary, output_attentions=True, use_cache=False)
            return full[:, :, 9, 2:9]
        torch.testing.assert_close(prior(hidden), prior(altered), rtol=0, atol=0)

    def test_all_caches_compact_and_decode_preserves_original_positions(self):
        for implementation in ("eager", "sdpa"):
            with self.subTest(backend=implementation):
                torch.manual_seed(676)
                core = configure(LlavaQwenModel(tiny_config(implementation)).eval())
                observed = []
                def observe(layer, args, kwargs):
                    observed.append((kwargs["position_ids"].clone(), kwargs["output_attentions"]))
                handles = [layer.register_forward_pre_hook(observe, with_kwargs=True) for layer in core.layers]
                try:
                    with torch.no_grad():
                        result = core(input_ids=torch.arange(1, 13)[None], past_key_values=DynamicCache(), use_cache=True)
                        self.assertEqual(result.last_hidden_state.shape[1], 7)  # floor(7*.4)=2, plus five text
                        self.assertEqual([result.past_key_values.get_seq_length(i) for i in range(4)], [7] * 4)
                        stage = core.get_priortr_stages()[0]
                        self.assertEqual(sum(stage["mask"]["keep"]), 2)
                        selected = torch.tensor(stage["mask"]["keep"]).nonzero().flatten() + 2
                        expected_ids = torch.cat((torch.arange(2), selected, torch.arange(9, 12)))[None]
                        self.assertTrue(torch.equal(observed[2][0], expected_ids))
                        for token, position in ((13, 12), (14, 13), (15, 14)):
                            observed.clear()
                            result = core(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
                            self.assertTrue(all(ids.tolist() == [[position]] for ids, _ in observed))
                            self.assertTrue(all(not requested for _, requested in observed))
                        self.assertEqual(len(core.get_priortr_stats()["layers"]), 4)
                finally:
                    for handle in handles:
                        handle.remove()

    def test_cached_decode_matches_independent_frozen_decision_full_prefix(self):
        torch.manual_seed(677)
        core = configure(LlavaQwenModel(tiny_config("sdpa")).eval())
        reference = Qwen2Model(tiny_config("eager")).eval()
        reference.load_state_dict(core.state_dict())
        tokens = torch.arange(1, 13)[None]
        with torch.no_grad():
            result = core(input_ids=tokens, use_cache=True)
            mask = torch.tensor(core.get_priortr_stages()[0]["mask"]["keep"])
            removed = (~mask).nonzero().flatten() + 2
            for token in (13, 14, 15):
                result = core(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
                tokens = torch.cat((tokens, torch.tensor([[token]])), 1)
                hidden = reference.embed_tokens(tokens)
                ids = torch.arange(tokens.shape[1])[None]
                for index, layer in enumerate(reference.layers):
                    bias = causal_bias(hidden)
                    if index < 2:
                        # Original prefill queries saw ALL keys. New decode queries only
                        # see retained cached keys, including at shallow layers.
                        bias[:, :, 12:, removed] = torch.finfo(hidden.dtype).min
                    hidden = layer(hidden, attention_mask=bias, position_ids=ids,
                                   position_embeddings=reference.rotary_emb(hidden, ids),
                                   use_cache=False, output_attentions=False)[0]
                    if index == 1:
                        select = torch.cat((torch.arange(2), mask.nonzero().flatten() + 2,
                                            torch.arange(9, hidden.shape[1])))
                        hidden, ids = hidden[:, select], ids[:, select]
                expected = reference.norm(hidden)
                torch.testing.assert_close(result.last_hidden_state[:, -1], expected[:, -1], rtol=1e-5, atol=1e-6)

    def test_real_llava_generate_zero_identity_and_repeatable_nonzero(self):
        torch.manual_seed(678)
        config = tiny_config("sdpa")
        config.eos_token_id, config.pad_token_id = None, 0
        config.image_aspect_ratio, config.mm_patch_merge_type = "anyres_max_9", "spatial_unpad"
        config.image_grid_pinpoints = [[8, 8], [16, 8]]
        config.tokenizer_model_max_length, config.tokenizer_padding_side = 128, "right"
        config.use_pos_skipping = False
        model = LlavaQwenForCausalLM(config).eval()
        core = model.get_model()
        core.vision_tower = types.SimpleNamespace(num_patches_per_side=2, image_size=8)
        core.image_newline = torch.nn.Parameter(torch.zeros(32))
        features = torch.randn(3, 4, 32)
        options = dict(inputs=torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3, 4]]),
                       images=[torch.zeros(3, 3, 8, 8)], image_sizes=[(16, 8)],
                       do_sample=False, max_new_tokens=4, use_cache=True)
        with patch.object(model, "encode_images", return_value=features), torch.no_grad():
            core.configure_fastv(enabled=False)
            baseline = model.generate(**options)
            core.configure_priortr(enabled=False, layer=2, keep_ratio=1)
            with patch("llava.model.language_model.priortr.priortr_scores", side_effect=AssertionError("0% scored")):
                zero = model.generate(**options)
            self.assertTrue(torch.equal(baseline, zero))
            core.configure_priortr(enabled=True, layer=2, keep_ratio=.3)
            first = model.generate(**options)
            self.assertTrue(torch.equal(first, model.generate(**options)))
            from llava_pruning.benchmark_timing import generate_with_benchmark
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                    patch("torch.cuda.max_memory_allocated", return_value=100), \
                    patch("torch.cuda.max_memory_reserved", return_value=200):
                measured, timing = generate_with_benchmark(model, options)
            self.assertTrue(torch.equal(first, measured))
            self.assertEqual(timing["generated_tokens"], 4)
            self.assertGreaterEqual(timing["lat_seconds"], 0)
            self.assertLessEqual(timing["lat_seconds"], timing["generation_seconds"])
            self.assertEqual(first.shape[-1], 4)
            stats = core.get_priortr_stats()
            self.assertEqual(stats["original_image_tokens"], 14)
            self.assertEqual(stats["final_llm_visual_tokens"], 4)
            self.assertEqual(stats["position_policy"], "preserve_original")
            with self.assertRaisesRegex(ValueError, "greedy"):
                model.generate(**dict(options, num_beams=2))

    def test_config_source_and_method_switches(self):
        method = replace(load_method("priortr"), rate_semantics="final_cumulative")
        self.assertEqual(method.query_aggregation, "last")
        self.assertTrue(method.details()["source"]["commit"])
        core = LlavaQwenModel(tiny_config()).eval()
        for next_method in ("fastv", "vico", "pyramiddrop"):
            method.configure(core, 70, capture_attention=False)
            if next_method == "fastv":
                core.configure_fastv(enabled=False)
            else:
                getattr(core, "configure_" + next_method)(enabled=True, layers=(1, 2, 3), keep_ratios=(.8, .5, .2))
            self.assertFalse(core.priortr_enabled)
            self.assertFalse(core.priortr_configured)
        with self.assertRaisesRegex(ValueError, "BEFORE"):
            method.validate_prompt("Question? <image>\n", 70)
        tokenizer = types.SimpleNamespace(encode=lambda *args, **kwargs: [5])
        method.validate_token_boundary(torch.tensor([[1, IMAGE_TOKEN_INDEX, 5, 8]]), tokenizer, IMAGE_TOKEN_INDEX, 70)
        with self.assertRaisesRegex(ValueError, "separator"):
            method.validate_token_boundary(torch.tensor([[1, IMAGE_TOKEN_INDEX, 8, 5]]), tokenizer, IMAGE_TOKEN_INDEX, 70)
        for query in ("question", "last"):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "config.json"
                path.write_text(json.dumps(dict(layer=2, prune_rates=[0, 70], query_aggregation=query)), encoding="utf-8")
                self.assertEqual(load_method("priortr", path).rates, (0, 70))


if __name__ == "__main__":
    unittest.main()
