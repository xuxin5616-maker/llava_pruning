"""Real, tiny Qwen2 numerical coverage; no checkpoint or multimodal imports.

The attention oracle is Qwen2's eager attention, and the decoding oracle is a
fresh causal prefix pass using captured pruning decisions. Neither oracle calls
the PyramidDrop ranking or forward implementation.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


# Loading this leaf module avoids importing llava's vision/audio dependencies.
_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = "_pyramiddrop_test_decoder"
_DIRECTORY = _ROOT / "vendor/llava/llava/model/language_model"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_DIRECTORY)]
    sys.modules[_PACKAGE] = package
_SPEC = importlib.util.spec_from_file_location(
    f"{_PACKAGE}.pyramiddrop", _DIRECTORY / "pyramiddrop.py")
pyramiddrop = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pyramiddrop
_SPEC.loader.exec_module(pyramiddrop)


def tiny_config(implementation="eager"):
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=56,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0, use_cache=True,
    )
    config._attn_implementation = implementation
    return config


class TinyPyramidDrop(pyramiddrop.PyramidDropMixin, Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.init_pyramiddrop()

    def forward(self, *args, **kwargs):
        if self.pyramiddrop_enabled:
            return self.pyramiddrop_forward(*args, **kwargs)
        return super().forward(*args, **kwargs)


def configure(model, *, layers=(1, 2, 3), ratios=(0.8, 0.5, 0.2),
              capture=True, span=(2, 9), length=11):
    model.configure_pyramiddrop(enabled=True, layers=layers, keep_ratios=ratios,
                               capture_stages=capture)
    model.set_pyramiddrop_image_spans([[span]], length)
    return model


def cache_lengths(cache):
    if isinstance(cache, DynamicCache):
        return [cache.get_seq_length(i) for i in range(4)]
    return [entry[0].shape[-2] for entry in cache]


def causal_bias(hidden):
    length = hidden.shape[1]
    disallowed = torch.ones(length, length, device=hidden.device, dtype=torch.bool).triu(1)
    return torch.zeros(1, 1, length, length, device=hidden.device,
                       dtype=hidden.dtype).masked_fill(disallowed, torch.finfo(hidden.dtype).min)


@torch.no_grad()
def eager_last_row(layer, hidden, rotary):
    """Reference the complete Qwen2 attention matrix, including GQA and RoPE."""
    _, attention, _ = layer.self_attn(
        layer.input_layernorm(hidden), attention_mask=causal_bias(hidden),
        position_ids=torch.arange(hidden.shape[1], device=hidden.device)[None],
        position_embeddings=rotary, output_attentions=True, use_cache=False,
    )
    return attention[:, :, -1, :].mean(dim=1)


@torch.no_grad()
def recompute_prefix(model, tokens, stages, span):
    """Run the entire causal prefix with frozen choices, without any KV cache."""
    hidden = model.embed_tokens(tokens)
    start, end = span
    active = torch.arange(end - start, device=hidden.device)
    choices = {stage["after_layer"]: stage["mask"]["keep"] for stage in stages}
    for index, layer in enumerate(model.layers):
        ids = torch.arange(hidden.shape[1], device=hidden.device)[None]
        rotary = model.rotary_emb(hidden, ids)
        hidden = layer(hidden, attention_mask=causal_bias(hidden), position_ids=ids,
                       position_embeddings=rotary, use_cache=False,
                       output_attentions=False)[0]
        if index + 1 in choices:
            original_keep = torch.tensor(choices[index + 1], device=hidden.device)
            local_keep = original_keep[active].nonzero(as_tuple=False).flatten()
            selection = torch.cat((torch.arange(start, device=hidden.device),
                                   start + local_keep,
                                   torch.arange(start + len(active), hidden.shape[1],
                                                device=hidden.device)))
            hidden = hidden.index_select(1, selection)
            active = active[local_keep]
    return model.norm(hidden)


class PyramidDropAttentionTests(unittest.TestCase):
    def test_bfloat16_scores_preserve_original_eager_attention_precision(self):
        torch.manual_seed(420)
        config = tiny_config()
        layer = Qwen2DecoderLayer(config, layer_idx=1).to(torch.bfloat16).eval()
        core = Qwen2Model(config).to(torch.bfloat16).eval()
        hidden = torch.randn(1, 11, config.hidden_size).to(torch.bfloat16)
        rotary = core.rotary_emb(hidden, torch.arange(11)[None])
        with torch.no_grad():
            actual = pyramiddrop.last_prompt_attention(layer, hidden, rotary)
            expected = eager_last_row(layer, hidden, rotary)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gqa_score_matches_complete_eager_attention_with_nonuniform_rope(self):
        torch.manual_seed(421)
        config = tiny_config()
        layer = Qwen2DecoderLayer(config, layer_idx=2).eval()
        core = Qwen2Model(config).eval()
        hidden = torch.randn(1, 9, config.hidden_size)
        snapshot = hidden.clone()
        # A nonuniform grid catches ignored/double-applied position embeddings.
        ids = torch.tensor([[0, 1, 3, 4, 8, 13, 21, 34, 55]])
        rotary = core.rotary_emb(hidden, ids)
        with torch.no_grad():
            actual = pyramiddrop.last_prompt_attention(layer, hidden, rotary)
            expected = eager_last_row(layer, hidden, rotary)
        self.assertEqual(layer.self_attn.num_heads, 4)
        self.assertEqual(layer.self_attn.num_key_value_heads, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        self.assertTrue(torch.equal(hidden, snapshot))
        self.assertTrue(torch.isfinite(actual).all())

    def test_stage_scores_use_next_layer_and_keep_text_and_visual_order(self):
        torch.manual_seed(422)
        core = configure(TinyPyramidDrop(tiny_config()).eval(), layers=(1,), ratios=(0.57,))
        snapshots = {}
        def completed(module, args, output):
            snapshots["boundary"] = output[0].detach().clone()
        def entering(module, args, kwargs):
            snapshots["next_input"] = args[0].detach().clone()
        handles = [core.layers[0].register_forward_hook(completed),
                   core.layers[1].register_forward_pre_hook(entering, with_kwargs=True)]
        try:
            with torch.no_grad():
                core(input_ids=torch.arange(1, 12)[None], use_cache=True)
            full = snapshots["boundary"]
            rotary = core.rotary_emb(full, torch.arange(11)[None])
            expected = eager_last_row(core.layers[1], full, rotary)[0, 2:9]
            previous_layer_scores = eager_last_row(core.layers[0], full, rotary)[0, 2:9]
            self.assertFalse(torch.allclose(expected, previous_layer_scores, rtol=1e-4, atol=1e-6))
            stage = core.get_pyramiddrop_stages()[0]
            actual = torch.tensor(stage["attention"]["scores"])
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            # floor(7 * .57) is 3; ceil would incorrectly keep 4.
            selected = expected.topk(3).indices.sort().values
            original_keep = torch.zeros(7, dtype=torch.bool)
            original_keep[selected] = True
            self.assertEqual(stage["mask"]["keep"], original_keep.tolist())
            all_indices = torch.cat((torch.tensor([0, 1]), selected + 2, torch.tensor([9, 10])))
            self.assertTrue(torch.equal(snapshots["next_input"], full[:, all_indices]))
        finally:
            for handle in handles:
                handle.remove()


class PyramidDropDecoderTests(unittest.TestCase):
    def test_floor_budgets_nested_masks_and_compact_positions_for_both_backends(self):
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                torch.manual_seed(423)
                core = configure(TinyPyramidDrop(tiny_config(backend)).eval())
                observed = []
                def entering(module, args, kwargs):
                    observed.append((args[0].shape[1], kwargs["position_ids"].clone(),
                                     kwargs["cache_position"].clone(), kwargs["output_attentions"]))
                handles = [layer.register_forward_pre_hook(entering, with_kwargs=True)
                           for layer in core.layers]
                try:
                    with torch.no_grad():
                        result = core(input_ids=torch.arange(1, 12)[None], use_cache=True)
                    self.assertEqual([row[0] for row in observed], [11, 9, 7, 5])
                    self.assertEqual(cache_lengths(result.past_key_values), [11, 9, 7, 5])
                    self.assertEqual(result.last_hidden_state.shape[1], 5)
                    for length, ids, positions, output_attentions in observed:
                        self.assertEqual(ids.tolist(), [list(range(length))])
                        self.assertEqual(positions.tolist(), list(range(length)))
                        self.assertFalse(output_attentions)
                    stages = core.get_pyramiddrop_stages()
                    self.assertEqual([sum(stage["mask"]["keep"]) for stage in stages], [5, 3, 1])
                    self.assertEqual([sum(stage["attention"]["valid"]) for stage in stages], [7, 5, 3])
                    for earlier, later in zip(stages, stages[1:]):
                        self.assertTrue(all(not b or a for a, b in
                                            zip(earlier["mask"]["keep"], later["mask"]["keep"])))
                    stats = core.get_pyramiddrop_stats()
                    self.assertEqual(stats["original_image_tokens"], 7)
                    self.assertEqual([row["image_tokens_in"] for row in stats["layers"]], [7, 5, 3, 1])
                    self.assertEqual([row["image_tokens_out"] for row in stats["layers"]], [5, 3, 1, 1])
                finally:
                    for handle in handles:
                        handle.remove()

    def test_cached_decode_matches_fresh_full_prefix_with_the_same_decisions(self):
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                torch.manual_seed(424)
                core = configure(TinyPyramidDrop(tiny_config(backend)).eval())
                oracle = TinyPyramidDrop(tiny_config("eager")).eval()
                oracle.load_state_dict(core.state_dict())
                tokens = torch.arange(1, 12)[None]
                with torch.no_grad():
                    result = core(input_ids=tokens, use_cache=True)
                    stages = core.get_pyramiddrop_stages()
                    expected = recompute_prefix(oracle, tokens, stages, (2, 9))
                    torch.testing.assert_close(result.last_hidden_state, expected, rtol=1e-5, atol=1e-6)
                    for step, token in enumerate((12, 13, 14), start=1):
                        observed = []
                        def entering(module, args, kwargs):
                            observed.append((kwargs["position_ids"].item(),
                                             kwargs["cache_position"].item()))
                        handles = [layer.register_forward_pre_hook(entering, with_kwargs=True)
                                   for layer in core.layers]
                        try:
                            result = core(input_ids=torch.tensor([[token]]),
                                          past_key_values=result.past_key_values,
                                          attention_mask=torch.ones(1, 11 + step),
                                          position_ids=torch.tensor([[10 + step]]),
                                          cache_position=torch.tensor([10 + step]), use_cache=True)
                        finally:
                            for handle in handles:
                                handle.remove()
                        expected_positions = [10 + step, 8 + step, 6 + step, 4 + step]
                        self.assertEqual(observed, [(p, p) for p in expected_positions])
                        self.assertEqual(cache_lengths(result.past_key_values),
                                         [11 + step, 9 + step, 7 + step, 5 + step])
                        tokens = torch.cat((tokens, torch.tensor([[token]])), dim=1)
                        expected = recompute_prefix(oracle, tokens, stages, (2, 9))
                        torch.testing.assert_close(result.last_hidden_state[:, -1], expected[:, -1],
                                                   rtol=1e-5, atol=1e-6)
                    self.assertEqual(len(core.get_pyramiddrop_stages()), 3)
                    self.assertEqual(len(core.get_pyramiddrop_stats()["layers"]), 4)

    def test_consecutive_prompts_reset_decisions_and_sample_statistics(self):
        torch.manual_seed(425)
        core = configure(TinyPyramidDrop(tiny_config()).eval())
        with torch.no_grad():
            core(input_ids=torch.arange(1, 12)[None], use_cache=True)
            core.set_pyramiddrop_image_spans([[(1, 6)]], 9)
            result = core(input_ids=torch.arange(20, 29)[None], use_cache=True)
            fresh = configure(TinyPyramidDrop(tiny_config()).eval(), span=(1, 6), length=9)
            fresh.load_state_dict(core.state_dict())
            expected = fresh(input_ids=torch.arange(20, 29)[None], use_cache=True)
        self.assertTrue(torch.equal(result.last_hidden_state, expected.last_hidden_state))
        self.assertEqual(cache_lengths(result.past_key_values), [9, 8, 6, 5])
        self.assertEqual(core.get_pyramiddrop_stats()["original_image_tokens"], 5)
        self.assertEqual(len(core.get_pyramiddrop_stats()["layers"]), 4)
        self.assertEqual(len(core.get_pyramiddrop_stages()), 3)
        self.assertTrue(all(len(stage["mask"]["keep"]) == 5 for stage in core.get_pyramiddrop_stages()))

    def test_all_one_ratios_use_bitwise_original_prefill_and_decode(self):
        torch.manual_seed(426)
        reference = Qwen2Model(tiny_config("sdpa")).eval()
        core = configure(TinyPyramidDrop(tiny_config("sdpa")).eval(), ratios=(1, 1, 1))
        core.load_state_dict(reference.state_dict())
        self.assertFalse(core.pyramiddrop_enabled)
        with patch.object(pyramiddrop, "last_prompt_attention", side_effect=AssertionError("baseline ranked")), \
                torch.no_grad():
            tokens = torch.arange(1, 12)[None]
            expected = reference(input_ids=tokens, past_key_values=DynamicCache(), use_cache=True)
            actual = core(input_ids=tokens, past_key_values=DynamicCache(), use_cache=True)
            self.assertTrue(torch.equal(actual.last_hidden_state, expected.last_hidden_state))
            for token in (12, 13):
                expected = reference(input_ids=torch.tensor([[token]]),
                                     past_key_values=expected.past_key_values, use_cache=True)
                actual = core(input_ids=torch.tensor([[token]]),
                              past_key_values=actual.past_key_values, use_cache=True)
                self.assertTrue(torch.equal(actual.last_hidden_state, expected.last_hidden_state))

    def test_real_hf_generate_uses_dynamic_cache_resets_samples_and_rejects_beams(self):
        torch.manual_seed(427)
        config = tiny_config("sdpa")
        config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
        model = Qwen2ForCausalLM(config).eval()
        model.model = configure(TinyPyramidDrop(config).eval())
        tokens = torch.arange(1, 12)[None]
        with torch.no_grad():
            first = model.generate(input_ids=tokens, do_sample=False,
                                   max_new_tokens=4, use_cache=True)
            second = model.generate(input_ids=tokens, do_sample=False,
                                    max_new_tokens=4, use_cache=True)
        self.assertEqual(first.shape, (1, 15))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(model.model.get_pyramiddrop_stages()), 3)
        self.assertEqual(len(model.model.get_pyramiddrop_stats()["layers"]), 4)
        with self.assertRaises((ValueError, NotImplementedError)):
            model.generate(input_ids=tokens, num_beams=2, max_new_tokens=2,
                           do_sample=False, use_cache=True)

    def test_unsupported_cache_beam_expansion_padding_and_decode_chunk_fail(self):
        tokens = torch.arange(1, 12)[None]
        invalid_calls = [
            {"input_ids": tokens, "use_cache": False},
            {"input_ids": tokens, "past_key_values": ()},
            {"input_ids": tokens.expand(2, -1)},
            {"input_ids": tokens, "attention_mask": torch.tensor([[0] + [1] * 10])},
            {"input_ids": tokens, "attention_mask": torch.tensor([[1] * 10 + [0]])},
            {"input_ids": tokens, "output_attentions": True},
        ]
        for call in invalid_calls:
            with self.subTest(arguments=list(call)):
                core = configure(TinyPyramidDrop(tiny_config()).eval())
                with self.assertRaises((ValueError, NotImplementedError)):
                    core(**call)
        core = configure(TinyPyramidDrop(tiny_config()).eval())
        static = StaticCache(core.config, max_batch_size=1, max_cache_len=64,
                             device="cpu", dtype=torch.float32)
        with self.assertRaises((ValueError, NotImplementedError)):
            core(input_ids=tokens, past_key_values=static, use_cache=True)
        with torch.no_grad():
            result = core(input_ids=tokens, use_cache=True)
        with self.assertRaises((ValueError, NotImplementedError)):
            core(input_ids=torch.tensor([[12, 13]]), past_key_values=result.past_key_values)

    def test_invalid_schedule_is_rejected_before_forward(self):
        for layers, ratios in (((4,), (0.5,)), ((2, 1), (0.5, 0.2)),
                               ((1, 1), (0.5, 0.2)), ((0,), (0.5,)),
                               ((1, 2), (0.2, 0.5)), ((1,), (0,)),
                               ((1,), (float("nan"),)), ((1, 2), (0.5,))):
            with self.subTest(layers=layers, ratios=ratios):
                core = TinyPyramidDrop(tiny_config()).eval()
                with self.assertRaises(ValueError):
                    core.configure_pyramiddrop(enabled=True, layers=layers, keep_ratios=ratios)


if __name__ == "__main__":
    unittest.main()
