"""Tiny real-Qwen2 tests: no model downloads; CUDA coverage when available."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Model

from llava_pruning.methods import load_method, resolve_method_config
from test_fastv_attention import tiny_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/llava"))
from llava.model.language_model.llava_qwen import LlavaQwenModel, LlavaQwenForCausalLM
from llava.model.language_model.fastv_attention import last_prompt_attention


def configure(model, enabled=True, capture=True):
    model.configure_vico(enabled=enabled, layers=[1, 2, 3],
                         keep_ratios=[0.75, 0.5, 0.25] if enabled else [1, 1, 1],
                         capture_visualization=capture, capture_attention=capture)
    model.set_vico_image_spans([[(1, 9)]], 11)


def fixed_scores(layer, hidden, rotary, attention_mask=None):
    return torch.arange(hidden.shape[1], device=hidden.device, dtype=torch.float32)[None]


class ViCoMethodTests(unittest.TestCase):
    def test_default_config_is_method_specific_and_rates_are_cumulative(self):
        method = load_method("vico")
        self.assertEqual(method.layers, (8, 16, 24))
        self.assertEqual(method.rates, tuple(range(0, 100, 10)))
        self.assertEqual(method.visualize_rates, {10, 30, 50, 70, 90})
        self.assertEqual(method.keep_ratios(0), (1, 1, 1))
        ratios = method.keep_ratios(90)
        self.assertAlmostEqual(ratios[0], 0.1 ** (1/3))
        self.assertAlmostEqual(ratios[1], 0.1 ** (2/3))
        self.assertAlmostEqual(ratios[-1], 0.1)
        self.assertEqual(resolve_method_config("vico").name, "vico.json")
        self.assertEqual(resolve_method_config("fastv").name, "fastv.json")
        with self.assertRaises(ValueError):
            load_method("missing")

    def test_wrong_config_and_invalid_layer_or_rate_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, "Unknown ViCo config"):
            load_method("vico", resolve_method_config("fastv"))
        base = {"layers": [8, 16, 24], "prune_rates": [0, 90], "visualize_rates": [90]}
        for changes in ({"layers": []}, {"layers": [8, 8]}, {"layers": [24, 8]},
                        {"layers": [0]}, {"prune_rates": [100]}, {"prune_rates": [True]},
                        {"visualize_rates": [0]}, {"min_tokens": 0},
                        {"rate_semantics": "per_stage"}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "method.json"
                config.write_text(json.dumps({**base, **changes}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_method("vico", config)


class ViCoDecoderTests(unittest.TestCase):
    def test_zero_rate_is_bitwise_original_prefill_and_cached_decode(self):
        torch.manual_seed(71)
        reference = Qwen2Model(tiny_config("sdpa")).eval()
        model = LlavaQwenModel(tiny_config("sdpa")).eval()
        model.load_state_dict(reference.state_dict())
        configure(model, enabled=False)
        tokens = torch.arange(1, 12)[None]
        with patch("llava.model.language_model.vico.last_prompt_attention",
                   side_effect=AssertionError("zero-rate attention scoring")), torch.no_grad():
            expected, actual = reference(input_ids=tokens), model(input_ids=tokens)
            self.assertTrue(torch.equal(expected.last_hidden_state, actual.last_hidden_state))
            for token in (12, 13):
                expected = reference(input_ids=torch.tensor([[token]]), past_key_values=expected.past_key_values)
                actual = model(input_ids=torch.tensor([[token]]), past_key_values=actual.past_key_values)
                self.assertTrue(torch.equal(expected.last_hidden_state, actual.last_hidden_state))
        rows = model.get_vico_stats()["layers"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["image_tokens_in"] == row["image_tokens_out"] == 8 for row in rows))

    def test_physical_lengths_positions_nested_masks_and_three_independent_scores(self):
        model = LlavaQwenModel(tiny_config("sdpa")).eval()
        configure(model)
        observed = []
        def observe(module, args, kwargs):
            observed.append((args[0].shape[1], kwargs["position_ids"].tolist(), kwargs["output_attentions"]))
        handles = [layer.register_forward_pre_hook(observe, with_kwargs=True) for layer in model.layers]
        try:
            with patch("llava.model.language_model.vico.last_prompt_attention", wraps=last_prompt_attention) as score, torch.no_grad():
                output = model(input_ids=torch.arange(1, 12)[None])
                self.assertEqual(score.call_count, 3)
                self.assertEqual([item[0] for item in observed], [11, 9, 7, 5])
                for length, positions, attentions in observed:
                    self.assertEqual(positions, [list(range(length))])
                    self.assertFalse(attentions)
                self.assertEqual([item[0].shape[-2] for item in output.past_key_values], [11, 9, 7, 5])
                self.assertEqual(output.last_hidden_state.shape[1], 5)
                observed.clear()
                output = model(input_ids=torch.tensor([[12]]), past_key_values=output.past_key_values,
                               attention_mask=torch.ones(1, 12), position_ids=torch.tensor([[11]]))
                self.assertEqual(score.call_count, 3)
                self.assertEqual([item[1] for item in observed], [[[11]], [[9]], [[7]], [[5]]])
                self.assertEqual([item[0].shape[-2] for item in output.past_key_values], [12, 10, 8, 6])
                self.assertTrue(torch.isfinite(output.last_hidden_state).all())
            stages = model.get_vico_stages()
            self.assertEqual([sum(stage["mask"]["keep"]) for stage in stages], [6, 4, 2])
            self.assertEqual([sum(stage["attention"]["valid"]) for stage in stages], [8, 6, 4])
            for earlier, later in zip(stages, stages[1:]):
                self.assertTrue(all(not b or a for a, b in zip(earlier["mask"]["keep"], later["mask"]["keep"])))
            self.assertEqual(model.get_vico_stats()["layers"][-1]["cumulative_prune_rate"], 75)
        finally:
            for handle in handles:
                handle.remove()

    def test_cache_matches_recomputed_prefix_with_fixed_decisions(self):
        torch.manual_seed(74)
        model = LlavaQwenModel(tiny_config("eager")).eval()
        full = LlavaQwenModel(tiny_config("eager")).eval()
        full.load_state_dict(model.state_dict())
        configure(model, capture=False)
        tokens = torch.arange(1, 12)[None]
        with patch("llava.model.language_model.vico.last_prompt_attention", side_effect=fixed_scores), torch.no_grad():
            output = model(input_ids=tokens)
            for token in (12, 13, 14):
                output = model(input_ids=torch.tensor([[token]]), past_key_values=output.past_key_values)
                tokens = torch.cat((tokens, torch.tensor([[token]])), dim=1)
                configure(full, capture=False)
                expected = full(input_ids=tokens, use_cache=False)
                torch.testing.assert_close(output.last_hidden_state[:, -1], expected.last_hidden_state[:, -1],
                                           rtol=1e-5, atol=1e-6)
        self.assertEqual(model._vico_stages, [])

    def test_method_switch_clears_all_sample_state_and_guards_limitations(self):
        model = LlavaQwenModel(tiny_config()).eval()
        configure(model)
        with torch.no_grad():
            model(input_ids=torch.arange(1, 12)[None])
        self.assertTrue(model._vico_stages)
        model.configure_fastv(enabled=True)
        self.assertFalse(model.vico_enabled)
        self.assertFalse(model.vico_configured)
        self.assertEqual(model._vico_stages, [])
        configure(model)
        self.assertFalse(model.fastv_enabled)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            model(input_ids=torch.ones(2, 11, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "unpadded"):
            model(input_ids=torch.ones(1, 11, dtype=torch.long), attention_mask=torch.zeros(1, 11))
        with self.assertRaises(ValueError):
            model.configure_vico(enabled=True, layers=[4], keep_ratios=[0.5])

    def test_actual_hf_generate_runs_multistage_cache_and_resets_between_images(self):
        config = tiny_config("sdpa")
        config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
        model = LlavaQwenForCausalLM(config).eval()
        core = model.get_model()
        def pack(inputs, positions, mask, past, labels, images, modalities, image_sizes=None):
            if images is None:
                return inputs, positions, mask, past, None, labels
            core.set_vico_image_spans([[(1, 9)]], inputs.shape[1])
            return None, None, None, None, core.embed_tokens(inputs), None
        configure(core)
        tokens = torch.arange(1, 12)[None]
        with patch.object(model, "prepare_inputs_labels_for_multimodal", side_effect=pack), torch.no_grad():
            first = model.generate(inputs=tokens, images=torch.zeros(1), do_sample=False, max_new_tokens=4, use_cache=True)
            second = model.generate(inputs=tokens, images=torch.zeros(1), do_sample=False, max_new_tokens=4, use_cache=True)
        self.assertEqual(first.shape, (1, 4))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(core.get_vico_stats()["layers"]), 4)
        self.assertEqual(len(core.get_vico_stages()), 3)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flash_attn"),
                     "Requires CUDA + FlashAttention2 on the inference server")
class ViCoCudaTests(unittest.TestCase):
    def test_fp16_flash_zero_rate_identity_and_shortened_caches(self):
        config = tiny_config("flash_attention_2")
        config.torch_dtype = torch.float16
        reference = Qwen2Model(copy.deepcopy(config)).cuda().half().eval()
        model = LlavaQwenModel(copy.deepcopy(config)).cuda().half().eval()
        model.load_state_dict(reference.state_dict())
        tokens = torch.arange(1, 12, device="cuda")[None]
        configure(model, enabled=False)
        with torch.no_grad():
            self.assertTrue(torch.equal(reference(input_ids=tokens).last_hidden_state,
                                        model(input_ids=tokens).last_hidden_state))
            configure(model)
            output = model(input_ids=tokens)
            self.assertEqual([item[0].shape[-2] for item in output.past_key_values], [11, 9, 7, 5])
            for token in (12, 13, 14):
                output = model(input_ids=torch.tensor([[token]], device="cuda"), past_key_values=output.past_key_values)
                self.assertTrue(torch.isfinite(output.last_hidden_state).all())
                self.assertEqual(output.last_hidden_state.dtype, torch.float16)
        self.assertTrue(all(layer.self_attn.__class__.__name__ == "Qwen2FlashAttention2" for layer in model.layers))


if __name__ == "__main__":
    unittest.main()
