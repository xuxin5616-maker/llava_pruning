"""CPU numerical tests plus optional CUDA/FlashAttention2 integration tests.

Run with the pinned inference environment; no checkpoint download is needed.
"""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "llava"))
from llava.model.language_model.fastv_attention import last_prompt_attention
from llava.model.language_model.llava_qwen import LlavaQwenModel


def tiny_config(implementation="eager"):
    config = Qwen2Config(
        vocab_size=40, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0, use_cache=True,
    )
    config._attn_implementation = implementation
    return config


class ScoreTests(unittest.TestCase):
    def test_independent_scores_match_eager_reference_with_gqa_and_padding(self):
        torch.manual_seed(7)
        config = tiny_config()
        layer = Qwen2DecoderLayer(config, layer_idx=1).eval()
        rotary = Qwen2Model(config).rotary_emb
        hidden = torch.randn(3, 7, 32)
        # Include no padding, left padding, and right padding in one batch.
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 0, 0]])
        position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
        embeddings = rotary(hidden, position_ids)
        snapshot = hidden.clone()
        observed = last_prompt_attention(layer, hidden, embeddings, mask)
        positions = torch.arange(7)
        causal = positions[None, :] <= positions[:, None]
        visible = causal[None, None] & mask[:, None, None].bool()
        bias = torch.zeros(3, 1, 7, 7).masked_fill(~visible, torch.finfo(hidden.dtype).min)
        with torch.no_grad():
            _, full, _ = layer.self_attn(
                layer.input_layernorm(hidden), attention_mask=bias,
                position_ids=position_ids, position_embeddings=embeddings,
                output_attentions=True, use_cache=False,
            )
        expected = full.mean(1)[torch.arange(3), torch.tensor([6, 6, 4])]
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-7)
        self.assertTrue(torch.equal(hidden, snapshot))
        self.assertEqual(observed.dtype, torch.float32)
        self.assertTrue(torch.equal(observed[mask == 0], torch.zeros_like(observed[mask == 0])))

    def test_empty_prompt_and_nonfinite_scores_fail_explicitly(self):
        config = tiny_config()
        layer = Qwen2DecoderLayer(config, layer_idx=1).eval()
        hidden = torch.randn(1, 4, 32)
        positions = torch.arange(4)[None]
        embeddings = Qwen2Model(config).rotary_emb(hidden, positions)
        with self.assertRaises(ValueError):
            last_prompt_attention(layer, hidden, embeddings, torch.zeros(1, 4))
        hidden.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            last_prompt_attention(layer, hidden, embeddings)


class DecoderTests(unittest.TestCase):
    def test_zero_rate_is_original_forward_in_prefill_and_cached_decoding(self):
        torch.manual_seed(9)
        reference = Qwen2Model(tiny_config("sdpa")).eval()
        model = LlavaQwenModel(tiny_config("sdpa")).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        model.configure_fastv(enabled=False)
        # Even stale metadata must not route zero rate through scoring/masking.
        model.set_fastv_image_spans([[(1, 5)]])
        with patch("llava.model.language_model.llava_qwen.last_prompt_attention",
                   side_effect=AssertionError("zero rate scored attention")), \
             patch.object(model, "_apply_fastv_mask",
                          side_effect=AssertionError("zero rate applied pruning")), \
             torch.no_grad():
            tokens = torch.tensor([[3, 4, 5, 6, 7, 8]])
            expected = reference(input_ids=tokens, use_cache=True)
            actual = model(input_ids=tokens, use_cache=True)
            self.assertTrue(torch.equal(actual.last_hidden_state, expected.last_hidden_state))
            for token in (9, 10):
                expected = reference(input_ids=torch.tensor([[token]]),
                                     past_key_values=expected.past_key_values, use_cache=True)
                actual = model(input_ids=torch.tensor([[token]]),
                               past_key_values=actual.past_key_values, use_cache=True)
                self.assertTrue(torch.equal(actual.last_hidden_state, expected.last_hidden_state))

    def test_nonzero_scores_once_without_requesting_attention_and_resets(self):
        torch.manual_seed(11)
        model = LlavaQwenModel(tiny_config("sdpa")).eval()
        model.configure_fastv(enabled=True, layer=2, keep_ratio=0.5,
                              preserve_image_newline=False, capture_attention=True)
        model.set_fastv_image_spans([[(1, 5)]])
        seen = []

        def observe(module, args, kwargs):
            seen.append(kwargs.get("output_attentions"))

        handles = [layer.register_forward_pre_hook(observe, with_kwargs=True) for layer in model.layers]
        try:
            with patch("llava.model.language_model.llava_qwen.last_prompt_attention",
                       wraps=last_prompt_attention) as score, torch.no_grad():
                output = model(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]), use_cache=True)
                model(input_ids=torch.tensor([[7]]), past_key_values=output.past_key_values,
                      use_cache=True)
                self.assertEqual(score.call_count, 1)
            self.assertTrue(all(flag is False for flag in seen))
            keep = model.get_fastv_image_masks()[0][0]["keep"]
            self.assertEqual(sum(keep), 2)
            self.assertEqual(len(model.get_fastv_image_attentions()[0][0]["scores"]), 4)
            self.assertEqual(model.get_fastv_stats()["attention_source"], "independent_qk")
            model.configure_fastv(enabled=False)
            self.assertIsNone(model._fastv_keep_mask)
            self.assertEqual(model._fastv_image_attentions, [])
        finally:
            for handle in handles:
                handle.remove()

    def test_flash_mask_is_2d_preserves_padding_and_keeps_generated_tokens(self):
        model = LlavaQwenModel(tiny_config()).eval()
        model.config._attn_implementation = "flash_attention_2"
        model.configure_fastv(enabled=True, layer=2, keep_ratio=0.5)
        model._fastv_keep_mask = torch.tensor([[True, False, True, False, True]])
        base = torch.tensor([[0, 1, 1, 1, 1]])
        snapshot = base.clone()
        actual = model._apply_fastv_mask(base, 5)
        self.assertEqual(actual.tolist(), [[False, False, True, False, True]])
        self.assertTrue(torch.equal(base, snapshot))
        extended = model._apply_fastv_mask(None, 7)
        self.assertEqual(extended.tolist(), [[True, False, True, False, True, True, True]])


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flash_attn"),
                     "CUDA + flash-attn required; run this test on the inference server")
class FlashCudaTests(unittest.TestCase):
    def test_flash_zero_rate_bitwise_and_nonzero_cache(self):
        torch.manual_seed(13)
        config = tiny_config("flash_attention_2")
        config.torch_dtype = torch.float16
        reference = Qwen2Model(copy.deepcopy(config)).cuda().half().eval()
        model = LlavaQwenModel(copy.deepcopy(config)).cuda().half().eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
        with torch.no_grad():
            expected = reference(input_ids=tokens, use_cache=True)
            actual = model(input_ids=tokens, use_cache=True)
            self.assertTrue(torch.equal(actual.last_hidden_state, expected.last_hidden_state))
            expected = reference(input_ids=tokens[:, -1:], past_key_values=expected.past_key_values)
            actual = model(input_ids=tokens[:, -1:], past_key_values=actual.past_key_values)
            self.assertTrue(torch.equal(actual.last_hidden_state, expected.last_hidden_state))
            model.configure_fastv(enabled=True, layer=2, keep_ratio=0.5,
                                  preserve_image_newline=False)
            model.set_fastv_image_spans([[(1, 5)]])
            output = model(input_ids=tokens, use_cache=True)
            self.assertTrue(torch.isfinite(output.last_hidden_state).all())
            output = model(input_ids=tokens[:, -1:], past_key_values=output.past_key_values)
            self.assertTrue(torch.isfinite(output.last_hidden_state).all())
            self.assertEqual(sum(model.get_fastv_image_masks()[0][0]["keep"]), 2)
            self.assertTrue(all(layer.self_attn.__class__.__name__ == "Qwen2FlashAttention2"
                                for layer in model.layers))


if __name__ == "__main__":
    unittest.main()
