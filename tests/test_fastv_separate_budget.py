"""Actual AnyRes packing, independent budgets and cached decoding; CPU only."""

import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache

from test_fastv_attention import tiny_config
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.fastv_budget import base_anyres_groups
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenModel
from llava_pruning.methods import FastVMethod


PROJECT = Path(__file__).resolve().parents[1]


class SeparateBudgetTests(unittest.TestCase):
    def test_base_and_anyres_have_independent_topk_even_when_base_scores_dominate(self):
        core = LlavaQwenModel(tiny_config()).eval()
        core.configure_fastv(enabled=True, keep_ratio=.3, budget_mode="base_anyres")
        layout = base_anyres_groups(4, 2, 4)
        core.set_fastv_image_spans([[(1, 15)]])
        core.set_fastv_view_groups([[layout]])
        scores = torch.arange(17, dtype=torch.float32)[None]
        scores[0, 1:5] += 1000  # A joint pool would disproportionately select Base.
        core._build_fastv_keep_mask(scores)
        mask = core._fastv_keep_mask[0]
        self.assertEqual(mask[1:5].tolist(), [False, False, True, True])
        highres = torch.tensor(layout["groups"][1]["indices"]) + 1
        self.assertEqual(highres[mask[highres]].tolist(), [11, 12, 13])
        self.assertTrue(mask[[0, 9, 14, 15, 16]].all())  # text + both row newlines
        stats = core.get_fastv_stats()["batch"][0]
        self.assertEqual([g["kept_tokens"] for g in stats["groups"]], [2, 3])
        self.assertEqual(stats["protected_newline_tokens"], 2)

    def test_fractional_roundoff_does_not_retain_an_extra_token(self):
        core = LlavaQwenModel(tiny_config()).eval()
        core.configure_fastv(enabled=True, keep_ratio=1 - 70 / 100, budget_mode="base_anyres")
        layout = base_anyres_groups(10, 2, 5)
        count = layout["packed_tokens"]
        core.set_fastv_image_spans([[(0, count)]])
        core.set_fastv_view_groups([[layout]])
        core._build_fastv_keep_mask(torch.arange(count, dtype=torch.float32)[None])
        self.assertEqual([g["kept_tokens"] for g in core.get_fastv_stats()["batch"][0]["groups"]], [3, 3])

    def make_model(self, size, grid, order, extra):
        config = tiny_config("sdpa")
        config.image_aspect_ratio = "anyres_max_9"
        config.image_grid_pinpoints = [list(grid)]
        config.mm_patch_merge_type = "spatial_unpad_add_newl" if extra else "spatial_unpad"
        config.image_token_order = order
        config.tokenizer_model_max_length = 4096
        config.tokenizer_padding_side = "right"
        config.use_pos_skipping = False
        model = LlavaQwenForCausalLM(config).eval()
        core = model.get_model()
        core.vision_tower = SimpleNamespace(num_patches_per_side=2, image_size=8)
        core.image_newline = torch.nn.Parameter(torch.full((32,), -777.0))
        view_count = 1 + grid[0] // 8 * (grid[1] // 8)
        features = torch.arange(view_count * 4 * 32, dtype=torch.float32).reshape(view_count, 4, 32) / 100
        core.configure_fastv(enabled=True, keep_ratio=.3, budget_mode="base_anyres")
        with patch.object(model, "encode_images", return_value=features):
            packed = model.prepare_inputs_labels_for_multimodal(
                torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]]), None, None, None, None,
                [torch.zeros(view_count, 3, 8, 8)], image_sizes=[size])[4]
        return model, packed

    def test_real_packer_uses_unpadded_resized_geometry_and_both_token_orders(self):
        torch.manual_seed(54)
        cases = [((16, 8), (16, 8)), ((16, 8), (16, 16)), ((32, 32), (32, 32))]
        for size, grid in cases:
            for order in ("base_first", "anyres_first"):
                for extra in (False, True):
                    with self.subTest(size=size, grid=grid, order=order, extra=extra):
                        model, packed = self.make_model(size, grid, order, extra)
                        core = model.get_model()
                        [(start, end)] = core._fastv_image_spans[0]
                        image = packed[0, start:end]
                        layout = core._fastv_view_groups[0][0]
                        spatial = sorted(i for group in layout["groups"] for i in group["indices"])
                        newlines = (image == -777).all(-1)
                        # Independently identify actual newline feature vectors in the packed tensor.
                        self.assertEqual(newlines.logical_not().nonzero().flatten().tolist(), spatial)
                        self.assertEqual(len(layout["groups"][0]["indices"]), 4)
                        core._build_fastv_keep_mask(torch.arange(packed.shape[1], dtype=torch.float32)[None])
                        local = core._fastv_keep_mask[0, start:end]
                        self.assertTrue(local[newlines].all())
                        for group in layout["groups"]:
                            indices = torch.tensor(group["indices"])
                            self.assertEqual(int(local[indices].sum()), math.ceil(len(indices) * .3 - 1e-10))

    def test_real_forward_ranks_once_and_keeps_full_cache_lengths(self):
        model, packed = self.make_model((16, 8), (16, 8), "base_first", False)
        core = model.get_model()
        from llava.model.language_model.fastv_attention import last_prompt_attention
        with patch("llava.model.language_model.llava_qwen.last_prompt_attention",
                   wraps=last_prompt_attention) as ranking, torch.no_grad():
            prefill = core(inputs_embeds=packed, past_key_values=DynamicCache(), use_cache=True)
            core(input_ids=torch.tensor([[4]]), past_key_values=prefill.past_key_values, use_cache=True)
        self.assertEqual(ranking.call_count, 1)
        self.assertEqual(prefill.past_key_values.get_seq_length(), packed.shape[1] + 1)
        self.assertEqual([g["kept_tokens"] for g in core.get_fastv_stats()["batch"][0]["groups"]], [2, 3])

    def test_mode_switch_and_config_defaults_preserve_original_joint_budget(self):
        method = FastVMethod.from_config(PROJECT / "configs/fastv70_base_anyres.json")
        original = FastVMethod.from_config(PROJECT / "configs/fastv.json")
        core = LlavaQwenModel(tiny_config()).eval()
        method.configure(core, 70, capture_attention=False)
        self.assertEqual(core.fastv_budget_mode, "base_anyres")
        original.configure(core, 70, capture_attention=False)
        self.assertEqual(core.fastv_budget_mode, "joint")
        self.assertEqual(core._fastv_view_groups, [])
        with self.assertRaises(ValueError):
            core.configure_fastv(budget_mode="base_anyres", preserve_image_newline=False)


if __name__ == "__main__":
    unittest.main()
