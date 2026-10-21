"""Real multimodal packer and LlavaQwen generation with synthetic vision features."""

import types
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from test_reference import ToyPacking
from test_pyramiddrop import tiny_config
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM


class PyramidDropPackingTests(unittest.TestCase):
    def test_shared_visualizer_labels_pyramiddrop_and_saves_only_two_png_types(self):
        from PIL import Image
        from llava_pruning.visualization import save_vico_visualizations
        from test_vico_visualization import sample_stages

        metadata = {"mode": "anyres_max_9", "original_size": [16, 8],
                    "grid_patches": [2, 1], "roi_boxes": [], "final_newline": False}
        with tempfile.TemporaryDirectory() as directory:
            paths = save_vico_visualizations(
                Image.new("RGB", (16, 8), "white"), metadata, sample_stages(14),
                layer_stats=[], base_grid=2, patch_size=4, output_dir=directory,
                sample_id="sample", save_prune=True, save_attention=True,
                prediction_label="GT: Normal | Pred: Abnormal", method_name="PyramidDrop")
            folder = Path(paths[0]["directory"])
            self.assertEqual({path.name for path in folder.glob("*.png")},
                             {"comparison.png", "attention_overlay.png"})
            saved = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["method"], "pyramiddrop")
            self.assertEqual(len(saved["stages"]), 3)

    def test_switching_pruning_methods_clears_pyramiddrop_state(self):
        model = LlavaQwenForCausalLM(tiny_config("sdpa")).eval()
        core = model.get_model()
        options = dict(enabled=True, layers=(1, 2, 3), keep_ratios=(.75, .5, .25))
        core.configure_pyramiddrop(**options)
        core.set_pyramiddrop_image_spans([[(1, 9)]], 11)
        core.configure_fastv(enabled=False)
        self.assertFalse(core.pyramiddrop_enabled)
        self.assertFalse(core.pyramiddrop_configured)
        self.assertEqual(core._pyramiddrop_image_spans, [])
        core.configure_pyramiddrop(**options)
        core.configure_vico(**options)
        self.assertFalse(core.pyramiddrop_enabled)
        self.assertFalse(core.pyramiddrop_configured)
        self.assertTrue(core.vico_enabled)
        core.configure_pyramiddrop(**options)
        self.assertFalse(core.vico_enabled)
        self.assertFalse(core.vico_configured)

    def test_real_packer_reports_actual_dense_spans_and_clipped_length(self):
        for limit, expected_end, expected_length in ((128, 15, 17), (10, 10, 10)):
            with self.subTest(limit=limit):
                toy = ToyPacking(torch.float32)
                toy.config.tokenizer_model_max_length = limit
                toy.core.pyramiddrop_configured = True
                captured = []
                toy.core.set_pyramiddrop_image_spans = lambda spans, length: captured.append((spans, length))
                tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                result = toy.prepare_inputs_labels_for_multimodal(
                    tokens, None, None, None, None, [torch.zeros(3, 3, 8, 8)],
                    modalities=["image"], image_sizes=[(16, 8)])
                self.assertEqual(captured, [([[(1, expected_end)]], expected_length)])
                self.assertEqual(result[4].shape[1], expected_length)

    def test_actual_llava_generate_packs_then_prunes_and_resets_between_samples(self):
        torch.manual_seed(552)
        model = LlavaQwenForCausalLM(tiny_config("sdpa")).eval()
        core = model.get_model()
        model.config.image_aspect_ratio = "anyres_max_9"
        model.config.mm_patch_merge_type = "spatial_unpad"
        model.config.image_grid_pinpoints = [[8, 8], [16, 8]]
        model.config.tokenizer_model_max_length = 128
        model.config.tokenizer_padding_side = "right"
        model.config.use_pos_skipping = False
        core.vision_tower = SimpleNamespace(num_patches_per_side=2, image_size=8)
        core.image_newline = nn.Parameter(torch.arange(32, dtype=torch.float32) / 32)
        features = torch.randn(3, 4, 32)
        model.encode_images = types.MethodType(lambda self, images: features[:images.shape[0]], model)
        core.configure_pyramiddrop(enabled=True, layers=(1, 2, 3), keep_ratios=(0.75, 0.5, 0.25), capture_stages=True)
        inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        options = dict(inputs=inputs, images=[torch.zeros(3, 3, 8, 8)], image_sizes=[(16, 8)],
                       do_sample=False, use_cache=True, max_new_tokens=3, pad_token_id=0)
        first = model.generate(**options)
        stats = core.get_pyramiddrop_stats()
        self.assertEqual(stats["original_image_tokens"], 14)
        self.assertEqual(stats["original_prompt_tokens"], 17)
        self.assertEqual([row["image_tokens_after"] for row in stats["stages"]], [10, 7, 3])
        self.assertEqual([row["image_tokens_in"] for row in stats["layers"]], [14, 10, 7, 3])
        self.assertEqual(core.get_pyramiddrop_stages()[0]["mask"]["span"], [1, 15])
        self.assertEqual(first.shape[-1], 3)
        self.assertTrue(torch.equal(first, model.generate(**options)))
        self.assertEqual(len(core.get_pyramiddrop_stats()["stages"]), 3)
        core.configure_pyramiddrop(enabled=False, layers=(1, 2, 3), keep_ratios=(1, 1, 1))
        zero_output = model.generate(**options)
        self.assertEqual(core.get_pyramiddrop_stats()["mode"], "disabled_baseline")
        self.assertEqual(core.get_pyramiddrop_stats()["original_image_tokens"], 14)
        core.configure_fastv(enabled=False)
        native_output = model.generate(**options)
        self.assertTrue(torch.equal(zero_output, native_output))


if __name__ == "__main__":
    unittest.main()
