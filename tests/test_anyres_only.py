"""No-Base ablation: preserve the anyres tiles and their token identities.

Uses real preprocessing, multimodal packing and tiny Qwen generation on CPU;
does not download weights or claim to validate 7B CUDA accuracy/performance.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from run import build_parser
from llava_pruning.backend import configure_image_mode
from llava_pruning.roi import choose_roi
from llava_pruning import visualization as vis
from test_reference import ToyPacking
from test_fastv_attention import tiny_config
from test_vico_visualization import sample_stages
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.language_model.fastv_attention import last_prompt_attention
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor


def metadata(size=(16, 8), grid=(2, 1), final_newline=False):
    return {"mode": "anyres_only", "original_size": list(size), "grid_patches": list(grid),
            "image_token_order": "base_first", "final_newline": final_newline,
            "roi_boxes": [], "processed_view_sizes": [[grid[0] * 8, grid[1] * 8]]}


class AnyresOnlyInputTests(unittest.TestCase):
    def test_cli_and_config_preserve_precision_and_checkpoint_merge(self):
        args = ["--model-path", "checkpoint", "--input-json", "input.jsonl", "--data-root", "data"]
        self.assertEqual(build_parser().parse_args(args + ["--roi-mode", "anyres_only"]).roi_mode, "anyres_only")
        self.assertEqual(build_parser().parse_args(args).roi_mode, "randomroi")
        for merge in ("spatial_unpad", "spatial_unpad_add_newl"):
            config = SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type=merge,
                                     torch_dtype=torch.float16, _attn_implementation="flash_attention_2")
            configure_image_mode(config, "anyres_only")
            self.assertEqual(config.image_aspect_ratio, "anyres_only")
            self.assertEqual(config.mm_patch_merge_type, merge)
            self.assertEqual(config.torch_dtype, torch.float16)
            self.assertEqual(config._attn_implementation, "flash_attention_2")
            with self.assertRaisesRegex(ValueError, "anyres_max_9"):
                configure_image_mode(config, "anyres_only", "anyres_first")
        for merge in ("flat", "spatial_unpad_nobase", "spatial_unpad_ex_base_copy"):
            with self.assertRaisesRegex(ValueError, "anyres_only requires"):
                configure_image_mode(SimpleNamespace(mm_patch_merge_type=merge), "anyres_only")
        self.assertEqual(choose_roi(object(), "anyres_only"), ("anyres_only", None, None))

    def test_pixels_exactly_equal_original_tiles_and_base_is_never_processed(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad",
                                 image_grid_pinpoints=[[28, 28], [56, 28], [28, 56], [56, 56]])
        rng = np.random.RandomState(17)
        for size in ((28, 28), (63, 29), (29, 63), (49, 53)):
            with self.subTest(size=size):
                image = Image.fromarray(rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8))
                before = np.array(image)
                config.image_aspect_ratio = "anyres_max_9"
                reference, ref_metadata = process_images([image], processor, config, return_pro_data=True)
                config.image_aspect_ratio = "anyres_only"
                with patch.object(processor, "preprocess", wraps=processor.preprocess) as preprocess:
                    actual, actual_metadata = process_images([image], processor, config, return_pro_data=True,
                                                              masks=[object()], boxes_list=[object()])
                self.assertTrue(torch.equal(actual, reference[:, 1:]))
                self.assertEqual(preprocess.call_count, reference.shape[1] - 1)
                self.assertGreaterEqual(preprocess.call_count, 1)
                expected_metadata = copy.deepcopy(ref_metadata)
                expected_metadata[0]["mode"] = "anyres_only"
                expected_metadata[0]["processed_view_sizes"] = expected_metadata[0]["processed_view_sizes"][1:]
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(np.asarray(image), before)
                self.assertTrue(torch.equal(process_images([image], processor, config), actual))

    def test_mixed_sizes_use_tile_lists_without_dropping_single_tile_sample(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="anyres_only", mm_patch_merge_type="spatial_unpad",
                                 image_grid_pinpoints=[[28, 28], [56, 28]])
        images = [Image.new("RGB", (28, 28), "red"), Image.new("RGB", (56, 28), "blue")]
        pixels, info = process_images(images, processor, config, return_pro_data=True)
        self.assertIsInstance(pixels, list)
        self.assertEqual([x.shape[0] for x in pixels], [1, 2])
        self.assertEqual([m["grid_patches"] for m in info], [[1, 1], [2, 1]])


class AnyresOnlyPackingTests(unittest.TestCase):
    def pack(self, toy, images, size, modality="image"):
        tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        return toy.prepare_inputs_labels_for_multimodal(
            tokens, torch.arange(4)[None], torch.ones_like(tokens, dtype=torch.bool),
            None, tokens.clone(), images, modalities=[modality], image_sizes=[size])

    def test_exact_packed_token_identity_matches_anyres_with_only_base_removed(self):
        cases = (((8, 8), (1, 1)), ((16, 8), (2, 1)), ((8, 16), (1, 2)),
                 ((23, 11), (3, 2)), ((11, 23), (2, 3)), ((48, 48), (6, 6)))
        for dtype in (torch.float32, torch.float16):
            for size, grid in cases:
                # PyTorch 2.1 CPU bilinear interpolation lacks FP16 support.
                if dtype == torch.float16 and grid == (6, 6):
                    continue
                for final_newline in (False, True):
                    for as_list in (False, True):
                        with self.subTest(dtype=dtype, size=size, grid=grid,
                                          final_newline=final_newline, as_list=as_list):
                            toy = ToyPacking(dtype)
                            toy.config.image_grid_pinpoints = [[grid[0] * 8, grid[1] * 8]]
                            if final_newline:
                                toy.config.mm_patch_merge_type += "_add_newl"
                            count = grid[0] * grid[1]
                            # Feature identities depend on input pixels, not batch indices.
                            patches = torch.arange(16, dtype=dtype).reshape(1, 4, 4) / 20
                            def encode(pixels):
                                return pixels[:, :1, 0, 0, None] + patches
                            tiles = torch.arange(1, count + 1, dtype=dtype)[:, None, None, None].expand(-1, 3, 8, 8)
                            original = torch.cat((torch.full((1, 3, 8, 8), -100, dtype=dtype), tiles))
                            with patch.object(toy, "encode_images", side_effect=encode):
                                before = self.pack(toy, [original] if as_list else original[None], size)
                            configure_image_mode(toy.config, "anyres_only")
                            spans, vico_spans = [], []
                            toy.core.fastv_enabled = True
                            toy.core.set_fastv_image_spans = lambda value: spans.append(value)
                            toy.core.vico_configured = True
                            toy.core.set_vico_image_spans = lambda value, length: vico_spans.append((value, length))
                            with patch.object(toy, "encode_images", side_effect=encode) as encoder:
                                after = self.pack(toy, [tiles] if as_list else tiles[None], size)
                            encoder.assert_called_once()
                            self.assertEqual(encoder.call_args.args[0].shape[0], count)
                            expected = torch.cat((before[4][:, :1], before[4][:, 5:]), dim=1)
                            torch.testing.assert_close(after[4], expected, rtol=0, atol=0)
                            torch.testing.assert_close(after[4][0, -3], toy.core.image_newline, rtol=0, atol=0)
                            length = expected.shape[1]
                            self.assertTrue(torch.equal(after[1][0], torch.arange(length)))
                            self.assertTrue(after[2].all())
                            self.assertTrue(torch.equal(after[5], torch.cat((before[5][:, :1], before[5][:, 5:]), dim=1)))
                            self.assertEqual(spans, [[[(1, length - 2)]]])
                            self.assertEqual(vico_spans, [([[(1, length - 2)]], length)])
                            # Rendering must partition the same real packed token span.
                            info = metadata(size, grid, final_newline)
                            mask = {"span": [1, length - 2], "keep": [True] * (length - 3)}
                            prepared = vis.prepare_image_masks(size, info, mask, 2, 4)
                            self.assertEqual(prepared["sequence_tokens"], length - 3)
                            self.assertEqual(len(prepared["views"]), 1)
                            self.assertEqual(prepared["views"][0]["metadata"]["token_offset"], 0)

    def test_bad_packing_tile_count_order_and_video_are_rejected(self):
        for merge, order, images, size, modality in (
                ("flat", "base_first", torch.zeros(1, 2, 3, 8, 8), (16, 8), "image"),
                ("spatial_unpad", "anyres_first", torch.zeros(1, 2, 3, 8, 8), (16, 8), "image"),
                ("spatial_unpad", "base_first", torch.zeros(1, 3, 8, 8), (16, 8), "image"),
                ("spatial_unpad", "base_first", torch.zeros(1, 2, 3, 8, 8), (16, 8), "video"),
                ("spatial_unpad", "base_first", torch.zeros(1, 3, 3, 8, 8), (16, 8), "image")):
            toy = ToyPacking(torch.float32)
            toy.config.image_aspect_ratio = "anyres_only"
            toy.config.mm_patch_merge_type = merge
            toy.config.image_token_order = order
            with self.subTest(merge=merge, order=order, modality=modality), self.assertRaisesRegex(ValueError, "anyres_only"):
                self.pack(toy, images, size, modality)

    def test_real_qwen_generation_for_both_methods_zero_nonzero_and_single_tile(self):
        for method in ("fastv", "vico"):
            for rate in (0, 50):
                for count in (1, 2):
                    with self.subTest(method=method, rate=rate, count=count):
                        config = tiny_config("sdpa")
                        config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
                        config.mm_patch_merge_type = "spatial_unpad"
                        config.image_grid_pinpoints = [[8, 8], [16, 8]]
                        configure_image_mode(config, "anyres_only")
                        model = LlavaQwenForCausalLM(config).eval()
                        core = model.get_model()
                        core.image_newline = torch.nn.Parameter(torch.zeros(config.hidden_size))
                        tower = SimpleNamespace(num_patches_per_side=2, image_size=8)
                        features = torch.arange(4 * config.hidden_size).reshape(1, 4, config.hidden_size).float() / 128
                        if method == "fastv":
                            core.configure_fastv(enabled=rate != 0, layer=2, keep_ratio=1 - rate / 100,
                                                 capture_attention=True, preserve_image_newline=True)
                        else:
                            core.configure_vico(enabled=rate != 0, layers=[1, 2, 3],
                                                keep_ratios=[(1 - rate / 100) ** (j / 3) for j in (1, 2, 3)],
                                                capture_visualization=True, capture_attention=True)
                        score_path = ("llava.model.language_model.llava_qwen.last_prompt_attention"
                                      if method == "fastv" else "llava.model.language_model.vico.last_prompt_attention")
                        with patch.object(model, "get_vision_tower", return_value=tower), \
                                patch.object(model, "encode_images", side_effect=lambda x: features.repeat(x.shape[0], 1, 1)) as encoder, \
                                patch(score_path, wraps=last_prompt_attention) as score, torch.no_grad():
                            options = dict(inputs=torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]]),
                                           images=torch.zeros(1, count, 3, 8, 8), image_sizes=[(count * 8, 8)],
                                           do_sample=False, max_new_tokens=3, use_cache=True)
                            first, second = model.generate(**options), model.generate(**options)
                        self.assertEqual(first.shape, (1, 3))
                        self.assertTrue(torch.equal(first, second))
                        self.assertTrue(all(call.args[0].shape[0] == count for call in encoder.call_args_list))
                        self.assertEqual(score.call_count, 0 if rate == 0 else (2 if method == "fastv" else 6))
                        if rate:
                            image_tokens = 4 * count + 2  # two row-newlines, no Base
                            if method == "fastv":
                                mask = core.get_fastv_image_masks()[0][0]
                                self.assertEqual(mask["span"], [1, 1 + image_tokens])
                                self.assertTrue(mask["keep"][-1])
                            else:
                                stages = core.get_vico_stages()
                                self.assertEqual(len(stages), 3)
                                self.assertTrue(all(len(s["mask"]["keep"]) == image_tokens for s in stages))


class AnyresOnlyVisualizationTests(unittest.TestCase):
    def test_layout_matches_original_anyres_with_offsets_shifted_and_no_base(self):
        for final in (False, True):
            base = [True, False, True, False]
            tiles = [True, False, True, False, True, False, True, False, True, False] + ([False] if final else [])
            meta = metadata(final_newline=final)
            original_meta = {**meta, "mode": "anyres_max_9"}
            before = vis.prepare_image_masks((16, 8), original_meta,
                                             {"span": [1, 5 + len(tiles)], "keep": base + tiles}, 2, 4)
            after = vis.prepare_image_masks((16, 8), meta, {"span": [1, 1 + len(tiles)], "keep": tiles}, 2, 4)
            self.assertEqual(after["patch_tokens"], 8)
            self.assertEqual(after["newline_tokens"], 2 + final)
            self.assertEqual(after["pruned_newline_tokens"], 1 + final)
            self.assertEqual(after["views"][0]["metadata"]["row_newline_kept"], [True, False])
            np.testing.assert_array_equal(after["views"][0]["pruned"], before["views"][1]["pruned"])
            np.testing.assert_array_equal(after["combined_pruned"], before["views"][1]["pruned"])

    def test_comparison_has_only_original_and_anyres_no_combined_or_global_column(self):
        source = Image.new("RGB", (16, 8), (80, 120, 200))
        keep = [True] * 10
        for index in (0, 1, 5, 6):
            keep[index] = False
        mask = {"span": [1, 11], "keep": keep}
        prepared = vis.prepare_image_masks(source.size, metadata(), mask, 2, 4)
        with patch.object(ImageDraw.ImageDraw, "multiline_text", autospec=True) as text:
            image = vis.build_anyres_only_comparison(source, prepared)
        captions = [call.args[2] for call in text.call_args_list]
        self.assertEqual(image.width, 960)
        self.assertEqual(len(captions), 3)  # two panel labels and footer
        self.assertIn("anyres only", captions[1])
        self.assertIn("4/8 (50.00%)", captions[1])
        self.assertNotIn("Combined", " ".join(captions))
        self.assertIn("No Base view", captions[-1])
        x = (480 - source.width) // 2
        self.assertEqual(image.getpixel((x + 1, 63)), (80, 120, 200))
        self.assertEqual(image.getpixel((480 + x + 1, 63)), (0, 0, 0))

    def test_attention_has_one_view_and_scores_start_at_zero_not_after_base(self):
        source = Image.new("RGB", (16, 8), (80, 120, 200))
        mask = {"span": [1, 11], "keep": [True] * 10}
        prepared = vis.prepare_image_masks(source.size, metadata(), mask, 2, 4)
        attention = {"span": mask["span"], "scores": [1.0] * 10, "valid": [True] * 10}
        attention["valid"][0] = False
        image = vis.build_attention_overlay(source, attention, mask, prepared)
        self.assertEqual(image.size, (16, 50))
        self.assertEqual(image.getpixel((1, 43)), (96, 96, 96))
        self.assertNotEqual(image.getpixel((5, 43)), (96, 96, 96))
        with self.assertRaisesRegex(ValueError, "Expected"):
            vis.prepare_image_masks(source.size, metadata(), {"span": [1, 15], "keep": [True] * 14}, 2, 4)
        with self.assertRaisesRegex(ValueError, "no Base"):
            vis.prepare_image_masks(source.size, {**metadata(), "image_token_order": "anyres_first"}, mask, 2, 4)

    def test_both_methods_save_two_png_types_and_only_anyres_count_summaries(self):
        source = Image.new("RGB", (16, 8), (80, 120, 200))
        for method in ("fastv", "vico"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                common = dict(base_grid=2, patch_size=4, output_dir=directory, sample_id="only_tiles",
                              save_prune=True, save_attention=True, prediction_label="GT: Normal | Pred: Abnormal")
                if method == "fastv":
                    mask = {"span": [1, 11], "keep": [True] * 10}
                    paths = vis.save_fastv_visualizations(
                        [source], [metadata()], [mask], fastv_layer=2, keep_ratio=0.5,
                        image_attentions=[{"span": mask["span"], "scores": [0.1] * 10}], **common)
                else:
                    paths = vis.save_vico_visualizations(source, metadata(), sample_stages(10), layer_stats=[], **common)
                folder = Path(paths[0]["directory"])
                self.assertEqual({p.name for p in folder.glob("*.png")}, {"comparison.png", "attention_overlay.png"})
                saved = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
                for summary in [saved] if method == "fastv" else saved["stages"]:
                    self.assertEqual([v["name"] for v in summary["views"]], ["anyres"])
                    self.assertEqual(summary["sequence_tokens"], 10)
                    self.assertEqual(summary["patch_tokens"], 8)
                    self.assertNotIn("mask", summary)
                    self.assertNotIn("attention", summary)


if __name__ == "__main__":
    unittest.main()
