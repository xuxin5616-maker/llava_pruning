"""Base-copy pixels, real multimodal packing/decoder integration and projections.

CPU tests use tiny deterministic vision features; no weights are downloaded.
They do not claim to validate 7B accuracy or CUDA/FlashAttention performance.
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
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.mm_utils import process_images
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor


def metadata(size=(16, 8), processed=8):
    return {"mode": "ex_base_copy", "original_size": list(size), "roi_boxes": [],
            "image_token_order": "base_first", "base_view_count": 3,
            "final_newline": True, "processed_view_sizes": [[processed, processed]] * 3}


def image_mask(removed=(), count=13):
    return {"span": [1, 1 + count], "keep": [i not in removed for i in range(count)]}


class BaseCopyInputTests(unittest.TestCase):
    def test_cli_accepts_mode_and_defaults_do_not_change(self):
        args = ["--model-path", "checkpoint", "--input-json", "input.jsonl", "--data-root", "dataset"]
        self.assertEqual(build_parser().parse_args(args).roi_mode, "randomroi")
        self.assertEqual(build_parser().parse_args(args + ["--roi-mode", "ex_base_copy"]).roi_mode,
                         "ex_base_copy")

    def test_config_changes_only_image_fields_and_rejects_anyres_order(self):
        original = SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad",
                                   torch_dtype=torch.float16, _attn_implementation="flash_attention_2")
        config = copy.copy(original)
        configure_image_mode(config, "ex_base_copy")
        self.assertEqual(config.image_aspect_ratio, "ex_base_copy")
        self.assertEqual(config.mm_patch_merge_type, "spatial_unpad_ex_base_copy")
        self.assertEqual(config.image_token_order, "base_first")
        self.assertEqual(config.torch_dtype, original.torch_dtype)
        self.assertEqual(config._attn_implementation, original._attn_implementation)
        for mode, aspect, merge in (("anyres_max_9", "anyres_max_9", "spatial_unpad"),
                                    ("randomroi", "randomroi", "spatial_avgpool_auto_unpad_add_newl"),
                                    ("randompatch", "randomroi", "spatial_avgpool_auto_unpad_add_newl")):
            config = copy.copy(original)
            configure_image_mode(config, mode)
            self.assertEqual((config.image_aspect_ratio, config.mm_patch_merge_type), (aspect, merge))
        with self.assertRaisesRegex(ValueError, "anyres_max_9"):
            configure_image_mode(copy.copy(original), "ex_base_copy", "anyres_first")
        with self.assertRaisesRegex(ValueError, "Unknown ROI mode"):
            configure_image_mode(copy.copy(original), "bad_mode")

    def test_annotation_source_is_ignored(self):
        # Not even sample.mask/bbox attributes are accessed in this mode.
        self.assertEqual(choose_roi(object(), "ex_base_copy"), ("ex_base_copy", None, None))

    def test_all_three_pixels_equal_existing_anyres_base_for_all_aspect_ratios(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad",
                                 image_grid_pinpoints=[[28, 28], [56, 28], [28, 56], [56, 56]])
        rng = np.random.RandomState(11)
        for width, height in ((28, 28), (63, 29), (29, 63), (53, 49)):
            with self.subTest(size=(width, height)):
                config.image_aspect_ratio = "anyres_max_9"
                image = Image.fromarray(rng.randint(0, 256, (height, width, 3), dtype=np.uint8))
                source = np.array(image)
                reference = process_images([image], processor, config)
                config.image_aspect_ratio = "ex_base_copy"
                # Nonexistent annotation objects and no grid lookup must be harmless.
                with patch("llava.mm_utils.process_anyres_image", side_effect=AssertionError("anyres")), \
                        patch("llava.mm_utils.process_randomroi_image", side_effect=AssertionError("ROI")):
                    actual, info = process_images([image], processor, config, masks=[object()],
                                                  boxes_list=[object()], return_pro_data=True)
                self.assertEqual(tuple(actual.shape), (1, 3, 3, 28, 28))
                for index in range(3):
                    self.assertTrue(torch.equal(actual[0, index], reference[0, 0]))
                self.assertEqual(info, [metadata(image.size, 28)])
                np.testing.assert_array_equal(np.array(image), source)
                self.assertTrue(torch.equal(process_images([image], processor, config), actual))

    def test_multiple_samples_each_receive_three_independent_views(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="ex_base_copy")  # no grid_pinpoints needed
        images = [Image.new("RGB", (29, 63), "red"), Image.new("RGB", (63, 29), "blue")]
        actual, info = process_images(images, processor, config, return_pro_data=True)
        self.assertEqual(tuple(actual.shape), (2, 3, 3, 28, 28))
        self.assertFalse(torch.equal(actual[0], actual[1]))
        for index in range(2):
            self.assertTrue(torch.equal(actual[index, 0], actual[index, 2]))
            self.assertEqual(info[index]["original_size"], list(images[index].size))


class BaseCopyPackingTests(unittest.TestCase):
    def pack(self, toy, images, size=(16, 8), modality="image"):
        tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        return toy.prepare_inputs_labels_for_multimodal(
            tokens, torch.arange(4)[None], torch.ones_like(tokens, dtype=torch.bool),
            None, tokens.clone(), images, modalities=[modality], image_sizes=[size])

    def test_preserves_all_three_feature_sequences_newline_text_positions_and_spans(self):
        for dtype in (torch.float16, torch.float32):
            for as_list in (False, True):
                for size in ((16, 8), (8, 16), (8, 8)):
                    with self.subTest(dtype=dtype, as_list=as_list, size=size):
                        toy = ToyPacking(dtype)
                        configure_image_mode(toy.config, "ex_base_copy")
                        spans, vico_spans = [], []
                        toy.core.fastv_enabled = True
                        toy.core.set_fastv_image_spans = lambda value: spans.append(value)
                        toy.core.vico_configured = True
                        toy.core.set_vico_image_spans = lambda value, length: vico_spans.append((value, length))
                        pixels = torch.zeros(3, 3, 8, 8, dtype=dtype)
                        with patch.object(toy, "encode_images", wraps=toy.encode_images) as encode:
                            packed = self.pack(toy, [pixels] if as_list else pixels[None], size)
                        encode.assert_called_once()
                        self.assertEqual(encode.call_args.args[0].shape[0], 3)
                        expected = torch.cat((toy.encode_images(pixels).flatten(0, 1),
                                              toy.core.image_newline[None]))
                        torch.testing.assert_close(packed[4][0, 1:14], expected, rtol=0, atol=0)
                        torch.testing.assert_close(packed[4][0, [0, 14, 15]],
                                                   toy.core.embed_tokens(torch.tensor([1, 2, 3])), rtol=0, atol=0)
                        self.assertTrue(torch.equal(packed[1][0], torch.arange(16)))
                        self.assertTrue(packed[2].all())
                        self.assertEqual(packed[5][0].tolist(), [1] + [IGNORE_INDEX] * 13 + [2, 3])
                        self.assertEqual(spans, [[[(1, 14)]]])
                        self.assertEqual(vico_spans, [([[(1, 14)]], 16)])

    def test_invalid_inputs_fail_instead_of_silently_using_anyres_or_single_image_path(self):
        for count in (1, 2, 4):
            toy = ToyPacking(torch.float32)
            configure_image_mode(toy.config, "ex_base_copy")
            with self.assertRaisesRegex(ValueError, "exactly three"):
                self.pack(toy, torch.zeros(1, count, 3, 8, 8))
        for bad_field, value in (("image_token_order", "anyres_first"),
                                 ("mm_patch_merge_type", "spatial_unpad")):
            toy = ToyPacking(torch.float32)
            configure_image_mode(toy.config, "ex_base_copy")
            setattr(toy.config, bad_field, value)
            with self.assertRaisesRegex(ValueError, "ex_base_copy requires"):
                self.pack(toy, torch.zeros(1, 3, 3, 8, 8))
        toy = ToyPacking(torch.float32)
        configure_image_mode(toy.config, "ex_base_copy")
        with self.assertRaisesRegex(ValueError, "ex_base_copy requires"):
            self.pack(toy, torch.zeros(1, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, "ex_base_copy requires"):
            self.pack(toy, torch.zeros(1, 3, 3, 8, 8), modality="video")

    def test_real_generate_uses_three_views_with_both_methods_and_zero_rate(self):
        for method in ("fastv", "vico"):
            for rate in (0, 50):
                with self.subTest(method=method, rate=rate):
                    config = tiny_config("sdpa")
                    config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
                    configure_image_mode(config, "ex_base_copy")
                    model = LlavaQwenForCausalLM(config).eval()
                    core = model.get_model()
                    core.image_newline = torch.nn.Parameter(torch.zeros(config.hidden_size))
                    tower = SimpleNamespace(num_patches_per_side=2, image_size=8)
                    features = torch.arange(4 * config.hidden_size).reshape(1, 4, config.hidden_size).float() / 128
                    if method == "fastv":
                        core.configure_fastv(enabled=rate != 0, layer=2, keep_ratio=1 - rate / 100,
                                             capture_attention=True, preserve_image_newline=True)
                    else:
                        final = 1 - rate / 100
                        core.configure_vico(enabled=rate != 0, layers=[1, 2, 3],
                                            keep_ratios=[final ** (j / 3) for j in (1, 2, 3)],
                                            capture_visualization=True, capture_attention=True)
                    tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                    score_path = ("llava.model.language_model.llava_qwen.last_prompt_attention"
                                  if method == "fastv" else "llava.model.language_model.vico.last_prompt_attention")
                    from llava.model.language_model.fastv_attention import last_prompt_attention
                    with patch.object(model, "get_vision_tower", return_value=tower), \
                            patch.object(model, "encode_images", side_effect=lambda x: features.repeat(x.shape[0], 1, 1)) as encode, \
                            patch(score_path, wraps=last_prompt_attention) as score, torch.no_grad():
                        options = dict(inputs=tokens, images=torch.zeros(1, 3, 3, 8, 8),
                                       image_sizes=[(16, 8)], do_sample=False, max_new_tokens=3, use_cache=True)
                        first = model.generate(**options)
                        second = model.generate(**options)
                    self.assertEqual(first.shape, (1, 3))
                    self.assertTrue(torch.equal(first, second))
                    self.assertTrue(all(call.args[0].shape[0] == 3 for call in encode.call_args_list))
                    self.assertEqual(score.call_count, 0 if rate == 0 else (2 if method == "fastv" else 6))
                    if rate:
                        if method == "fastv":
                            masks = core.get_fastv_image_masks()[0]
                            self.assertEqual(masks[0]["span"], [1, 14])
                            self.assertTrue(masks[0]["keep"][-1])
                        else:
                            stages = core.get_vico_stages()
                            self.assertEqual([s["image_tokens_after"] for s in stages], [11, 9, 7])
                            self.assertTrue(all(len(s["mask"]["keep"]) == 13 for s in stages))


class BaseCopyVisualizationTests(unittest.TestCase):
    def test_offsets_counts_overlap_and_structural_newline(self):
        # Half of Base 2 removed alone must not darken the combined panel.
        prepared = vis.prepare_image_masks((16, 8), metadata(), image_mask((4, 6, 12)), 2, 4)
        self.assertEqual([v["metadata"]["token_offset"] for v in prepared["views"]], [0, 4, 8])
        self.assertEqual([v["metadata"]["pruned_patch_tokens"] for v in prepared["views"]], [0, 2, 0])
        self.assertEqual(prepared["patch_tokens"], 12)
        self.assertEqual(prepared["sequence_tokens"], 13)
        self.assertEqual(prepared["pruned_newline_tokens"], 1)
        self.assertFalse(prepared["combined_pruned"].any())
        prepared = vis.prepare_image_masks((16, 8), metadata(), image_mask((0, 2, 4, 6, 8, 10)), 2, 4)
        self.assertTrue(prepared["combined_pruned"][:, :8].all())
        self.assertFalse(prepared["combined_pruned"][:, 8:].any())

    def test_siglip_384_counts_and_uncovered_patch_margins(self):
        count = 3 * 27 * 27 + 1
        prepared = vis.prepare_image_masks((384, 384), metadata((384, 384), 384),
                                           image_mask(range(count), count), 27, 14)
        self.assertEqual(prepared["sequence_tokens"], 2188)
        self.assertEqual(prepared["patch_tokens"], 2187)
        self.assertEqual([v["metadata"]["token_count"] for v in prepared["views"]], [729] * 3)
        self.assertTrue(prepared["combined_pruned"][:378, :378].all())
        self.assertFalse(prepared["covered"][378:, :].any())
        self.assertFalse(prepared["covered"][:, 378:].any())

    def test_bad_metadata_or_truncated_tokens_are_rejected(self):
        for bad in ({"base_view_count": 4}, {"processed_view_sizes": [[8, 8]] * 2},
                    {"processed_view_sizes": [[8, 8], [8, 8], [9, 9]]},
                    {"final_newline": False}, {"image_token_order": "anyres_first"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                vis.prepare_image_masks((16, 8), {**metadata(), **bad}, image_mask(), 2, 4)
        with self.assertRaisesRegex(ValueError, "Expected"):
            vis.prepare_image_masks((16, 8), metadata(), image_mask(count=12), 2, 4)

    def test_attention_does_not_stretch_patch_support_into_siglip_margins(self):
        image = Image.new("RGB", (384, 384), (80, 120, 200))
        count = 2188
        mask = image_mask(count=count)
        prepared = vis.prepare_image_masks(image.size, metadata(image.size, 384), mask, 27, 14)
        attention = {"span": mask["span"], "scores": [1.0] * count, "valid": [False] * count}
        panel = vis.build_attention_overlay(image, attention, mask, prepared)
        pixels = np.asarray(panel)
        for index in range(3):
            column = pixels[418:802, index * 392:index * 392 + 384]
            self.assertTrue((column[:378, :378] == 96).all())
            np.testing.assert_array_equal(column[378:, :], np.asarray(image)[378:, :])
            np.testing.assert_array_equal(column[:, 378:], np.asarray(image)[:, 378:])

    def test_comparison_has_three_distinct_views_and_unmodified_original(self):
        image = Image.new("RGB", (16, 8), (80, 120, 200))
        prepared = vis.prepare_image_masks(image.size, metadata(), image_mask((4, 6)), 2, 4)
        with patch.object(ImageDraw.ImageDraw, "multiline_text", autospec=True) as text:
            panel = vis.build_ex_base_copy_comparison(image, prepared)
        captions = [call.args[2] for call in text.call_args_list]
        self.assertEqual(panel.width, 5 * 280)
        for index in (1, 2, 3):
            self.assertTrue(captions[index].startswith(f"Base {index}\n"))
        self.assertIn("2/4 (50.00%)", captions[2])
        self.assertIn("ALL views", captions[4])
        # Pixel centers in each displayed view: Base 2 alone is black on the left.
        x = (280 - 16) // 2
        for column in range(5):
            expected = (0, 0, 0) if column == 2 else (80, 120, 200)
            self.assertEqual(panel.getpixel((column * 280 + x + 1, 63)), expected)

    def test_attention_uses_each_copys_own_scores_and_gray_for_missing_tokens(self):
        image = Image.new("RGB", (16, 8), (80, 120, 200))
        mask = image_mask()
        prepared = vis.prepare_image_masks(image.size, metadata(), mask, 2, 4)
        attention = {"span": mask["span"], "scores": [0] * 4 + [0.5] * 4 + [1] * 4 + [1000],
                     "valid": [True] * 8 + [False] * 4 + [True]}
        panel = vis.build_attention_overlay(image, attention, mask, prepared)
        # Structural newline score does not control the spatial normalization.
        self.assertNotEqual(panel.getpixel((1, 43)), panel.getpixel((17, 43)))
        self.assertEqual(panel.getpixel((33, 43)), (96, 96, 96))
        self.assertEqual(panel.getpixel((1, 27)), panel.getpixel((17, 27)))
        self.assertEqual(panel.getpixel((17, 27)), panel.getpixel((33, 27)))

    def test_save_both_methods_two_pngs_only_with_three_scalar_view_summaries(self):
        image = Image.new("RGB", (16, 8), (80, 120, 200))
        for method in ("fastv", "vico"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                if method == "fastv":
                    mask = image_mask((4, 6))
                    paths = vis.save_fastv_visualizations(
                        [image], [metadata()], [mask], base_grid=2, patch_size=4, output_dir=directory,
                        sample_id="copy", fastv_layer=2, keep_ratio=0.5,
                        image_attentions=[{"span": mask["span"], "scores": [0.1] * 13}],
                        save_prune=True, save_attention=True, prediction_label="GT: Normal | Pred: Abnormal")
                else:
                    paths = vis.save_vico_visualizations(
                        image, metadata(), sample_stages(13), layer_stats=[], base_grid=2, patch_size=4,
                        output_dir=directory, sample_id="copy", save_prune=True, save_attention=True,
                        prediction_label="GT: Normal | Pred: Abnormal")
                folder = Path(paths[0]["directory"])
                self.assertEqual({p.name for p in folder.glob("*.png")}, {"comparison.png", "attention_overlay.png"})
                saved = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
                summaries = [saved] if method == "fastv" else saved["stages"]
                for summary in summaries:
                    self.assertEqual([v["name"] for v in summary["views"]], ["Base 1", "Base 2", "Base 3"])
                    self.assertEqual([v["token_count"] for v in summary["views"]], [4, 4, 4])
                    self.assertNotIn("mask", summary)
                    self.assertNotIn("attention", summary)


if __name__ == "__main__":
    unittest.main()
