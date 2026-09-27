import tempfile
import unittest
from pathlib import Path

from PIL import Image

from llava_pruning.visualization import (build_anyres_layout, prepare_image_masks,
                                         save_fastv_visualizations)


class VisualizationTests(unittest.TestCase):
    def test_anyres_layout_and_row_newlines(self):
        image = Image.new("RGB", (16, 8), "white")
        crop = {"mode": "anyres_max_9", "original_size": [16, 8],
                "roi_boxes": [], "grid_patches": [2, 1]}
        layouts = build_anyres_layout(crop, base_grid=2, patch_size=4)
        self.assertEqual(layouts[1]["grid_shape"], [2, 4])
        # Four global patches + two rows of four patches/one newline; spatial_unpad has no final newline.
        mask = {"span": [0, 14], "keep": [True] * 14}
        mask["keep"][4] = False
        prepared = prepare_image_masks(image.size, crop, mask, 2, 4)
        self.assertEqual(prepared["patch_tokens"], 12)
        self.assertEqual(prepared["sequence_tokens"], 14)
        self.assertEqual(prepared["pruned_patch_tokens"], 1)
        self.assertEqual(prepared["views"][1]["metadata"]["row_newline_kept"], [True, True])
        attention = {"span": [0, 14], "scores": [0.1] * 14}
        with tempfile.TemporaryDirectory() as directory:
            saved, = save_fastv_visualizations(
                [image], [crop], [mask], base_grid=2, patch_size=4,
                output_dir=directory, sample_id="anyres", fastv_layer=2,
                keep_ratio=0.9, image_attentions=[attention],
                save_prune=True, save_attention=True,
            )
            self.assertTrue(Path(saved["comparison"]).is_file())
            self.assertTrue(Path(saved["attention_overlay"]).is_file())
            self.assertEqual({path.name for path in Path(saved["directory"]).glob("*.png")},
                             {"comparison.png", "attention_overlay.png"})
            self.assertTrue((Path(saved["directory"]) / "decisions.json").is_file())

    def test_randomroi_saves_only_two_images_even_with_crops(self):
        image = Image.new("RGB", (8, 8), "white")
        crop = {"original_size": [8, 8], "roi_boxes": [[0, 0, 4, 4]],
                "processed_view_sizes": [[8, 8], [8, 8]]}
        mask = {"span": [0, 9], "keep": [True, False] * 4 + [True]}
        attention = {"span": [0, 9], "scores": [0.1] * 9}
        with tempfile.TemporaryDirectory() as directory:
            saved, = save_fastv_visualizations(
                [image], [crop], [mask], base_grid=2, patch_size=4,
                output_dir=directory, sample_id="roi", fastv_layer=2,
                keep_ratio=0.5, image_attentions=[attention],
                save_prune=True, save_attention=True,
            )
            self.assertEqual({path.name for path in Path(directory).rglob("*.png")},
                             {"comparison.png", "attention_overlay.png"})
            self.assertNotIn("blackout", saved)

    def test_anyres_max_nine_downsampling(self):
        crop = {"mode": "anyres_max_9", "original_size": [16, 16],
                "roi_boxes": [], "grid_patches": [4, 4]}
        layouts = build_anyres_layout(crop, base_grid=2, patch_size=4)
        self.assertEqual(layouts[1]["grid_shape"], [6, 6])
        self.assertEqual(layouts[1]["token_count"], 36)

    def test_independent_save_switches(self):
        image = Image.new("RGB", (8, 8), "white")
        crop = {"original_size": [8, 8], "roi_boxes": [],
                "processed_view_sizes": [[8, 8]]}
        mask = {"span": [0, 5], "keep": [True, False, True, False, True]}
        attention = {"span": [0, 5], "scores": [0.1, 0.2, 0.3, 0.4, 0.0]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            prune, = save_fastv_visualizations(
                [image], [crop], [mask], base_grid=2, patch_size=4,
                output_dir=output, sample_id="108", fastv_layer=2,
                keep_ratio=0.5, save_prune=True, save_attention=False,
            )
            self.assertTrue(Path(prune["comparison"]).is_file())
            self.assertIsNone(prune["attention_overlay"])
            self.assertEqual({path.name for path in Path(prune["directory"]).glob("*.png")},
                             {"comparison.png"})
            attention_only, = save_fastv_visualizations(
                [image], [crop], [mask], base_grid=2, patch_size=4,
                output_dir=output, sample_id="108", fastv_layer=2,
                keep_ratio=0.5, image_attentions=[attention],
                save_prune=False, save_attention=True,
            )
            self.assertIsNone(attention_only["comparison"])
            self.assertTrue(Path(attention_only["attention_overlay"]).is_file())
            self.assertTrue(Path(attention_only["directory"]).name == "image_0")
            self.assertEqual({path.name for path in Path(attention_only["directory"]).glob("*.png")},
                             {"attention_overlay.png"})


if __name__ == "__main__":
    unittest.main()
