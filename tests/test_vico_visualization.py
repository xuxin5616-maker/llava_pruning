import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from llava_pruning.visualization import (build_attention_overlay, prepare_image_masks,
                                         save_vico_visualizations)


def sample_stages(count):
    previous = count
    result = []
    for layer, after in zip((8, 16, 24), (max(1, count * 3 // 4), max(1, count // 2), max(1, count // 4))):
        result.append({"after_layer": layer, "scoring_layer": layer + 1,
                       "image_tokens_before": previous, "image_tokens_after": after,
                       "removed_this_stage": previous - after,
                       "cumulative_prune_rate": 100 * (1 - after / count),
                       "mask": {"span": [3, 3 + count], "keep": [i < after for i in range(count)]},
                       "attention": {"span": [3, 3 + count],
                                     "scores": [1 / previous if i < previous else 0 for i in range(count)],
                                     "valid": [i < previous for i in range(count)]}})
        previous = after
    return result


class ViCoVisualizationTests(unittest.TestCase):
    def test_anyres_and_randomroi_two_multistage_pngs_only(self):
        cases = [((16, 8), {"mode": "anyres_max_9", "original_size": [16, 8],
                            "grid_patches": [2, 1], "roi_boxes": [], "final_newline": False}, 14),
                 ((8, 8), {"original_size": [8, 8], "roi_boxes": [[0, 0, 4, 4]],
                            "processed_view_sizes": [[8, 8], [8, 8]]}, 9)]
        for size, metadata, count in cases:
            for save_prune, save_attention in ((True, True), (True, False), (False, True)):
                with self.subTest(size=size, prune=save_prune, attention=save_attention), tempfile.TemporaryDirectory() as directory:
                    stages = sample_stages(count)
                    original_stages = copy.deepcopy(stages)
                    paths = save_vico_visualizations(
                        Image.new("RGB", size, "white"), metadata, stages, layer_stats=[{"layer": 1}],
                        base_grid=2, patch_size=4, output_dir=directory, sample_id="000000108",
                        save_prune=save_prune, save_attention=save_attention)
                    folder = Path(paths[0]["directory"])
                    expected = ({"comparison.png"} if save_prune else set()) | ({"attention_overlay.png"} if save_attention else set())
                    self.assertEqual({p.name for p in folder.glob("*.png")}, expected)
                    saved = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
                    self.assertEqual([s["after_layer"] for s in saved["stages"]], [8, 16, 24])
                    self.assertEqual(set(saved), {"sample_id", "method", "stages", "image_token_order"})
                    self.assertEqual(saved["image_token_order"], metadata.get("image_token_order", "base_first"))
                    self.assertEqual(stages, original_stages)
                    for summary, stage in zip(saved["stages"], stages):
                        self.assertNotIn("mask", summary)
                        self.assertNotIn("attention", summary)
                        self.assertEqual(summary["image_tokens_after"], stage["image_tokens_after"])
                        self.assertTrue(all(isinstance(value, (int, float)) for key, value in summary.items()
                                            if key != "views"))
                        for view in summary["views"]:
                            self.assertEqual(set(view), {"name", "token_count", "kept_patch_tokens",
                                                         "pruned_patch_tokens", "prune_rate_percent"})
                    if save_attention:
                        with Image.open(folder / "attention_overlay.png") as image:
                            self.assertGreater(image.height, 3 * 52)
                            self.assertGreaterEqual(image.width, 1000)

    def test_absent_attention_is_gray_not_low_score(self):
        image = Image.new("RGB", (8, 8), "white")
        metadata = {"original_size": [8, 8], "roi_boxes": [], "processed_view_sizes": [[8, 8]]}
        stage = sample_stages(5)[-1]
        stage["attention"]["valid"] = [False] * 5
        prepared = prepare_image_masks(image.size, metadata, stage["mask"], 2, 4)
        overlay = build_attention_overlay(image, stage["attention"], stage["mask"], prepared)
        pixels = np.asarray(overlay)
        self.assertTrue((pixels[-8:, :8] == 96).all())

    def test_non_nested_masks_fail_before_writing_files(self):
        metadata = {"original_size": [8, 8], "roi_boxes": [], "processed_view_sizes": [[8, 8]]}
        stages = sample_stages(5)
        stages[-1]["mask"]["keep"][-1] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "restore previously pruned"):
                save_vico_visualizations(Image.new("RGB", (8, 8)), metadata, stages, layer_stats=[],
                                         base_grid=2, patch_size=4, output_dir=directory, sample_id="one")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
