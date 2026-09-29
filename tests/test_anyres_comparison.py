import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from llava_pruning import visualization


class AnyresComparisonTests(unittest.TestCase):
    """Check the distinction between per-view token decisions and overlap."""

    def setUp(self):
        self.image = Image.new("RGB", (16, 8), (80, 120, 200))
        self.metadata = {
            "mode": "anyres_max_9", "original_size": [16, 8],
            "grid_patches": [2, 1], "final_newline": False,
        }
        # Four global tokens, then two rows of four spatial tokens + newline.
        self.global_indices = (0, 1, 2, 3)
        self.highres_indices = (4, 5, 6, 7, 9, 10, 11, 12)
        self.left_highres_indices = (4, 5, 9, 10)

    def mask(self, removed=(), *, final_newline=False):
        count = 14 + int(final_newline)
        keep = [True] * count
        for index in removed:
            keep[index] = False
        return {"span": [3, 3 + count], "keep": keep}

    def prepare(self, removed=(), *, final_newline=False):
        return visualization.prepare_image_masks(
            self.image.size, {**self.metadata, "final_newline": final_newline},
            self.mask(removed, final_newline=final_newline), 2, 4,
        )

    def assert_view_counts(self, views, global_pruned, highres_pruned):
        self.assertEqual([view["name"] for view in views], ["global", "anyres"])
        for view, total, removed in zip(views, (4, 8), (global_pruned, highres_pruned)):
            self.assertEqual(view["token_count"], total)
            self.assertEqual(view["pruned_patch_tokens"], removed)
            self.assertEqual(view["kept_patch_tokens"], total - removed)
            self.assertAlmostEqual(view["prune_rate_percent"], 100 * removed / total)

    def assert_compact_saved_views(self, views):
        for view in views:
            self.assertEqual(set(view), {"name", "token_count", "kept_patch_tokens",
                                         "pruned_patch_tokens", "prune_rate_percent"})

    def capture_comparison(self, prepared):
        panels = []
        original_paste = Image.Image.paste

        def capture_paste(canvas, image, *args, **kwargs):
            panels.append(np.array(image))
            return original_paste(canvas, image, *args, **kwargs)

        with patch.object(Image.Image, "paste", autospec=True, side_effect=capture_paste), \
                patch.object(ImageDraw.ImageDraw, "multiline_text", autospec=True) as text:
            comparison = visualization.build_anyres_comparison(self.image, prepared)
        return comparison, panels, [call.args[2] for call in text.call_args_list]

    def test_each_view_remains_visible_when_overlap_hides_its_pruning(self):
        left_half = np.zeros((8, 16), dtype=bool)
        left_half[:, :8] = True
        cases = (
            (self.left_highres_indices, False, True, False, 0, 4),
            ((0, 2), True, False, False, 2, 0),
            ((0, 2) + self.left_highres_indices, True, True, True, 2, 4),
        )
        for removed, global_black, highres_black, combined_black, global_count, highres_count in cases:
            with self.subTest(removed=removed):
                prepared = self.prepare(removed)
                self.assert_view_counts(
                    [view["metadata"] for view in prepared["views"]], global_count, highres_count,
                )
                comparison, panels, captions = self.capture_comparison(prepared)
                self.assertIsInstance(comparison, Image.Image)
                self.assertEqual(len(panels), 4)
                np.testing.assert_array_equal(panels[0], np.asarray(self.image))
                for panel, should_black in zip(panels[1:], (global_black, highres_black, combined_black)):
                    expected = np.array(self.image)
                    if should_black:
                        expected[left_half] = 0
                    np.testing.assert_array_equal(panel, expected)
                np.testing.assert_array_equal(prepared["combined_pruned"], left_half & combined_black)
                self.assertIn("Global", captions[1])
                self.assertIn("anyres", captions[2].lower())
                self.assertIn("not the token pruning rate", captions[3].lower())
        np.testing.assert_array_equal(np.asarray(self.image), np.full((8, 16, 3), (80, 120, 200)))

    def test_row_and_final_newlines_are_excluded_from_per_view_rates(self):
        for final_newline in (False, True):
            with self.subTest(final_newline=final_newline):
                newlines = (8, 13, 14) if final_newline else (8, 13)
                prepared = self.prepare(newlines, final_newline=final_newline)
                self.assertEqual(prepared["patch_tokens"], 12)
                self.assertEqual(prepared["pruned_patch_tokens"], 0)
                self.assertEqual(prepared["sequence_tokens"], 12 + len(newlines))
                self.assertEqual(prepared["pruned_sequence_tokens"], len(newlines))
                self.assertEqual(prepared["newline_tokens"], len(newlines))
                self.assertEqual(prepared["pruned_newline_tokens"], len(newlines))
                self.assertIs(prepared["image_newline_kept"], False if final_newline else None)
                self.assertEqual(prepared["views"][1]["metadata"]["row_newline_kept"], [False, False])
                self.assert_view_counts([view["metadata"] for view in prepared["views"]], 0, 0)
                self.assertFalse(prepared["combined_pruned"].any())
                for view in prepared["views"]:
                    self.assertFalse(view["pruned"].any())

    def test_zero_and_full_spatial_pruning_have_exact_rates_and_masks(self):
        spatial = self.global_indices + self.highres_indices
        for removed, global_count, highres_count, percentage in (((), 0, 0, "0.00%"), (spatial, 4, 8, "100.00%")):
            with self.subTest(removed=removed):
                prepared = self.prepare(removed, final_newline=True)
                self.assert_view_counts([view["metadata"] for view in prepared["views"]], global_count, highres_count)
                self.assertEqual(prepared["pruned_sequence_tokens"], len(removed))
                self.assertEqual(prepared["pruned_newline_tokens"], 0)
                self.assertIs(prepared["image_newline_kept"], True)
                _, panels, captions = self.capture_comparison(prepared)
                for panel in panels[1:]:
                    np.testing.assert_array_equal(panel, np.zeros_like(panel) if removed else np.asarray(self.image))
                for caption, count, total in zip(captions[1:3], (global_count, highres_count), (4, 8)):
                    self.assertIn(f"{count}/{total}", caption)
                    self.assertIn(percentage, caption)

    def test_captions_show_separate_spatial_and_sequence_denominators(self):
        prepared = self.prepare(self.left_highres_indices + (8, 13), final_newline=True)
        _, _, captions = self.capture_comparison(prepared)
        self.assertIn("0/4", captions[1])
        self.assertIn("0.00%", captions[1])
        self.assertIn("4/8", captions[2])
        self.assertIn("50.00%", captions[2])
        self.assertIn("6/15", captions[-1])
        self.assertIn("40.00%", captions[-1])
        self.assertIn("4/12", captions[-1])
        self.assertIn("2/3", captions[-1])

    def test_fastv_saves_two_pngs_and_auditable_view_counts(self):
        mask = self.mask(self.left_highres_indices + (8, 13))
        attention = {"span": mask["span"], "scores": [0.1] * 14}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(visualization, "build_anyres_comparison", wraps=visualization.build_anyres_comparison) as build:
            saved, = visualization.save_fastv_visualizations(
                [self.image], [self.metadata], [mask], base_grid=2, patch_size=4,
                output_dir=directory, sample_id="anyres", fastv_layer=2, keep_ratio=8 / 14,
                image_attentions=[attention], save_prune=True, save_attention=True,
            )
            build.assert_called_once()
            folder = Path(saved["directory"])
            self.assertEqual({path.name for path in folder.glob("*.png")}, {"comparison.png", "attention_overlay.png"})
            metadata = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
            self.assert_view_counts(metadata["views"], 0, 4)
            self.assert_compact_saved_views(metadata["views"])
            self.assertEqual(metadata["method"], "fastv")
            self.assertNotIn("image_span", metadata)
            self.assertEqual(metadata["sequence_tokens"], 14)
            self.assertEqual(metadata["pruned_sequence_tokens"], 6)
            self.assertEqual(metadata["pruned_patch_tokens"], 4)
            self.assertEqual(metadata["newline_tokens"], 2)
            self.assertEqual(metadata["pruned_newline_tokens"], 2)

    def test_vico_saves_per_stage_counts_and_two_pngs(self):
        metadata = {**self.metadata, "final_newline": True}
        stages = []
        previous = [True] * 15
        removals = (self.left_highres_indices + (8, 13),
                    self.left_highres_indices + (0, 2, 6, 11, 8, 13, 14))
        for layer, removed in zip((8, 16), removals):
            mask = self.mask(removed, final_newline=True)
            before, after = sum(previous), sum(mask["keep"])
            stages.append({
                "after_layer": layer, "scoring_layer": layer + 1,
                "image_tokens_before": before, "image_tokens_after": after,
                "removed_this_stage": before - after,
                "cumulative_prune_rate": 100 * (1 - after / 15), "mask": mask,
                "attention": {"span": mask["span"], "valid": previous,
                              "scores": [0.1 if keep else 0.0 for keep in previous]},
            })
            previous = mask["keep"]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(visualization, "build_anyres_comparison", wraps=visualization.build_anyres_comparison) as build:
            saved, = visualization.save_vico_visualizations(
                self.image, metadata, stages, layer_stats=[], base_grid=2, patch_size=4,
                output_dir=directory, sample_id="stages", save_prune=True, save_attention=True,
            )
            self.assertEqual(build.call_count, 2)
            folder = Path(saved["directory"])
            self.assertEqual({path.name for path in folder.glob("*.png")}, {"comparison.png", "attention_overlay.png"})
            decisions = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
            for stage, expected in zip(decisions["stages"], ((0, 4, 2), (2, 6, 3))):
                global_count, highres_count, newline_count = expected
                self.assert_view_counts(stage["views"], global_count, highres_count)
                self.assert_compact_saved_views(stage["views"])
                self.assertNotIn("mask", stage)
                self.assertNotIn("attention", stage)
                self.assertEqual(stage["sequence_tokens"], 15)
                self.assertEqual(stage["patch_tokens"], 12)
                self.assertEqual(stage["newline_tokens"], 3)
                self.assertEqual(stage["pruned_patch_tokens"], global_count + highres_count)
                self.assertEqual(stage["pruned_newline_tokens"], newline_count)
                self.assertEqual(stage["pruned_sequence_tokens"], global_count + highres_count + newline_count)
                self.assertEqual(stage["pruned_sequence_tokens"], 15 - stage["image_tokens_after"])


if __name__ == "__main__":
    unittest.main()
