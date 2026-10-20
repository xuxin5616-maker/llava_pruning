"""Read-only geometry diagnostics; do not change the model/preprocessor output."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import runVisionZip as vz
from llava import mm_utils
from llava.model.llava_arch import unpad_image
import test_visionzip as visionzip_tests


def metadata(size, grid):
    return {"original_size": list(size), "grid_patches": list(grid),
            "processed_view_sizes": [[384, 384], [grid[0] * 384, grid[1] * 384]]}


class PaddingAuditTests(unittest.TestCase):
    def test_grid_is_not_fixed_three_by_three(self):
        pinpoints = [[384 * w, 384 * h] for w in range(1, 7) for h in range(1, 7)]
        for original, grid in (((384, 384), (1, 1)), ((1024, 1024), (3, 3)),
                               ((1000, 600), (3, 2)), ((600, 1000), (2, 3))):
            self.assertEqual(mm_utils.get_anyres_image_grid_shape(original, pinpoints, 384), grid)

    def test_observer_preserves_pixels_and_restores_function(self):
        original = mm_utils.resize_and_pad_image
        pixels = np.random.RandomState(5).randint(1, 256, (600, 1000, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        expected = original(image, (1152, 768))
        with vz.observe_anyres_padding() as calls:
            result = mm_utils.resize_and_pad_image(image, (1152, 768))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
        self.assertIs(mm_utils.resize_and_pad_image, original)
        self.assertEqual(calls, [((1000, 600), (1152, 768), (1152, 768))])
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with vz.observe_anyres_padding():
                raise RuntimeError("synthetic")
        self.assertIs(mm_utils.resize_and_pad_image, original)

    def test_padding_applied_is_separate_from_unpad_skipped(self):
        calls = [((1000, 600), (1152, 768), (1152, 768))]
        for full_grid in (False, True):
            audit = vz.audit_anyres_geometry(metadata((1000, 600), (3, 2)), 27, full_grid, calls)
            self.assertTrue(audit["padding_required"])
            self.assertTrue(audit["padding_call_observed"])
            self.assertTrue(audit["padding_output_size_matches"])
            self.assertFalse(audit["padding_missing_or_mismatch"])
            self.assertEqual([audit[f"padding_{edge}"] for edge in ("left", "top", "right", "bottom")],
                             [0, 38, 0, 38])
            self.assertEqual(audit["baseline_unpad_tokens"], 486)
            self.assertEqual(audit["unpad_executed"], full_grid)
            self.assertEqual(audit["unpad_required_but_skipped"], not full_grid)

    def test_required_but_missing_or_wrong_size_is_flagged(self):
        for calls in ([], [((1000, 600), (1152, 768), (1152, 692))],
                      [((384, 384), (384, 384), (384, 384))]):
            audit = vz.audit_anyres_geometry(metadata((1000, 600), (3, 2)), 27, False, calls)
            self.assertTrue(audit["padding_missing_or_mismatch"])
        audit = vz.audit_anyres_geometry(metadata((384, 384), (1, 1)), 27, False, [])
        self.assertFalse(audit["padding_required"])
        self.assertFalse(audit["padding_missing_or_mismatch"])
        self.assertIsNone(audit["padding_output_size_matches"])
        self.assertFalse(audit["unpad_required_but_skipped"])

    def test_dense_unpad_count_matches_vendor_including_odd_padding(self):
        for size, grid in (((1000, 600), (3, 2)), ((600, 1000), (2, 3)),
                           ((1000, 666), (3, 2)), ((666, 1000), (2, 3)),
                           ((384, 384), (1, 1)), ((2000, 2000), (6, 6))):
            raw = torch.zeros(1, grid[1] * 27, grid[0] * 27)
            audit = vz.audit_anyres_geometry(metadata(size, grid), 27, False, [])
            self.assertEqual(audit["baseline_unpad_tokens"], raw.numel() - unpad_image(raw, size).numel())

    def test_run_saves_audit_with_visualizations_disabled(self):
        helper = visionzip_tests.RunnerTests()
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory).resolve(), []
            args = helper.fixture(root, rate=1, count=2)
            parent = helper.backend(calls)

            class GeometryBackend(parent):
                def generate(self, sample, prompt, method, *positional, **kwargs):
                    result = super().generate(sample, prompt, method, *positional, **kwargs)
                    self.vision_tower.num_patches_per_side = 27
                    method.adapter.crop_metadata = metadata((1000, 600), (3, 2))
                    result["stats"]["full_grid_identity"] = False
                    # Deliberately omit the required preprocessing call.
                    return result

            with helper.fake_cuda(), contextlib.redirect_stdout(io.StringIO()):
                output = vz.run_experiment(args, backend_factory=GeometryBackend)
            with (output / "prune_01/geometry_audit.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)  # excludes warmups
            self.assertTrue(all(row["padding_missing_or_mismatch"] == "True" for row in rows))
            self.assertTrue(all(row["unpad_required_but_skipped"] == "True" for row in rows))
            self.assertEqual(rows[0]["anyres_grid_width"], "3")
            records = [json.loads(line) for line in (output / "prune_01/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(records[0]["pruning"]["geometry_audit"]["padding_missing_or_mismatch"])


if __name__ == "__main__":
    unittest.main()
