"""Row-marker placement only; no checkpoint/CUDA or accuracy claims."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import runVisionZip as vz
from llava.mm_utils import (get_anyres_image_grid_shape, resize_and_pad_image,
                            select_best_resolution)


class SparseNewlineTests(unittest.TestCase):
    def test_original_global_rows_empty_rows_and_context_order(self):
        # 2x2 tiles, 4x4 tokens/tile. Rows 1/2/5/6 have no dominant tokens.
        for dtype in (torch.float16, torch.float32):
            for merge in ("spatial_unpad", "spatial_unpad_add_newl"):
                views, side, dominant, contextual, hidden = 5, 4, 3, 2, 8
                features = torch.arange(views * 5 * hidden, dtype=dtype).reshape(views, 5, hidden)
                indices = torch.tensor([[0, 1, 15]] * views)
                decisions = vz.TokenDecisions(side ** 2, 5, indices)
                newline = torch.full((hidden,), -1000., dtype=dtype)
                expected = [features[0]]
                token_rows = [[] for _ in range(8)]
                for tile in range(4):
                    for slot, local in enumerate(indices[tile + 1].tolist()):
                        row = (tile // 2) * side + local // side
                        col = (tile % 2) * side + local % side
                        token_rows[row].append((col, features[tile + 1, slot]))
                for row in token_rows:
                    expected += [value[None] for _, value in sorted(row, key=lambda pair: pair[0])]
                    expected.append(newline[None])
                expected.append(features[1:, dominant:].reshape(-1, hidden))
                if merge.endswith("add_newl"):
                    expected.append(newline[None])
                with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("No GPU-to-CPU packing")):
                    actual = vz.pack_sparse_views(features, decisions, (2, 2), dominant,
                                                  image_newline=newline, merge_type=merge)
                self.assertTrue(torch.equal(actual, torch.cat(expected)))
                legacy = vz.pack_sparse_views(features, decisions, (2, 2), dominant)
                markers = (actual == newline).all(dim=1)
                self.assertEqual(int(markers.sum()), 8 + int(merge.endswith("add_newl")))
                self.assertTrue(torch.equal(actual[~markers], legacy))
                self.assertEqual(actual.dtype, dtype)

    def test_dense_grid_is_exact_reference_when_no_unpadding_or_resize_is_needed(self):
        for grid in ((1, 1), (2, 2), (3, 1), (1, 3)):
            for merge in ("spatial_unpad", "spatial_unpad_add_newl"):
                views = 1 + grid[0] * grid[1]
                features = torch.arange(views * 4 * 8).reshape(views, 4, 8).float()
                decisions = vz.TokenDecisions(4, 4, torch.arange(4).expand(views, -1))
                newline = torch.arange(8).float() - 20
                actual = vz.pack_sparse_views(features, decisions, grid, 4,
                                              image_newline=newline, merge_type=merge)
                expected, _ = vz.pack_full_views(features, decisions, grid,
                                                 (grid[0] * 8, grid[1] * 8), newline, merge)
                self.assertTrue(torch.equal(actual, expected))

    def test_no_dominants_still_emits_every_row_and_preserves_contexts(self):
        features = torch.randn(3, 1, 8)
        decisions = vz.TokenDecisions(4, 1, torch.empty(3, 0, dtype=torch.long))
        newline = torch.arange(8).float()
        actual = vz.pack_sparse_views(features, decisions, (2, 1), 0, image_newline=newline)
        self.assertTrue(torch.equal(actual, torch.cat((features[0], newline[None].expand(2, -1),
                                                       features[1:].flatten(0, 1)))))

    def test_bad_newline_or_packing_is_rejected(self):
        features = torch.randn(2, 1, 8)
        decisions = vz.TokenDecisions(4, 1, torch.zeros(2, 1, dtype=torch.long))
        for newline, merge in ((torch.ones(7), "spatial_unpad"),
                                (torch.ones(1, 8), "spatial_unpad"),
                                (torch.ones(8), "flat")):
            with self.assertRaisesRegex(ValueError, "Sparse row newlines"):
                vz.pack_sparse_views(features, decisions, (1, 1), 1,
                                     image_newline=newline, merge_type=merge)

    def test_metadata_distinguishes_default_and_legacy(self):
        for enabled in (True, False):
            meta = vz.source_metadata(1, sparse_row_newlines=enabled)
            self.assertEqual(meta["sparse_row_newlines"], enabled)
            self.assertIn("row newlines" if enabled else "legacy", meta["packing"])
            self.assertIn("local adaptation", meta["sparse_newline_policy"])
            self.assertIn("unpad/max9/newlines", vz.source_metadata(0, sparse_row_newlines=enabled)["packing"])


class Image384GeometryTests(unittest.TestCase):
    def test_original_384_has_no_padding_or_max9_resize_with_user_pinpoints(self):
        pinpoints = [[384 * w, 384 * h] for w in range(1, 7) for h in range(1, 7)]
        size = (384, 384)
        self.assertEqual(select_best_resolution(size, pinpoints), size)
        self.assertEqual(get_anyres_image_grid_shape(size, pinpoints, 384), (1, 1))
        pixels = np.random.RandomState(7).randint(1, 256, (384, 384, 3), dtype=np.uint8)
        padded = resize_and_pad_image(Image.fromarray(pixels), size)
        np.testing.assert_array_equal(np.asarray(padded), pixels)
        features = torch.randn(2, 729, 4)
        decisions = vz.TokenDecisions(729, 729, torch.arange(729).expand(2, -1))
        with patch("torch.nn.functional.interpolate", side_effect=AssertionError("Unexpected max9 resize")):
            packed, count = vz.pack_full_views(features, decisions, (1, 1), size,
                                               torch.zeros(4), "spatial_unpad")
        self.assertEqual(count, 27)
        self.assertEqual(packed.shape[0], 1485)


if __name__ == "__main__":
    unittest.main()
