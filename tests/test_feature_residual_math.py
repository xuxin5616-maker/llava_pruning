"""CPU numerical tests against direct, center-excluding neighborhood loops."""
import unittest

import numpy as np

from feature_residual_math import WINDOWS, local_residuals, stage_residuals, stitch_tokens, tile_validity
from llava_pruning.score_visualization import ScoreGeometry


def reference(features, valid):
    features = np.asarray(features, dtype=np.float64)
    h, w, _ = features.shape
    l2 = np.full((len(WINDOWS), h, w), np.nan)
    cosine, counts = l2.copy(), np.zeros(l2.shape, dtype=np.int32)
    for wi, window in enumerate(WINDOWS):
        radius = window // 2
        for y in range(h):
            for x in range(w):
                if not valid[y, x]:
                    continue
                neighbors = [features[ny, nx]
                             for ny in range(max(0, y - radius), min(h, y + radius + 1))
                             for nx in range(max(0, x - radius), min(w, x + radius + 1))
                             if (ny, nx) != (y, x) and valid[ny, nx]]
                counts[wi, y, x] = len(neighbors)
                if not neighbors:
                    continue
                mean = np.mean(neighbors, axis=0)
                center = features[y, x]
                l2[wi, y, x] = np.linalg.norm(center - mean)
                product = np.linalg.norm(center) * np.linalg.norm(mean)
                if product > 0:
                    cosine[wi, y, x] = 1 - np.clip(np.dot(center, mean) / product, -1, 1)
    return {"l2": l2, "cosine": cosine, "neighbor_counts": counts}


class ResidualMathTests(unittest.TestCase):
    def test_all_windows_masked_edges_and_channel_chunking(self):
        rng = np.random.default_rng(8)
        features = rng.normal(size=(8, 11, 17)).astype(np.float32)
        valid = rng.random((8, 11)) > .25
        expected = reference(features, valid)
        before = features.copy()
        for chunk in (1, 5, 128):
            actual = local_residuals(features, valid, channel_chunk=chunk)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(features, before)
        self.assertEqual(actual["l2"].dtype, np.float32)

    def test_raw_mean_no_normalization_and_excludes_center(self):
        # Middle vector is [1,0]; neighbor RAW mean is [5,.5], not [.5,.5].
        features = np.array([[[10., 0.], [1., 0.], [0., 1.]]])
        actual = local_residuals(features)
        self.assertAlmostEqual(float(actual["l2"][0, 0, 1]), np.sqrt(16.25), places=6)
        self.assertAlmostEqual(float(actual["cosine"][0, 0, 1]), 1 - 5 / np.sqrt(25.25), places=7)
        scaled = local_residuals(features * 3)
        np.testing.assert_allclose(scaled["l2"], actual["l2"] * 3, rtol=1e-6)
        np.testing.assert_allclose(scaled["cosine"], actual["cosine"], atol=1e-7)

    def test_zero_vectors_and_cancelled_mean(self):
        result = local_residuals(np.zeros((3, 3, 2)))
        self.assertTrue(np.isnan(result["cosine"]).all())
        np.testing.assert_array_equal(result["l2"], 0)
        self.assertEqual(result["neighbor_counts"][0, 0, 0], 3)
        self.assertEqual(result["neighbor_counts"][0, 1, 1], 8)
        result = local_residuals(np.array([[[1., 0.], [0., 2.], [-1., 0.]]]))
        self.assertTrue(np.isnan(result["cosine"][:, 0, 1]).all())
        np.testing.assert_array_equal(result["l2"][:, 0, 1], 2)

    def test_no_neighbors_and_invalid_centers(self):
        for valid in (np.array([[True]]), np.array([[False]])):
            result = local_residuals(np.ones((1, 1, 2)), valid)
            self.assertTrue(np.isnan(result["l2"]).all())
            self.assertTrue(np.isnan(result["cosine"]).all())
            self.assertFalse(result["neighbor_counts"].any())
        features = np.ones((3, 3, 2))
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        features[~mask] = 10000  # Padding must not enter the mean.
        self.assertTrue(np.isnan(local_residuals(features, mask)["l2"]).all())

    def test_tile_order_and_cross_crop_neighbors(self):
        geometry = ScoreGeometry.create((56, 56), (2, 2), 28, 14)
        features = np.zeros((5, 4, 2), dtype=np.float32)
        features[0] = 99  # Base must not affect AnyRes neighbors.
        for ti in range(4):
            features[ti + 1] = [ti + 1, 1]
        stitched = stitch_tokens(features[1:], geometry)
        np.testing.assert_array_equal(stitched[..., 0], [[1, 1, 2, 2], [1, 1, 2, 2],
                                                        [3, 3, 4, 4], [3, 3, 4, 4]])
        actual = stage_residuals(features, geometry)
        expected = reference(stitched, np.ones((4, 4), dtype=bool))
        for wi in range(3):
            for kind in ("l2", "cosine"):
                np.testing.assert_allclose(stitch_tokens(actual["tile_" + kind][wi], geometry),
                                           expected[kind][wi], atol=1e-7)
        self.assertGreater(actual["tile_l2"][0, 0, 1], 0)
        np.testing.assert_array_equal(actual["base_l2"], 0)

    def test_full_patch_content_mask_with_partial_padding_and_gaps(self):
        geometry = ScoreGeometry.create((384, 200), (2, 1), 384, 14)
        valid = tile_validity(geometry).reshape(2, 27, 27)
        for ti in range(2):
            left, top, right, bottom = geometry.tile_content_box(ti)
            for y in range(27):
                for x in range(27):
                    self.assertEqual(valid[ti, y, x],
                                     left <= x * 14 and top <= y * 14 and
                                     (x + 1) * 14 <= right and (y + 1) * 14 <= bottom)
        self.assertTrue(valid.any())
        self.assertFalse(valid.all())
        features = np.ones((3, 729, 2), dtype=np.float32)
        features[1:][~valid.reshape(2, 729)] = 1000
        actual = stage_residuals(features, geometry)
        np.testing.assert_array_equal(actual["tile_valid"], valid.reshape(2, 729))
        self.assertTrue(np.isnan(actual["tile_l2"][:, ~valid.reshape(2, 729)]).all())
        np.testing.assert_allclose(actual["tile_l2"][:, valid.reshape(2, 729)], 0)

    def test_invalid_shapes_and_nonfinite(self):
        for bad in (np.ones((2, 2)), np.zeros((0, 2, 3)), np.full((2, 2, 2), np.inf),
                    np.full((2, 2, 2), np.nan), np.ones((2, 2, 2), dtype=complex)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                local_residuals(bad)
        with self.assertRaises(ValueError):
            local_residuals(np.ones((2, 2, 3)), np.ones((3, 2)))
        with self.assertRaises(ValueError):
            local_residuals(np.ones((2, 2, 3)), windows=(2,))
        geometry = ScoreGeometry.create((28, 28), (1, 1), 28, 14)
        with self.assertRaises(ValueError):
            stage_residuals(np.ones((1, 4, 3)), geometry)


if __name__ == "__main__":
    unittest.main()
