"""CPU-only regression tests for coarse plain entropy / local deviation modes."""

import builtins
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import runVFlowOpt as vis


def fixture(root, size=(56, 28), grid=(2, 1)):
    data = root / "data"
    data.mkdir()
    pixels = np.arange(size[0] * size[1] * 3, dtype=np.uint16).reshape(size[1], size[0], 3) % 256
    image_path = data / "source.png"
    Image.fromarray(pixels.astype(np.uint8)).save(image_path)
    results = root / "results"
    folder = results / "000001_test"
    folder.mkdir(parents=True)
    geometry = vis.ScoreGeometry.create(size, grid, 384, 14)
    metadata = {"id": "test", "image": str(image_path), "gt": 0, "geometry": geometry.__dict__}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (results / "run.json").write_text(json.dumps({"data_root": str(data)}), encoding="utf-8")
    (folder / "scores.npz.log").write_bytes(b"not an NPZ: cached scores must not be read")
    return results, folder, image_path, geometry


class EntropyTests(unittest.TestCase):
    def test_constant_binary_and_spatial_arrangement(self):
        pixels = np.zeros((14, 14, 3), dtype=np.uint8)
        np.testing.assert_array_equal(vis.patch_entropy(pixels, 14), [0])
        pixels[7:] = 255
        np.testing.assert_allclose(vis.patch_entropy(pixels, 14), [math.log(2)], rtol=1e-6)
        checker = ((np.indices((14, 14)).sum(axis=0) % 2) * 255).astype(np.uint8)
        np.testing.assert_array_equal(vis.patch_entropy(pixels, 14),
                                      vis.patch_entropy(np.repeat(checker[..., None], 3, -1), 14))

    def test_arithmetic_rgb_and_legacy_remainder_support(self):
        pixels = np.zeros((384, 384, 3), dtype=np.uint8)
        pixels[:378, :378] = [255, 0, 0]
        pixels[:189, :378] = [0, 255, 0]
        pixels[378:] = 255
        pixels[:, 378:] = 255
        np.testing.assert_array_equal(vis.patch_entropy(pixels, 14), np.zeros(729))

    def test_bit_softmax_per_view_stable_nonmutating(self):
        bits = np.array([[0., 1., 2.], [3., 4., 5.]])
        raw = bits * math.log(2)
        original = raw.copy()
        expected = np.exp(bits) / np.exp(bits).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(vis.entropy_softmax(raw), expected, rtol=1e-6)
        np.testing.assert_allclose(vis.entropy_softmax(raw + 10000), expected, rtol=1e-6)
        np.testing.assert_array_equal(raw, original)
        raw[1, 0] += 100
        np.testing.assert_allclose(vis.entropy_softmax(raw)[0], expected[0], rtol=1e-6)

    def test_uniform_weights_and_invalid_inputs(self):
        np.testing.assert_allclose(vis.entropy_softmax(np.zeros(36)), 1 / 36)
        np.testing.assert_allclose(vis.entropy_softmax(np.zeros((9, 4))), .25)
        for invalid in ([], 1., [np.nan], [np.inf], [-1.], np.zeros((1, 1, 2))):
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                vis.entropy_softmax(invalid)


class GridTests(unittest.TestCase):
    def test_model_geometry_unchanged_and_last_pixel_covered(self):
        source = vis.ScoreGeometry.create((384, 384), (3, 3), 384, 14)
        coarse = vis.entropy_geometry(source)
        self.assertEqual(source.patch_size, 14)
        self.assertEqual((coarse.patch_size, coarse.token_side), (192, 2))
        self.assertEqual(coarse.resized_size, source.resized_size)
        for block, count in ((64, 36), (192, 4)):
            pixels = np.zeros((384, 384, 3), dtype=np.uint8)
            pixels[-1, -1] = 255
            values = vis.patch_entropy(pixels, block)
            self.assertEqual(values.shape, (count,))
            self.assertGreater(values[-1], 0)
            self.assertEqual(np.count_nonzero(values), 1)
        heat = vis.base_score_map(np.arange(36), coarse)
        self.assertEqual(heat.shape, (384, 384))
        self.assertEqual((heat[-1, -1], heat[63, 63], heat[64, 64]), (35, 0, 7))

    def test_nine_crops_form_6x6_in_correct_order_without_gaps(self):
        g = vis.entropy_geometry(vis.ScoreGeometry.create((384, 384), (3, 3), 384, 14))
        values = np.arange(36).reshape(9, 4)
        expected = np.block([[values[r * 3 + c].reshape(2, 2) for c in range(3)] for r in range(3)])
        np.testing.assert_array_equal(vis.assemble_patch_grid(values, g), expected)
        np.testing.assert_array_equal(vis.split_patch_grid(expected, g), values)
        np.testing.assert_array_equal(vis.stitch_score_map(values, g), expected.repeat(192, 0).repeat(192, 1))

    def test_other_crop_layouts_are_preserved(self):
        g = vis.entropy_geometry(vis.ScoreGeometry.create((56, 28), (2, 1), 384, 14))
        self.assertEqual(vis.assemble_patch_grid(np.arange(8).reshape(2, 4), g).shape, (2, 4))
        self.assertEqual(vis.stitch_score_map(np.arange(8).reshape(2, 4), g).shape, (384, 768))

    def test_recomputed_histogram_not_averaged_old_entropy(self):
        pixels = np.zeros((384, 384, 3), dtype=np.uint8)
        pixels[96:192, :192] = 255
        self.assertAlmostEqual(float(vis.patch_entropy(pixels, 192)[0]), math.log(2), places=6)
        self.assertEqual(float(vis.patch_entropy(pixels[:192, :192], 96).mean()), 0.)


class LocalTests(unittest.TestCase):
    def test_local_median_matches_scalar_reference_cross_crop_boundaries(self):
        g = vis.entropy_geometry(vis.ScoreGeometry.create((384, 384), (3, 3), 384, 14))
        grid = np.arange(36, dtype=float).reshape(6, 6) / 10
        grid[2, 2] = 5.5
        d, medians, counts, valid = vis.local_entropy_deviation(vis.split_patch_grid(grid, g), g)
        self.assertEqual(d.shape, (3, 9, 4))
        self.assertTrue(valid.all())
        for wi, window in enumerate((3, 5, 7)):
            radius = window // 2
            dg, mg, ng = (vis.assemble_patch_grid(array[wi], g) for array in (d, medians, counts))
            for y in range(6):
                for x in range(6):
                    neighbors = [grid[j, i] for j in range(max(0, y-radius), min(6, y+radius+1))
                                 for i in range(max(0, x-radius), min(6, x+radius+1)) if (j, i) != (y, x)]
                    self.assertAlmostEqual(float(mg[y, x]), np.median(neighbors), places=6)
                    self.assertAlmostEqual(float(dg[y, x]), abs(grid[y, x] - np.median(neighbors)), places=6)
                    self.assertEqual(ng[y, x], len(neighbors))
        self.assertEqual(vis.assemble_patch_grid(counts[0], g)[1, 1], 8)

    def test_partial_padding_and_no_neighbors_are_nan_not_zero(self):
        g = vis.entropy_geometry(vis.ScoreGeometry.create((56, 1), (2, 4), 384, 14))
        self.assertTrue(vis.visible_blocks(g).any())
        self.assertFalse(vis.full_content_tokens(g).any())
        d, medians, counts, _ = vis.local_entropy_deviation(np.zeros((8, 4)), g)
        self.assertTrue(np.isnan(d).all())
        self.assertTrue(np.isnan(medians).all())
        self.assertFalse(counts.any())
        self.assertTrue(np.isnan(vis.deviation_softmax(d)).all())

    def test_defined_softmax_independent_per_window_and_crop(self):
        values = np.array([[[0., 1., np.nan, 2.], [np.nan] * 4], [[1., 2., 3., 4.], [0.] * 4]])
        weights = vis.deviation_softmax(values)
        np.testing.assert_allclose(weights[0, 0, [0, 1, 3]], vis.entropy_softmax([0, 1, 2]))
        self.assertTrue(np.isnan(weights[0, 1]).all())
        np.testing.assert_allclose(weights[1].sum(axis=-1), 1., atol=1e-7)
        np.testing.assert_allclose(weights[1, 1], .25)


class DisplayTests(unittest.TestCase):
    def test_plain_padding_remains_in_softmax_denominator(self):
        g = vis.entropy_geometry(vis.ScoreGeometry.create((56, 1), (2, 4), 384, 14))
        raw, valid = np.zeros((8, 4)), vis.visible_blocks(g)
        raw[valid] = 2.
        _, display, _ = vis.plain_score_display(np.zeros(36), raw, g, "softmax")
        np.testing.assert_allclose(display[valid], vis.entropy_softmax(raw)[valid])
        self.assertTrue(np.isnan(display[~valid]).all())
        for i in range(8):
            if valid[i].any():
                self.assertLess(float(np.nansum(display[i])), 1.)

    def test_plain_minmax_shared_and_constant(self):
        g = vis.entropy_geometry(vis.ScoreGeometry.create((56, 28), (2, 1), 384, 14))
        base, tiles = np.linspace(1, 3, 36), np.array([[2., 3., 4., 5.], [1., 2., 3., 4.]])
        b, t, spec = vis.plain_score_display(base, tiles, g, "minmax")
        np.testing.assert_allclose(b, (base - 1) / 4)
        np.testing.assert_allclose(t, (tiles - 1) / 4)
        self.assertEqual(spec["shared_reference_nats"], (1., 5.))
        b, t, _ = vis.plain_score_display(np.zeros(36), np.zeros((2, 4)), g, "minmax")
        np.testing.assert_array_equal(b, np.full(36, .5))
        np.testing.assert_array_equal(t, np.full((2, 4), .5))

    def test_local_minmax_separate_base_and_joint_windows(self):
        base = np.linspace(1, 3, 36)
        d = np.array([[[2., 3., 4., np.nan]], [[3., 4., 5., 6.]], [[1., 2., 3., 4.]]])
        b, t, spec = vis.score_display(base, d, "minmax")
        np.testing.assert_allclose(b, (base - 1) / 2)
        np.testing.assert_allclose(t, (d - 1) / 5, atol=1e-7)
        self.assertEqual(spec["anyres_reference_nats"], (1., 6.))

    def test_plain_rendered_pixels_labels_and_separate_ranges(self):
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        from matplotlib.figure import Figure
        source_g = vis.ScoreGeometry.create((56, 28), (2, 1), 384, 14)
        g = vis.entropy_geometry(source_g)
        image = Image.new("RGB", (56, 28), "gray")
        base, _ = vis.prepare_entropy_views(image, source_g)
        raw_b, raw_t = np.linspace(0, 3, 36), np.array([[0., 1., 3., 5.], [2., 1., 0., 3.]])
        b, t, setup = vis.plain_score_display(raw_b, raw_t, g, "softmax")
        def check(fig, path, **kwargs):
            self.assertEqual(len(fig.axes), 6)
            self.assertIn("6x6", fig.axes[1].get_title())
            self.assertIn("2x2 per crop", fig.axes[3].get_title())
            self.assertIn("same color does NOT imply", fig._supxlabel.get_text())
            for axis, name, heat in ((1, "base", vis.base_score_map(b, g)), (3, "anyres", vis.stitch_score_map(t, g))):
                actual = np.asarray(fig.axes[axis].images[1].get_array())
                expected = colormaps[vis.COLORMAP](Normalize(*setup[name + "_limits"])(heat))
                np.testing.assert_allclose(actual[..., :3], expected[..., :3])
                np.testing.assert_allclose(actual[..., 3], vis.OVERLAY_ALPHA)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, "savefig", check):
            result = vis.draw_figures(image, base, raw_b, raw_t, g, tmp, "Synthetic", "softmax", "plain")
        self.assertEqual(result["figures"], ["entropy_overview.png"])


class RunTests(unittest.TestCase):
    def test_plain_all_scales_no_model_no_neighbors_sources_untouched(self):
        for scale in ("softmax", "raw", "minmax", "sample", "fixed", "layer"):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                results, folder, image_path, source_g = fixture(root)
                hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
                original_import = builtins.__import__
                def forbid_model(name, *args, **kwargs):
                    if name.split(".")[0] in {"torch", "transformers", "accelerate", "llava"}:
                        raise AssertionError("Unexpected model import: " + name)
                    return original_import(name, *args, **kwargs)
                with patch("builtins.__import__", side_effect=forbid_model), patch.object(
                        vis, "local_entropy_deviation", side_effect=AssertionError("plain computed neighbors")):
                    output = vis.run(results, color_scale=scale)
                self.assertEqual(output.name, "results_vflowopt_plain_" + scale)
                target = output / folder.name
                self.assertEqual({p.name for p in target.iterdir()},
                                 {"entropy_overview.png", "entropy.npz", "entropy_patch_means.csv", "metadata.json"})
                with Image.open(target / "entropy_overview.png") as png:
                    self.assertEqual(png.size, (2400, 750))
                manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
                saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
                self.assertTrue(manifest["complete"])
                self.assertFalse(manifest["model_loaded"])
                self.assertFalse(manifest["source_scores_read"])
                self.assertEqual(saved["entropy_mode"], "plain")
                self.assertEqual(saved["geometry"]["patch_size"], 14)
                self.assertEqual(saved["entropy_geometry"]["patch_size"], 192)
                self.assertEqual(saved["stitched_anyres_grid_shape"], [2, 4])
                with Image.open(image_path) as source:
                    base, tiles = vis.prepare_entropy_views(source.convert("RGB"), source_g)
                with np.load(target / "entropy.npz", allow_pickle=False) as a:
                    np.testing.assert_array_equal(a["base_entropy"], vis.patch_entropy(base, 64))
                    np.testing.assert_array_equal(a["tile_entropy"], [vis.patch_entropy(tile, 192) for tile in tiles])
                    self.assertEqual(a["base_entropy"].shape, (36,))
                    self.assertEqual(a["tile_entropy"].shape, (2, 4))
                    self.assertEqual(str(a["entropy_mode"]), "plain")
                    self.assertNotIn("tile_deviation", a.files)
                    np.testing.assert_allclose(a["base_softmax"].sum(), 1., atol=1e-7)
                    np.testing.assert_allclose(a["tile_entropy_softmax"].sum(axis=-1), 1., atol=1e-7)
                with (target / "entropy_patch_means.csv").open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([int(r["total_blocks"]) for r in rows], [36, 4, 4])
                np.testing.assert_allclose([float(r["mean_softmax_weight"]) for r in rows], [1 / 36, .25, .25])
                for p, digest in hashes.items():
                    self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
                with self.assertRaises(FileExistsError):
                    vis.run(results, color_scale=scale)

    def test_local_three_scales_six_figures_and_9crop_data(self):
        for scale in ("softmax", "minmax", "raw"):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as tmp:
                results, folder, _, _ = fixture(Path(tmp), (56, 56), (3, 3))
                output = vis.run(results, color_scale=scale, entropy_mode="local")
                target = output / folder.name
                self.assertEqual({p.name for p in target.glob("*.png")}, {
                    "entropy_overview.png", "entropy_raw_overview.png", "base_entropy_6x6.png",
                    "anyres_deviation_3x3.png", "anyres_deviation_5x5.png", "anyres_deviation_7x7.png"})
                with np.load(target / "entropy.npz", allow_pickle=False) as a:
                    self.assertEqual(a["tile_entropy"].shape, (9, 4))
                    self.assertEqual(a["tile_deviation"].shape, (3, 9, 4))
                    np.testing.assert_allclose(a["tile_deviation_softmax"].sum(axis=-1), 1., atol=1e-7)
                    self.assertEqual(int(a["patch_size"]), 192)
                    self.assertEqual(int(a["source_model_patch_size"]), 14)
                saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["stitched_anyres_grid_shape"], [6, 6])
                self.assertEqual(saved["entropy_mode"], "local")
                with (target / "entropy_patch_means.csv").open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 28)
                self.assertEqual({r["total_blocks"] for r in rows[1:]}, {"4"})

    def test_cli_and_invalid_options_before_writes(self):
        parser = vis.build_parser()
        self.assertEqual(parser.parse_args([]).entropy_mode, "plain")
        self.assertEqual(parser.parse_args(["--entropy-mode", "local"]).entropy_mode, "local")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new"
            for kwargs in ({"entropy_mode": "wrong"}, {"color_scale": "wrong"}, {"limit": 0}):
                with self.assertRaises(ValueError):
                    vis.run(Path(tmp) / "missing", target, **kwargs)
                self.assertFalse(target.exists())
        version = subprocess.run([sys.executable, str(Path(vis.__file__)), "--version"],
                                 capture_output=True, text=True, check=True)
        self.assertIn("2.1-plain-local-coarse-grids", version.stdout)

    def test_main_passes_mode_and_default_all_samples(self):
        with patch.object(sys, "argv", ["runVFlowOpt.py", "--results-dir", "source", "--entropy-mode", "local"]), \
                patch.object(vis, "run") as call:
            vis.main()
        self.assertEqual(call.call_args.args[-1], "local")
        self.assertIsNone(call.call_args.args[-2])


if __name__ == "__main__":
    unittest.main()
