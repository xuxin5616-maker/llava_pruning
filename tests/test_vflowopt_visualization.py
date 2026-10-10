"""CPU-only regression tests for whole AnyRes 12x12 / Base 18x18 entropy grids."""

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


class GlobalGridTests(unittest.TestCase):
    def test_18_grid_covers_every_base_pixel_in_21_22_intervals(self):
        pixels = np.zeros((384, 384, 3), dtype=np.uint8)
        pixels[-1, -1] = 255
        h, x, y = vis.global_grid_entropy(pixels, 18)
        self.assertEqual(h.shape, (18, 18))
        self.assertEqual(set(np.diff(x)), {21, 22})
        self.assertEqual((x[0], x[-1], y[0], y[-1]), (0, 384, 0, 384))
        self.assertGreater(h[-1, -1], 0.)
        self.assertEqual(np.count_nonzero(h), 1)
        heat = vis.global_grid_score_map(h, x, y)
        self.assertEqual(heat.shape, (384, 384))
        self.assertEqual(heat[-1, -1], h[-1, -1])
        # A nonuniform block is normalized by its actual 22*22 pixel count.
        counts = np.bincount(pixels[-22:, -22:, 0].ravel(), minlength=256)
        p = counts[counts > 0] / 484
        self.assertAlmostEqual(float(h[-1, -1]), float(-(p * np.log(p)).sum()), places=7)

    def test_anyres_always_12_total_including_nondivisible_crop_layout(self):
        for grid in ((3, 3), (2, 1), (5, 2)):
            with self.subTest(grid=grid):
                g = vis.ScoreGeometry.create((56, 28), grid, 384, 14)
                image = Image.new("RGB", (56, 28), (10, 20, 30))
                _, tiles = vis.prepare_entropy_views(image, g)
                mosaic = vis.reconstruct_mosaic(tiles, g)
                self.assertEqual(mosaic.shape, (grid[1] * 384, grid[0] * 384, 3))
                h, x, y = vis.global_grid_entropy(mosaic, 12)
                self.assertEqual(h.shape, (12, 12))
                self.assertEqual(g.patch_size, 14)
                left, top = g.paste_xy
                width, height = g.resized_size
                heat = vis.global_grid_score_map(np.arange(144).reshape(12, 12), x, y,
                                                 (left, top, left + width, top + height))
                self.assertEqual(heat.shape, (height, width))
                np.testing.assert_array_equal(
                    vis.global_grid_score_map(h, x, y),
                    h.repeat(np.diff(y), 0).repeat(np.diff(x), 1))

    def test_mosaic_crop_order_and_global_histogram_not_average(self):
        g = vis.ScoreGeometry.create((384, 384), (3, 3), 384, 14)
        tiles = [Image.new("RGB", (384, 384), (i, i, i)) for i in range(9)]
        mosaic = vis.reconstruct_mosaic(tiles, g)
        for i in range(9):
            r, c = divmod(i, 3)
            np.testing.assert_array_equal(mosaic[r * 384, c * 384], [i] * 3)
        pixels = np.zeros((384, 384, 3), dtype=np.uint8)
        pixels[16:32, :32] = 255
        h, _, _ = vis.global_grid_entropy(pixels, 12)
        self.assertAlmostEqual(float(h[0, 0]), math.log(2), places=6)
        self.assertEqual(float(vis.patch_entropy(pixels[:32, :32], 16).mean()), 0.)

    def test_geometric_masks_distinguish_partial_and_full_padding(self):
        edges = np.array([0, 10, 20, 30])
        visible, full = vis.global_grid_masks(edges, edges, (5, 5, 25, 25))
        self.assertTrue(visible.all())
        self.assertEqual(full.sum(), 1)
        self.assertTrue(full[1, 1])
        for args in ((10, 0), (10, 11), (0, 1), (10, 1.5), (True, 1)):
            with self.assertRaises(ValueError):
                vis.equal_grid_edges(*args)


class LocalTests(unittest.TestCase):
    def test_same_neighbor_rule_for_12_and_18_matches_scalar_reference(self):
        for side in (12, 18):
            grid = np.arange(side * side, dtype=float).reshape(side, side) / (side * side)
            grid[2, 2] = 4.
            d, medians, counts = vis.global_grid_deviation(grid)
            self.assertEqual(d.shape, (3, side, side))
            for wi, window in enumerate((3, 5, 7)):
                radius = window // 2
                for y, x in ((0, 0), (2, 2), (3, 3), (side-1, side-1)):
                    neighbors = [grid[j, i] for j in range(max(0, y-radius), min(side, y+radius+1))
                                 for i in range(max(0, x-radius), min(side, x+radius+1)) if (j, i) != (y, x)]
                    self.assertAlmostEqual(float(medians[wi, y, x]), float(np.median(neighbors)), places=6)
                    self.assertAlmostEqual(float(d[wi, y, x]), abs(grid[y, x] - np.median(neighbors)), places=6)
                    self.assertEqual(counts[wi, y, x], len(neighbors))
            self.assertEqual(counts[0, 3, 3], 8)  # crosses former 3x3 crop boundaries

    def test_padding_and_isolated_center_are_undefined_not_zero(self):
        h = np.ones((12, 12))
        valid = np.zeros((12, 12), dtype=bool)
        valid[4, 4] = True
        d, medians, counts = vis.global_grid_deviation(h, valid)
        self.assertTrue(np.isnan(d).all())
        self.assertTrue(np.isnan(medians).all())
        self.assertFalse(counts.any())
        self.assertTrue(np.isnan(vis.whole_grid_softmax(d)).all())
        view = vis.analyze_view(np.zeros((384, 768, 3), dtype=np.uint8), 12, (0, 190, 768, 194), "local")
        self.assertTrue(np.isnan(view["scores"]).all())

    def test_base_is_deviation_not_raw_entropy(self):
        y, x = np.indices((384, 384))
        pixels = np.repeat(((x + y) % 256).astype(np.uint8)[..., None], 3, -1)
        view = vis.analyze_view(pixels, 18, (0, 0, 384, 384), "local")
        expected, median, count = vis.global_grid_deviation(view["entropy"])
        np.testing.assert_array_equal(view["scores"], expected)
        np.testing.assert_array_equal(view["neighbor_median"], median)
        self.assertFalse(np.allclose(view["scores"][0], view["entropy"]))
        self.assertEqual(view["scores"].shape, (3, 18, 18))


class NormalizationTests(unittest.TestCase):
    def test_whole_grid_softmax_not_per_row_crop_or_joined_window(self):
        raw = np.zeros((3, 12, 12))
        raw[0, 0, 0] = 3.
        weights = vis.whole_grid_softmax(raw)
        np.testing.assert_allclose(weights.sum(axis=(1, 2)), 1., atol=1e-7)
        expected = vis.entropy_softmax(raw[0].ravel()).reshape(12, 12)
        np.testing.assert_array_equal(weights[0], expected)
        self.assertLess(float(weights[0, :4, :4].sum()), 1.)  # one old crop isn't a group
        np.testing.assert_allclose(weights[1:], 1 / 144)
        np.testing.assert_allclose(vis.whole_grid_softmax(np.zeros((3, 18, 18))), 1 / 324)
        changed = raw.copy()
        changed[1, 1, 1] = 5
        np.testing.assert_array_equal(vis.whole_grid_softmax(changed)[0], weights[0])

    def test_whole_grid_nan_support_and_invalids(self):
        scores = np.full((3, 12, 12), np.nan)
        scores[0, 0, :2] = [0., 1.]
        weights = vis.whole_grid_softmax(scores)
        np.testing.assert_allclose(weights[0, 0, :2], vis.entropy_softmax([0., 1.]))
        self.assertTrue(np.isnan(weights[1:]).all())
        for invalid in ([], [1, 2], [[-1.]], [[np.inf]], np.zeros((1, 1, 1, 1))):
            with self.assertRaises(ValueError):
                vis.whole_grid_softmax(invalid)

    def test_display_minmax_raw_softmax_and_constant_unchanged(self):
        b = vis.analyze_view(np.zeros((384, 384, 3), dtype=np.uint8), 18, (0, 0, 384, 384), "local")
        a = vis.analyze_view(np.zeros((384, 384, 3), dtype=np.uint8), 12, (0, 0, 384, 384), "local")
        views = {"base": b, "anyres": a}
        for name, side in (("base", 18), ("anyres", 12)):
            views[name]["scores"][:] = np.linspace(1, 5 if name == "base" else 3, 3 * side * side).reshape(3, side, side)
            views[name]["softmax"] = vis.whole_grid_softmax(views[name]["scores"])
        original = {k: v["scores"].copy() for k, v in views.items()}
        vis.display_scores(views, "minmax", "local")
        np.testing.assert_allclose(b["display"], (b["scores"] - 1) / 4, atol=1e-7)
        np.testing.assert_allclose(a["display"], (a["scores"] - 1) / 2, atol=1e-7)
        vis.display_scores(views, "raw", "local")
        for k, v in views.items():
            np.testing.assert_array_equal(v["display"], original[k])
        setup = vis.display_scores(views, "softmax", "local")
        for k, v in views.items():
            np.testing.assert_array_equal(v["display"], v["softmax"])
            np.testing.assert_array_equal(v["scores"], original[k])
        self.assertNotEqual(setup["base"]["limits"], setup["anyres"]["limits"])

    def test_plain_skips_neighbors_and_keeps_padding_in_denominator(self):
        pixels = np.zeros((384, 384, 3), dtype=np.uint8)
        with patch.object(vis, "global_grid_deviation", side_effect=AssertionError("Plain computed neighbors")):
            a = vis.analyze_view(pixels, 12, (0, 128, 384, 256), "plain")
        b = vis.analyze_view(pixels, 18, (0, 0, 384, 384), "plain")
        views = {"base": b, "anyres": a}
        np.testing.assert_allclose(a["softmax"], 1 / 144)
        vis.display_scores(views, "softmax", "plain")
        self.assertLess(float(np.nansum(a["display"])), 1)
        vis.display_scores(views, "minmax", "plain")
        self.assertTrue(np.all(a["display"][np.isfinite(a["display"])] == .5))


class RenderTests(unittest.TestCase):
    def test_exactly_two_global_pngs_with_3_windows_and_correct_pixels(self):
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        from matplotlib.figure import Figure
        image = Image.new("RGB", (384, 384), "gray")
        views = {name: vis.analyze_view(image, side, (0, 0, 384, 384), "local")
                 for name, side in (("base", 18), ("anyres", 12))}
        setups = vis.display_scores(views, "softmax", "local")
        calls = []
        def check(fig, path, **kwargs):
            name = "base" if Path(path).name.startswith("base") else "anyres"
            calls.append(Path(path).name)
            self.assertEqual(len(fig.axes), 5)
            self.assertEqual([len(a.images) for a in fig.axes[:4]], [1, 2, 2, 2])
            for i, window in enumerate((3, 5, 7)):
                self.assertIn(f"{window}x{window} neighborhood", fig.axes[i+1].get_title())
                v = views[name]
                heat = vis.global_grid_score_map(v["display"][i], v["x_edges"], v["y_edges"])
                expected = colormaps[vis.COLORMAP](Normalize(*setups[name]["limits"])(heat))
                actual = np.asarray(fig.axes[i+1].images[1].get_array())
                np.testing.assert_allclose(actual[..., :3], expected[..., :3])
                np.testing.assert_allclose(actual[..., 3], vis.OVERLAY_ALPHA)
            self.assertIn("WHOLE grid", fig._supxlabel.get_text())
            self.assertIn("Neighborhood entropy deviation", fig._suptitle.get_text())
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, "savefig", check):
            files = vis.draw_overviews(image, image, views, setups, tmp, "Synthetic", "local")
        self.assertEqual(files, list(vis.FIGURES))
        self.assertEqual(calls, list(vis.FIGURES))


class RunTests(unittest.TestCase):
    def test_local_and_plain_all_display_modes_no_model_and_sources_untouched(self):
        for mode in ("local", "plain"):
            for scale in ("softmax", "raw", "minmax", "sample", "fixed", "layer"):
                with self.subTest(mode=mode, scale=scale), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    results, folder, image_path, geometry = fixture(root, (56, 56), (3, 3))
                    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
                    original_import = builtins.__import__
                    def forbid_model(name, *args, **kwargs):
                        if name.split(".")[0] in {"torch", "transformers", "accelerate", "llava"}:
                            raise AssertionError("Unexpected model import: " + name)
                        return original_import(name, *args, **kwargs)
                    with patch("builtins.__import__", side_effect=forbid_model):
                        output = vis.run(results, color_scale=scale, entropy_mode=mode)
                    target = output / folder.name
                    self.assertEqual({p.name for p in target.iterdir()},
                                     set(vis.FIGURES) | {"entropy.npz", "entropy_patch_means.csv", "metadata.json"})
                    for file in vis.FIGURES:
                        with Image.open(target / file) as png:
                            self.assertEqual(png.size, (2400 if mode == "local" else 1350, 750))
                    saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
                    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
                    self.assertEqual(saved["geometry"]["patch_size"], 14)
                    self.assertEqual(saved["analysis_grids"]["base"]["shape"], [18, 18])
                    self.assertEqual(saved["analysis_grids"]["anyres"]["shape"], [12, 12])
                    self.assertTrue(manifest["complete"])
                    self.assertEqual(manifest["figures_per_sample"], list(vis.FIGURES))
                    self.assertFalse(manifest["source_scores_read"])
                    self.assertFalse(manifest["model_loaded"])
                    with np.load(target / "entropy.npz", allow_pickle=False) as a:
                        n = 3 if mode == "local" else 1
                        self.assertEqual(a["base_entropy"].shape, (18, 18))
                        self.assertEqual(a["anyres_entropy"].shape, (12, 12))
                        self.assertEqual(a["base_scores"].shape, (n, 18, 18))
                        self.assertEqual(a["anyres_scores"].shape, (n, 12, 12))
                        np.testing.assert_allclose(a["base_softmax"].sum(axis=(1, 2)), 1., atol=2e-7)
                        np.testing.assert_allclose(a["anyres_softmax"].sum(axis=(1, 2)), 1., atol=2e-7)
                        if mode == "local":
                            # Deviation is computed before the median is rounded to FP32
                            # for storage; allow that half-ULP rounding near zero.
                            np.testing.assert_allclose(a["base_deviation"], np.abs(a["base_entropy"] - a["base_neighbor_median"]), atol=3e-7)
                            np.testing.assert_allclose(a["anyres_deviation"], np.abs(a["anyres_entropy"] - a["anyres_neighbor_median"]), atol=3e-7)
                        else:
                            self.assertNotIn("base_deviation", a.files)
                    with (target / "entropy_patch_means.csv").open(encoding="utf-8-sig", newline="") as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(len(rows), 6 if mode == "local" else 2)
                    for row in rows:
                        self.assertAlmostEqual(float(row["softmax_weight_sum"]), 1., places=6)
                        self.assertEqual(int(row["total_blocks"]), 324 if row["view"] == "base" else 144)
                    for path, digest in hashes.items():
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                    with self.assertRaises(FileExistsError):
                        vis.run(results, color_scale=scale, entropy_mode=mode)

    def test_all_undefined_anyres_still_renders_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, folder, _, _ = fixture(Path(tmp), (56, 1), (2, 4))
            output = vis.run(results)
            target = output / folder.name
            with np.load(target / "entropy.npz", allow_pickle=False) as a:
                self.assertTrue(np.isnan(a["anyres_scores"]).all())
                self.assertTrue(np.isnan(a["anyres_softmax"]).all())
                self.assertTrue(np.isfinite(a["base_scores"]).all())
            saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            self.assertIsNone(saved["display"]["anyres"]["observed_range"])
            self.assertEqual(saved["display"]["anyres"]["defined_counts"], [0, 0, 0])

    def test_cli_defaults_local_modes_version_and_fail_before_writing(self):
        parser = vis.build_parser()
        self.assertEqual(parser.parse_args([]).entropy_mode, "local")
        self.assertEqual(parser.parse_args(["--entropy-mode", "plain"]).entropy_mode, "plain")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new"
            for kwargs in ({"entropy_mode": "wrong"}, {"color_scale": "wrong"}, {"limit": 0}):
                with self.assertRaises(ValueError):
                    vis.run(Path(tmp) / "missing", target, **kwargs)
                self.assertFalse(target.exists())
        version = subprocess.run([sys.executable, str(Path(vis.__file__)), "--version"],
                                 capture_output=True, text=True, check=True)
        self.assertIn("3.0-global12-base18-two-overviews", version.stdout)
        with patch.object(sys, "argv", ["runVFlowOpt.py", "--results-dir", "source"]), patch.object(vis, "run") as call:
            vis.main()
        self.assertEqual(call.call_args.args[-1], "local")
        self.assertIsNone(call.call_args.args[-2])


if __name__ == "__main__":
    unittest.main()
