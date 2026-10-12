"""CPU tests: Base AND whole AnyRes each use BOTH 12x12 and 18x18 grids."""
import contextlib
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import run_entropy_18_12 as vis

PROJECT = Path(__file__).resolve().parents[1]


def geometry_for(size, grid=(3, 3)):
    width, height = size
    canvas_width, canvas_height = grid[0] * 384, grid[1] * 384
    scale_x, scale_y = canvas_width / width, canvas_height / height
    if scale_x < scale_y:
        resized = (canvas_width, min(math.ceil(height * scale_x), canvas_height))
    else:
        resized = (min(math.ceil(width * scale_y), canvas_width), canvas_height)
    return vis.Geometry(size, grid, 384, 14, resized,
                        ((canvas_width - resized[0]) // 2, (canvas_height - resized[1]) // 2))


def fixture(root, size=(120, 90), grid=(3, 3), sample_count=1):
    """Synthetic image fixture, no clinical/model/performance interpretation."""
    data = root / "data"
    (data / "imgs").mkdir(parents=True)
    rng = np.random.default_rng(18)
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[:] = [90, 110, 130]
    pixels[size[1] // 4:3 * size[1] // 4, size[0] // 4:3 * size[0] // 4] = rng.integers(
        0, 256, (size[1] // 2, size[0] // 2, 3), dtype=np.uint8)
    image_path = data / "imgs" / "test.png"
    Image.fromarray(pixels).save(image_path)
    geometry = geometry_for(size, grid)
    results = root / "results"
    results.mkdir()
    (results / "run.json").write_text(json.dumps({"data_root": str(data)}), encoding="utf-8")
    for i in range(sample_count):
        folder = results / f"{i + 1:06d}_example"
        folder.mkdir()
        (folder / "metadata.json").write_text(json.dumps({
            "id": f"synthetic_{i}", "gt": 0, "image": str(image_path), "geometry": asdict(geometry),
        }), encoding="utf-8")
        # A corrupt, suffix-added score archive proves we do not use or rename it.
        (folder / "scores.npz.log").write_bytes(b"not a score archive")
    return results, data, image_path, geometry


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class EntropyMathTests(unittest.TestCase):
    def test_base_edges_exact_coverage_21_22_pixels(self):
        edges = vis.grid_edges(384, 18)
        self.assertEqual((edges[0], edges[-1]), (0, 384))
        widths, counts = np.unique(np.diff(edges), return_counts=True)
        np.testing.assert_array_equal(widths, [21, 22])
        np.testing.assert_array_equal(counts, [12, 6])
        expanded = vis.expand_grid(np.arange(324).reshape(18, 18), edges, edges)
        self.assertEqual(expanded.shape, (384, 384))
        self.assertEqual(expanded[-1, -1], 323)
        areas = np.diff(edges)[:, None] * np.diff(edges)[None, :]
        np.testing.assert_array_equal(np.bincount(expanded.ravel()), areas.ravel())

    def test_histogram_entropy_not_feature_or_cross_entropy(self):
        pixels = np.zeros((36, 36, 3), dtype=np.uint8)
        result = vis.grid_entropy(pixels, 18)
        np.testing.assert_array_equal(result["entropy_nats"], 0)
        pixels[::2] = 255
        np.testing.assert_allclose(vis.grid_entropy(pixels, 18)["entropy_nats"], math.log(2), atol=1e-12)
        # Arithmetic gray treats red and green identically (both have mean 85).
        pixels[::2] = [255, 0, 0]
        pixels[1::2] = [0, 255, 0]
        np.testing.assert_array_equal(vis.grid_entropy(pixels, 18)["entropy_nats"], 0)

    def test_cell_rgba_expansion_matches_scalar_mapping(self):
        values = np.arange(12 * 12 * 4, dtype=np.uint16).astype(np.uint8).reshape(12, 12, 4)
        x_edges, y_edges = vis.grid_edges(384, 12), vis.grid_edges(768, 12)
        rgba = vis.expand_grid(values, x_edges, y_edges)
        self.assertEqual(rgba.shape, (768, 384, 4))
        self.assertEqual(rgba.dtype, np.uint8)
        for channel in range(4):
            np.testing.assert_array_equal(rgba[..., channel], vis.expand_grid(values[..., channel], x_edges, y_edges))

    def test_rgb_preprocessing_matches_existing_anyres_helpers(self):
        # Import the real helper only in this test; the delivered script never
        # imports Torch or LLaVA. Compare pixels, not just compatible shapes.
        sys.path.insert(0, str(PROJECT / "vendor" / "llava"))
        from llava.mm_utils import resize_and_pad_image
        rng = np.random.default_rng(13)
        for size, grid in (((63, 100), (3, 3)), ((100, 63), (2, 1))):
            image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
            geometry = geometry_for(size, grid)
            base, mosaic = vis.prepare_views(image, geometry)
            np.testing.assert_array_equal(base, image.resize((384, 384)))
            np.testing.assert_array_equal(mosaic, resize_and_pad_image(image, geometry.canvas_size))

    def test_partial_padding_excluded_from_both_center_and_neighbor_sets(self):
        pixels = np.zeros((24, 24, 3), dtype=np.uint8)
        result = vis.grid_entropy(pixels, 12, (3, 5, 21, 23))
        self.assertEqual(int(result["valid"].sum()), 8 * 8)
        result["entropy_nats"][~result["valid"]] = 1000
        deviations, medians, counts = vis.neighborhood_deviation(result["entropy_nats"], result["valid"])
        np.testing.assert_array_equal(deviations[:, result["valid"]], 0)
        np.testing.assert_array_equal(medians[:, result["valid"]], 0)
        self.assertTrue(np.isnan(deviations[:, ~result["valid"]]).all())
        np.testing.assert_array_equal(counts[:, ~result["valid"]], 0)

    def test_center_excluded_and_median_not_mean(self):
        entropy = np.zeros((3, 3))
        entropy[1, 1] = 9
        entropy[0, 0] = 8
        entropy[0, 1] = 3
        deviation, median, counts = vis.neighborhood_deviation(entropy, np.ones((3, 3), dtype=bool))
        np.testing.assert_array_equal(median[:, 1, 1], 0)
        np.testing.assert_array_equal(deviation[:, 1, 1], 9)
        self.assertEqual(counts[0, 1, 1], 8)
        self.assertEqual(counts[0, 0, 0], 3)

    def test_three_windows_match_direct_reference_for_both_grids(self):
        rng = np.random.default_rng(12)
        for side in (12, 18):
            entropy = rng.uniform(0, 8, (side, side))
            valid = rng.random((side, side)) > .15
            actual, medians, counts = vis.neighborhood_deviation(entropy, valid)
            for wi, window in enumerate((3, 5, 7)):
                r = window // 2
                for y, x in np.argwhere(valid):
                    neighbors = [entropy[ny, nx]
                                 for ny in range(max(0, y - r), min(side, y + r + 1))
                                 for nx in range(max(0, x - r), min(side, x + r + 1))
                                 if (ny, nx) != (y, x) and valid[ny, nx]]
                    self.assertEqual(counts[wi, y, x], len(neighbors))
                    self.assertAlmostEqual(medians[wi, y, x], np.median(neighbors))
                    self.assertAlmostEqual(actual[wi, y, x], abs(entropy[y, x] - np.median(neighbors)))

    def test_uniform_softmax_whole_view_not_rows_or_crops(self):
        for side, n in ((18, 324), (12, 144)):
            deviations, _, _ = vis.neighborhood_deviation(np.zeros((side, side)), np.ones((side, side), bool))
            weights = vis.view_softmax(deviations)
            np.testing.assert_allclose(weights, 1 / n)
            np.testing.assert_allclose(weights.sum(axis=(1, 2)), 1)
            self.assertEqual(weights.shape, (3, side, side))
        anyres = vis.view_softmax(np.zeros((3, 12, 12)))
        # Each old crop in a 3x3 layout covers 4x4 analysis cells, sums to 1/9.
        self.assertAlmostEqual(anyres[0, :4, :4].sum(), 1 / 9)

    def test_softmax_bits_stable_per_window_and_view(self):
        values = np.array([[[0., 1.], [2., np.nan]], [[4., 3.], [0., 1.]]]) * math.log(2)
        before = values.copy()
        weights = vis.view_softmax(values)
        expected = np.exp([0., 1., 2.]) / np.exp([0., 1., 2.]).sum()
        np.testing.assert_allclose(weights[0][np.isfinite(weights[0])], expected)
        shifted = vis.view_softmax(values + 10000)
        np.testing.assert_allclose(weights, shifted, atol=1e-12)
        changed = values.copy()
        changed[0, 0, 0] += 10
        np.testing.assert_array_equal(weights[1], vis.view_softmax(changed)[1])
        np.testing.assert_array_equal(values, before)
        self.assertTrue(np.isnan(weights[0, 1, 1]))

    def test_undefined_neighbors_not_fabricated_uniform_values(self):
        valid = np.zeros((12, 12), bool)
        valid[4, 4] = True
        deviation, median, counts = vis.neighborhood_deviation(np.zeros((12, 12)), valid)
        self.assertTrue(np.isnan(deviation).all())
        self.assertTrue(np.isnan(median).all())
        self.assertFalse(counts.any())
        self.assertTrue(np.isnan(vis.view_softmax(deviation)).all())
        self.assertEqual(vis.color_limits(vis.view_softmax(deviation)), (0, 1))

    def test_anyres_fixed_total_for_different_crop_layouts(self):
        for grid in ((1, 1), (3, 3), (5, 2)):
            image = Image.new("RGB", (100, 100), (45, 55, 65))
            geometry = geometry_for(image.size, grid)
            base, mosaic = vis.prepare_views(image, geometry)
            self.assertEqual(base.size, (384, 384))
            self.assertEqual(mosaic.size, (grid[0] * 384, grid[1] * 384))
            results = vis.analyze_grids(mosaic, geometry.content_box)
            for side in (12, 18):
                self.assertEqual(results[side]["entropy_nats"].size, side ** 2)
                self.assertEqual(results[side]["softmax"].shape, (3, side, side))

    def test_both_views_recompute_each_grid_and_normalize_independently(self):
        image = Image.new("RGB", (90, 90), (45, 55, 65))
        geometry = geometry_for(image.size, (3, 3))
        base, mosaic = vis.prepare_views(image, geometry)
        with patch.object(vis, "analyze_view", wraps=vis.analyze_view) as analyze:
            groups = {"Base": vis.analyze_grids(base),
                      "AnyRes": vis.analyze_grids(mosaic, geometry.content_box)}
        self.assertEqual([call.args[1] for call in analyze.call_args_list], [12, 18, 12, 18])
        for view in groups.values():
            self.assertEqual(set(view), {12, 18})
            for side in (12, 18):
                np.testing.assert_allclose(view[side]["softmax"], 1 / side ** 2)
                np.testing.assert_allclose(view[side]["softmax"].sum(axis=(1, 2)), 1)

    def test_bad_numeric_inputs_rejected(self):
        with self.assertRaises(ValueError):
            vis.grid_edges(10, 18)
        with self.assertRaises(ValueError):
            vis.grid_entropy(np.ones((24, 24, 3), dtype=float), 12)
        with self.assertRaises(ValueError):
            vis.neighborhood_deviation(np.full((2, 2), np.inf), np.ones((2, 2)))
        for bad in (np.zeros((2, 2)), np.full((3, 2, 2), -1.), np.full((3, 2, 2), np.inf)):
            with self.assertRaises(ValueError):
                vis.view_softmax(bad)


class InputAndOutputTests(unittest.TestCase):
    def test_cached_scores_never_read_or_renamed_all_samples_default(self):
        with tempfile.TemporaryDirectory() as temp:
            results, data, image, geometry = fixture(Path(temp), sample_count=2)
            before = tree_hashes(results)
            with patch("numpy.load", side_effect=AssertionError("score NPZ read")):
                root, samples = vis.prepare_inputs(results, data)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["geometry"], geometry)
            self.assertEqual(len(vis.prepare_inputs(results, limit=1)[1]), 1)
            self.assertEqual(len(vis.prepare_inputs(samples[0]["folder"], data)[1]), 1)
            self.assertEqual(tree_hashes(results), before)
            for limit in (0, -1, True, 1.2):
                with self.assertRaises(ValueError):
                    vis.prepare_inputs(results, limit=limit)

    def test_geometry_validation_resize_rounding_and_content_box(self):
        for size in ((100, 63), (63, 100), (100, 100)):
            geometry = geometry_for(size)
            self.assertEqual(vis.Geometry.from_metadata({"geometry": asdict(geometry)}), geometry)
        bad = asdict(geometry)
        bad["paste_xy"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            vis.Geometry.from_metadata({"geometry": bad})
        bad = asdict(geometry)
        bad["tile_size"] = 448
        with self.assertRaisesRegex(ValueError, "384/14"):
            vis.Geometry.from_metadata({"geometry": bad})

    def test_missing_metadata_images_and_changed_sizes_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results, data, image, _ = fixture(root)
            output = root / "output"
            Image.new("RGB", (2, 2)).save(image)
            with self.assertRaisesRegex(ValueError, "size differs"):
                vis.run(results, output)
            self.assertFalse(output.exists())
            image.unlink()
            with self.assertRaises(FileNotFoundError):
                vis.run(results, output)
            self.assertFalse(output.exists())
            (results / "000001_example" / "metadata.json").unlink()
            with self.assertRaises(FileNotFoundError):
                vis.prepare_inputs(results)

    def test_cross_platform_relocation_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "imgs").mkdir()
            image = root / "imgs" / "test.png"
            Image.new("RGB", (8, 8)).save(image)
            for raw, saved in (("/old/dataset/imgs/test.png", "/old/dataset"),
                               (r"C:\old\dataset\imgs\test.png", r"C:\old\dataset")):
                self.assertEqual(vis.resolve_image({"image": raw}, {"data_root": saved}, root), image.resolve())
            self.assertEqual(vis.resolve_image({"image": "/missing/test.png"}, {}, root), image.resolve())
            Image.new("RGB", (8, 8)).save(root / "test.png")
            with self.assertRaisesRegex(ValueError, "Multiple"):
                vis.resolve_image({"image": "/missing/test.png"}, {}, root)
            with self.assertRaisesRegex(ValueError, "escapes"):
                vis.resolve_image({"image": "/old/../outside.png"}, {"data_root": "/old"}, root)

    def test_output_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            for target in (source, source.parent, source / "child"):
                with self.assertRaises(ValueError):
                    vis.check_output(target, source)
            output = Path(temp) / "output"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                vis.check_output(output, source)

    def test_two_overviews_only_real_rendering_and_embedded_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results, data, image_path, _ = fixture(root)
            before = tree_hashes(results)
            image_before = image_path.read_bytes()
            with patch("numpy.load", side_effect=AssertionError("score NPZ read")):
                output = vis.run(results, root / "output", data)
            folder = output / "000001_example"
            self.assertEqual({p.name for p in folder.iterdir()}, {"base_entropy_overview.png", "anyres_entropy_overview.png"})
            for name, view in (("base_entropy_overview.png", "Base"), ("anyres_entropy_overview.png", "AnyRes")):
                with Image.open(folder / name) as figure:
                    self.assertEqual(figure.size, (2400, 1410))
                    embedded = json.loads(figure.info["EntropyAnalysis"])
                self.assertEqual(embedded["schema_version"], 2)
                self.assertEqual(embedded["view"], view)
                self.assertEqual(embedded["grid_order"], [12, 18])
                self.assertEqual(set(embedded["grids"]), {"12", "18"})
                self.assertEqual(embedded["definition"]["colormap"], "jet")
                for side in (12, 18):
                    row = embedded["grids"][str(side)]
                    self.assertEqual(row["analysis_grid"], [side, side])
                    weights = np.asarray(row["arrays"]["softmax"], dtype=float)
                    self.assertEqual(weights.shape, (3, side, side))
                    np.testing.assert_allclose(np.nansum(weights, axis=(1, 2)), 1)
                    deviations = np.asarray(row["arrays"]["deviation_nats"], dtype=float)
                    # Each grid uses DEVIATION, not ordinary entropy or another grid's values.
                    np.testing.assert_allclose(weights, vis.view_softmax(deviations))
                    np.testing.assert_allclose(row["color_limits"], vis.color_limits(weights))
            manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_samples"], 1)
            self.assertFalse(manifest["definition"]["model_loaded"])
            self.assertEqual(manifest["definition"]["four_groups"], ["Base12", "Base18", "AnyRes12", "AnyRes18"])
            self.assertEqual(tree_hashes(results), before)
            self.assertEqual(image_path.read_bytes(), image_before)
            with self.assertRaises(FileExistsError):
                vis.run(results, output)

    def test_failure_manifest_not_false_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results, _, _, _ = fixture(root)
            with patch.object(vis, "draw_overview", side_effect=RuntimeError("plot failed")):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    vis.run(results, root / "output")
            manifest = json.loads((root / "output" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_samples"], 0)

    def test_all_panels_keep_original_aspect_after_heatmap_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Image.new("RGB", (60, 120), (40, 40, 40))
            analyses = vis.analyze_grids(image)

            def check_axes(figure, *args, **kwargs):
                self.assertEqual(len(figure.axes), 10)  # 8 panels + 2 colorbars
                for ax in figure.axes[:8]:
                    self.assertEqual(ax.get_aspect(), 2.)

            with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=check_axes) as save:
                vis.draw_overview(image, analyses, Path(temp) / "aspect.png", "AnyRes", "synthetic", {})
            save.assert_called_once()

    def test_cli_import_is_standalone_and_no_torch(self):
        code = "import run_entropy_18_12; import sys; assert 'torch' not in sys.modules; assert 'llava_pruning' not in sys.modules; print(run_entropy_18_12.SCRIPT_VERSION)"
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2.0-both-views-grids12-18", result.stdout)
        self.assertIsNone(vis.build_parser().parse_args([]).limit)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            vis.build_parser().parse_args(["--limit", "0"])


if __name__ == "__main__":
    unittest.main()
