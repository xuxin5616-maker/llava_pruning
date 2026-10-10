"""Offline cache aggregation, renamed archive and spatial rendering tests."""

import builtins
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import visualize_patch_means as vis


def fixture(root, name="000001_000000108", suffix="scores.npz"):
    data = root / "data"
    data.mkdir(exist_ok=True)
    image = data / "source.png"
    Image.new("RGB", (56, 28), "steelblue").save(image)
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "run.json").write_text(json.dumps({"data_root": str(data)}), encoding="utf-8")
    folder = results / name
    folder.mkdir()
    metadata = {"id": "000000108", "image": str(image), "gt": 1,
                "score_shape": [5, 2, 4],
                "geometry": {"original_size": [56, 28], "grid": [2, 1], "tile_size": 28,
                             "patch_size": 14, "resized_size": [56, 28], "paste_xy": [0, 0]}}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    global_scores = np.tile(np.array([[[-1, -0.8, -0.6, -0.4], [-0.5, -0.3, -0.1, 0.1]]], dtype=np.float32), (5, 1, 1))
    local_scores = np.clip(global_scores - 0.1, -1, 1)
    # Write the same original archive bytes under any suffix, including .npy.log.
    with (folder / suffix).open("wb") as stream:
        np.savez_compressed(stream, global_scores=global_scores, local_scores=local_scores, stages=vis.CACHE_STAGES)
    return results, folder, metadata, {"global": global_scores, "local": local_scores}


class CachedMeanTests(unittest.TestCase):
    def test_restore_log_suffix_preserves_bytes_and_is_idempotent(self):
        for suffix in ("scores.npz.log", "scores.npy.log", "scores.npz.LOG", "scores.npz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                _, folder, meta, _ = fixture(Path(tmp), suffix=suffix)
                source = folder / suffix
                before = source.read_bytes()
                vis.load_scores(source, vis.validate_geometry(meta), meta)
                restored = vis.restore_score_filename(source)
                expected = source.with_suffix("") if source.suffix.lower() == ".log" else source
                self.assertEqual(restored, expected)
                self.assertEqual(restored.read_bytes(), before)
                if restored != source:
                    self.assertFalse(source.exists())
                self.assertEqual(vis.restore_score_filename(restored), restored)

    def test_restore_name_conflict_never_overwrites_either_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, _, _ = fixture(Path(tmp), suffix="scores.npz.log")
            source = folder / "scores.npz.log"
            before = source.read_bytes()
            target = folder / "scores.npz"
            target.write_bytes(b"existing data must be preserved")
            with self.assertRaisesRegex(FileExistsError, "destination already exists"):
                vis.restore_score_filename(source)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(target.read_bytes(), b"existing data must be preserved")

    def test_npz_and_renamed_log_archives_identical(self):
        for suffix in ("scores.npz", "scores.npz.log", "scores.npy", "scores.npy.log"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                results, folder, metadata, expected = fixture(Path(tmp), suffix=suffix)
                source = vis.find_score_file(folder)
                self.assertEqual(source.name, suffix)
                loaded = vis.load_scores(source, vis.validate_geometry(metadata), metadata)
                means, counts = vis.aggregate_scores(loaded)
                self.assertEqual(set(loaded), {"global"})
                for kind in vis.KINDS:
                    np.testing.assert_array_equal(loaded[kind], expected[kind][:4])
                    np.testing.assert_allclose(means[kind], expected[kind][:4].mean(axis=-1), atol=1e-7)
                    self.assertTrue(np.all(counts[kind] == 4))
                self.assertEqual(vis.sample_folders(results), [folder])
                self.assertEqual(vis.sample_folders(folder), [folder])

    def test_average_all_scores_not_features_or_visible_only(self):
        scores = {"global": np.array([[[-1., -1., 0., 0.]]])}
        means, counts = vis.aggregate_scores(scores)
        self.assertEqual(means["global"][0, 0], -0.5)
        self.assertEqual(counts["global"][0, 0], 4)

    def test_missing_values_counts_and_fully_missing(self):
        scores = {"global": np.array([[[-1., np.nan, -0.5], [np.nan] * 3]])}
        means, counts = vis.aggregate_scores(scores)
        self.assertEqual(means["global"][0, 0], -0.75)
        for kind in vis.KINDS:
            self.assertTrue(np.isnan(means[kind][0, 1]))
            np.testing.assert_array_equal(counts[kind], [[2, 0]])

    def test_reordered_stage_axis_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, meta, scores = fixture(Path(tmp))
            order = [4, 2, 0, 3, 1]
            # Distinct layer values catch an accidental label-only reordering.
            scores = {k: values * np.linspace(.1, 1, 5)[:, None, None] for k, values in scores.items()}
            np.savez(folder / "scores.npz", global_scores=scores["global"][order],
                     local_scores=scores["local"][order], stages=np.array(vis.CACHE_STAGES)[order])
            actual = vis.load_scores(folder / "scores.npz", vis.validate_geometry(meta), meta)
            for kind in vis.KINDS:
                np.testing.assert_array_equal(actual[kind], scores[kind][:4])

    def test_local_is_optional_and_never_read_even_when_unusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, meta, scores = fixture(Path(tmp))
            source = folder / "scores.npz"
            for extras in ({}, {"local_scores": np.array([object()], dtype=object)},
                           {"local_scores": np.full((5, 2, 4), np.inf)}):
                with self.subTest(local_fields=list(extras)):
                    np.savez(source, global_scores=scores["global"], stages=vis.CACHE_STAGES, **extras)
                    before = source.read_bytes()
                    loaded = vis.load_scores(source, vis.validate_geometry(meta), meta)
                    self.assertEqual(set(loaded), {"global"})
                    np.testing.assert_array_equal(loaded["global"], scores["global"][:4])
                    self.assertEqual(source.read_bytes(), before)
            np.savez(source, local_scores=scores["local"], stages=vis.CACHE_STAGES)
            with self.assertRaisesRegex(ValueError, "missing score fields.*global_scores"):
                vis.load_scores(source, vis.validate_geometry(meta), meta)

    def test_projector_values_do_not_affect_retained_scores_or_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, meta, scores = fixture(Path(tmp))
            source = folder / "scores.npz"
            geometry = vis.validate_geometry(meta)
            baseline = vis.load_scores(source, geometry, meta)
            for kind in vis.KINDS:
                scores[kind][4] = np.inf  # Even unusable projector scores are irrelevant.
            np.savez(source, global_scores=scores["global"], local_scores=scores["local"], stages=vis.CACHE_STAGES)
            before = source.read_bytes()
            loaded = vis.load_scores(source, geometry, meta)
            self.assertEqual(source.read_bytes(), before)
            for kind in vis.KINDS:
                np.testing.assert_array_equal(loaded[kind], baseline[kind])
            means, counts = vis.aggregate_scores(loaded)
            displayed, _ = vis.build_display_means(means, counts)
            limits = vis.color_limits(displayed, vis.tile_rectangles(geometry), "sample")
            self.assertTrue(all(np.isfinite(limits)))
            self.assertNotIn("projector", vis.STAGES)

    def test_cache_without_projector_supported_and_missing_or_duplicate_layers_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, meta, scores = fixture(Path(tmp))
            source = folder / "scores.npz"
            geometry = vis.validate_geometry(meta)
            meta["score_shape"] = [4, 2, 4]
            order = [3, 1, 0, 2]
            np.savez(source, global_scores=scores["global"][order], local_scores=scores["local"][order],
                     stages=np.array(vis.SIGLIP_STAGES)[order])
            actual = vis.load_scores(source, geometry, meta)
            for kind in vis.KINDS:
                np.testing.assert_array_equal(actual[kind], scores[kind][:4])
            for stages in (vis.SIGLIP_STAGES[:3] + ("projector",), vis.SIGLIP_STAGES[:3] + ("siglip_07",)):
                np.savez(source, global_scores=scores["global"][:4], local_scores=scores["local"][:4], stages=stages)
                with self.assertRaisesRegex(ValueError, "Expected SigLIP"):
                    vis.load_scores(source, geometry, meta)

    def test_ambiguous_files_and_invalid_archives_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, meta, _ = fixture(Path(tmp))
            geometry = vis.validate_geometry(meta)
            extra = folder / "scores.npy.log"
            extra.write_bytes(b"this is a text log, not scores")
            with self.assertRaisesRegex(ValueError, "Multiple score"):
                vis.find_score_file(folder)
            with self.assertRaisesRegex(ValueError, "binary score archive"):
                vis.load_scores(extra, geometry, meta)
            plain = folder / "plain.npy"
            np.save(plain, np.ones((5, 2, 4)))
            with self.assertRaisesRegex(ValueError, "single NPY array"):
                vis.load_scores(plain, geometry, meta)
            object_file = folder / "object.npy"
            np.save(object_file, {"dangerous": "not loaded"})
            with self.assertRaises(ValueError):
                vis.load_scores(object_file, geometry, meta)

    def test_wrong_shapes_nonfinite_scores_and_stage_names_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, meta, scores = fixture(Path(tmp))
            source = folder / "scores.npz"
            geometry = vis.validate_geometry(meta)
            for values, stages in ((np.ones((5, 1, 4)), vis.CACHE_STAGES),
                                   (np.full((5, 2, 4), np.inf), vis.CACHE_STAGES),
                                   (np.full((5, 2, 4), 2.), vis.CACHE_STAGES),
                                   (scores["global"], ("other",) * 5)):
                np.savez(source, global_scores=values, local_scores=scores["local"], stages=stages)
                with self.assertRaises(ValueError):
                    vis.load_scores(source, geometry, meta)


class CrossLayerMeanTests(unittest.TestCase):
    def test_equal_layer_weights_preserve_raw_means_without_normalization(self):
        raw = np.array([[-.9, -.8, -.7], [-.1, -.5, -.9], [-.2, -.4, -.6], [-.6, -.2, -.4]])
        means = {"global": raw.copy()}
        counts = {kind: np.array([[4, 1, 4], [2, 4, 1], [3, 3, 3], [4, 4, 4]]) for kind in vis.KINDS}
        originals = {kind: values.copy() for kind, values in means.items()}
        displayed, displayed_counts = vis.build_display_means(means, counts)
        self.assertEqual(set(displayed), {"global"})
        for kind in vis.KINDS:
            np.testing.assert_array_equal(means[kind], originals[kind])
            np.testing.assert_array_equal(displayed[kind][:4], means[kind])
            for row, num in enumerate((2, 3, 4), start=4):
                np.testing.assert_allclose(displayed[kind][row], means[kind][:num].mean(axis=0))
                np.testing.assert_array_equal(displayed_counts[kind][row], counts[kind][:num].sum(axis=0))
        np.testing.assert_allclose(displayed["global"][4], [-.5, -.65, -.8])
        self.assertTrue(np.all(displayed["global"] < 0))
        pooled_tokens = (means["global"][:2] * counts["global"][:2]).sum(axis=0) / counts["global"][:2].sum(axis=0)
        self.assertFalse(np.allclose(displayed["global"][4], pooled_tokens))

    def test_missing_layer_propagates_without_silently_changing_group(self):
        means = {kind: np.tile([[-.8, -.5, -.2]], (4, 1)) for kind in vis.KINDS}
        counts = {kind: np.full((4, 3), 4) for kind in vis.KINDS}
        for kind in vis.KINDS:
            means[kind][2, 0] = np.nan
            means[kind][0, 1] = np.nan
            counts[kind][2, 0] = 0
            counts[kind][0, 1] = 0
        displayed, displayed_counts = vis.build_display_means(means, counts)
        for kind in vis.KINDS:
            self.assertEqual(displayed[kind][4, 0], -.8)  # First 2 layers unaffected.
            self.assertTrue(np.isnan(displayed[kind][5:, 0]).all())
            self.assertTrue(np.isnan(displayed[kind][4:, 1]).all())
            self.assertEqual(displayed[kind][3, 1], -.5)
            self.assertEqual(displayed_counts[kind][5, 0], 8)

    def test_reject_accidentally_including_projector_in_group_inputs(self):
        means = {kind: np.ones((5, 2)) for kind in vis.KINDS}
        counts = {kind: np.ones((5, 2), dtype=int) for kind in vis.KINDS}
        with self.assertRaisesRegex(ValueError, "exactly SigLIP"):
            vis.build_display_means(means, counts)


class ScaledScoreTests(unittest.TestCase):
    def test_fixed_linear_mapping_endpoints_missing_and_raw_preservation(self):
        raw = np.array([[-1., -.8, -.2, 0., .6, 1., np.nan]])
        original = raw.copy()
        scaled = vis.scale_scores_01(raw)
        np.testing.assert_allclose(scaled, [[0., .1, .4, .5, .8, 1., np.nan]], equal_nan=True)
        np.testing.assert_array_equal(raw, original)
        # Constant rows are mapped by the same formula, never forced to 0.5.
        np.testing.assert_allclose(vis.scale_scores_01(np.full((7, 3), -.8)), .1)

    def test_roundoff_only_clipping_and_invalid_scores_rejected(self):
        np.testing.assert_array_equal(vis.scale_scores_01([-1.0000005, 1.0000005]), [0, 1])
        for invalid in (np.inf, -np.inf, 1.01, -1.01):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "out-of-range"):
                vis.scale_scores_01([invalid])

    def test_cross_layer_average_is_scaled_once_after_averaging(self):
        raw = np.array([[-.9, -.5], [-.5, .1], [.1, .5], [.5, .9]])
        means, _ = vis.build_display_means({"global": raw}, {"global": np.ones((4, 2), dtype=int)})
        actual = vis.scale_scores_01(means["global"])
        for row, count in enumerate((2, 3, 4), start=4):
            np.testing.assert_allclose(actual[row], (raw[:count].mean(axis=0) + 1) / 2)
            np.testing.assert_allclose(actual[row], vis.scale_scores_01(raw[:count]).mean(axis=0))

    def test_color_limits_preserve_raw_values_and_ignore_local(self):
        rectangles = [(0, 0, 1/3, 1)] * 3
        means = {"global": np.array([-.95 + i*.1 + .02*np.array([0, .25, 1])
                                     for i in range(len(vis.STAGES))]),
                 "local": np.full((len(vis.STAGES), 3), np.inf)}
        original = means["global"].copy()
        self.assertEqual(vis.color_limits(means, rectangles, "fixed"), (0, 1))
        np.testing.assert_allclose(vis.color_limits(means, rectangles, "sample"),
                                   [(original.min() + 1) / 2, (original.max() + 1) / 2])
        np.testing.assert_array_equal(means["global"], original)

    def test_raw_tile_averages_and_constant_rows_are_not_rescaled(self):
        scores = {"global": np.tile([[[-1., 1.], [0., .5], [.5, .5]]], (4, 1, 1))}
        raw_means, counts = vis.aggregate_scores(scores)
        means, _ = vis.build_display_means(raw_means, counts)
        np.testing.assert_allclose(means["global"], np.tile([0, .25, .5], (len(vis.STAGES), 1)))
        raw_means["global"][:] = -.6
        means, _ = vis.build_display_means(raw_means, counts)
        np.testing.assert_allclose(means["global"], -.6)
        np.testing.assert_allclose(vis.color_limits(means, [(0, 0, 1, 1)] * 3, "sample"), [.195, .205])
        np.testing.assert_allclose(means["global"], -.6)

    def test_removed_layer_mode_is_rejected_without_writing(self):
        with self.assertRaises(ValueError):
            vis.color_limits({"global": np.ones((7, 1))}, [(0, 0, 1, 1)], "layer")
        with tempfile.TemporaryDirectory() as tmp:
            results, _, _, _ = fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "per-layer normalization was removed"):
                vis.run(results, color_scale="layer")
            self.assertFalse((Path(tmp) / "results_patch_means").exists())

    def test_csv_raw_and_scaled_global_means_including_invisible_and_missing(self):
        geometry = vis.validate_geometry({"geometry": {
            "original_size": [56, 1], "grid": [2, 4], "tile_size": 28,
            "patch_size": 14, "resized_size": [56, 1], "paste_xy": [0, 55]}})
        means = {kind: np.ones((len(vis.STAGES), 8)) for kind in vis.KINDS}
        counts = {kind: np.full((len(vis.STAGES), 8), 4) for kind in vis.KINDS}
        for kind in vis.KINDS:
            means[kind][:, 2:4] = [-.9, -.7]
        means["global"][1, 2] = np.nan
        counts["global"][1, 2] = 0
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "values.csv"
            vis.write_values(output, means, counts, geometry, 4)
            with output.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 56)
            self.assertEqual(rows[0]["global_mean"], "1.0")
            self.assertEqual(rows[0]["global_score_01"], "1.0")
            self.assertEqual(rows[0]["visible"], "False")
            self.assertEqual(float(rows[2]["global_mean"]), -.9)
            self.assertEqual(float(rows[3]["global_mean"]), -.7)
            self.assertAlmostEqual(float(rows[2]["global_score_01"]), .05)
            self.assertAlmostEqual(float(rows[3]["global_score_01"]), .15)
            self.assertEqual(rows[10]["global_mean"], "")
            self.assertEqual(rows[10]["global_score_01"], "")
            self.assertEqual(rows[10]["global_valid_tokens"], "0")
            self.assertFalse(any("local" in key or "normalized" in key for key in rows[0]))


class OfflineGeometryTests(unittest.TestCase):
    def test_rectangles_tile_the_image_without_stretched_padding(self):
        geometry = {"original_size": [31, 17], "grid": [2, 2], "tile_size": 28,
                    "patch_size": 14, "resized_size": [56, 31], "paste_xy": [0, 12]}
        actual = vis.tile_rectangles(vis.validate_geometry({"geometry": geometry}))
        np.testing.assert_allclose(actual, [(0, 0, .5, 16/31), (.5, 0, .5, 16/31),
                                            (0, 16/31, .5, 15/31), (.5, 16/31, .5, 15/31)])
        self.assertAlmostEqual(sum(w*h for _, _, w, h in actual), 1.)
        geometry["paste_xy"] = [0, 13]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            vis.validate_geometry({"geometry": geometry})

    def test_invisible_tiles_preserved_but_do_not_change_color_scale(self):
        geometry = {"original_size": [56, 1], "grid": [2, 4], "tile_size": 28,
                    "patch_size": 14, "resized_size": [56, 1], "paste_xy": [0, 55]}
        rectangles = vis.tile_rectangles(vis.validate_geometry({"geometry": geometry}))
        self.assertEqual([i for i, r in enumerate(rectangles) if r], [2, 3])
        means = {kind: np.ones((len(vis.STAGES), 8)) for kind in vis.KINDS}
        means["global"][:, 2:4] = [-.9, -.7]
        np.testing.assert_allclose(vis.color_limits(means, rectangles, "sample"), [.05, .15])
        self.assertEqual(vis.color_limits(means, rectangles, "fixed"), (0, 1))
        for values in means.values():
            values[:] = np.nan
        self.assertEqual(vis.color_limits(means, rectangles, "sample"), (0, 1))

    def test_cross_platform_image_relocation_preserves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "imgs").mkdir()
            image = root / "imgs" / "a.png"
            Image.new("RGB", (4, 4)).save(image)
            for old_root, old_image in (("/server/data", "/server/data/imgs/a.png"),
                                        ("C:\\old\\data", "C:\\old\\data\\imgs\\a.png")):
                actual = vis.resolve_image({"image": old_image}, {"data_root": old_root}, root)
                self.assertEqual(actual, image.resolve())
            self.assertEqual(vis.resolve_image({"image": "/old/a.png"}, {}, root), image.resolve())
            with self.assertRaisesRegex(ValueError, "outside"):
                vis.resolve_image({"image": "/elsewhere/a.png"}, {"data_root": "/server/data"}, root)
            with self.assertRaisesRegex(ValueError, "escapes"):
                vis.resolve_image({"image": "/server/data/../../secret.png"}, {"data_root": "/server/data"}, root)


class OfflineRunnerTests(unittest.TestCase):
    def test_saved_script_reports_scaled_global_version_and_help(self):
        script = str(Path(vis.__file__).resolve())
        version = subprocess.run([sys.executable, script, "--version"],
                                 check=True, capture_output=True, text=True)
        self.assertIn("4.2-global-horizontal", version.stdout)
        help_result = subprocess.run([sys.executable, script, "--help"],
                                     check=True, capture_output=True, text=True)
        self.assertIn("{fixed,sample}", help_result.stdout)
        self.assertNotIn("{layer", help_result.stdout)

    def test_full_run_renames_archive_cpu_only_and_contents_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, folder, _, expected = fixture(root, suffix="scores.npz.log")
            sources = [path for path in root.rglob("*") if path.is_file()]
            digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
            original_import = builtins.__import__
            def no_model_import(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "transformers", "llava", "llava_pruning", "accelerate"}:
                    raise AssertionError(f"Offline visualization must not import {name}")
                return original_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=no_model_import):
                output = vis.run(results)
            self.assertEqual(output, (root / "results_patch_means").resolve())
            target = output / folder.name
            with Image.open(target / "patch_means.png") as png:
                self.assertEqual(png.size, (2400, 1350))
            with (target / "patch_means.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 14)
            self.assertEqual([row["stage"] for row in rows[::2]], list(vis.STAGES))
            self.assertNotIn("projector", {row["stage"] for row in rows})
            self.assertEqual(rows[-1]["source_layers"], "+".join(vis.SIGLIP_STAGES))
            self.assertEqual(rows[-1]["layer_count"], "4")
            self.assertEqual(rows[-1]["global_valid_tokens"], "16")
            self.assertEqual(rows[-1]["total_tokens"], "16")
            self.assertEqual(rows[0]["stage"], "siglip_07")
            self.assertAlmostEqual(float(rows[0]["global_mean"]), expected["global"][0, 0].mean(), places=6)
            for row in rows:
                self.assertAlmostEqual(float(row["global_score_01"]), (float(row["global_mean"]) + 1) / 2)
            self.assertFalse(any("local" in key or "normalized" in key for key in rows[0]))
            manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["model_loaded"])
            self.assertEqual(manifest["colormap"], "jet")
            self.assertEqual(manifest["overlay_alpha"], .7)
            self.assertEqual(manifest["color_scale"], "fixed")
            self.assertEqual(manifest["script_version"], vis.SCRIPT_VERSION)
            self.assertEqual(manifest["figure_layout"]["rows"], 2)
            self.assertEqual(manifest["figure_layout"]["columns"], 4)
            self.assertFalse(manifest["figure_layout"]["original_tile_labels"])
            self.assertEqual(manifest["figure_layout"]["panels"], [list(row) for row in vis.PANEL_STAGES])
            self.assertEqual(manifest["score_kind"], "global")
            self.assertEqual(manifest["score_normalization"], "fixed_linear_0_1")
            self.assertEqual(manifest["score_scaling"], vis.SCORE_SCALING)
            self.assertNotIn("layer_normalization", manifest)
            self.assertEqual(manifest["stages"], list(vis.STAGES))
            self.assertEqual(manifest["cross_layer_average"]["groups"]["mean_first_4"], list(vis.SIGLIP_STAGES))
            original_source = folder / "scores.npz.log"
            restored_source = folder / "scores.npz"
            self.assertFalse(original_source.exists())
            self.assertTrue(restored_source.is_file())
            for path, digest in digests.items():
                actual = restored_source if path == original_source else path
                self.assertEqual(hashlib.sha256(actual.read_bytes()).hexdigest(), digest)
            saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(saved["source_scores"]), restored_source.resolve())
            self.assertEqual(saved["color_limits"], [0, 1])
            self.assertEqual(saved["display_values"], "global_score_01")
            self.assertEqual(saved["script_version"], vis.SCRIPT_VERSION)
            self.assertEqual(saved["figure_layout"], manifest["figure_layout"])
            self.assertEqual(saved["score_normalization"], "fixed_linear_0_1")
            self.assertEqual(saved["score_scaling"]["formula"], "(raw_score + 1) / 2")
            self.assertNotIn("layer_raw_ranges", saved)
            with self.assertRaises(FileExistsError):
                vis.run(results)

    def test_invalid_archive_keeps_original_log_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, folder, _, _ = fixture(Path(tmp), suffix="scores.npz.log")
            source = folder / "scores.npz.log"
            source.write_bytes(b"not an archive")
            with self.assertRaisesRegex(ValueError, "binary score archive"):
                vis.run(results)
            self.assertEqual(source.read_bytes(), b"not an archive")
            self.assertFalse((folder / "scores.npz").exists())

    def test_uniform_patch_color_missing_gray_and_common_norm(self):
        from matplotlib.figure import Figure
        from matplotlib import colormaps
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, meta, _ = fixture(root)
            means = {kind: np.tile([[-.5, np.nan]], (len(vis.STAGES), 1)) for kind in vis.KINDS}
            def check_figure(fig, *args, **kwargs):
                self.assertEqual(len(fig.axes), 9)  # 8 panels + a shared colorbar
                self.assertEqual([ax.get_title() for ax in fig.axes[:8]],
                                 [*vis.LABELS[:4], "Original", *vis.LABELS[4:]])
                for ax in fig.axes[:8]:
                    patches = [p for p in ax.patches if p.get_fill()]
                    if patches:
                        self.assertEqual(len(patches), 2)
                        np.testing.assert_allclose(patches[0].get_facecolor(), (*colormaps["jet"](.25)[:3], .7))
                        np.testing.assert_allclose(patches[1].get_facecolor(), (184/255, 184/255, 184/255, 1.))
            with patch.object(Figure, "savefig", check_figure):
                vis.render_figure(Image.new("RGB", (56, 28)), means, vis.validate_geometry(meta),
                                  "test", None, root / "figure.png", "fixed")

    def test_default_render_uses_scaled_labels_and_preserves_colors(self):
        from matplotlib.figure import Figure
        from matplotlib import colormaps
        with tempfile.TemporaryDirectory() as tmp:
            _, _, meta, _ = fixture(Path(tmp))
            means = {
                "global": np.array([[-.95 + .1*i, -.85 + .1*i] for i in range(len(vis.STAGES))]),
            }
            original = means["global"].copy()
            def check_figure(fig, *args, **kwargs):
                self.assertEqual(len(fig.axes), 9)
                for row, panel in enumerate((0, 1, 2, 3, 5, 6, 7)):
                    ax = fig.axes[panel]
                    patches = [p for p in ax.patches if p.get_fill()]
                    for tile_patch, value, text in zip(patches, means["global"][row], ax.texts):
                        palette_value = (value + 1) / 2
                        np.testing.assert_allclose(tile_patch.get_facecolor(), (*colormaps["jet"](palette_value)[:3], .7))
                        self.assertTrue(text.get_text().endswith(f"\n{palette_value:.4f}"))
                np.testing.assert_allclose(fig.axes[-1].get_xlim(), [0, 1])
                self.assertIn("(raw mean -cos score + 1) / 2", fig.axes[-1].get_xlabel())
                self.assertIn("No per-layer min-max", fig._supxlabel.get_text())
                self.assertIn("Local and projector excluded", fig._supxlabel.get_text())
            with patch.object(Figure, "savefig", check_figure):
                limits = vis.render_figure(Image.new("RGB", (56, 28)), means, vis.validate_geometry(meta),
                                          "test", None, Path(tmp) / "figure.png")
            self.assertEqual(limits, [0, 1])
            np.testing.assert_array_equal(means["global"], original)

    def test_compact_horizontal_layout_original_has_boundaries_but_no_labels(self):
        from matplotlib.figure import Figure
        with tempfile.TemporaryDirectory() as tmp:
            _, _, meta, _ = fixture(Path(tmp))
            means = {"global": np.full((len(vis.STAGES), 2), -.5)}
            def check_figure(fig, *args, **kwargs):
                fig.canvas.draw()
                original = fig.axes[4]
                self.assertEqual(original.get_title(), "Original")
                self.assertEqual(len(original.texts), 0)
                self.assertEqual(len(original.patches), 2)
                self.assertTrue(all(not patch.get_fill() for patch in original.patches))
                self.assertEqual(sum(ax.get_title() == "Original" for ax in fig.axes), 1)
                positions = [ax.get_position() for ax in fig.axes[:8]]
                for row in (positions[:4], positions[4:]):
                    self.assertTrue(all(left.x1 < right.x0 for left, right in zip(row, row[1:])))
                    np.testing.assert_allclose([pos.y0 for pos in row], row[0].y0)
                self.assertGreater(positions[0].y0, positions[4].y1)
                for panel in (0, 1, 2, 3, 5, 6, 7):
                    self.assertEqual([text.get_text() for text in fig.axes[panel].texts], ["1\n0.2500", "2\n0.2500"])
                # Inspect actual rendered titles/colorbar label for canvas clipping.
                renderer = fig.canvas.get_renderer()
                for artist in [ax.title for ax in fig.axes[:8]] + [fig.axes[-1].xaxis.label, fig._suptitle, fig._supxlabel]:
                    bounds = artist.get_window_extent(renderer)
                    self.assertGreaterEqual(bounds.x0, 0)
                    self.assertGreaterEqual(bounds.y0, 0)
                    self.assertLessEqual(bounds.x1, fig.bbox.width)
                    self.assertLessEqual(bounds.y1, fig.bbox.height)
            with patch.object(Figure, "savefig", check_figure):
                vis.render_figure(Image.new("RGB", (56, 28)), means, vis.validate_geometry(meta),
                                  "layout-check", 0, Path(tmp) / "figure.png")

    def test_wrong_image_size_and_output_overlap_fail_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, _, meta, _ = fixture(root)
            for target in (results, results / "redraw", root):
                with self.assertRaisesRegex(ValueError, "separate"):
                    vis.run(results, target)
            Image.new("RGB", (10, 10)).save(meta["image"])
            with self.assertRaisesRegex(ValueError, "size no longer matches"):
                vis.run(results)
            self.assertFalse((root / "results_patch_means").exists())

    def test_incomplete_cache_does_not_silently_skip_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, _, _, _ = fixture(Path(tmp))
            incomplete = results / "000002_incomplete"
            incomplete.mkdir()
            np.savez(incomplete / "scores.npz", data=np.zeros(4))
            with self.assertRaises(FileNotFoundError):
                vis.run(results)

    def test_configuration_at_top_and_cli_override(self):
        with patch.object(vis, "RESULTS_DIR", "my/results"), patch.object(vis, "OUTPUT_DIR", ""), \
             patch.object(vis, "DATA_ROOT", ""), patch.object(vis, "run") as run:
            with patch("sys.argv", ["visualize_patch_means.py"]):
                vis.main()
                self.assertEqual(run.call_args.args[0], Path(vis.__file__).resolve().parent / "my/results")
                self.assertEqual(run.call_args.args[-1], "fixed")
            with patch("sys.argv", ["visualize_patch_means.py", "--results-dir", "other", "--color-scale", "sample"]):
                vis.main()
                self.assertEqual(run.call_args.args, (Path("other"), None, None, "sample"))


if __name__ == "__main__":
    unittest.main()
