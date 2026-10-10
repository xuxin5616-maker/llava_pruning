"""CPU tests for the integrated feature-residual CLI, cache metadata and plots."""
import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import runVFlowOpt as vis
from feature_residual_math import stage_residuals
from feature_residual_capture import STAGES, LAYERS

PROJECT = Path(__file__).resolve().parents[1]


def fixture(root):
    data = root / "data"
    data.mkdir()
    image_path = data / "source.png"
    pixels = np.zeros((120, 240, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(240)
    pixels[..., 1] = np.arange(120)[:, None]
    pixels[40:70, 100:130] = [255, 255, 255]
    Image.fromarray(pixels).save(image_path)
    geometry = vis.ScoreGeometry.create((240, 120), (2, 1), 384, 14)
    results = root / "results"
    folder = results / "000001_example"
    folder.mkdir(parents=True)
    metadata = {"id": "example", "image": str(image_path), "gt": 0, "geometry": asdict(geometry)}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (results / "run.json").write_text(json.dumps({"data_root": str(data)}), encoding="utf-8")
    (folder / "scores.npz.log").write_bytes(b"not NPZ: must never be read or renamed")
    records = [{"question_id": str(i), "image": "source.png", "gt": i % 2} for i in range(3)]
    input_json = root / "input.json"
    input_json.write_text(json.dumps(records), encoding="utf-8")
    return image_path, geometry, results, input_json, data


def synthetic_scores(geometry):
    """SYNTHETIC test vectors, not a checkpoint inference result."""
    y, x = np.indices((27, 27))
    features = np.ones((1 + geometry.tile_count, 729, 4), dtype=np.float32)
    features[..., 1] = x.reshape(-1)[None] / 27
    features[..., 2] = y.reshape(-1)[None] / 27
    features[:, 9 * 27 + 8:9 * 27 + 12, 3] = 8
    stages = [stage_residuals(features * (1 + i * .2), geometry) for i in range(len(STAGES))]
    return {key: np.stack([stage[key] for stage in stages]) for key in stages[0]}


class InputAndDisplayTests(unittest.TestCase):
    def test_cached_paths_only_and_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            image, geometry, results, input_json, data = fixture(Path(temp))
            cache = results / "000001_example" / "scores.npz.log"
            before = cache.read_bytes()
            with patch("numpy.load", side_effect=AssertionError("cached scalars were read")):
                samples, run, source = vis.prepare_inputs(results_dir=results)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["saved_geometry"], asdict(geometry))
            self.assertEqual(cache.read_bytes(), before)
            self.assertEqual(len(vis.prepare_inputs(input_json=input_json, data_root=data)[0]), 3)
            self.assertEqual(len(vis.prepare_inputs(input_json=input_json, data_root=data, limit=1)[0]), 1)
            for limit in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    vis.prepare_inputs(results_dir=results, limit=limit)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                vis.prepare_inputs(results_dir=results, input_json=input_json)
            with self.assertRaisesRegex(ValueError, "data-root"):
                vis.prepare_inputs(input_json=input_json)
            Image.new("RGB", (20, 20)).save(image)
            with self.assertRaisesRegex(ValueError, "size changed"):
                vis.prepare_inputs(results_dir=results)

    def test_no_output_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            for dest in (source, source / "child", source.parent):
                with self.assertRaises(ValueError):
                    vis.check_output(dest, source)
            output = Path(temp) / "output"
            self.assertEqual(vis.check_output(output, source), output.resolve())
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                vis.check_output(output, source)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_joint_minmax_not_independent_crop_and_raw_preserved(self):
        base = np.array([[2., 4., np.nan]], dtype=np.float32)
        tiles = np.array([[[0., 8., np.nan]]], dtype=np.float32)
        original = base.copy()
        b, t, limits, raw_range = vis.display_scores(base, tiles, "minmax")
        np.testing.assert_allclose(b[0, :2], [.25, .5])
        np.testing.assert_allclose(t[0, 0, :2], [0, 1])
        self.assertEqual(limits, (0, 1))
        self.assertEqual(raw_range, [0, 8])
        np.testing.assert_array_equal(base, original)
        b, t, limits, _ = vis.display_scores(base, tiles, "raw")
        np.testing.assert_array_equal(b, base)
        np.testing.assert_array_equal(t, tiles)
        self.assertEqual(limits, (0, 8))
        b, _, _, _ = vis.display_scores(np.array([2., np.nan]), np.array([2.]), "minmax")
        self.assertEqual(b[0], .5)
        self.assertTrue(np.isnan(b[1]))
        self.assertIsNone(vis.display_scores(np.array([np.nan]), np.array([np.nan]), "raw")[3])
        with self.assertRaises(ValueError):
            vis.display_scores(np.array([np.nan]), np.array([np.nan]), "softmax")
        with self.assertRaises(ValueError):
            vis.display_scores(np.array([np.inf]), np.ones(1), "raw")

    def test_native_pixel_support_and_nan_padding(self):
        geometry = vis.ScoreGeometry.create((384, 384), (1, 1), 384, 14)
        result = vis.base_score_map(np.ones(729), geometry)
        self.assertEqual(result.shape, (384, 384))
        np.testing.assert_array_equal(result[:378, :378], 1)
        self.assertTrue(np.isnan(result[378:]).all())
        self.assertTrue(np.isnan(result[:, 378:]).all())

    def test_cli_defaults_and_help_do_not_load_torch(self):
        args = vis.build_parser().parse_args([])
        self.assertIsNone(args.limit)
        self.assertEqual(args.metric, "l2")
        self.assertEqual(args.color_scale, "raw")
        for args in (["--metric", "entropy"], ["--color-scale", "softmax"],
                     ["--results-dir", "a", "--input-json", "b"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                vis.build_parser().parse_args(args)
        code = "import runVFlowOpt; import sys; assert 'torch' not in sys.modules; print(runVFlowOpt.SCRIPT_VERSION)"
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3.1-integrated-feature-residual", result.stdout)

    def test_saved_figures_raw_arrays_metadata_csv_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path, geometry, _, _, _ = fixture(Path(temp))
            scores = synthetic_scores(geometry)
            output = Path(temp) / "plots"
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            shapes = {stage: [3, 729, 4] for stage in STAGES}
            vis.save_sample(image, image.resize((384, 384)), scores, shapes, geometry,
                            {"id": "synthetic", "image": image_path, "gt": 0},
                            output, "cosine", "minmax")
            figures = sorted(output.glob("*.png"))
            self.assertEqual(len(figures), 4)
            with Image.open(figures[0]) as figure:
                self.assertEqual(figure.size, (2400, 1320))
            with np.load(output / "residuals.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["base_l2"], scores["base_l2"])
                np.testing.assert_array_equal(saved["tile_cosine"], scores["tile_cosine"])
                np.testing.assert_array_equal(saved["layers"], LAYERS)
                np.testing.assert_array_equal(saved["windows"], [3, 5, 7])
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["metric"], "cosine")
            self.assertEqual(metadata["display_ranges"][STAGES[0]]["colorbar_limits"], [0, 1])
            self.assertEqual(metadata["colormap"], "jet")
            with (output / "crop_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4 * 3 * 3)
            with self.assertRaises(FileExistsError):
                vis.draw_figures(image, image, scores, geometry, output, "same", "cosine", "minmax")


class RunIntegrationTests(unittest.TestCase):
    def test_orchestration_manifest_and_no_llm_forward(self):
        import torch
        from test_feature_residual_capture import TinyTower
        with tempfile.TemporaryDirectory() as temp:
            image_path, geometry, _, input_json, data = fixture(Path(temp))
            checkpoint = Path(temp) / "checkpoint"
            checkpoint.mkdir()
            output = Path(temp) / "output"
            tower = TinyTower().eval()
            tower.device, tower.dtype = torch.device("cpu"), torch.float32
            model = SimpleNamespace(config=SimpleNamespace())
            backend = SimpleNamespace(vision_tower=tower, model=model, processor=object(),
                                      inference_config={"model_dtype": "unchanged"})
            scores = synthetic_scores(geometry)
            shapes = {stage: [3, 729, 4] for stage in STAGES}
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            prepared = (torch.ones(3, 3, 384, 384), image.resize((384, 384)), [], geometry)
            # GPU/checkpoint loading is mocked; real numerical/capture tests are separate.
            with patch("llava_pruning.backend.LlavaBackend", return_value=backend) as loader, \
                 patch("llava.model.multimodal_encoder.siglip_encoder.SigLipVisionTower", TinyTower), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=1), \
                 patch.object(vis, "prepare_views", return_value=prepared), \
                 patch.object(vis, "capture_stage_scores", return_value=(scores, shapes)), \
                 patch.object(vis, "save_sample") as save:
                vis.run(str(checkpoint), output, input_json=input_json, data_root=data, limit=2)
            loader.assert_called_once_with(checkpoint.resolve(), roi_mode="anyres_max_9")
            self.assertEqual(save.call_count, 2)
            manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_samples"], 2)
            self.assertFalse(manifest["llm_forward"])
            self.assertFalse(manifest["projector_forward"])
            self.assertEqual(manifest["loader_config"]["model_dtype"], "unchanged")
            # Errors are recorded rather than mislabeled as complete.
            with patch("llava_pruning.backend.LlavaBackend", return_value=backend), \
                 patch("llava.model.multimodal_encoder.siglip_encoder.SigLipVisionTower", TinyTower), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=1), \
                 patch.object(vis, "prepare_views", side_effect=ValueError("test geometry failure")):
                with self.assertRaisesRegex(ValueError, "test geometry failure"):
                    vis.run(str(checkpoint), Path(temp) / "failed", input_json=input_json, data_root=data)
            failed = json.loads((Path(temp) / "failed" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["completed_samples"], 0)


if __name__ == "__main__":
    unittest.main()
