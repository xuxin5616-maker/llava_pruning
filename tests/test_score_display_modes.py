"""Display-mode integration: same means/PNG as the offline script, one forward."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import visualize_patch_means as offline
import visualize_scores as cli
from llava_pruning.score_visualization import (
    FEATURE_NORMALIZATION, ScoreGeometry, _display_options,
    run_score_visualization, save_sample,
)
from test_score_visualization import TinyProbeModel, TinyTower, SigLipImageProcessor


class DisplayModeTests(unittest.TestCase):
    def test_cli_mode_defaults_explicit_scales_and_forwarding(self):
        argv = ["--model-path", "m", "--input-json", "i", "--data-root", "d",
                "--output-dir", "o"]
        parser = cli.build_parser()
        args = parser.parse_args(argv)
        self.assertEqual(_display_options(args.display_mode, args.color_scale)[0], "sample")
        args = parser.parse_args(argv + ["--display-mode", "patch-means"])
        self.assertEqual(_display_options(args.display_mode, args.color_scale)[0], "fixed")
        for mode in ("tokens", "patch-means"):
            for scale in ("sample", "fixed"):
                self.assertEqual(_display_options(mode, scale)[0], scale)
        with patch.object(sys, "argv", ["visualize_scores.py", *argv, "--display-mode", "patch-means"]), \
             patch("llava_pruning.score_visualization.run_score_visualization") as runner:
            cli.main()
        self.assertEqual(runner.call_args.kwargs["display_mode"], "patch-means")
        self.assertIsNone(runner.call_args.kwargs["color_scale"])

    def test_invalid_options_fail_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("llava_pruning.backend.LlavaBackend") as loader:
            root = Path(tmp)
            for kwargs in ({"display_mode": "typo"}, {"color_scale": "typo"}):
                with self.assertRaises(ValueError):
                    run_score_visualization(root, root / "missing", root, root / "output", **kwargs)
            loader.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_online_and_offline_png_csv_are_identical(self):
        for requested_scale, effective_scale in ((None, "fixed"), ("sample", "sample")):
            with self.subTest(scale=requested_scale), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = Image.fromarray(np.arange(55 * 31 * 3, dtype=np.uint16)
                                       .reshape(31, 55, 3).astype(np.uint8))
                path = root / "source.png"
                image.save(path)
                geometry = ScoreGeometry.create(image.size, (2, 2), 28, 14)
                base = image.resize((28, 28))
                scores = {"global": np.linspace(-1, 1, 80, dtype=np.float32).reshape(5, 4, 4),
                          "local": np.full((5, 4, 4), np.nan, dtype=np.float32)}
                scores["global"][4] = np.nan  # projector must not affect averages/colors
                scores["global"][0, 1, 0] = np.nan
                scores["global"][2, 2] = np.nan  # undefined layer -> undefined group
                sample = {"id": "sample", "image": path, "gt": 1}
                folder = root / "online" / "000001_sample"
                with patch("llava_pruning.score_visualization.draw_figures",
                           side_effect=AssertionError("Token renderer must not run")):
                    save_sample(image, base, [base] * 4, scores, {}, geometry, sample, folder,
                                requested_scale, display_mode="patch-means")
                self.assertEqual({p.name for p in folder.iterdir()},
                                 {"patch_means.png", "patch_means.csv", "scores.npz", "metadata.json"})
                metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["display_mode"], "patch-means")
                self.assertEqual(metadata["color_scale"], effective_scale)
                self.assertEqual(metadata["feature_normalization"], FEATURE_NORMALIZATION)
                self.assertEqual(metadata["displayed_stages"], list(offline.STAGES))
                self.assertEqual(metadata["score_scaling"], offline.SCORE_SCALING)
                hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in folder.iterdir()}
                redrawn = offline.run(folder, root / "redrawn", color_scale=effective_scale) / folder.name
                self.assertEqual((folder / "patch_means.csv").read_bytes(),
                                 (redrawn / "patch_means.csv").read_bytes())
                with Image.open(folder / "patch_means.png") as first, \
                     Image.open(redrawn / "patch_means.png") as second:
                    self.assertEqual(first.size, (2400, 1350))
                    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
                for p, digest in hashes.items():
                    self.assertEqual(hashlib.sha256(p.read_bytes()).digest(), digest)
                with np.load(folder / "scores.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive["global_scores"], scores["global"])
                    self.assertEqual(len(archive["stages"]), 5)
                with self.assertRaises(FileExistsError):
                    save_sample(image, base, [base] * 4, scores, {}, geometry, sample, folder,
                                display_mode="patch-means")

    def test_patch_mean_runner_encodes_once_and_no_llm_or_token_png(self):
        model = TinyProbeModel().eval()
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(image_grid_pinpoints=[[28, 28], [56, 28]])
        backend = SimpleNamespace(model=model, vision_tower=model.tower,
                                  processor=SigLipImageProcessor(size=(28, 28),
                                            crop_size={"height": 28, "width": 28}),
                                  inference_config={"vision_dtype": "float32"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (51, 24), "orange").save(root / "source.png")
            questions = root / "questions.json"
            questions.write_text(json.dumps([{"id": "one", "image": "source.png", "gt": 0}]), encoding="utf-8")
            output = root / "result"
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=1), \
                 patch("llava_pruning.backend.LlavaBackend", return_value=backend) as loader, \
                 patch("llava.model.multimodal_encoder.siglip_encoder.SigLipVisionTower", TinyTower), \
                 patch.object(model, "encode_images", wraps=model.encode_images) as encoder, \
                 patch("llava_pruning.score_visualization.draw_figures",
                       side_effect=AssertionError("Wrong renderer")):
                run_score_visualization(root, questions, root, output, display_mode="patch-means")
            loader.assert_called_once()
            encoder.assert_called_once()
            self.assertTrue(all(not module._forward_hooks for module in model.modules()))
            saved = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["display_mode"], "patch-means")
            self.assertEqual(saved["color_scale"], "fixed")
            self.assertFalse(saved["llm_generation"])
            self.assertFalse(saved["pruning"])
            self.assertEqual(saved["figure_layout"]["rows"], 2)
            self.assertEqual(saved["figure_layout"]["columns"], 4)
            self.assertEqual([p.name for p in output.rglob("*.png")], ["patch_means.png"])


if __name__ == "__main__":
    unittest.main()
