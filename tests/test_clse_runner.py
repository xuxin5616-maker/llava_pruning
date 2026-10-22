"""CLSE public run.py integration: budgets, shared metrics, two PNG types."""

import contextlib
import csv
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import run as cli
import test_run_benchmark_metrics as common
from test_clse import toy_model
from llava_pruning.methods import load_method
from llava_pruning.runner import run


class CLSERunnerTests(unittest.TestCase):
    def backend(self, fixture, calls, warmups):
        parent = fixture.backend(calls, warmups)
        class Backend(parent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.core = types.SimpleNamespace(layers=[None] * 28)
                self.model = types.SimpleNamespace(get_model=lambda: self.core)

            def generate(self, sample, prompt, method, rate, mode, **kwargs):
                result = super().generate(sample, prompt, method, rate, mode, **kwargs)
                budget = method.resolve_budget(rate, 28)
                original = 100 + int(sample.sample_id)
                retained = max(1, math.floor(original * (100 - budget["requested_layer_prune_rate"]) / 100 + 1e-10))
                rows = []
                for layer in range(1, 29):
                    before = original if layer == 1 else retained
                    rows.append(dict(layer=layer, image_tokens_in=before, image_tokens_out=retained,
                                     sequence_tokens_in=before + 10, sequence_tokens_out=retained + 10,
                                     removed_after_layer=before - retained,
                                     cumulative_prune_rate=100 * (1 - retained / original)))
                raw = dict(original_image_tokens=original, final_llm_visual_tokens=retained, layers=rows)
                self.core.get_clse_stats = lambda: raw
                result["stats"] = method.stats(self.core, rate)
                return result
        return Backend

    def test_cli_all_samples_average_budgets_and_lat_mem_without_pngs(self):
        fixture = common.RunBenchmarkTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            options = fixture.fixture(root, count=4, rates=(0, 50, 70, 90), method="clse", warmup=2)
            options["roi_mode"] = "anyres_max_9"
            args = ["run.py", "--model-path", str(root), "--input-json", str(options["input_json"]),
                    "--data-root", str(root), "--method", "clse", "--method-config", str(options["method_config"]),
                    "--roi-mode", "anyres_max_9", "--no-sample", "--no-plots", "--benchmark-metrics", "--warmup", "2",
                    "--seed", "42", "--output-dir", str(options["output_dir"])]
            with fixture.fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(fixture, calls, 2)), \
                    patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(len(calls), 24)  # 4 settings x (2 warmups + ALL 4 input samples)
            for rate in (0, 50, 70, 90):
                folder = options["output_dir"] / f"prune_{rate:02d}"
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(metrics["evaluated_samples"], 4)
                self.assertEqual(metrics["warmup_samples"], 2)
                self.assertEqual(metrics["accuracy"], 1)
                self.assertEqual(metrics["precision"], 1)
                self.assertEqual(metrics["recall"], 1)
                self.assertEqual(metrics["tnr"], 1)
                self.assertEqual(metrics["pruning_layer"], 1)
                self.assertEqual(metrics["target_average_prune_rate"], rate)
                self.assertAlmostEqual(metrics["requested_layer_prune_rate"], rate * 28 / 27)
                self.assertAlmostEqual(metrics["lat_mean_ms"], 150)
                self.assertAlmostEqual(metrics["mem_peak_allocated_gib"], 1 + rate / 100)
                actual = []
                for index, record in enumerate(records):
                    count = 100 + index
                    retained = math.floor(count * (1 - rate / 100 * 28 / 27) + 1e-10)
                    average = 100 * (1 - (count + 27 * retained) / (28 * count))
                    actual.append(average)
                    self.assertAlmostEqual(record["actual_average_prune_rate"], average)
                    self.assertNotIn("scores", record)
                    self.assertNotIn("stages", record)
                self.assertAlmostEqual(metrics["actual_average_prune_rate"], sum(actual) / 4)
                with (folder / "layer_tokens.csv").open(encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 4 * 28)
                self.assertFalse((folder / "visualizations").exists())
            with (options["output_dir"] / "summary.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            metadata = json.loads((options["output_dir"] / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["resolved_budgets"]), 4)
            self.assertEqual(metadata["method_details"]["source"]["commit"], "cfc9710abe4a1bd5480e521f3acd86ded4855ff3")
            self.assertIn("none", metadata["inference"]["attention_source"])
            self.assertIn("FP32 scoring branch", metadata["inference"]["score_dtype"])

    def test_invalid_modes_rejected_before_loading(self):
        fixture = common.RunBenchmarkTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            options = fixture.fixture(Path(directory), rates=(50,), method="clse")
            with patch("llava_pruning.runner.LlavaBackend") as backend:
                with self.assertRaisesRegex(ValueError, "anyres_max_9"):
                    run(**options)
                options.update(roi_mode="anyres_max_9", no_sample=False, benchmark_metrics=False)
                with self.assertRaisesRegex(ValueError, "greedy"):
                    run(**options)
                backend.assert_not_called()

    def test_real_captured_stage_writes_two_png_types_and_compact_metadata(self):
        torch.manual_seed(795)
        method = replace(load_method("clse"), rates=(30,), visualize_rates=frozenset({30}))
        model, features, options = toy_model()
        core = model.get_model()
        method.configure(core, 30, capture_visualization=True, capture_attention=True)
        with patch.object(model, "encode_images", return_value=features), torch.no_grad():
            model.generate(**options)
        result = dict(image=Image.new("RGB", (16, 8), "white"), answer="B", stats=method.stats(core, 30),
                      stages=core.get_clse_stages(), crop_metadata=[dict(mode="anyres_max_9", original_size=[16, 8],
                                                                         grid_patches=[2, 1], roi_boxes=[], final_newline=False)])
        tower = types.SimpleNamespace(num_patches_per_side=2, config=types.SimpleNamespace(patch_size=4))
        with tempfile.TemporaryDirectory() as directory:
            paths = method.visualize(result=result, vision_tower=tower, output_dir=directory, sample_id="000000108",
                                     save_prune=True, save_attention=True, ground_truth=0)
            folder = Path(paths[0]["directory"])
            self.assertEqual({path.name for path in folder.glob("*.png")}, {"comparison.png", "attention_overlay.png"})
            metadata = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["method"], "clse")
            self.assertEqual(metadata["stages"][0]["after_layer"], 1)
            self.assertEqual(metadata["stages"][0]["pruned_newline_tokens"], 0)
            self.assertNotIn("scores", metadata["stages"][0])


if __name__ == "__main__":
    unittest.main()
