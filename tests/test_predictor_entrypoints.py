"""CPU smoke of the actual training CLI plus public evaluation/reporting."""

import contextlib
import csv
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

import train_attention_predictor as training_cli
import run as evaluation_cli
import test_run_benchmark_metrics as common
from test_clse import toy_model, layout
from llava_pruning.predictor_training import extract_teacher
from llava_pruning.methods import load_method


class PredictorEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_training_cli_writes_checkpoints_history_and_portable_method_config_without_gt(self):
        torch.manual_seed(818)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candle").mkdir()
            (root / "split_csv").mkdir()
            with (root / "split_csv/1cls.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["object", "image"])
                writer.writeheader()
                for index in range(10):
                    Image.new("RGB", (16, 8), "white").save(root / "candle" / f"{index}.png")
                    writer.writerow(dict(object="candle", image=f"candle/{index}.png"))
            model, features, options = toy_model()
            core = model.get_model()
            core.config.num_hidden_layers = 6
            for index in (4, 5):
                core.layers.append(type(core.layers[0])(core.config, layer_idx=index))
            class Backend:
                def __init__(self, *args, **kwargs):
                    self.model = model
                    self.inference_config = {"model_dtype": "float32", "attention": "sdpa_fixture"}
            packed = torch.randn(1, 19, 32)
            def example(backend, sample, teacher_layer):
                return extract_teacher(backend.model.get_model(), packed, (2, 16), layout(), teacher_layer=teacher_layer)
            args = ["train_attention_predictor.py", "--model-path", str(root), "--data-root", str(root),
                    "--teacher-layer", "3", "--epochs", "1",
                    "--student-precision", "float32", "--gradient-accumulation", "4",
                    "--output-dir", str(root / "trained")]
            with patch("sys.argv", args), patch("llava_pruning.backend.LlavaBackend", Backend), \
                    patch("benchmark_fastv.validate_single_gpu"), patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.manual_seed_all"), \
                    patch("torch.cuda.synchronize"), patch("llava_pruning.predictor_training.prepare_teacher_example", side_effect=example), \
                    contextlib.redirect_stdout(io.StringIO()) as logs:
                training_cli.main()
            self.assertIn("phase_ETA=", logs.getvalue())
            self.assertIn("total_ETA=", logs.getvalue())
            self.assertIn("overall=100.0%", logs.getvalue())
            self.assertIn("loss=", logs.getvalue())
            self.assertIn("mean_loss=", logs.getvalue())
            self.assertIn("Best validation loss=", logs.getvalue())
            self.assertIn("copied native blocks 3/4/5", logs.getvalue())
            trained = root / "trained"
            self.assertTrue((trained / "best.pt").is_file())
            self.assertTrue((trained / "last.pt").is_file())
            saved = torch.load(trained / "best.pt", weights_only=True)
            self.assertEqual(saved["version"], 2)
            self.assertEqual(saved["architecture"]["source_layers"], (3, 4, 5))
            self.assertEqual(saved["architecture"]["qwen_config"]["hidden_size"], 32)
            self.assertNotIn("project.weight", saved["state_dict"])
            method = load_method("attention_predictor", trained / "method.json")
            self.assertEqual(method.rates, (50, 70, 90))
            self.assertEqual(method.checkpoint, (trained / "best.pt").resolve())
            with (trained / "history.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["split"] for row in rows], ["train", "validation"])
            self.assertEqual([int(row["samples"]) for row in rows], [9, 1])
            self.assertTrue(all(math.isfinite(float(row["kl"])) for row in rows))
            train = [json.loads(line) for line in (trained / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            validation = [json.loads(line) for line in (trained / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertFalse({row["image"] for row in train} & {row["image"] for row in validation})
            self.assertTrue(all("gt" not in row for row in train + validation))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            status = json.loads((trained / "training_status.json").read_text(encoding="utf-8"))
            self.assertTrue(status["complete"])

    def test_public_evaluation_cli_uses_layer2_average_and_existing_lat_mem_protocol(self):
        fixture = common.RunBenchmarkTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            options = fixture.fixture(root, count=4, rates=(50, 70, 90), method="priortr", warmup=2)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"not loaded by mocked backend")
            config = dict(checkpoint="best.pt", layer=2, rate_semantics="layer_average",
                          prune_rates=[50, 70, 90], visualize_rates=[])
            (root / "method.json").write_text(json.dumps(config), encoding="utf-8")
            parent = fixture.backend(calls, warmups=2)
            class Backend(parent):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.core = types.SimpleNamespace(layers=[None] * 28)
                    self.model = types.SimpleNamespace(get_model=lambda: self.core)

                def generate(self, sample, prompt, method, rate, mode, **kwargs):
                    result = super().generate(sample, prompt, method, rate, mode, **kwargs)
                    before = 100 + int(sample.sample_id)
                    budget = method.resolve_budget(rate, 28)
                    after = max(1, math.floor(before * (100 - budget["requested_layer_prune_rate"]) / 100))
                    rows = []
                    for layer in range(1, 29):
                        count = before if layer <= 2 else after
                        out = before if layer < 2 else after
                        rows.append(dict(layer=layer, image_tokens_in=count, image_tokens_out=out,
                                         sequence_tokens_in=count + 10, sequence_tokens_out=out + 10,
                                         removed_after_layer=count - out, cumulative_prune_rate=100 * (1 - out / before)))
                    self.core.get_clse_stats = lambda: dict(original_image_tokens=before, final_llm_visual_tokens=after, layers=rows)
                    result["stats"] = method.stats(self.core, rate)
                    return result
            args = ["run.py", "--model-path", str(root), "--input-json", str(options["input_json"]),
                    "--data-root", str(root), "--method", "attention_predictor", "--method-config", str(root / "method.json"),
                    "--roi-mode", "anyres_max_9", "--no-plots", "--benchmark-metrics", "--warmup", "2",
                    "--output-dir", str(options["output_dir"])]
            with fixture.fake_gpu(), patch("llava_pruning.runner.LlavaBackend", Backend), patch("sys.argv", args), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation_cli.main()
            self.assertEqual(len(calls), 18)
            for rate in (50, 70, 90):
                path = options["output_dir"] / f"prune_{rate:02d}"
                metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["evaluated_samples"], 4)
                self.assertEqual(metrics["warmup_samples"], 2)
                self.assertEqual(metrics["pruning_layer"], 2)
                self.assertEqual(metrics["target_average_prune_rate"], rate)
                self.assertAlmostEqual(metrics["requested_layer_prune_rate"], rate * 28 / 26)
                self.assertEqual(metrics["accuracy"], 1)
                self.assertAlmostEqual(metrics["lat_mean_ms"], 150)
                self.assertAlmostEqual(metrics["mem_peak_allocated_gib"], 1 + rate / 100)
                self.assertFalse((path / "visualizations").exists())
            metadata = json.loads((options["output_dir"] / "run.json").read_text(encoding="utf-8"))
            self.assertIn("trained", metadata["inference"]["attention_source"])
            self.assertEqual(len(metadata["method_details"]["checkpoint_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
