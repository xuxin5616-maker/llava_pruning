"""Fractional FastV budgets, lossless output names, and Lat/Mem integration."""

import contextlib
import csv
from decimal import Decimal, ROUND_CEILING
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run as cli
from llava_pruning.methods import load_method
from llava_pruning.metric_plot import load_metric_rows, save_metric_plot
from llava_pruning.rates import format_prune_rate
import test_run_benchmark_metrics as benchmark_tests


PROJECT = Path(__file__).resolve().parents[1]
RATES = (53.85, 75.38, 96.92)


class DecimalFastVTests(unittest.TestCase):
    def test_fractional_config_and_invalid_values(self):
        method = load_method("fastv", PROJECT / "configs/fastv_decimal.json")
        self.assertEqual(method.rates, RATES)
        self.assertEqual(method.layer, 2)
        self.assertFalse(method.visualize_rates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method.json"
            config = dict(layer=2, prune_rates=list(RATES), visualize_rates=[53.85])
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_method("fastv", path).visualize_rates, {53.85})
            invalid = ([True], [False], [float("nan")], [float("inf")], [-.01],
                       [100], ["53.85"], [50, 50.0], [[53.85]], [])
            for rates in invalid:
                with self.subTest(rates=rates):
                    path.write_text(json.dumps(dict(config, prune_rates=rates)), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_method("fastv", path)
            path.write_text(json.dumps(dict(config, visualize_rates=[True])), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_method("fastv", path)

    def test_folder_names_preserve_decimals_and_old_integer_names(self):
        self.assertEqual([format_prune_rate(r) for r in (0, 10, 50, 50.0, *RATES)],
                         ["00", "10", "50", "50", "53.85", "75.38", "96.92"])
        close_rates = (53.85, 53.851, 53.85000000000001)
        self.assertEqual(len({format_prune_rate(r) for r in close_rates}), 3)

    def test_real_fastv_mask_receives_decimal_ratio_and_retains_top_tokens(self):
        import torch
        from test_fastv_attention import tiny_config
        from llava.model.language_model.llava_qwen import LlavaQwenModel

        core = LlavaQwenModel(tiny_config()).eval()
        method = load_method("fastv", PROJECT / "configs/fastv_decimal.json")
        scores = torch.arange(731, dtype=torch.float32)[None]
        for rate in RATES:
            with self.subTest(rate=rate):
                method.configure(core, rate, capture_attention=False)
                core.set_fastv_image_spans([[(1, 730)]])  # 729 image positions; final one protected
                core._build_fastv_keep_mask(scores)
                mask = core._fastv_keep_mask[0]
                expected_ratio = (Decimal(100) - Decimal(str(rate))) / 100
                selected = int((728 * expected_ratio).to_integral_value(rounding=ROUND_CEILING))
                self.assertAlmostEqual(core.fastv_keep_ratio, float(expected_ratio))
                self.assertEqual(int(mask[1:730].sum()), selected + 1)
                self.assertTrue(mask[0] and mask[730] and mask[729])
                self.assertFalse(mask[1:729-selected].any())
                self.assertTrue(mask[729-selected:729].all())
                self.assertEqual(core.fastv_layer, 2)

    def test_cli_warmup_measurements_reports_and_plot_keep_exact_rates(self):
        fixture = benchmark_tests.RunBenchmarkTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            options = fixture.fixture(root, count=4, rates=RATES, warmup=2)
            output = options["output_dir"]
            arguments = ["run.py", "--model-path", str(root), "--input-json", str(options["input_json"]),
                         "--data-root", str(root), "--prompt-version", "v0", "--roi-mode", "anyres_max_9",
                         "--method", "fastv", "--method-config", str(PROJECT / "configs/fastv_decimal.json"),
                         "--no-sample", "--no-plots", "--benchmark-metrics", "--warmup", "2",
                         "--seed", "42", "--output-dir", str(output)]
            with fixture.fake_gpu(), patch("sys.argv", arguments), \
                    patch("llava_pruning.runner.LlavaBackend", fixture.backend(calls, 2)), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual([rate for _, rate, _, _ in calls], [r for r in RATES for _ in range(6)])
            self.assertEqual({mode for _, _, mode, _ in calls}, {"anyres_max_9"})
            self.assertEqual({p.name for p in output.iterdir() if p.is_dir()},
                             {f"prune_{r}" for r in RATES})
            for rate in RATES:
                folder = output / f"prune_{rate}"
                records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["question_id"] for r in records], ["0", "1", "2", "3"])
                self.assertTrue(all(r["prune_rate"] == rate for r in records))
                self.assertTrue(all(r["lat_seconds"] < 1 for r in records))  # warmup Lat=8 s excluded
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                self.assertTrue(metrics["complete"])
                self.assertEqual(metrics["evaluated_samples"], 4)
                self.assertEqual(metrics["warmup_samples"], 2)
                self.assertAlmostEqual(metrics["lat_mean_ms"], 150)
                self.assertAlmostEqual(metrics["mem_peak_allocated_gib"], 1 + rate / 100)
                self.assertEqual([metrics[k] for k in ("accuracy", "precision", "recall", "tnr")], [1] * 4)
            with (output / "summary.csv").open(encoding="utf-8", newline="") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual([r["prune_rate"] for r in summary], [str(r) for r in RATES])
            self.assertFalse(list(output.rglob("*.png")))
            rows = load_metric_rows(output / "summary.csv")
            self.assertEqual([r["prune_rate"] for r in rows], list(RATES))
            figure = save_metric_plot(output / "summary.csv")
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 1000)
            import matplotlib.pyplot as plt
            from llava_pruning.metric_plot import build_metric_figure
            chart = build_metric_figure(rows)
            try:
                for line in chart.axes[0].lines:
                    self.assertEqual(list(line.get_xdata()), list(RATES))
            finally:
                plt.close(chart)


if __name__ == "__main__":
    unittest.main()
