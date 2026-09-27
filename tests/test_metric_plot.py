import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from run import main
from test_runner import FakeBackend
from llava_pruning.metric_plot import build_metric_figure, check_plot_dependencies, load_metric_rows, save_metric_plot


class MetricPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        check_plot_dependencies()

    def summary_file(self, root):
        source = root / "summary.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("prune_rate", "complete", "accuracy", "precision", "recall", "tnr"))
            writer.writeheader()
            for rate in range(90, -1, -10):
                writer.writerow(dict(prune_rate=rate, complete=True, accuracy=0.90, precision=0.92,
                                     recall=0.93, tnr=0.85))
        return source

    def test_four_curves_percentage_conversion_and_fixed_axis(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            rows = load_metric_rows(self.summary_file(Path(directory)))
            self.assertEqual([row["prune_rate"] for row in rows], list(range(0, 100, 10)))
            figure = build_metric_figure(rows)
            try:
                axis, = figure.axes
                self.assertEqual(axis.get_ylim(), (50, 100))
                self.assertEqual(len(axis.lines), 4)
                for line, expected in zip(axis.lines, (90, 92, 93, 85)):
                    self.assertTrue(all(abs(value - expected) < 1e-10 for value in line.get_ydata()))
                    self.assertEqual(list(line.get_xdata()), list(range(0, 100, 10)))
            finally:
                plt.close(figure)

    def test_below_axis_values_are_not_clamped_and_undefined_points_are_gaps(self):
        import matplotlib.pyplot as plt
        rows = [{"prune_rate": 0, "complete": True, "accuracy": 0.4, "precision": None, "recall": 0.7, "tnr": 0.6},
                {"prune_rate": 10, "complete": False, "accuracy": 0.8, "precision": 0.8, "recall": 0.8, "tnr": 0.8}]
        with self.assertWarnsRegex(UserWarning, "below the 50% plot limit"):
            figure = build_metric_figure(rows)
        try:
            self.assertEqual(figure.axes[0].lines[0].get_ydata()[0], 40)
            self.assertTrue(math.isnan(figure.axes[0].lines[0].get_ydata()[1]))
            self.assertTrue(math.isnan(figure.axes[0].lines[1].get_ydata()[0]))
            self.assertIn("outside the displayed range", figure.texts[0].get_text())
        finally:
            plt.close(figure)

    def test_png_saved_beside_summary_without_modifying_source_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.summary_file(root)
            original = source.read_bytes()
            result = save_metric_plot(source)
            self.assertEqual(result, root / "metrics_vs_pruning.png")
            self.assertEqual(source.read_bytes(), original)
            with Image.open(result) as figure:
                self.assertGreater(figure.width, 2000)
                self.assertGreater(figure.height, 1000)

    def test_old_accuracy_only_csv_is_rejected_instead_of_inventing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.csv"
            source.write_text("prune_rate,accuracy\n0,0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks the new metric columns"):
                load_metric_rows(source)


class RunMetricPlotTests(unittest.TestCase):
    def test_run_cli_draws_after_full_sweep_without_changing_data_or_sample_visualization_flags(self):
        FakeBackend.calls.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "image.png")
            source = root / "input.json"
            # More than 100 records ensures run.py has no inherited ex.py limit.
            source.write_text(json.dumps([
                {"question_id": str(index), "image": "image.png", "gt": 1,
                 "origin_path": "screw/test/bad.png"} for index in range(101)
            ]), encoding="utf-8")
            output = root / "result"
            args = ["run.py", "--model-path", str(root), "--input-json", str(source), "--data-root", str(root),
                    "--roi-mode", "anyres_max_9", "--no-sample", "--seed", "42", "--output-dir", str(output)]
            with patch("sys.argv", args), patch("llava_pruning.runner.LlavaBackend", FakeBackend), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(len(FakeBackend.calls), 1010)
            for rate in range(0, 100, 10):
                self.assertEqual([call for call in FakeBackend.calls if call[0] == rate],
                                 [(rate, 42 + index, False) for index in range(101)])
            self.assertTrue((output / "metrics_vs_pruning.png").is_file())
            self.assertEqual(list(output.rglob("visualizations")), [])
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(int(row["evaluated_samples"]) == 101 for row in rows))
            self.assertTrue(all(row["tnr"] == "" for row in rows))
            self.assertTrue(all(float(row["precision"]) == 1 for row in rows))

    def test_missing_plot_dependency_fails_before_model_run(self):
        with patch("sys.argv", ["run.py", "--model-path", "model", "--input-json", "input.json", "--data-root", "data"]), \
             patch("run.check_plot_dependencies", side_effect=RuntimeError("missing matplotlib")), \
             patch("run.run") as run_model, self.assertRaisesRegex(RuntimeError, "missing matplotlib"):
            main()
        run_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
