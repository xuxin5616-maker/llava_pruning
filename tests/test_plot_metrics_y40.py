import contextlib
import csv
import hashlib
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

from PIL import Image

import plot_metrics_y40 as redraw


class RedrawY40Tests(unittest.TestCase):
    def summary(self, folder):
        source = folder / "summary.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["prune_rate", "complete", "accuracy", "precision", "recall", "tnr"])
            for rate in range(90, -1, -10):
                writer.writerow([rate, True, 0.9, 0.85, 0.8, 0.45])
        return source

    def test_original_style_and_40_percent_axis_without_changing_values(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            rows = redraw.load_rows(self.summary(Path(directory)).read_text(encoding="utf-8"))
            with warnings.catch_warnings(record=True) as caught:
                figure = redraw.build_figure(rows)
            try:
                axis, = figure.axes
                self.assertEqual(axis.get_ylim(), (40, 100))
                self.assertEqual(list(axis.get_yticks()), list(range(40, 101, 5)))
                self.assertEqual(len(axis.lines), 4)
                for line, (key, label, color, marker, linestyle) in zip(axis.lines, redraw.SERIES):
                    self.assertEqual(list(line.get_xdata()), list(range(0, 100, 10)))
                    self.assertTrue(all(abs(a - row[key] * 100) < 1e-9 for a, row in zip(line.get_ydata(), rows)))
                    self.assertEqual((line.get_label(), line.get_color(), line.get_marker(), line.get_linestyle()),
                                     (label, color, marker, linestyle))
                self.assertFalse(any("metric point" in str(item.message) for item in caught))
            finally:
                plt.close(figure)

    def test_below_40_not_clamped_undefined_and_incomplete_are_gaps(self):
        import matplotlib.pyplot as plt
        rows = redraw.load_rows("prune_rate,complete,accuracy,precision,recall,tnr\n"
                                "0,True,0.35,,0.7,0.6\n10,False,0.9,0.9,0.9,0.9\n")
        with self.assertWarnsRegex(UserWarning, "below 40%"):
            figure = redraw.build_figure(rows)
        try:
            self.assertEqual(figure.axes[0].lines[0].get_ydata()[0], 35)
            self.assertTrue(math.isnan(figure.axes[0].lines[0].get_ydata()[1]))
            self.assertTrue(math.isnan(figure.axes[0].lines[1].get_ydata()[0]))
            self.assertIn("below 40%", figure.texts[0].get_text())
        finally:
            plt.close(figure)

    def test_repeat_exports_preserve_csv_predictions_old_and_new_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = self.summary(folder)
            old = folder / "metrics_vs_pruning.png"
            old.write_bytes(b"old figure")
            prediction = folder / "predictions.jsonl"
            prediction.write_bytes(b"original predictions")
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            destination = redraw.redraw(folder)
            first = destination.read_bytes()
            repeated = redraw.redraw(source)
            self.assertEqual(destination.name, "metrics_vs_pruning_y40.png")
            self.assertEqual(repeated.name, "metrics_vs_pruning_y40_2.png")
            self.assertEqual(destination.read_bytes(), first)
            for name, content in before.items():
                self.assertEqual((folder / name).read_bytes(), content)
            with Image.open(destination) as image:
                self.assertEqual(image.size, (2640, 1620))
                self.assertAlmostEqual(image.info["dpi"][0], 300, places=1)
                self.assertEqual(image.info["SourceSHA256"], hashlib.sha256(before["summary.csv"]).hexdigest())

    def test_invalid_or_missing_input_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(FileNotFoundError):
                redraw.redraw(folder)
            source = folder / "summary.csv"
            source.write_text("prune_rate,accuracy\n0,0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing metric columns"):
                redraw.redraw(source)
            header = "prune_rate,complete,accuracy,precision,recall,tnr\n"
            for body in ("", "0,True,90,0.8,0.7,0.6\n", "0,True,nan,0.8,0.7,0.6\n",
                         "0,maybe,0.9,0.8,0.7,0.6\n", "0,True,0.9,0.8,0.7\n",
                         "0,True,0.9,0.8,0.7,0.6\n0,True,0.9,0.8,0.7,0.6\n"):
                with self.subTest(body=body), self.assertRaises(ValueError):
                    redraw.load_rows(header + body)
            self.assertEqual(list(folder.glob("*.png")), [])

    def test_single_file_cli_does_not_import_model_or_project_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = self.summary(folder)
            script = Path(redraw.__file__).resolve()
            # Run from a fresh process, blocking even an attempted model import.
            command = (
                "import builtins,runpy,sys; original=builtins.__import__; "
                "blocked={'torch','transformers','llava','llava_pruning','run'}; "
                "exec(\"def guarded(name,*a,**kw):\\n if name.split('.')[0] in blocked: "
                "raise AssertionError('Model import forbidden: '+name)\\n return original(name,*a,**kw)\"); "
                "builtins.__import__=guarded; "
                f"sys.argv=[{str(script)!r},{str(source)!r}]; runpy.run_path({str(script)!r},run_name='__main__')"
            )
            result = subprocess.run([sys.executable, "-c", command], cwd=folder,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("No inference", result.stdout)
            self.assertTrue((folder / "metrics_vs_pruning_y40.png").is_file())

    def test_no_argument_reads_summary_configured_at_top(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = self.summary(folder)
            with patch.object(redraw, "SUMMARY_CSV", str(source)), contextlib.redirect_stdout(io.StringIO()):
                redraw.main([])
            self.assertTrue((folder / "metrics_vs_pruning_y40.png").is_file())

    def test_relative_configuration_is_anchored_to_script_and_empty_path_fails(self):
        with patch.object(redraw, "SUMMARY_CSV", "output/custom/summary.csv"):
            self.assertEqual(redraw.configured_input(),
                             Path(redraw.__file__).resolve().parent / "output/custom/summary.csv")
        with patch.object(redraw, "SUMMARY_CSV", " "), self.assertRaisesRegex(ValueError, "Set SUMMARY_CSV"):
            redraw.redraw()


if __name__ == "__main__":
    unittest.main()
