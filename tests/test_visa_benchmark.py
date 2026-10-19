"""Small filesystem fixtures; no real dataset, model download or GPU required."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

import benchmark_fastv as benchmark
from llava_pruning import visa_data
from llava_pruning.data import load_samples
from llava_pruning.prompts import GENERIC_PROMPT_V0, MVTEC_PROMPT_V0


class VisaDataTests(unittest.TestCase):
    def image(self, root, relative):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
        return path

    def csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["object", "split", "label", "image", "mask"])
            writer.writerows(rows)
        return path

    def test_directory_layouts_read_only_test_and_never_masks(self):
        for prefix in ("", "1cls/"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for category in ("candle", "pcb1"):
                    for label in ("good", "bad"):
                        self.image(root, f"{prefix}{category}/test/{label}/001.png")
                    self.image(root, f"{prefix}{category}/train/good/002.png")
                    self.image(root, f"{prefix}{category}/ground_truth/bad/001.png")
                samples, metadata = visa_data.load_visa_test(root)
                self.assertEqual(len(samples), 4)
                self.assertEqual((metadata["normal"], metadata["abnormal"]), (2, 2))
                self.assertEqual(metadata["layout"], "test_directories")
                self.assertEqual(metadata["categories"], {"candle": 2, "pcb1": 2})
                self.assertEqual(len({sample.sample_id for sample in samples}), 4)
                for sample in samples:
                    self.assertIn("/test/", sample.origin_path)
                    self.assertIsNone(sample.mask)
                    self.assertIsNone(sample.bbox)
                    self.assertEqual(sample.text, GENERIC_PROMPT_V0)
                    self.assertEqual(sample.gt, int("/bad/" in sample.origin_path))

    def test_raw_csv_filters_train_and_validation_without_needing_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = "candle/Data/Images/Normal/a.JPG"
            bad = "candle/Data/Images/Anomaly/b.JPG"
            self.image(root, good)
            self.image(root, bad)
            self.csv(root / "split_csv/1cls.csv", [
                ["candle", "train", "normal", "candle/Data/Images/Normal/train.JPG", ""],
                ["candle", "val", "normal", "candle/Data/Images/Normal/val.JPG", ""],
                ["candle", "test", "normal", good, ""],
                ["candle", "test", "anomaly", bad, "missing-mask.png"],
            ])
            samples, info = visa_data.load_visa_test(root)
            self.assertEqual(len(samples), 2)
            self.assertEqual({s.source["image"]: s.gt for s in samples}, {good: 0, bad: 1})
            self.assertEqual(info["split_source"], "local_csv")
            self.assertEqual(len(info["split_sha256"]), 64)
            self.assertTrue(all(s.mask is None for s in samples))

    def test_bundled_snapshot_has_official_test_counts_and_unique_paths(self):
        with visa_data.BUNDLED_SPLIT.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2162)
        self.assertEqual(sum(row["label"] == "normal" for row in rows), 962)
        self.assertEqual(sum(row["label"] == "anomaly" for row in rows), 1200)
        self.assertEqual({row["object"] for row in rows}, visa_data.CATEGORIES)
        self.assertEqual({row["split"] for row in rows}, {"test"})
        self.assertEqual(len({row["image"] for row in rows}), 2162)
        self.assertTrue(all("mask" not in row for row in rows))

    def test_explicit_then_directory_then_local_then_bundled_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = "candle/Data/Images/Normal/a.JPG"
            self.image(root, good)
            bundled = self.csv(root / "bundled.csv", [["candle", "test", "normal", good, ""]])
            with patch.object(visa_data, "BUNDLED_SPLIT", bundled):
                samples, info = visa_data.load_visa_test(root)
                self.assertEqual(info["split_source"], "bundled_official_1cls_test")
                self.csv(root / "split_csv/1cls.csv", [["candle", "test", "normal", good, ""]])
                self.assertEqual(visa_data.load_visa_test(root)[1]["split_source"], "local_csv")
                self.image(root, "candle/test/good/different.png")
                self.assertEqual(visa_data.load_visa_test(root)[1]["layout"], "test_directories")
                selected, explicit = visa_data.load_visa_test(root, bundled)
                self.assertEqual(selected[0].source["image"], good)
                self.assertEqual(explicit["layout"], "raw_csv")

    def test_invalid_missing_duplicate_and_overlapping_csv_samples_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = "candle/Data/Images/Normal/a.JPG"
            self.image(root, good)
            normal = ["candle", "test", "normal", good, ""]
            failures = [
                ([normal, normal], "Duplicate"),
                ([normal, ["candle", "train", "normal", good, ""]], "overlap"),
                ([["candle", "test", "anomaly", good, ""]], "conflicts"),
                ([["candle", "test", "unknown", good, ""]], "Invalid VisA label"),
                ([["candle", "testing", "normal", good, ""]], "Invalid category or split"),
                ([["candle", "test", "normal", "../outside.png", ""]], "relative"),
                ([["candle", "test", "normal", "/outside.png", ""]], "relative"),
                ([["candle", "test", "normal", "candle/Data/Images/Normal/missing.JPG", ""]], "not found"),
                ([["candle", "train", "normal", good, ""]], "empty"),
            ]
            for rows, message in failures:
                with self.subTest(message=message):
                    path = self.csv(root / "explicit.csv", rows)
                    with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                        visa_data.load_visa_test(root, path)

    def test_partial_ambiguous_and_unknown_label_test_layouts_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.image(root, "candle/test/good/a.png")
            self.image(root, "pcb1/train/good/a.png")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                visa_data.load_visa_test(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.image(root, "candle/test/good/a.png")
            self.image(root, "1cls/candle/test/good/a.png")
            with self.assertRaisesRegex(ValueError, "Multiple"):
                visa_data.load_visa_test(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.image(root, "candle/test/unknown/a.png")
            with self.assertRaisesRegex(ValueError, "Unknown VisA test label"):
                visa_data.load_visa_test(root)


class VisaBenchmarkTests(unittest.TestCase):
    def test_prompt_is_v0_with_only_category_clause_removed(self):
        for category, prompt in MVTEC_PROMPT_V0.items():
            if category == "hazelnut":
                self.assertEqual(GENERIC_PROMPT_V0, prompt.replace(" on the hazelnut", ""))
        sample = types.SimpleNamespace(category="hazelnut", text="Label leakage: the answer is A")
        self.assertEqual(benchmark.benchmark_prompt(sample, "v0_generic"), GENERIC_PROMPT_V0)
        self.assertEqual(benchmark.benchmark_prompt(sample, "v0"), MVTEC_PROMPT_V0["hazelnut"])

    def test_visatest_snapshot_and_all_seven_workers_use_generic_prompt(self):
        calls = []

        class Backend:
            def __init__(self, *args, **kwargs):
                self.model = types.SimpleNamespace(get_model=lambda: types.SimpleNamespace(layers=[0, 1, 2, 3]))
                self.inference_config = {"test": True}

            def generate(self, sample, prompt, method, rate, mode, **kwargs):
                calls.append((sample, prompt, method.layer, rate, mode, kwargs))
                return {"answer": "A" if sample.gt else "B", "lat_seconds": 0.1,
                        "generation_seconds": 0.2, "generated_tokens": 2,
                        "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "VisA"
            helper = VisaDataTests()
            helper.image(data, "candle/test/good/a.png")
            helper.image(data, "candle/test/bad/a.png")
            helper.image(data, "candle/train/good/not_test.png")
            model = root / "model"
            model.mkdir()
            args = benchmark.build_parser().parse_args([
                "--dataset", "visa", "--data-root", str(data), "--model-path", str(model),
                "--output-dir", str(root / "output"), "--warmup", "1",
            ])
            output, spec = benchmark.prepare(args)
            self.assertEqual(spec["sample_count"], 2)
            self.assertEqual(spec["prompt_version"], "v0_generic")
            self.assertEqual(spec["prompt_text"], GENERIC_PROMPT_V0)
            self.assertEqual(spec["dataset"], "visa")
            self.assertEqual(len(spec["jobs"]), 7)
            self.assertTrue((output / "input_visa_test.jsonl").is_file())
            loaded = load_samples(spec["input_json"], spec["data_root"])
            self.assertEqual({s.gt for s in loaded}, {0, 1})
            self.assertTrue(all(s.text == GENERIC_PROMPT_V0 and s.mask is None for s in loaded))
            self.assertFalse(list(data.rglob("*.json*")))
            reports = []
            with patch("llava_pruning.backend.LlavaBackend", Backend), \
                 patch.object(benchmark, "validate_single_gpu"), \
                 patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace(total_memory=1000)), \
                 contextlib.redirect_stdout(io.StringIO()):
                for job in spec["jobs"]:
                    self.assertEqual(benchmark.run_worker(output / "run.json", job["prune_rate"], job["fastv_layer"]), 0)
                    report = json.loads((output / job["output_subdir"] / "metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["dataset"], "visa")
                    self.assertEqual(report["evaluated_samples"], 2)
                    self.assertEqual(report["unparsed"], 0)
                    for name in ("accuracy", "precision", "recall", "tnr"):
                        self.assertEqual(report[name], 1)
                    reports.append(report)
            self.assertEqual(len(calls), 7 * 3)
            for sample, prompt, layer, rate, mode, options in calls:
                self.assertEqual(prompt, GENERIC_PROMPT_V0)
                self.assertIsNone(sample.mask)
                self.assertEqual(mode, "anyres_max_9")
                self.assertFalse(options["do_sample"])
                self.assertFalse(options["capture_visualization"])
                self.assertFalse(options["capture_attention"])
                self.assertTrue(options["include_pruning_time"])
            benchmark.write_summary(output, reports)
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 7)
            self.assertEqual({row["dataset"] for row in rows}, {"visa"})

    def test_default_dataset_and_server_roots_are_explicit(self):
        parser = benchmark.build_parser()
        self.assertEqual(parser.parse_args([]).dataset, "mvtec")
        self.assertEqual(benchmark.DEFAULT_DATA_ROOTS["visa"], "/home/yz/xxy/data/datasets/VisA")
        self.assertIsNone(parser.parse_args(["--dataset", "visa"]).input_json)


if __name__ == "__main__":
    unittest.main()
