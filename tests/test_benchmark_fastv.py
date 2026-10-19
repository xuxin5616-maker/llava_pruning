import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import benchmark_fastv as benchmark


class BenchmarkTests(unittest.TestCase):
    def fixture(self, root, count=103):
        model = root / "model"
        model.mkdir()
        Image.new("RGB", (8, 8)).save(root / "image.png")
        source = root / "questions.jsonl"
        source.write_text("".join(json.dumps({
            "question_id": str(index), "image": "image.png", "gt": index % 2,
            "origin_path": "screw/test/good/one.png",
        }) + "\n" for index in range(count)), encoding="utf-8")
        config = root / "fastv.json"
        config.write_text(json.dumps({"layer": 8, "prune_rates": [10, 30],
                                      "visualize_rates": [10], "min_tokens": 1,
                                      "preserve_image_newline": True}), encoding="utf-8")
        args = benchmark.build_parser().parse_args([
            "--model-path", str(model), "--input-json", str(source), "--data-root", str(root),
            "--method-config", str(config), "--output-dir", str(root / "result"),
        ])
        return args, config

    def test_preflight_full_dataset_and_effective_config_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config = self.fixture(root)
            before = config.read_bytes()
            output, spec = benchmark.prepare(args)
            self.assertEqual(spec["sample_count"], 103)
            self.assertEqual(spec["prune_rates"], [0, 50, 70, 90])
            self.assertEqual(spec["fastv_layers"], [2, 3])
            self.assertEqual(len(spec["jobs"]), 7)
            for layer in (2, 3):
                configured = spec["fastv_configs"][str(layer)]
                self.assertEqual(configured["fastv"]["visualize_rates"], [])
                self.assertEqual(configured["fastv"]["layer"], layer)
                self.assertEqual(json.loads(Path(configured["method_config"]).read_text(encoding="utf-8")),
                                 configured["fastv"])
            self.assertEqual(config.read_bytes(), before)
            self.assertFalse(spec["do_sample"])
            self.assertTrue(spec["include_pruning_time"])
            self.assertFalse(spec["save_visualizations"])
            self.assertEqual(spec["roi_mode"], "anyres_max_9")
            with self.assertRaises(FileExistsError):
                benchmark.prepare(args)

    def test_invalid_warmup_and_seed_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory), count=1)
            for name, value in (("warmup", 0), ("gpu", -1), ("seed", -1), ("seed", 2 ** 32)):
                modified = types.SimpleNamespace(**dict(vars(args), **{name: value}))
                with self.assertRaises(ValueError):
                    benchmark.prepare(modified)
            self.assertFalse(Path(args.output_dir).exists())

    def test_measurement_summary_peak_not_mean_and_warmup_selection(self):
        self.assertEqual(benchmark.warmup_indices(3, 10), [0, 1, 2])
        self.assertEqual(benchmark.warmup_indices(103, 3), [0, 51, 102])
        rows = [dict(lat_seconds=lat, generation_seconds=lat + 1, generated_tokens=2,
                     peak_allocated_bytes=mem, peak_reserved_bytes=mem * 2)
                for lat, mem in ((1, 10), (3, 30))]
        summary = benchmark.measurement_summary(rows, 100)
        self.assertEqual(summary["lat_mean_ms"], 2000)
        self.assertEqual(summary["lat_p50_ms"], 2000)
        self.assertEqual(summary["lat_p95_ms"], 2900)
        self.assertEqual(summary["mem_peak_allocated_pct"], 30)
        self.assertEqual(summary["peak_allocated_bytes"], 30)
        self.assertEqual(benchmark.measurement_summary([], None), {})

    def test_worker_evaluates_all_images_and_excludes_warmup_for_all_seven_jobs(self):
        calls = []

        class Backend:
            def __init__(self, *args, **kwargs):
                self.core = types.SimpleNamespace(layers=[1, 2, 3, 4], configure_fastv=Mock())
                self.model = types.SimpleNamespace(get_model=lambda: self.core)
                self.inference_config = {"attention": "fake_test", "model_dtype": "fake_test"}

            def generate(self, sample, prompt, method, rate, mode, **kwargs):
                method.configure(self.core, rate, capture_attention=False, capture_visualization=False)
                calls.append((sample.sample_id, rate, self.core.configure_fastv.call_args.kwargs, kwargs))
                return {"answer": "unknown" if sample.sample_id == "0" else ("A. defect" if sample.gt else "B. normal"),
                        "lat_seconds": 0.1, "generation_seconds": 0.2, "generated_tokens": 3,
                        "peak_allocated_bytes": benchmark.GIB, "peak_reserved_bytes": 2 * benchmark.GIB}

        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            output, spec = benchmark.prepare(args)
            with patch("llava_pruning.backend.LlavaBackend", Backend), \
                 patch.object(benchmark, "validate_single_gpu"), \
                 patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace(total_memory=10 * benchmark.GIB)), \
                 contextlib.redirect_stdout(io.StringIO()):
                for job in spec["jobs"]:
                    rate, layer = job["prune_rate"], job["fastv_layer"]
                    self.assertEqual(benchmark.run_worker(output / "run.json", rate, layer), 0)
                    folder = output / job["output_subdir"]
                    report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["fastv_layer"], layer)
                    self.assertEqual(report["prune_rate"], rate)
                    self.assertEqual(report["model"], job["model"])
                    self.assertEqual(report["evaluated_samples"], 103)
                    self.assertEqual(report["warmup_samples"], 10)
                    self.assertEqual(report["unparsed"], 1)
                    self.assertEqual(report["accuracy"], 102 / 103)
                    self.assertEqual(report["precision"], 1)
                    self.assertEqual(report["recall"], 1)
                    self.assertEqual(report["tnr"], 1)
                    self.assertEqual(report["lat_mean_ms"], 100)
                    self.assertEqual(report["mem_peak_allocated_pct"], 10)
                    self.assertTrue(report["complete"])
                    rows = [json.loads(line) for line in
                            (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(rows), 103)
                    self.assertEqual(rows[-1]["question_id"], "102")
                    self.assertEqual(rows[1]["answer"], "A. defect")
            self.assertEqual(len(calls), (103 + 10) * 7)
            for job_index, job in enumerate(spec["jobs"]):
                for _, rate, actual, _ in calls[job_index * 113:(job_index + 1) * 113]:
                    self.assertEqual(rate, job["prune_rate"])
                    self.assertEqual(actual["layer"], job["fastv_layer"] or 2)
                    self.assertEqual(actual["enabled"], rate != 0)
                    self.assertEqual(actual["keep_ratio"], 1 - rate / 100)
            for _, _, _, options in calls:
                self.assertTrue(options["benchmark_metrics"])
                self.assertTrue(options["include_pruning_time"])
                self.assertFalse(options["capture_visualization"])
                self.assertFalse(options["capture_attention"])
                self.assertFalse(options["do_sample"])
            self.assertFalse(list(output.rglob("*.png")))

    def test_changed_dataset_produces_failed_report_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory), count=1)
            output, spec = benchmark.prepare(args)
            Path(spec["input_json"]).write_text("[]", encoding="utf-8")
            with patch("llava_pruning.backend.LlavaBackend") as backend, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark.run_worker(output / "run.json", 0), 1)
                backend.assert_not_called()
            report = json.loads((output / "prune_00" / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(report["complete"])
            self.assertEqual(report["status"], "failed")
            self.assertNotIn("lat_mean_ms", report)

    def test_launch_serial_order_continues_failures_and_exports_seven_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory), count=1)
            jobs = []

            def child(manifest, rate, gpu, log_path, *, layer=None):
                jobs.append((layer, rate))
                job = benchmark.job_info(rate, layer)
                self.assertEqual(log_path.name, f"{job['job_id']}.log")
                self.assertEqual(gpu, 5)
                if layer == 2 and rate == 70:
                    return 1
                folder = Path(manifest).parent / job["output_subdir"]
                folder.mkdir(parents=True)
                benchmark.save_json(folder / "metrics.json", {
                    **job, "status": "success", "complete": True,
                    "accuracy": 0.9, "precision": 0.8, "recall": 0.7, "tnr": None,
                    "lat_mean_ms": 100 if rate == 0 else 100 * layer, "mem_peak_allocated_gib": 1,
                    "mem_peak_allocated_pct": 10, "unparsed": 0,
                })
                return 0

            with patch.object(benchmark, "run_child", side_effect=child), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.launch(args), 1)
            self.assertEqual(jobs, [(None, 0), (2, 50), (2, 70), (2, 90), (3, 50), (3, 70), (3, 90)])
            with (Path(args.output_dir) / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 7)
            self.assertEqual([row["fastv_layer"] for row in rows], ["", "2", "2", "2", "3", "3", "3"])
            self.assertEqual(rows[0]["lat_speedup_vs_baseline"], "1.0")
            self.assertEqual(rows[1]["lat_speedup_vs_baseline"], "0.5")
            self.assertEqual(rows[2]["status"], "failed")
            self.assertEqual(rows[2]["lat_mean_ms"], "")
            self.assertEqual(rows[3]["tnr"], "")
            self.assertEqual(float(rows[4]["lat_speedup_vs_baseline"]), 1 / 3)
            self.assertEqual(rows[5]["status"], "success")  # Layer 2 failure does not hide layer 3.

    def test_child_sets_single_gpu_before_import_and_tees_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(stdout=io.StringIO("progress\n"))
            process.wait.return_value = 0
            process.poll.return_value = 0
            with patch.object(subprocess, "Popen", return_value=process) as start, \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), \
                 contextlib.redirect_stdout(io.StringIO()) as terminal:
                self.assertEqual(benchmark.run_child("run.json", 50, 5, Path(directory) / "worker.log", layer=3), 0)
            self.assertEqual(start.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "5")
            self.assertIn("--worker-rate", start.call_args.args[0])
            self.assertEqual(start.call_args.args[0][-2:], ["--worker-layer", "3"])
            self.assertEqual(terminal.getvalue(), "progress\n")
            self.assertEqual((Path(directory) / "worker.log").read_text(encoding="utf-8"), "progress\n")
            process.terminate.assert_not_called()

    def test_baseline_once_unique_paths_and_invalid_layer_rate_pairs(self):
        jobs = benchmark.experiment_jobs()
        self.assertEqual(sum(job["prune_rate"] == 0 for job in jobs), 1)
        self.assertIsNone(jobs[0]["fastv_layer"])
        self.assertEqual(len({job["output_subdir"] for job in jobs}), 7)
        self.assertEqual(len({job["job_id"] for job in jobs}), 7)
        for rate, layer in ((0, 2), (0, 3), (50, None), (50, 4), (10, 2)):
            with self.subTest(rate=rate, layer=layer), self.assertRaises(ValueError):
                benchmark.job_info(rate, layer)

    def test_wrong_layer_in_effective_config_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory), count=1)
            output, spec = benchmark.prepare(args)
            config = Path(spec["fastv_configs"]["3"]["method_config"])
            altered = json.loads(config.read_text(encoding="utf-8"))
            altered["layer"] = 2
            benchmark.save_json(config, altered)
            with patch("llava_pruning.backend.LlavaBackend") as backend, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark.run_worker(output / "run.json", 50, 3), 1)
                backend.assert_not_called()
            report = json.loads((output / "layer_3" / "prune_50" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fastv_layer"], 3)
            self.assertEqual(report["status"], "failed")
            self.assertIn("Effective FastV configuration changed", report["error"])

    def test_launcher_import_does_not_load_torch_or_numpy(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import benchmark_fastv; "
                                 "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                                cwd=benchmark.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
