import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ex


class ExperimentTests(unittest.TestCase):
    def make_jobs(self, root, count=2, gpus=(4, 5)):
        # Reverse IDs verify that selection follows file order, not ID sorting.
        records = [{"question_id": str(index), "image": f"{index}.png"}
                   for index in range(count, 0, -1)]
        (root / "questions.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8")
        args = ex.build_parser().parse_args([
            "--model-path", str(root), "--input-json", str(root / "questions.jsonl"),
            "--data-root", str(root), "--seed", "42",
            "--gpus", *map(str, gpus),
        ])
        return ex.prepare_jobs(args, root)

    def test_jobs_split_rates_and_keep_original_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = self.make_jobs(Path(directory))
            self.assertEqual([(job["gpu"], job["prune_rate"]) for job in jobs],
                             [(4 + index % 2, index * 10) for index in range(10)])
            self.assertEqual(jobs[0]["input_json"], jobs[1]["input_json"])
            for job in jobs:
                config = json.loads(Path(job["method_config"]).read_text(encoding="utf-8"))
                self.assertEqual(config["layer"], 2)
                self.assertEqual(config["prune_rates"], [job["prune_rate"]])
                self.assertEqual(config["visualize_rates"], [])
                self.assertEqual(job["seed"], 42)
                self.assertEqual(job["sample_limit"], 100)
                self.assertEqual(job["selected_samples"], 2)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}):
                self.assertEqual(ex.worker_environment(4)["CUDA_VISIBLE_DEVICES"], "4")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "5")

    def test_both_jobs_use_only_first_100_in_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = self.make_jobs(root, count=105)
            original = (root / "questions.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(original.splitlines()), 105)
            for job in jobs:
                selected = json.loads(Path(job["input_json"]).read_text(encoding="utf-8"))
                self.assertEqual(len(selected), 100)
                self.assertEqual([row["question_id"] for row in selected],
                                 [str(index) for index in range(105, 5, -1)])
                self.assertEqual(job["selected_samples"], 100)
                self.assertEqual(job["source_input_json"], str((root / "questions.jsonl").resolve()))

    def test_jsonl_stops_before_later_records_and_ignores_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text('\n{"question_id": "id"}\n' * 100 + 'invalid later record',
                            encoding="utf-8")
            self.assertEqual(len(ex.read_first_samples(path)), 100)

    def test_json_array_prefix_and_short_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            for count in (3, 105):
                with self.subTest(count=count):
                    records = [{"question_id": str(index)} for index in range(count)]
                    ex.write_json(path, records)
                    self.assertEqual(ex.read_first_samples(path), records[:100])
            for invalid in ([], {"samples": []}):
                ex.write_json(path, invalid)
                with self.assertRaises(ValueError):
                    ex.read_first_samples(path)

    def test_worker_measures_peaks_and_selected_samples_generation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.make_jobs(Path(directory))[1]
            cuda = SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 1,
                set_device=Mock(), reset_peak_memory_stats=Mock(),
                get_device_name=lambda index: "test GPU", synchronize=Mock(),
                is_initialized=lambda: True,
                max_memory_allocated=lambda index: 3 * ex.GIB,
                max_memory_reserved=lambda index: 4 * ex.GIB,
            )

            def fake_run(**kwargs):
                cuda.reset_peak_memory_stats.assert_called_once_with(0)
                self.assertTrue(kwargs["no_sample"])
                self.assertFalse(kwargs["save_prune_vis"])
                self.assertFalse(kwargs["save_attention_vis"])
                self.assertEqual(kwargs["roi_mode"], "anyres_max_9")
                self.assertEqual(kwargs["prompt_version"], "v0")
                self.assertEqual(kwargs["input_json"], job["input_json"])
                self.assertEqual(len(json.loads(Path(kwargs["input_json"]).read_text(encoding="utf-8"))), 2)
                folder = Path(kwargs["output_dir"]) / "prune_10"
                folder.mkdir(parents=True)
                ex.write_json(folder / "metrics.json", {
                    "accuracy": 0.5, "evaluated_samples": 2, "complete": True,
                })
                (folder / "predictions.jsonl").write_text(
                    '{"generation_seconds": 1.25}\n{"generation_seconds": 2.5}\n', encoding="utf-8")

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}), \
                 patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda),
                                            "llava_pruning.runner": SimpleNamespace(run=fake_run)}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ex.run_worker(job["spec_path"]), 0)
            report = json.loads(Path(job["resource_report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["peak_allocated_gib"], 3)
            self.assertEqual(report["peak_reserved_gib"], 4)
            self.assertEqual(report["generation_seconds_sum"], 3.75)
            self.assertEqual(report["selected_samples"], 2)
            self.assertEqual(report["evaluated_samples"], 2)
            self.assertGreaterEqual(report["worker_wall_seconds"], 0)
            self.assertTrue(report["complete"])
            self.assertEqual(report["status"], "success")
            cuda.set_device.assert_called_once_with(0)
            cuda.synchronize.assert_called_once_with(0)

    def test_worker_failure_still_writes_resource_report(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.make_jobs(Path(directory))[0]
            cuda = SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}), \
                 patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ex.run_worker(job["spec_path"]), 1)
            report = json.loads(Path(job["resource_report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIsNone(report["peak_allocated_gib"])
            self.assertGreaterEqual(report["worker_wall_seconds"], 0)
            self.assertIn("CUDA", report["error"])

    def test_invalid_gpu_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for gpus in ((4, 4), (-1,)):
                with self.subTest(gpus=gpus), self.assertRaises(ValueError):
                    self.make_jobs(Path(directory), gpus=gpus)

    def check_scheduler(self, gpus, fail_rate=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = self.make_jobs(root, gpus=gpus)
            launched = []
            running = {}

            def spawn(command, **kwargs):
                job = jobs[len(launched)]
                self.assertNotIn(job["gpu"], running, "two workers overlapped on the same GPU")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(job["gpu"]))
                self.assertEqual(command[-1], job["spec_path"])
                failed = job["prune_rate"] == fail_rate
                ex.write_json(job["resource_report"], {
                    "status": "failed" if failed else "success", "complete": not failed,
                    "evaluated_samples": 1 if failed else 2,
                    "peak_allocated_gib": 3.0, "peak_reserved_gib": 4.0,
                })
                process = Mock(returncode=None)

                def poll():
                    if process.returncode is None:
                        self.assertGreaterEqual(len(launched), len(gpus), "waited before initial GPUs were launched")
                        del running[job["gpu"]]
                        process.returncode = 1 if failed else 0
                    return process.returncode

                process.poll.side_effect = poll
                launched.append(process)
                running[job["gpu"]] = process
                return process

            with patch("ex.subprocess.Popen", side_effect=spawn), \
                 patch("ex.time.sleep"), patch("ex.generate_reports") as reports, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ex.launch_jobs(jobs, root), 0 if fail_rate is None else 1)
            reports.assert_called_once_with(root / "benchmark.json")
            self.assertEqual(len(launched), 10)
            self.assertEqual(running, {})
            report = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["jobs"]), 10)
            self.assertFalse(report["settings"]["save_prune_vis"])
            self.assertFalse(report["settings"]["save_attention_vis"])
            self.assertEqual(report["settings"]["sample_limit"], 100)
            self.assertEqual(report["settings"]["selected_samples"], 2)
            self.assertEqual(report["settings"]["gpus"], list(gpus))
            self.assertEqual(report["settings"]["prune_rates"], list(range(0, 100, 10)))
            self.assertGreaterEqual(report["experiment_wall_seconds"], 0)
            self.assertTrue((root / "benchmark.csv").is_file())
            for job in report["jobs"]:
                self.assertGreaterEqual(job["total_seconds"], 0)
                self.assertEqual(job["status"], "failed" if job["prune_rate"] == fail_rate else "success")

    def test_launcher_sweeps_all_rates_without_overlapping_gpu_jobs(self):
        self.check_scheduler((4, 5))

    def test_single_gpu_sweep(self):
        self.check_scheduler((5,))

    def test_failed_job_does_not_prevent_remaining_rates_or_report(self):
        self.check_scheduler((4, 5), fail_rate=20)

    def test_launch_failure_keeps_raw_reports_and_stops_own_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = self.make_jobs(root)
            process = Mock(returncode=-15)
            process.poll.return_value = None
            with patch("ex.subprocess.Popen", side_effect=[process, OSError("test launch failure")]), \
                 patch("ex.generate_reports") as reports, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ex.launch_jobs(jobs, root), 1)
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)
            reports.assert_called_once()
            self.assertTrue((root / "benchmark.csv").is_file())
            summary = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["status"] == "failed" for row in summary["jobs"]))
            self.assertIsNone(summary["jobs"][-1]["total_seconds"])

    def test_report_only_does_not_launch_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["ex.py", "--report-only", directory]), \
                 patch("ex.generate_reports") as reports, patch("ex.subprocess.Popen") as popen, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ex.main(), 0)
            reports.assert_called_once_with(Path(directory).resolve() / "benchmark.json")
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
