"""CPU-only checks for the four-run launcher; no real model/GPU is loaded."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ablate_attention_predictor as ablation


class PredictorAblationTests(unittest.TestCase):
    def fixture(self, root, count=103):
        model, visa, test = (root / name for name in ("model", "visa", "mvtec"))
        for folder in (model, visa / "split_csv", test):
            folder.mkdir(parents=True)
        rows = []
        for category in ("candle", "cashew"):
            (visa / category).mkdir()
            for index in range(4):
                relative = f"{category}/{index}.png"
                (visa / relative).write_bytes(b"preflight only; no image decoding")
                rows.append(dict(object=category, image=relative, label="ignored"))
        with (visa / "split_csv/1cls.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["object", "image", "label"])
            writer.writeheader()
            writer.writerows(rows)
        records = []
        for index in range(count):
            (test / f"{index}.png").write_bytes(b"preflight only")
            records.append(dict(question_id=str(index), image=f"{index}.png", gt=index % 2,
                                origin_path=f"screw/test/good/{index}.png"))
        questions = root / "questions.jsonl"
        questions.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        argv = ["--gpus", "4", "5", "--model-path", str(model), "--visa-root", str(visa),
                "--input-json", str(questions), "--data-root", str(test), "--output-dir", str(root / "result")]
        args = ablation.build_parser().parse_args(argv)
        return argv, args, ablation.prepare(args)

    def training_output(self, plan, job):
        folder = Path(job["directory"]) / "train"
        folder.mkdir(parents=True, exist_ok=True)
        settings, data = plan["settings"], plan["data"]
        samples, _ = ablation.load_visa_distillation(settings["visa_root"], settings["visa_split_csv"])
        train, validation = ablation.split_samples(samples, .1, settings["seed"])
        for name, selected in (("train", train), ("validation", validation)):
            records = (dict(question_id=s.sample_id, image=s.origin_path, category=s.category,
                            text=ablation.GENERIC_PROMPT_V0) for s in selected)
            (folder / f"{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        ablation.write_json(folder / "train_run.json", dict(
            configuration=dict(epochs=settings["epochs"], seed=settings["seed"], teacher_layer=22,
                               gradient_accumulation=job["batch_size"], learning_rate=job["learning_rate"],
                               model_path=settings["model_path"], data_root=settings["visa_root"],
                               visa_split_csv=settings["visa_split_csv"], validation_fraction=.1,
                               weight_decay=0., adam_beta1=.9, adam_beta2=.98, warmup_ratio=.03,
                               max_grad_norm=1., student_precision="bfloat16"),
            version=2, student_source_layer=2, teacher_layer=22, backbone_frozen=True,
            architecture=dict(source_layers=[3, 4, 5]),
            training_samples=data["training_samples"], validation_samples=data["validation_samples"],
            source=dict(source_sha256=data["visa_csv_sha256"]), split_sha256=data["split_sha256"]))
        ablation.write_json(folder / "training_status.json", dict(
            complete=True, completed_epochs=settings["epochs"], best_validation_kl=1 / settings["epochs"]))
        with (folder / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "split", "kl"])
            writer.writeheader()
            writer.writerows(dict(epoch=epoch, split="validation", kl=1 / epoch)
                             for epoch in range(1, settings["epochs"] + 1))
        (folder / "best.pt").write_bytes(b"checkpoint fixture; never unpickled")

    def evaluation_output(self, plan, job):
        base, settings = Path(job["directory"]), plan["settings"]
        folder = base / "test"
        folder.mkdir(parents=True, exist_ok=True)
        config = dict(checkpoint=str(base / "train" / "best.pt"), layer=2, rate_semantics="layer_average",
                      selection_mode="base_guided", prune_rates=settings["prune_rates"], visualize_rates=[])
        ablation.write_json(base / "test_method.json", config)
        ablation.write_json(folder / "run.json", dict(
            model_path=settings["model_path"], input_json=settings["input_json"], data_root=settings["data_root"],
            method="attention_predictor", roi_mode="anyres_max_9", prompt_version="v0", image_token_order="base_first",
            save_prune_vis=False, save_attention_vis=False, warmup_requested=settings["warmup"],
            do_sample=False, include_pruning_time=True, benchmark_metrics=True, seed=settings["seed"],
            samples=plan["data"]["test_samples"], method_config=config))
        count = plan["data"]["test_samples"]
        metrics = []
        for rate in settings["prune_rates"]:
            item = dict(prune_rate=rate, complete=True, expected_samples=count, evaluated_samples=count,
                        labeled_samples=count, unparsed=1, parsed_samples=count - 1,
                        tp=40, fp=5, fn=10, tn=count - 56, correct=count - 16,
                        accuracy=(count - 16) / count, precision=40 / 45, recall=.8,
                        tnr=(count - 56) / (count - 51), lat_mean_ms=120., lat_p50_ms=119., lat_p95_ms=145.,
                        mem_peak_allocated_gib=20., mem_peak_reserved_gib=22., actual_average_prune_rate=rate)
            ablation.write_json(folder / f"prune_{ablation.format_prune_rate(rate)}" / "metrics.json", item)
            metrics.append(item)
        with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)

    def test_grid_defaults_full_data_and_no_preflight_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, args, plan = self.fixture(Path(directory))
            self.assertFalse(args.output_dir.exists())
            self.assertEqual([(j["batch_size"], j["learning_rate"]) for j in plan["experiments"]],
                             [(32, 2e-5), (32, 5e-5), (64, 2e-5), (64, 5e-5)])
            self.assertEqual(plan["data"]["test_samples"], 103)
            self.assertEqual(plan["data"]["training_samples"], 6)
            self.assertEqual(plan["data"]["validation_samples"], 2)
            self.assertEqual(plan["settings"]["epochs"], 5)
            self.assertEqual(plan["settings"]["prune_rates"], [50, 70, 90])
            self.assertEqual(plan["settings"]["eval_gpu"], 4)
            self.assertAlmostEqual(plan["settings"]["baseline_accuracy"], .9118)
            for job in plan["experiments"]:
                train = ablation.command_for(plan, job, "train")
                test = ablation.command_for(plan, job, "test")
                self.assertEqual(train[train.index("--gradient-accumulation") + 1], str(job["batch_size"]))
                self.assertEqual(train[train.index("--seed") + 1], "42")
                self.assertNotIn("--cache-dir", train)
                self.assertIn("--benchmark-metrics", test)
                self.assertIn("--no-plots", test)
                self.assertIn("--no-sample", test)
                self.assertNotIn("--save-prune-vis", test)
                self.assertNotIn("--sample-limit", test)
                self.assertEqual(test[test.index("--input-json") + 1], str(args.input_json.resolve()))

    def test_commands_are_accepted_by_actual_public_clis(self):
        import train_attention_predictor
        import run
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            for job in plan["experiments"]:
                train = train_attention_predictor.build_parser().parse_args(ablation.command_for(plan, job, "train")[3:])
                test = run.build_parser().parse_args(ablation.command_for(plan, job, "test")[3:])
                self.assertEqual(train.epochs, 5)
                self.assertEqual(train.gradient_accumulation, job["batch_size"])
                self.assertEqual(train.learning_rate, job["learning_rate"])
                self.assertTrue(test.benchmark_metrics)
                self.assertTrue(test.include_pruning_time)
                self.assertEqual(test.method, "attention_predictor")

    def test_metric_definitions_unparsed_and_retention_not_clamped(self):
        row = ablation.extended_metrics(dict(tp=2, fp=1, tn=2, fn=1, unparsed=2, accuracy=.5), .9118)
        self.assertAlmostEqual(row["f1"], 2 / 3)
        self.assertAlmostEqual(row["bacc"], 2 / 3)
        self.assertAlmostEqual(row["relative_accuracy_pct"], .5 / .9118 * 100)
        self.assertEqual(row["unparsed"], 2)
        perfect = ablation.extended_metrics(dict(tp=1, fp=0, tn=1, fn=0, accuracy=1.), .9118)
        self.assertGreater(perfect["relative_accuracy_pct"], 100)
        self.assertAlmostEqual(ablation.extended_metrics(dict(tp=1, fp=0, tn=1, fn=0, accuracy=.9118), .9118)
                               ["relative_accuracy_pct"], 100)

    def test_undefined_metrics_and_bad_units(self):
        empty = dict(tp=0, fp=0, tn=0, fn=0, accuracy=None)
        result = ablation.extended_metrics(empty, .9118)
        for key in ("f1", "bacc", "relative_accuracy_pct"):
            self.assertIsNone(result[key])
        result = ablation.extended_metrics(dict(tp=0, fp=0, tn=3, fn=2, accuracy=.6), .9118)
        self.assertEqual(result["f1"], 0)
        self.assertEqual(result["bacc"], .5)
        for baseline in (0, 91.18, float("nan")):
            with self.assertRaises(ValueError):
                ablation.extended_metrics(empty, baseline)
        with self.assertRaises(ValueError):
            ablation.extended_metrics(dict(empty, accuracy=91.18), .9118)

    def test_invalid_gpu_reference_and_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            _, args, _ = self.fixture(Path(directory))
            for field, value in (("gpus", None), ("gpus", [4, 4]), ("gpus", [-1, 5]),
                                 ("eval_gpu", 3), ("baseline_acc", 0), ("baseline_acc", 101),
                                 ("prune_rates", [50, 50]), ("prune_rates", [float("nan")]),
                                 ("prune_rates", [100])):
                with self.subTest(field=field, value=value):
                    changed = SimpleNamespace(**vars(args))
                    setattr(changed, field, value)
                    with self.assertRaises(ValueError):
                        ablation.prepare(changed)

    def test_dry_run_does_not_create_outputs_or_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            argv, args, _ = self.fixture(Path(directory))
            with patch.object(ablation.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(ablation.main(argv + ["--dry-run"]), 0)
            self.assertEqual(len(json.loads(output.getvalue())["commands"]), 4)
            self.assertFalse(args.output_dir.exists())
            popen.assert_not_called()

    def test_training_verifies_exact_split_and_completed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            job = plan["experiments"][0]
            self.training_output(plan, job)
            result = ablation.validate_training(plan, job)
            self.assertEqual(result, dict(best_epoch=5, validation_kl=.2))
            (Path(job["directory"]) / "train/train.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed split"):
                ablation.validate_training(plan, job)

    def test_enrichment_is_idempotent_and_keeps_latency_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            job = plan["experiments"][0]
            self.evaluation_output(plan, job)
            for _ in range(2):
                ablation.validate_test(plan, job)
            path = Path(job["directory"]) / "test"
            with (path / "summary.csv").open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
            self.assertTrue(set(ablation.EXTRA_FIELDS).issubset(rows[0]))
            self.assertEqual(float(rows[0]["lat_mean_ms"]), 120)
            self.assertEqual(float(rows[0]["mem_peak_allocated_gib"]), 20)
            metrics = ablation.read_json(path / "prune_50/metrics.json")
            self.assertAlmostEqual(metrics["f1"], 80 / 95)
            self.assertEqual(metrics["unparsed"], 1)
            self.assertEqual(metrics["baseline_accuracy"], .9118)

    def test_partial_test_is_not_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            job = plan["experiments"][0]
            self.evaluation_output(plan, job)
            path = Path(job["directory"]) / "test/prune_70/metrics.json"
            ablation.write_json(path, dict(ablation.read_json(path), complete=False))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                ablation.validate_test(plan, job)

    def run_launcher(self, plan, *, fail_name=None):
        active, events = {}, []
        by_folder = {Path(job["directory"]): job for job in plan["experiments"]}

        def spawn(command, **kwargs):
            folder = Path(command[command.index("--output-dir") + 1])
            phase, job = folder.name, by_folder[folder.parent]
            gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            self.assertNotIn(gpu, active)
            if phase == "test":
                self.assertFalse(active, "test overlapped training or another test")
                self.assertEqual(sum(event[0] == "train_done" for event in events), 4)
                self.assertEqual(gpu, plan["settings"]["eval_gpu"])
            active[gpu] = job["name"]
            events.append((f"{phase}_start", job["name"], gpu))
            process = SimpleNamespace(returncode=None, polls=0)

            def poll():
                if process.returncode is not None:
                    return process.returncode
                process.polls += 1
                if process.polls == 1:
                    return None
                failed = phase == "train" and job["name"] == fail_name
                if not failed:
                    if phase == "train":
                        self.training_output(plan, job)
                    else:
                        self.evaluation_output(plan, job)
                del active[gpu]
                events.append((f"{phase}_done", job["name"], gpu))
                process.returncode = 1 if failed else 0
                return process.returncode

            process.poll = poll
            return process

        with patch.object(ablation.subprocess, "Popen", side_effect=spawn), \
                patch.object(ablation.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            code = ablation.launch(plan)
        self.assertFalse(active)
        return code, events

    def test_two_training_slots_then_four_serial_tests_and_12_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            code, events = self.run_launcher(plan)
            self.assertEqual(code, 0)
            self.assertEqual([event[0] for event in events[:2]], ["train_start", "train_start"])
            self.assertEqual([event[2] for event in events[:2]], [4, 5])
            self.assertEqual(sum(event[0] == "test_start" for event in events), 4)
            path = Path(plan["output_dir"])
            with (path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["status"] == "success" for row in rows))
            self.assertTrue(all(row["f1"] and row["bacc"] and row["relative_accuracy_pct"] for row in rows))
            self.assertTrue(all(row["test_gpu"] == "4" for row in rows))
            self.assertFalse((path / ".launcher.lock").exists())

    def test_one_failed_training_does_not_stop_other_three(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            failed = plan["experiments"][1]["name"]
            code, events = self.run_launcher(plan, fail_name=failed)
            self.assertEqual(code, 1)
            self.assertEqual(sum(event[0] == "train_start" for event in events), 4)
            self.assertEqual(sum(event[0] == "test_start" for event in events), 3)
            self.assertEqual(plan["experiments"][1]["test"]["status"], "skipped")

    def test_resume_verifies_complete_stages_and_does_not_spawn_again(self):
        with tempfile.TemporaryDirectory() as directory:
            argv, _, plan = self.fixture(Path(directory))
            self.run_launcher(plan)
            with patch.object(ablation.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ablation.main(argv + ["--resume"]), 0)
            popen.assert_not_called()
            changed = dict(plan, data=dict(plan["data"], test_input_sha256="changed"))
            with self.assertRaisesRegex(ValueError, "identical"):
                ablation.load_or_create_plan(changed, resume=True)
            with self.assertRaises(FileExistsError):
                ablation.load_or_create_plan(plan)

    def test_incomplete_stage_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            plan["experiments"] = plan["experiments"][:1]
            folder = Path(plan["experiments"][0]["directory"]) / "train"
            folder.mkdir(parents=True)
            marker = folder / "partial.txt"
            marker.write_text("preserve me", encoding="utf-8")
            with patch.object(ablation.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
                ablation.run_phase(plan, "train", [4, 5])
            popen.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
            self.assertEqual(plan["experiments"][0]["train"]["status"], "failed")

    def test_report_only_uses_saved_results_without_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            self.run_launcher(plan)
            with patch.object(ablation.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ablation.main(["--report-only", plan["output_dir"]]), 0)
            popen.assert_not_called()

    def test_environment_does_not_mutate_parent_visibility(self):
        with patch.dict(ablation.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            self.assertEqual(ablation.worker_environment(5)["CUDA_VISIBLE_DEVICES"], "5")
            self.assertEqual(ablation.os.environ["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_keyboard_interrupt_stops_only_owned_children_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            processes, interrupt_sent = [], False

            def spawn(*args, **kwargs):
                process = Mock(returncode=None)

                def poll():
                    nonlocal interrupt_sent
                    if not interrupt_sent:
                        interrupt_sent = True
                        raise KeyboardInterrupt()
                    return process.returncode

                process.poll.side_effect = poll
                process.terminate.side_effect = lambda: setattr(process, "returncode", -15)
                processes.append(process)
                return process

            with patch.object(ablation.subprocess, "Popen", side_effect=spawn), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                ablation.launch(plan)
            self.assertEqual(len(processes), 2)
            for process in processes:
                process.terminate.assert_called_once()
                process.wait.assert_called_once_with(timeout=5)
            self.assertTrue(all(j["train"]["status"] == "interrupted" for j in plan["experiments"][:2]))
            self.assertFalse((Path(plan["output_dir"]) / ".launcher.lock").exists())
            self.assertTrue((Path(plan["output_dir"]) / "summary.csv").is_file())

    def test_existing_lock_is_not_removed_or_used(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.fixture(Path(directory))
            root = Path(plan["output_dir"])
            root.mkdir()
            lock = root / ".launcher.lock"
            lock.write_text("another launcher's lock", encoding="utf-8")
            with patch.object(ablation.subprocess, "Popen") as popen, self.assertRaisesRegex(FileExistsError, "lock exists"):
                ablation.launch(plan)
            self.assertEqual(lock.read_text(encoding="utf-8"), "another launcher's lock")
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
