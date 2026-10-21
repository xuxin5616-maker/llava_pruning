"""CPU checks for experiment plans and result bookkeeping; no model weights."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import runPyramidDrop as entry
import torch
from torch import nn

from test_pyramiddrop import TinyPyramidDrop, tiny_config


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = TinyPyramidDrop(tiny_config()).eval()

    def get_model(self):
        return self.core


class FakeBackend:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.kwargs = kwargs
        self.model = FakeModel()
        self.inference_config = {"attention": "eager", "test_backend": True}
        self.calls = []
        self.__class__.instances.append(self)

    def generate(self, sample, prompt, method, rate, roi_mode, **kwargs):
        self.calls.append((sample.sample_id, dict(method.setting), rate, roi_mode, kwargs))
        core = self.model.get_model()
        method.configure(core, rate, capture_visualization=kwargs["capture_visualization"],
                         capture_attention=kwargs["capture_attention"])
        core.set_pyramiddrop_image_spans([[(2, 9)]], 11)
        with torch.no_grad():
            core(input_ids=torch.arange(1, 12)[None], use_cache=True)
        result = {"answer": "A" if sample.gt else "B", "generation_seconds": 0.02,
                  "stats": method.stats(core, rate), "max_new_tokens": 512}
        if kwargs.get("benchmark_metrics"):
            ordinal = sum(call[1]["name"] == method.setting["name"] for call in self.calls)
            # The benchmark test requests one warmup; make contamination visible.
            warmup = ordinal == 1
            result.update(lat_seconds=5 if warmup else 0.007, generated_tokens=2,
                          peak_allocated_bytes=(8 if warmup else 1) * 1024**3,
                          peak_reserved_bytes=(9 if warmup else 2) * 1024**3,
                          generation_seconds=6 if warmup else 0.02)
        return result


class PlanTests(unittest.TestCase):
    def plan(self, *arguments, depth=28):
        return entry.resolve_plan(entry.build_parser().parse_args(list(arguments)), depth)

    def invalid(self, *arguments, depth=28):
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises((ValueError, SystemExit)):
            self.plan(*arguments, depth=depth)

    def test_official_preset_uses_actual_decoder_depth(self):
        [plan] = self.plan("--stage-keep-ratio", "0.5")
        self.assertEqual(tuple(plan["layers"]), (7, 14, 21))
        self.assertEqual(tuple(plan["keep_ratios"]), (0.5, 0.25, 0.125))
        self.assertIsNone(plan["prune_rate"])
        self.assertTrue(plan["enabled"])
        self.assertTrue(plan["name"])
        self.assertTrue(plan["rate_semantics"])
        [thirty_two] = self.plan("--stage-keep-ratio", "0.5", depth=32)
        self.assertEqual(tuple(thirty_two["layers"]), (8, 16, 24))

    def test_default_sweep_is_average_visual_token_retention(self):
        plans = self.plan()
        self.assertEqual([plan["average_keep_rate"] for plan in plans], [100, 50, 70, 90])
        self.assertFalse(plans[0]["enabled"])
        self.assertTrue(all(plan["enabled"] for plan in plans[1:]))
        self.assertTrue(all(plan["prune_rate"] is None for plan in plans))
        self.assertEqual(len({plan["name"] for plan in plans}), 4)

    def test_average_budget_uses_each_stage_width_including_unequal_depth(self):
        for depth in (28, 29, 32):
            plans = self.plan("--average-keep-rates", "50", "70", "90", "100", depth=depth)
            for plan in plans:
                with self.subTest(depth=depth, rate=plan["average_keep_rate"]):
                    boundaries = (0, *plan["layers"], depth)
                    widths = [b - a for a, b in zip(boundaries, boundaries[1:])]
                    fractions = (1.0, *plan["keep_ratios"])
                    actual_average = sum(w * f for w, f in zip(widths, fractions)) / depth
                    target = plan["average_keep_rate"] / 100
                    self.assertAlmostEqual(actual_average, target, places=10)
                    self.assertAlmostEqual(plan["expected_average_keep_ratio"], target, places=10)
                    retention = plan["keep_ratios"][0]
                    for index, actual in enumerate(plan["keep_ratios"], start=1):
                        self.assertAlmostEqual(actual, retention ** index, places=10)

    def test_average_budget_rejects_unreachable_and_ambiguous_requests(self):
        for rate in ("0", "20", "25", "-1", "101"):
            self.invalid("--average-keep-rates", rate)
        self.invalid("--average-keep-rates", "50", "50")
        self.invalid("--average-keep-rates", "50", "--stage-keep-ratio", "0.5")
        self.invalid("--average-keep-rates", "50", "--keep-ratios", "0.8", "0.5", "0.2")
        self.invalid("--average-keep-rates", "50", "--prune-rate", "50",
                     "--rate-semantics", "per_stage")
        self.invalid("--average-keep-rates", "50", "--rate-semantics", "final_cumulative")

    def test_close_fractional_targets_have_distinct_output_directories(self):
        plans = self.plan("--average-keep-rates", "50", "50.00001")
        self.assertNotEqual(plans[0]["name"], plans[1]["name"])

    def test_prevalidation_defers_reachability_until_actual_depth_is_known(self):
        args = entry.build_parser().parse_args([
            "--layers", "20", "40", "60", "--average-keep-rates", "30"])
        self.assertEqual(entry.resolve_plan(args, 61, validate_only=True), [])
        [plan] = entry.resolve_plan(args, 80)
        self.assertAlmostEqual(plan["expected_average_keep_ratio"], 0.3)
        with self.assertRaisesRegex(ValueError, "unreachable"):
            entry.resolve_plan(args, 61)

    def test_requested_final_cumulative_rates_end_at_requested_budget(self):
        plans = self.plan("--prune-rates", "0", "50", "70", "90",
                          "--rate-semantics", "final_cumulative")
        self.assertEqual([item["prune_rate"] for item in plans], [0, 50, 70, 90])
        self.assertEqual(len({item["name"] for item in plans}), 4)
        for item in plans:
            retained = 1 - item["prune_rate"] / 100
            self.assertEqual(tuple(item["layers"]), (7, 14, 21))
            for index, actual in enumerate(item["keep_ratios"], start=1):
                self.assertAlmostEqual(actual, retained ** (index / 3))
            self.assertEqual(item["rate_semantics"], "final_cumulative")
            self.assertEqual(item["enabled"], item["prune_rate"] != 0)

    def test_per_stage_rate_is_compounded_and_preserves_setting_order(self):
        plans = self.plan("--prune-rates", "90", "50",
                          "--rate-semantics", "per_stage")
        self.assertEqual([item["prune_rate"] for item in plans], [90, 50])
        for item in plans:
            retained = 1 - item["prune_rate"] / 100
            self.assertEqual(item["rate_semantics"], "per_stage")
            for index, actual in enumerate(item["keep_ratios"], start=1):
                self.assertAlmostEqual(actual, retained ** index)

    def test_stage_keep_ratio_and_absolute_custom_budgets(self):
        [stage] = self.plan("--stage-keep-ratio", "0.8")
        for actual, expected in zip(stage["keep_ratios"], (0.8, 0.64, 0.512)):
            self.assertAlmostEqual(actual, expected)
        [custom] = self.plan("--layers", "5", "10", "20",
                             "--keep-ratios", "0.8", "0.4", "0.2")
        self.assertEqual(tuple(custom["layers"]), (5, 10, 20))
        self.assertEqual(tuple(custom["keep_ratios"]), (0.8, 0.4, 0.2))
        self.assertIsNone(custom["prune_rate"])
        self.assertTrue(custom["enabled"])

    def test_rates_need_an_explicit_comparison_semantics(self):
        self.invalid("--prune-rate", "50")
        self.invalid("--prune-rates", "0", "50", "70", "90")
        self.invalid("--prune-rate", "50", "--prune-rates", "70",
                     "--rate-semantics", "per_stage")

    def test_invalid_rates_layers_and_ratios_are_rejected(self):
        cases = [
            ("--prune-rates", "50", "50", "--rate-semantics", "per_stage"),
            ("--prune-rate", "-1", "--rate-semantics", "per_stage"),
            ("--prune-rate", "100", "--rate-semantics", "per_stage"),
            ("--layers", "0", "14", "21"),
            ("--layers", "7", "14", "28"),
            ("--layers", "14", "7", "21"),
            ("--layers", "7", "7", "21"),
            ("--stage-keep-ratio", "0"),
            ("--stage-keep-ratio", "1.01"),
            ("--stage-keep-ratio", "nan"),
            ("--stage-keep-ratio", "inf"),
            ("--keep-ratios", "0.5", "0.25"),
            ("--keep-ratios", "0.5", "0.7", "0.25"),
            ("--keep-ratios", "0.5", "0.25", "0"),
            ("--keep-ratios", "0.5", "0.25", "nan"),
            ("--prune-rate", "50", "--rate-semantics", "per_stage",
             "--keep-ratios", "0.5", "0.25", "0.125"),
            ("--stage-keep-ratio", "0.8", "--keep-ratios", "0.5", "0.25", "0.125"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.invalid(*arguments)
        self.invalid(depth=0)

    def test_plan_only_needs_no_backend_or_cuda(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["runPyramidDrop.py", "--plan-only", "--model-depth", "28"]), \
                patch.object(entry, "run_experiment", side_effect=AssertionError("must not load a backend")), \
                patch.dict(sys.modules, {"torch": None, "transformers": None}), \
                contextlib.redirect_stdout(stdout):
            entry.main()
        plans = json.loads(stdout.getvalue())
        self.assertEqual([plan["average_keep_rate"] for plan in plans], [100, 50, 70, 90])
        self.assertTrue(all(plan["layers"] == [7, 14, 21] for plan in plans))


class EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        FakeBackend.instances.clear()

    def create_input(self, root):
        records = []
        for index, gt in enumerate((0, 1)):
            image = root / f"sample_{index}.png"
            image.write_bytes(b"the CPU backend does not read image pixels")
            records.append({"question_id": str(index), "image": image.name,
                            "category": "screw", "gt": gt})
        inputs = root / "questions.json"
        inputs.write_text(json.dumps(records), encoding="utf-8")
        return inputs

    def arguments(self, root, inputs, *extra):
        return entry.build_parser().parse_args([
            "--model-path", str(root / "checkpoint"), "--input-json", str(inputs),
            "--data-root", str(root), "--output-dir", str(root / "output"), *extra])

    def test_classification_sweep_reuses_backend_and_writes_complete_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self.create_input(root)
            args = self.arguments(root, inputs)
            expected_plans = entry.resolve_plan(args, 4)
            with contextlib.redirect_stdout(io.StringIO()):
                actual = entry.run_experiment(args, backend_factory=FakeBackend)
            output = root / "output"
            self.assertEqual(actual, output.resolve())
            self.assertEqual(len(FakeBackend.instances), 1)
            backend = FakeBackend.instances[0]
            self.assertEqual(backend.kwargs, {"roi_mode": "anyres_max_9", "image_token_order": "base_first"})
            self.assertEqual(len(backend.calls), 8)
            self.assertEqual([call[4]["random_seed"] for call in backend.calls], [42, 43] * 4)
            self.assertTrue(all(call[4]["include_pruning_time"] for call in backend.calls))
            self.assertTrue(all(not call[4]["do_sample"] for call in backend.calls))
            self.assertTrue(all(not call[4]["capture_visualization"] for call in backend.calls))
            self.assertTrue(all(not call[4]["capture_attention"] for call in backend.calls))
            # The loaded model has four layers, despite the preview default of 28.
            self.assertTrue(all(tuple(call[1]["layers"]) == (1, 2, 3) for call in backend.calls))
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["model_depth"], 4)
            self.assertEqual(run["samples"], 2)
            self.assertEqual(run["inference"]["attention"], "eager")
            self.assertEqual(run["settings"], json.loads(json.dumps(expected_plans)))
            self.assertTrue(run["source"]["commit"])
            with (output / "summary.csv").open(encoding="utf-8", newline="") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 4)
            for item, row in zip(expected_plans, summary):
                folder = output / item["name"]
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                self.assertTrue(metrics["complete"])
                self.assertEqual(metrics["expected_samples"], 2)
                self.assertEqual(metrics["evaluated_samples"], 2)
                self.assertEqual(metrics["accuracy"], 1.0)
                self.assertEqual(metrics["precision"], 1.0)
                self.assertEqual(metrics["recall"], 1.0)
                self.assertEqual(metrics["tnr"], 1.0)
                self.assertEqual(metrics["mean_generation_seconds"], 0.02)
                self.assertEqual(metrics["average_keep_rate"], item["average_keep_rate"])
                predictions = [json.loads(line) for line in
                               (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([record["prediction"] for record in predictions], [0, 1])
                final_count = max(1, int(7 * item["keep_ratios"][-1]))
                self.assertEqual(metrics["mean_final_llm_visual_tokens"], final_count)
                for record in predictions:
                    stats = record["pruning"]
                    self.assertEqual(stats["original_image_tokens"], 7)
                    self.assertEqual(stats["final_llm_visual_tokens"], final_count)
                    self.assertEqual(len(stats["layers"]), 4)
                    average = sum(layer["image_tokens_in"] for layer in stats["layers"]) / 4
                    self.assertEqual(record["layer_average_visual_tokens"], average)
                    self.assertEqual(metrics["mean_layer_average_visual_tokens"], average)
                    self.assertAlmostEqual(metrics["mean_actual_average_keep_ratio"], average / 7)
                self.assertEqual(row["setting"], item["name"])
                self.assertEqual(row["complete"], "True")
                self.assertEqual(float(row["average_keep_rate"]), item["average_keep_rate"])
                self.assertEqual(json.loads(row["keep_ratios"]), list(item["keep_ratios"]))
                with (folder / "classification_results.csv").open(encoding="utf-8-sig", newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 2)
            with self.assertRaises(FileExistsError):
                entry.run_experiment(args, backend_factory=FakeBackend)
            self.assertEqual(len(FakeBackend.instances), 1)

    def test_zero_rate_keeps_native_path_even_when_visualizations_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments(root, self.create_input(root), "--prune-rate", "0",
                                  "--rate-semantics", "final_cumulative", "--save-prune-vis",
                                  "--save-attention-vis")
            with contextlib.redirect_stdout(io.StringIO()):
                entry.run_experiment(args, backend_factory=FakeBackend)
            backend = FakeBackend.instances[0]
            self.assertTrue(all(not call[4]["capture_visualization"] for call in backend.calls))
            self.assertTrue(all(not call[4]["capture_attention"] for call in backend.calls))
            self.assertFalse(backend.model.get_model().pyramiddrop_enabled)
            self.assertFalse(backend.model.get_model().pyramiddrop_capture_stages)
            self.assertEqual(backend.model.get_model().get_pyramiddrop_stats()["mode"], "disabled_baseline")

    def test_warmup_is_excluded_from_predictions_and_metric_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments(root, self.create_input(root), "--stage-keep-ratio", "0.5",
                                  "--warmup", "1")
            with contextlib.redirect_stdout(io.StringIO()):
                output = entry.run_experiment(args, backend_factory=FakeBackend)
            self.assertEqual(len(FakeBackend.instances[0].calls), 3)
            [item] = entry.resolve_plan(args, 4)
            metrics = json.loads((output / item["name"] / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["evaluated_samples"], 2)
            self.assertEqual(metrics["mean_generation_seconds"], 0.02)
            self.assertEqual(len((output / item["name"] / "predictions.jsonl").read_text().splitlines()), 2)

    def test_interrupted_setting_leaves_incomplete_metrics_and_saved_predictions(self):
        original_generate = FakeBackend.generate
        def fail_second(backend, *args, **kwargs):
            if backend.calls:
                raise RuntimeError("simulated inference failure")
            return original_generate(backend, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments(root, self.create_input(root), "--average-keep-rates", "50")
            with patch.object(FakeBackend, "generate", fail_second), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
                entry.run_experiment(args, backend_factory=FakeBackend)
            [item] = entry.resolve_plan(args, 4)
            folder = root / "output" / item["name"]
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["complete"])
            self.assertEqual(metrics["expected_samples"], 2)
            self.assertEqual(metrics["evaluated_samples"], 1)
            self.assertEqual(len((folder / "predictions.jsonl").read_text().splitlines()), 1)
            with (root / "output" / "summary.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_benchmark_reuses_shared_protocol_and_excludes_each_settings_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments(root, self.create_input(root), "--prune-rates", "0", "50",
                                  "--rate-semantics", "final_cumulative",
                                  "--benchmark-metrics", "--warmup", "1")
            with patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.device_count", return_value=1), \
                    patch("benchmark_fastv.validate_single_gpu") as validate, \
                    patch("torch.cuda.get_device_properties",
                          return_value=types.SimpleNamespace(total_memory=10 * 1024**3)), \
                    patch.object(entry, "_reset_peak_memory", side_effect=AssertionError("reset is at model entry")), \
                    patch.object(entry, "_read_peak_memory", side_effect=AssertionError("reuse measured peaks")), \
                    contextlib.redirect_stdout(io.StringIO()):
                output = entry.run_experiment(args, backend_factory=FakeBackend)
            validate.assert_called_once()
            self.assertEqual(len(FakeBackend.instances[0].calls), 6)
            self.assertTrue(all(call[4]["benchmark_metrics"] for call in FakeBackend.instances[0].calls))
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            for rate, row in zip((0, 50), rows):
                metrics = json.loads((output / f"prune_{rate:02d}/metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["evaluated_samples"], 2)
                self.assertEqual(metrics["warmup_samples"], 1)
                self.assertEqual(metrics["accuracy"], 1)
                self.assertEqual(metrics["lat_mean_ms"], 7)
                self.assertEqual(metrics["mem_peak_allocated_gib"], 1)
                self.assertEqual(metrics["mem_peak_reserved_gib"], 2)
                self.assertEqual(metrics["mem_peak_allocated_pct"], 10)
                self.assertEqual(metrics["generation_mean_ms"], 20)
                self.assertEqual(float(row["lat_mean_ms"]), 7)
                records = [json.loads(line) for line in
                           (output / f"prune_{rate:02d}/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), 2)
                self.assertTrue(all(record["lat_seconds"] == 0.007 for record in records))
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(run["benchmark_metrics"])
            self.assertIn("first", run["benchmark_definitions"]["lat"])

    def test_benchmark_rejects_visualization_and_multiple_gpus_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self.create_input(root)
            args = self.arguments(root, inputs, "--benchmark-metrics", "--save-attention-vis")
            with self.assertRaisesRegex(ValueError, "visualization disabled"):
                entry.run_experiment(args, backend_factory=FakeBackend)
            args = self.arguments(root, inputs, "--benchmark-metrics")
            with patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.device_count", return_value=2), \
                    self.assertRaisesRegex(RuntimeError, "one visible CUDA GPU"):
                entry.run_experiment(args, backend_factory=FakeBackend)
            self.assertEqual(FakeBackend.instances, [])

    def test_invalid_experiment_inputs_fail_before_loading_backend(self):
        cases = [("--limit", "0"), ("--warmup", "-1"), ("--seed", "-1"),
                 ("--seed", str(2**32 - 1))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self.create_input(root)
            for extra in cases:
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    entry.run_experiment(self.arguments(root, inputs, *extra), backend_factory=FakeBackend)
            self.assertEqual(FakeBackend.instances, [])


if __name__ == "__main__":
    unittest.main()
