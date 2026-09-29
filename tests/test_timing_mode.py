"""CPU-only checks for diagnostic pruning-time subtraction and its CLI."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

import run as cli
from llava_pruning.backend import _generate_with_timing
from llava_pruning.runner import run
from test_runner import FakeBackend


TIMER_MODULE = "llava.model.language_model.pruning_timing"
BASIC_ARGUMENTS = ["--model-path", "checkpoint", "--input-json", "input.jsonl",
                   "--data-root", "dataset"]


class TimingCliTests(unittest.TestCase):
    def test_include_is_default_and_flags_select_mode(self):
        for flags, expected in (([], True), (["--include-pruning-time"], True),
                                (["--exclude-pruning-time"], False)):
            with self.subTest(flags=flags):
                args = cli.build_parser().parse_args(BASIC_ARGUMENTS + flags)
                self.assertIs(args.include_pruning_time, expected)

    def test_timing_flags_are_mutually_exclusive(self):
        for flags in (("--include-pruning-time", "--exclude-pruning-time"),
                      ("--exclude-pruning-time", "--include-pruning-time")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.build_parser().parse_args(BASIC_ARGUMENTS + list(flags))
                self.assertEqual(error.exception.code, 2)

    def test_main_forwards_each_timing_mode_to_runner(self):
        output = Path("mock-output")
        for flags, expected in (([], True), (["--include-pruning-time"], True),
                                (["--exclude-pruning-time"], False)):
            with self.subTest(flags=flags), \
                 patch.object(sys, "argv", ["run.py"] + BASIC_ARGUMENTS + flags), \
                 patch.object(cli, "check_plot_dependencies") as dependencies, \
                 patch.object(cli, "run", return_value=output) as execute, \
                 patch.object(cli, "save_metric_plot", return_value=output / "plot.png") as plot, \
                 contextlib.redirect_stdout(io.StringIO()):
                cli.main()
                dependencies.assert_called_once_with()
                self.assertIs(execute.call_args.kwargs["include_pruning_time"], expected)
                self.assertTrue(execute.call_args.kwargs["no_sample"])
                plot.assert_called_once_with(output / "summary.csv")


class GenerationTimingTests(unittest.TestCase):
    def setUp(self):
        self.previous_timer = object()
        self.core = types.SimpleNamespace(_pruning_timer=self.previous_timer)
        self.generated = object()
        self.model = types.SimpleNamespace(
            get_model=Mock(return_value=self.core),
            generate=Mock(return_value=self.generated),
        )
        self.options = {"inputs": object(), "use_cache": True, "do_sample": False}
        self.timer_factory = Mock(return_value=types.SimpleNamespace(seconds=0.375))
        fake_module = types.ModuleType(TIMER_MODULE)
        fake_module.PruningTimer = self.timer_factory
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, {TIMER_MODULE: fake_module}))
        self.synchronize = stack.enter_context(patch("torch.cuda.synchronize"))
        self.clock = stack.enter_context(patch("llava_pruning.backend.time.perf_counter",
                                               side_effect=[10.0, 12.0]))

    def test_default_measures_synchronized_generate_without_inner_timer(self):
        events = []
        readings = iter((10.0, 12.0))
        self.synchronize.side_effect = lambda: events.append("sync")

        def read_clock():
            events.append("clock")
            return next(readings)

        def generate(**options):
            events.append("generate")
            self.assertIsNone(self.core._pruning_timer)
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(options, self.options)
            return self.generated

        self.clock.side_effect = read_clock
        self.model.generate.side_effect = generate
        generated, timing = _generate_with_timing(self.model, self.options)
        self.assertIs(generated, self.generated)
        self.assertEqual(timing, {"generation_seconds": 2.0})
        self.assertEqual(events, ["sync", "clock", "generate", "sync", "clock"])
        self.timer_factory.assert_not_called()
        self.assertIs(self.core._pruning_timer, self.previous_timer)

    def test_exclude_attaches_timer_and_reports_elapsed_minus_pruning(self):
        timer = self.timer_factory.return_value

        def generate(**options):
            self.assertIs(self.core._pruning_timer, timer)
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(options, self.options)
            return self.generated

        self.model.generate.side_effect = generate
        generated, timing = _generate_with_timing(
            self.model, self.options, include_pruning_time=False)
        self.assertIs(generated, self.generated)
        self.assertEqual(timing, {
            "generation_seconds": 1.625,
            "generation_with_pruning_seconds": 2.0,
            "pruning_seconds": 0.375,
        })
        self.timer_factory.assert_called_once_with()
        self.assertEqual(self.synchronize.call_count, 2)
        self.model.generate.assert_called_once_with(**self.options)
        self.assertIs(self.core._pruning_timer, self.previous_timer)

    def test_failed_generate_restores_previous_timer_in_both_modes(self):
        for include in (True, False):
            with self.subTest(include=include):
                self.clock.side_effect = [10.0]
                failure = RuntimeError("generation failed")

                def generate(**options):
                    self.assertIs(self.core._pruning_timer,
                                  None if include else self.timer_factory.return_value)
                    raise failure

                self.model.generate.side_effect = generate
                with self.assertRaises(RuntimeError) as error:
                    _generate_with_timing(self.model, self.options,
                                          include_pruning_time=include)
                self.assertIs(error.exception, failure)
                self.assertIs(self.core._pruning_timer, self.previous_timer)

    def test_failed_synchronization_restores_previous_timer(self):
        for sync_results in ([RuntimeError("start sync failed")],
                             [None, RuntimeError("end sync failed")]):
            with self.subTest(sync_results=sync_results):
                self.synchronize.side_effect = sync_results
                self.clock.side_effect = [10.0]
                with self.assertRaisesRegex(RuntimeError, "sync failed"):
                    _generate_with_timing(self.model, self.options,
                                          include_pruning_time=False)
                self.assertIs(self.core._pruning_timer, self.previous_timer)

    def test_each_generate_gets_fresh_timer_and_default_disables_it(self):
        timers = [types.SimpleNamespace(seconds=0.5), types.SimpleNamespace(seconds=0.25)]
        self.timer_factory.side_effect = timers
        self.clock.side_effect = [10.0, 12.0, 20.0, 22.0, 30.0, 32.0]
        seen = []

        def generate(**options):
            seen.append(self.core._pruning_timer)
            return self.generated

        self.model.generate.side_effect = generate
        timings = []
        for include in (False, False, True):
            _, timing = _generate_with_timing(self.model, self.options,
                                              include_pruning_time=include)
            timings.append(timing)
            self.assertIs(self.core._pruning_timer, self.previous_timer)
        self.assertEqual(seen, timers + [None])
        self.assertEqual(self.timer_factory.call_count, 2)
        self.assertEqual([item["generation_seconds"] for item in timings], [1.5, 1.75, 2.0])
        self.assertEqual(timings[-1], {"generation_seconds": 2.0})

    def test_core_without_previous_timer_is_left_inactive(self):
        del self.core._pruning_timer
        _generate_with_timing(self.model, self.options, include_pruning_time=False)
        self.assertIsNone(getattr(self.core, "_pruning_timer", None))

    def test_invalid_outer_duration_fails_and_restores_timer(self):
        for elapsed in (-0.5, float("nan"), float("inf")):
            with self.subTest(elapsed=elapsed):
                self.clock.side_effect = [0.0, elapsed]
                with self.assertRaisesRegex(RuntimeError, "Invalid synchronized generation duration"):
                    _generate_with_timing(self.model, self.options,
                                          include_pruning_time=False)
                self.assertIs(self.core._pruning_timer, self.previous_timer)

    def test_invalid_pruning_duration_fails_without_clamping(self):
        for pruning in (-0.1, float("nan"), float("inf"), 2.1):
            with self.subTest(pruning=pruning):
                self.clock.side_effect = [10.0, 12.0]
                self.timer_factory.return_value.seconds = pruning
                with self.assertRaisesRegex(RuntimeError, "pruning duration is inconsistent"):
                    _generate_with_timing(self.model, self.options,
                                          include_pruning_time=False)
                self.assertIs(self.core._pruning_timer, self.previous_timer)

    def test_zero_and_full_pruning_duration_are_valid_boundaries(self):
        for pruning, elapsed in ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0)):
            with self.subTest(pruning=pruning, elapsed=elapsed):
                self.clock.side_effect = [0.0, elapsed]
                self.timer_factory.return_value.seconds = pruning
                _, timing = _generate_with_timing(self.model, self.options,
                                                  include_pruning_time=False)
                self.assertEqual(timing["generation_seconds"], elapsed - pruning)
                self.assertEqual(timing["generation_with_pruning_seconds"], elapsed)
                self.assertEqual(timing["pruning_seconds"], pruning)

    def test_nonboolean_mode_fails_before_generation(self):
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    _generate_with_timing(self.model, self.options,
                                          include_pruning_time=value)
        self.model.generate.assert_not_called()
        self.synchronize.assert_not_called()
        self.timer_factory.assert_not_called()
        self.assertIs(self.core._pruning_timer, self.previous_timer)

    def test_exclude_rejects_sharded_decoder_before_starting_but_default_is_unchanged(self):
        self.core.layers = [
            types.SimpleNamespace(parameters=lambda: [types.SimpleNamespace(device=torch.device("cuda:0"))]),
            types.SimpleNamespace(parameters=lambda: [types.SimpleNamespace(device=torch.device("cuda:1"))]),
        ]
        with self.assertRaisesRegex(ValueError, "decoder weights on one device"):
            _generate_with_timing(self.model, self.options, include_pruning_time=False)
        self.synchronize.assert_not_called()
        self.model.generate.assert_not_called()
        self.timer_factory.assert_not_called()
        self.assertIs(self.core._pruning_timer, self.previous_timer)
        _, timing = _generate_with_timing(self.model, self.options)
        self.assertEqual(timing, {"generation_seconds": 2.0})


class TimingRunnerTests(unittest.TestCase):
    def test_modes_preserve_answers_metrics_csv_and_only_extend_timing_fields(self):
        calls = []

        class TrackingBackend(FakeBackend):
            def generate(self, *args, **kwargs):
                calls.append(kwargs["include_pruning_time"])
                result = super().generate(*args, **kwargs)
                # A deterministic layer fixture also exercises the CSV writer.
                rate = args[3]
                result["stats"]["layers"] = [{
                    "layer": 2, "image_tokens_in": 100, "image_tokens_out": 100 - rate,
                    "sequence_tokens_in": 102, "sequence_tokens_out": 102 - rate,
                    "removed_after_layer": rate, "cumulative_prune_rate": rate,
                }]
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "image.png")
            source = root / "questions.jsonl"
            source.write_text(json.dumps({
                "question_id": "one", "image": "image.png",
                "origin_path": "screw/test/bad.png", "gt": 1,
            }) + "\n", encoding="utf-8")
            config = root / "fastv.json"
            config.write_text(json.dumps({
                "layer": 2, "prune_rates": [0, 10], "visualize_rates": [],
            }), encoding="utf-8")
            outputs = []
            for name, timing_options, expected_mode in (
                    ("default", {}, True),
                    ("include", {"include_pruning_time": True}, True),
                    ("exclude", {"include_pruning_time": False}, False)):
                output = root / name
                with patch("llava_pruning.runner.LlavaBackend", TrackingBackend), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run(model_path=root, input_json=source, data_root=root,
                        prompt_version="v0", method_name="fastv", method_config=config,
                        roi_mode="anyres_max_9", save_prune_vis=False,
                        save_attention_vis=False, output_dir=output, seed=42,
                        **timing_options)
                metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
                self.assertIs(metadata["include_pruning_time"], expected_mode)
                self.assertFalse(metadata["do_sample"])
                self.assertEqual(metadata["seed"], 42)
                outputs.append(output)
            self.assertEqual(calls, [True, True, True, True, False, False])
            for rate in (0, 10):
                folders = [output / f"prune_{rate:02d}" for output in outputs]
                rows = [json.loads((folder / "predictions.jsonl").read_text(encoding="utf-8"))
                        for folder in folders]
                expected = {
                    "question_id": "one", "image": str((root / "image.png").resolve()),
                    "origin_path": "screw/test/bad.png", "gt": 1,
                    "answer": "A", "prune_rate": rate, "method": "fastv",
                    "generation_seconds": 0.1, "max_new_tokens": 192,
                }
                self.assertEqual(rows[0], expected)
                self.assertEqual(rows[1], expected)
                pruning = 0.0 if rate == 0 else 0.02
                self.assertEqual(rows[2], {
                    **expected, "generation_seconds": 0.1 - pruning,
                    "generation_with_pruning_seconds": 0.1, "pruning_seconds": pruning,
                })
                self.assertEqual([len(row) for row in rows], [9, 9, 11])
                for filename in ("metrics.json", "layer_tokens.csv"):
                    contents = [(folder / filename).read_text(encoding="utf-8") for folder in folders]
                    self.assertEqual(contents, [contents[0]] * 3)
                metrics = json.loads((folders[0] / "metrics.json").read_text(encoding="utf-8"))
                self.assertTrue(metrics["complete"])
                self.assertEqual(metrics["accuracy"], 1.0)
            summaries = [(output / "summary.csv").read_text(encoding="utf-8") for output in outputs]
            self.assertEqual(summaries, [summaries[0]] * 3)

    def test_nonboolean_mode_fails_before_loading_samples_or_model(self):
        with patch("llava_pruning.runner.load_samples") as samples, \
             patch("llava_pruning.runner.LlavaBackend") as backend:
            for value in (None, 0, 1, "false"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                    run(model_path="checkpoint", input_json="input.jsonl", data_root="data",
                        prompt_version="v0", method_name="fastv", method_config=None,
                        roi_mode="randomroi", save_prune_vis=False, save_attention_vis=False,
                        output_dir="unused", include_pruning_time=value)
            samples.assert_not_called()
            backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
