"""run.py Lat/Mem integration; CPU fixtures, no checkpoint downloads."""

import contextlib
import csv
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import run as cli
from llava_pruning.runner import run


PROJECT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


class RunBenchmarkTests(unittest.TestCase):
    def fixture(self, root, count=4, rates=(0,), method="fastv", warmup=10):
        Image.new("RGB", (8, 8)).save(root / "image.png")
        source = root / "input.jsonl"
        source.write_text("".join(json.dumps({
            "question_id": str(index), "image": "image.png", "gt": index % 2,
            "origin_path": "screw/test/example.png",
        }) + "\n" for index in range(count)), encoding="utf-8")
        config = json.loads((PROJECT / "configs" / f"{method}.json").read_text(encoding="utf-8"))
        config.update(prune_rates=list(rates), visualize_rates=[])
        config_path = root / "method.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return dict(model_path=root, input_json=source, data_root=root, prompt_version="v0",
                    method_name=method, method_config=config_path, roi_mode="anyres_only",
                    save_prune_vis=False, save_attention_vis=False, seed=42,
                    output_dir=root / "out", benchmark_metrics=True, warmup=warmup)

    @contextlib.contextmanager
    def fake_gpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=1), \
                patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace(total_memory=10 * GIB)), \
                patch("benchmark_fastv.validate_single_gpu") as validate:
            yield validate

    def backend(self, calls, warmups, fail_on=None):
        class Backend:
            def __init__(self, model_path, roi_mode, image_token_order):
                self.model = object()
                self.counts = {}
                self.inference_config = {"attention": "fixture", "model_dtype": "unchanged"}

            def generate(self, sample, prompt, method, rate, mode, **kwargs):
                self.counts[rate] = self.counts.get(rate, 0) + 1
                ordinal = self.counts[rate]
                calls.append((sample.sample_id, rate, mode, kwargs))
                if fail_on == ordinal:
                    raise RuntimeError("synthetic failure")
                is_warmup = kwargs.get("benchmark_metrics", False) and ordinal <= warmups
                index = int(sample.sample_id)
                # Deliberately make warmup slower/bigger so accidental inclusion fails tests.
                result = dict(answer="A" if sample.gt else "B", generation_seconds=9 if is_warmup else .5,
                              max_new_tokens=512, stats={})
                if kwargs.get("benchmark_metrics", False):
                    result.update(lat_seconds=8 if is_warmup else .1 + .1 * (index % 2),
                                  generated_tokens=99 if is_warmup else 3,
                                  peak_allocated_bytes=(8 if is_warmup else 1 + rate / 100) * GIB,
                                  peak_reserved_bytes=(9 if is_warmup else 2 + rate / 100) * GIB)
                return result
        return Backend

    def test_user_cli_measures_all_samples_and_excludes_warmups(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            options = self.fixture(root, count=104)
            arguments = ["run.py", "--model-path", str(root), "--input-json", str(options["input_json"]),
                         "--data-root", str(root), "--prompt-version", "v0", "--roi-mode", "anyres_only",
                         "--method", "fastv", "--method-config", str(options["method_config"]),
                         "--no-sample", "--no-plots", "--benchmark-metrics", "--warmup", "10",
                         "--seed", "42", "--output-dir", str(options["output_dir"])]
            with self.fake_gpu() as validate, patch("sys.argv", arguments), \
                    patch("llava_pruning.runner.LlavaBackend", self.backend(calls, 10)), \
                    patch.object(cli, "check_plot_dependencies") as check_plots, \
                    patch.object(cli, "save_metric_plot") as plot, contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            check_plots.assert_not_called()
            plot.assert_not_called()
            validate.assert_called_once()
            self.assertEqual(len(calls), 114)
            self.assertTrue(all(mode == "anyres_only" for _, _, mode, _ in calls))
            for sample_id, _, _, kwargs in calls:
                self.assertTrue(kwargs["benchmark_metrics"])
                self.assertFalse(kwargs["do_sample"])
                self.assertFalse(kwargs["capture_visualization"])
                self.assertFalse(kwargs["capture_attention"])
                self.assertTrue(kwargs["include_pruning_time"])
                self.assertEqual(kwargs["random_seed"], 42 + int(sample_id))
            output = options["output_dir"]
            records = [json.loads(line) for line in (output / "prune_00/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["question_id"] for r in records], list(map(str, range(104))))
            self.assertEqual(set(records[0]) - {"question_id", "image", "origin_path", "gt", "answer",
                             "prune_rate", "method", "generation_seconds", "max_new_tokens"},
                             {"lat_seconds", "generated_tokens", "peak_allocated_bytes", "peak_reserved_bytes"})
            metrics = json.loads((output / "prune_00/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["evaluated_samples"], 104)
            self.assertEqual(metrics["warmup_samples"], 10)
            self.assertEqual(metrics["accuracy"], 1)
            self.assertAlmostEqual(metrics["lat_mean_ms"], 150)
            self.assertEqual(metrics["mem_peak_allocated_gib"], 1)
            self.assertEqual(metrics["mem_peak_reserved_gib"], 2)
            self.assertEqual(metrics["mem_peak_allocated_pct"], 10)
            self.assertEqual(metrics["generation_mean_ms"], 500)
            self.assertEqual(metrics["mean_generated_tokens"], 3)
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(float(summary[0]["lat_mean_ms"]), 150)
            self.assertEqual(float(summary[0]["mem_peak_allocated_gib"]), 1)
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["benchmark_metrics"])
            self.assertEqual(metadata["warmup_requested"], 10)
            self.assertIn("FULL answer", metadata["benchmark_definitions"]["memory"])
            self.assertFalse((output / "prune_00/visualizations").exists())

    def test_each_rate_has_its_own_warmup_and_summary_for_both_methods(self):
        for method in ("fastv", "vico"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root, calls = Path(directory), []
                options = self.fixture(root, count=2, rates=(0, 50), method=method)
                with self.fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(calls, 2)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    output = run(**options)
                self.assertEqual([rate for _, rate, _, _ in calls], [0] * 4 + [50] * 4)
                for rate in (0, 50):
                    metrics = json.loads((output / f"prune_{rate:02d}/metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(metrics["warmup_samples"], 2)
                    self.assertEqual(metrics["evaluated_samples"], 2)
                    self.assertEqual(metrics["mem_peak_allocated_gib"], 1 + rate / 100)

    def test_zero_warmup_and_legacy_path(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root, calls = Path(directory), []
                options = self.fixture(root, count=2, warmup=0)
                options.update(benchmark_metrics=enabled, roi_mode="base_only")
                with self.fake_gpu() as validate, \
                        patch("llava_pruning.runner.LlavaBackend", self.backend(calls, 0)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    output = run(**options)
                self.assertEqual(len(calls), 2)
                self.assertEqual(validate.call_count, int(enabled))
                metrics = json.loads((output / "prune_00/metrics.json").read_text(encoding="utf-8"))
                self.assertEqual("lat_mean_ms" in metrics, enabled)
                if enabled:
                    self.assertEqual(metrics["warmup_samples"], 0)
                else:
                    self.assertTrue(all("benchmark_metrics" not in kwargs for *_, kwargs in calls))
                    self.assertNotIn("mem_peak_allocated_gib", metrics)

    def test_invalid_combinations_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            options = self.fixture(Path(directory))
            for changes in ({"no_sample": False}, {"include_pruning_time": False},
                            {"save_prune_vis": True}, {"save_attention_vis": True},
                            {"warmup": -1}, {"warmup": True}, {"benchmark_metrics": "yes"}):
                with self.subTest(changes=changes), patch("llava_pruning.runner.LlavaBackend") as loader:
                    with self.assertRaises(ValueError):
                        run(**dict(options, **changes))
                    loader.assert_not_called()
            for available, count in ((False, 0), (True, 2)):
                with patch("torch.cuda.is_available", return_value=available), \
                        patch("torch.cuda.device_count", return_value=count), \
                        patch("llava_pruning.runner.LlavaBackend") as loader:
                    with self.assertRaisesRegex(RuntimeError, "one visible CUDA GPU"):
                        run(**options)
                    loader.assert_not_called()

    def test_offload_validation_rejects_before_outputs_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            options, calls = self.fixture(Path(directory)), []
            with self.fake_gpu() as validate, patch("llava_pruning.runner.LlavaBackend", self.backend(calls, 4)):
                validate.side_effect = RuntimeError("CPU/disk offloading")
                with self.assertRaisesRegex(RuntimeError, "offloading"):
                    run(**options)
            self.assertFalse(options["output_dir"].exists())
            self.assertFalse(calls)

    def test_partial_metrics_stay_incomplete_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            options, calls = self.fixture(Path(directory), warmup=1), []
            with self.fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(calls, 1, fail_on=3)), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                run(**options)
            metrics = json.loads((options["output_dir"] / "prune_00/metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["complete"])
            self.assertEqual(metrics["evaluated_samples"], 1)
            self.assertEqual(metrics["warmup_samples"], 1)
            self.assertEqual(metrics["lat_mean_ms"], 100)


class AnyresOnlyGenerationTests(unittest.TestCase):
    def test_first_token_observer_preserves_real_anyres_only_packing_and_answer(self):
        import torch
        from test_fastv_attention import tiny_config
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava_pruning.backend import configure_image_mode
        from llava_pruning.benchmark_timing import generate_with_benchmark

        torch.manual_seed(19)
        config = tiny_config("sdpa")  # CPU fixture only; production remains FP16/FlashAttention2.
        config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
        config.mm_patch_merge_type = "spatial_unpad"
        config.image_grid_pinpoints = [[8, 8], [16, 8]]
        configure_image_mode(config, "anyres_only")
        model = LlavaQwenForCausalLM(config).eval()
        model.get_model().configure_fastv(enabled=False)
        model.get_model().image_newline = torch.nn.Parameter(torch.zeros(config.hidden_size))
        tower = types.SimpleNamespace(num_patches_per_side=2, image_size=8)
        features = torch.randn(2, 4, config.hidden_size)  # two tiles, NO Base
        options = dict(inputs=torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]]),
                       images=torch.zeros(1, 2, 3, 8, 8), image_sizes=[(16, 8)],
                       do_sample=False, max_new_tokens=3, use_cache=True)
        with patch.object(model, "get_vision_tower", return_value=tower), \
                patch.object(model, "encode_images", return_value=features) as encoder, \
                patch("llava.model.language_model.llava_qwen.last_prompt_attention",
                      side_effect=AssertionError("0% must not compute pruning attention")):
            with torch.inference_mode():
                expected = model.generate(**options)
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats") as reset, \
                    patch("torch.cuda.max_memory_allocated", return_value=GIB), \
                    patch("torch.cuda.max_memory_reserved", return_value=2 * GIB), \
                    patch("llava_pruning.benchmark_timing.time.perf_counter", side_effect=[10, 12, 15]):
                actual, timing = generate_with_benchmark(model, options)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(actual.shape, (1, 3))
        self.assertEqual(encoder.call_count, 2)
        self.assertTrue(all(call.args[0].shape[0] == 2 for call in encoder.call_args_list))
        self.assertEqual(timing["lat_seconds"], 2)
        self.assertEqual(timing["generation_seconds"], 5)
        self.assertEqual(timing["generated_tokens"], 3)
        self.assertEqual(timing["peak_allocated_bytes"], GIB)
        self.assertEqual(options["max_new_tokens"], 3)
        self.assertEqual(model.config.image_aspect_ratio, "anyres_only")
        reset.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
