"""Base untouched; mapped Base attention ranks all AnyRes; clean Lat/Mem."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from test_clse import layout, toy_model
from test_attention_predictor import make_checkpoint
import test_run_benchmark_metrics as common
from llava.model.language_model.attention_predictor_selection import (
    base_full_count_budget, base_full_budget, base_guided_log_scores,
    select_base_full_tokens, native_anyres_counts)
from llava_pruning.runner import run
from llava_pruning.attention_predictor import AttentionPredictorMethod


class BaseFullSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_base_and_newlines_survive_with_original_total(self):
        geo = layout()
        mapped = base_guided_log_scores(torch.tensor([.1, .2, .3, .4]).log(), geo)
        indices = [i for grid in geo["grids"] for i in grid["indices"]]
        selected, plan = select_base_full_tokens(mapped, indices, 14, .5, geo)
        self.assertEqual(plan["target"], 7)
        self.assertEqual([g["kept_tokens"] for g in plan["groups"]], [4, 1])
        self.assertEqual(selected.tolist(), [0, 1, 2, 3, 8, 12, 13])
        self.assertEqual(plan["groups"][0]["actual_prune_rate"], 0)

    def test_exact_feasible_limit_allows_zero_anyres_but_never_discards_base(self):
        plan = base_full_count_budget(4, 8, 2, 6 / 14)
        self.assertEqual(plan["groups"][1]["kept_tokens"], 0)
        for ratio in (.4, .01):
            with self.assertRaisesRegex(ValueError, "cannot keep all"):
                base_full_count_budget(4, 8, 2, ratio)

    def test_30_50_70_remain_whole_image_layer_average_not_anyres_only(self):
        geo = layout(base=27, rows=81, cols=81)
        for rate in (30, 50, 70):
            keep_ratio = 1 - rate * 28 / 26 / 100
            plan = base_full_budget(geo, keep_ratio)
            target = math.floor(geo["packed_tokens"] * keep_ratio + 1e-10)
            self.assertEqual(plan["target"], target)
            self.assertEqual(plan["groups"][0]["kept_tokens"], 729)
            self.assertEqual(plan["groups"][1]["kept_tokens"], target - 729 - 81)
            actual = 100 * 26 / 28 * (1 - target / geo["packed_tokens"])
            self.assertLess(abs(actual - rate), 100 / geo["packed_tokens"])
        with self.assertRaises(ValueError):
            base_full_budget(geo, 1 - 90 * 28 / 26 / 100)

    def test_count_preflight_matches_native_unpad_max9_and_newlines(self):
        from llava.model.llava_arch import unpad_image
        for size, grid in (((384, 384), (1, 1)), ((1000, 600), (3, 2)),
                           ((600, 1000), (2, 3)), ((2000, 2000), (6, 6))):
            feature = unpad_image(torch.zeros(1, grid[1] * 27, grid[0] * 27), size)
            h, w = feature.shape[-2:]
            times = math.sqrt(h * w / (9 * 27 ** 2))
            if times > 1.1:
                feature = torch.nn.functional.interpolate(feature[None], [int(h // times), int(w // times)], mode="bilinear")[0]
            h, w = feature.shape[-2:]
            for final in (False, True):
                self.assertEqual(native_anyres_counts(27, grid, size, final_newline=final), (729, h * w, h + int(final)))

    def test_base_only_scores_are_requested_and_own_anyres_scores_cannot_change_rank(self):
        model, _, _ = toy_model()
        core, geo = model.get_model(), layout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            make_checkpoint(path, core)
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
            core.set_clse_image_layout([[(2, 16)]], 19, geo)
            h2, ids = torch.randn(1, 19, 32), torch.arange(19)[None]
            native_scores = torch.arange(19).float()
            def score(hidden, positions, indices):
                self.assertIs(hidden, h2)
                self.assertEqual(indices, [2, 3, 4, 5])
                return native_scores[indices]
            with patch.object(core.attention_predictor, "forward", side_effect=score) as predictor:
                before = core._clse_rank(h2, 2, 16, ids, None)
                native_scores[6:] = 10000
                after = core._clse_rank(h2, 2, 16, ids, None)
            self.assertEqual(predictor.call_count, 2)
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_zero_native_identity_generate_orders_cache_and_first_token_measurement(self):
        from llava_pruning.benchmark_timing import generate_with_benchmark
        for order in ("base_first", "anyres_first"):
            for extra in (False, True):
                model, features, options = toy_model(order=order, extra=extra)
                core = model.get_model()
                with tempfile.TemporaryDirectory() as directory, patch.object(model, "encode_images", return_value=features), torch.no_grad():
                    path = Path(directory) / "best.pt"
                    make_checkpoint(path, core)
                    baseline = model.generate(**options)
                    core.configure_attention_predictor(enabled=False, keep_ratio=1)
                    self.assertTrue(torch.equal(baseline, model.generate(**options)))
                    core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path,
                                                       capture_attention=True, capture_visualization=True)
                    output = model.generate(**options)
                    stats, stage = core.get_clse_stats(), core.get_clse_stages()[0]
                    self.assertEqual(stats["selection_mode"], "base_full")
                    self.assertNotIn("crop_budgets", stats)
                    self.assertEqual(stage["attention"]["score_kind"], "attention_predictor_base_full")
                    base = next(g for g in core._clse_layout["grids"] if g["name"] == "base")
                    self.assertTrue(all(stage["mask"]["keep"][i] for i in base["indices"]))
                    self.assertEqual(stats["selection_groups"][0]["kept_tokens"], 4)
                    core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
                    with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                            patch("torch.cuda.max_memory_allocated", return_value=123), \
                            patch("torch.cuda.max_memory_reserved", return_value=456):
                        measured, timing = generate_with_benchmark(model, options)
                    self.assertTrue(torch.equal(output, measured))
                    self.assertEqual(timing["generated_tokens"], 4)
                    self.assertLessEqual(timing["lat_seconds"], timing["generation_seconds"])
                    self.assertEqual(timing["peak_allocated_bytes"], 123)
                    self.assertEqual(timing["peak_reserved_bytes"], 456)
                    self.assertEqual(core.config._attn_implementation, "sdpa")


class BaseFullRunnerTests(unittest.TestCase):
    def fixture(self, root, *, visual=False):
        source = root / "questions.jsonl"
        Image.new("RGB", (8, 8)).save(root / "image.png")
        source.write_text("".join(json.dumps(dict(question_id=str(i), image="image.png", gt=i % 2,
                                                origin_path="screw/test/good/example.png")) + "\n" for i in range(3)), encoding="utf-8")
        (root / "best.pt").write_bytes(b"mocked backend only")
        config = root / "method.json"
        config.write_text(json.dumps(dict(checkpoint="best.pt", selection_mode="base_full",
                                          prune_rates=[30, 50, 70], visualize_rates=[30, 50, 70])), encoding="utf-8")
        return dict(model_path=root, input_json=source, data_root=root, prompt_version="v0",
                    method_name="attention_predictor", method_config=config, roi_mode="anyres_max_9",
                    save_prune_vis=visual, save_attention_vis=visual, benchmark_metrics=True, warmup=1,
                    seed=42, output_dir=root / "output")

    def backend(self, calls, *, grid=(24, 24), mismatched_vis=False):
        class Backend:
            def __init__(self, *args, **kwargs):
                self.core = types.SimpleNamespace(layers=[None] * 28, config=types.SimpleNamespace(
                    mm_patch_merge_type="spatial_unpad", image_grid_pinpoints=[list(grid)]))
                self.model = types.SimpleNamespace(get_model=lambda: self.core)
                self.vision_tower = types.SimpleNamespace(num_patches_per_side=2, image_size=8)

            def generate(self, sample, prompt, method, rate, mode, **kwargs):
                calls.append((sample.sample_id, rate, dict(kwargs)))
                counts = native_anyres_counts(2, (grid[0] // 8, grid[1] // 8), (8, 8))
                before = sum(counts)
                budget = method.resolve_budget(rate, 28)
                ratio = 1 - budget["requested_layer_prune_rate"] / 100
                plan = base_full_count_budget(*counts, ratio)
                after = plan["target"]
                rows = [dict(layer=i, image_tokens_in=before if i <= 2 else after,
                             image_tokens_out=before if i < 2 else after,
                             sequence_tokens_in=before + 5 if i <= 2 else after + 5,
                             sequence_tokens_out=before + 5 if i < 2 else after + 5,
                             removed_after_layer=before - after if i == 2 else 0,
                             cumulative_prune_rate=0 if i < 2 else 100 * (1 - after / before)) for i in range(1, 29)]
                self.core.get_clse_stats = lambda: dict(original_image_tokens=before, final_llm_visual_tokens=after,
                                                        layers=rows, selection_groups=plan["groups"],
                                                        common_spatial_keep_ratio=None, protected_newline_tokens=counts[2])
                result = dict(answer="A" if sample.gt else "B", generation_seconds=.5,
                              stats=method.stats(self.core, rate), max_new_tokens=512)
                if kwargs.get("benchmark_metrics"):
                    assert not kwargs["capture_visualization"] and not kwargs["capture_attention"]
                    result.update(lat_seconds=.125, generated_tokens=3,
                                  peak_allocated_bytes=2 * 1024 ** 3, peak_reserved_bytes=3 * 1024 ** 3)
                if kwargs["capture_visualization"]:
                    result["generation_seconds"] = 999  # Never contaminate recorded latency/time.
                    if mismatched_vis:
                        result["answer"] = "MISMATCH"
                return result
        return Backend

    def test_results_include_lat_mem_and_base_full_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            options = self.fixture(root)
            with common.RunBenchmarkTests().fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(calls)), contextlib.redirect_stdout(io.StringIO()):
                output = run(**options)
            self.assertEqual(len(calls), 12)  # 3*(one warmup + ALL three samples)
            with (output / "summary.csv").open(encoding="utf-8", newline="") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual([r["prune_rate"] for r in summary], ["30", "50", "70"])
            for rate, row in zip((30, 50, 70), summary):
                self.assertEqual(float(row["lat_mean_ms"]), 125)
                self.assertEqual(float(row["mem_peak_allocated_gib"]), 2)
                self.assertEqual(float(row["generation_mean_ms"]), 500)
                self.assertEqual(row["evaluated_samples"], "3")
                with (output / f"prune_{rate}/view_budgets.csv").open(encoding="utf-8", newline="") as stream:
                    groups = list(csv.DictReader(stream))
                for group in groups[::2]:
                    self.assertEqual(group["candidate_tokens"], group["kept_tokens"])
                    self.assertEqual(float(group["actual_prune_rate"]), 0)
                self.assertFalse((output / f"prune_{rate}/crop_budgets.csv").exists())
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["budget_preflight"]["checked_samples"], 3)

    def test_visualization_calls_are_after_all_measurements_and_do_not_change_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls, rendered = Path(directory), [], []
            options = self.fixture(root, visual=True)
            options["visualize_samples"] = ["0"]
            def render(**kwargs):
                rendered.append(kwargs["sample_id"])
                folder = kwargs["output_dir"] / ("sample_" + kwargs["sample_id"])
                folder.mkdir(parents=True)
                Image.new("RGB", (2, 2)).save(folder / "comparison.png")
            with common.RunBenchmarkTests().fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(calls)), \
                    patch.object(AttentionPredictorMethod, "visualize", side_effect=render), contextlib.redirect_stdout(io.StringIO()):
                output = run(**options)
            self.assertEqual(len(calls), 15)
            self.assertTrue(all(c[2].get("benchmark_metrics") for c in calls[:12]))
            self.assertTrue(all(not c[2].get("benchmark_metrics", False) and c[2]["capture_attention"] for c in calls[12:]))
            self.assertEqual(rendered, ["0"] * 3)
            self.assertEqual(len(list(output.rglob("comparison.png"))), 3)
            for rate in (30, 50, 70):
                metrics = json.loads((output / f"prune_{rate}/metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["lat_mean_ms"], 125)
                self.assertEqual(metrics["generation_mean_ms"], 500)
                self.assertEqual(metrics["mem_peak_allocated_gib"], 2)
            self.assertTrue((output / "visualization_verification.csv").is_file())

    def test_infeasible_image_fails_preflight_without_warmup_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            options = self.fixture(root)
            with common.RunBenchmarkTests().fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(calls, grid=(8, 8))), \
                    self.assertRaisesRegex(ValueError, "preflight failed"):
                run(**options)
            self.assertFalse(calls)
            self.assertFalse(options["output_dir"].exists())

    def test_mismatched_visualization_answer_does_not_replace_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            options = self.fixture(root, visual=True)
            options["visualize_samples"] = ["0"]
            with common.RunBenchmarkTests().fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(calls, mismatched_vis=True)), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "answer differs"):
                run(**options)
            metrics = json.loads((options["output_dir"] / "prune_70/metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["complete"])
            self.assertEqual(metrics["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
