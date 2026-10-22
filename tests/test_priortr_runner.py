"""PriorTR public CLI, shared Lat/Mem reporting and signed visualization."""

import contextlib
import csv
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import run as cli
import test_run_benchmark_metrics as common
from llava_pruning.runner import run
from llava_pruning.visualization import build_attention_overlay, prepare_image_masks
from llava_pruning.methods import load_method


class PriorTRRunnerTests(unittest.TestCase):
    def backend(self, fixture, calls, warmups):
        parent = fixture.backend(calls, warmups)
        class Backend(parent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.core = types.SimpleNamespace(layers=[None] * 28)
                self.model = types.SimpleNamespace(get_model=lambda: self.core)

            def generate(self, sample, prompt, method, rate, mode, **kwargs):
                result = super().generate(sample, prompt, method, rate, mode, **kwargs)
                budget = method.resolve_budget(rate, 28)
                count = 100 + int(sample.sample_id)
                retained = max(1, math.floor(count * (100 - budget["requested_layer_prune_rate"]) / 100))
                rows = []
                for layer in range(1, 29):
                    before = count if layer <= method.layer else retained
                    after = count if layer < method.layer else retained
                    rows.append(dict(layer=layer, image_tokens_in=before, image_tokens_out=after,
                                     sequence_tokens_in=before + 10, sequence_tokens_out=after + 10,
                                     removed_after_layer=before - after,
                                     cumulative_prune_rate=100 * (1 - after / count)))
                raw = dict(original_image_tokens=count, final_llm_visual_tokens=retained, layers=rows)
                self.core.get_priortr_stats = lambda: raw
                result["stats"] = method.stats(self.core, rate)
                return result
        return Backend

    def test_public_cli_and_all_rates_use_existing_benchmark_protocol(self):
        fixture = common.RunBenchmarkTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as temporary:
            root, calls = Path(temporary), []
            options = fixture.fixture(root, count=4, rates=(0, 50, 70, 90), method="priortr", warmup=2)
            config = json.loads(Path(options["method_config"]).read_text(encoding="utf-8"))
            config["rate_semantics"] = "final_cumulative"
            Path(options["method_config"]).write_text(json.dumps(config), encoding="utf-8")
            args = ["run.py", "--model-path", str(root), "--input-json", str(options["input_json"]),
                    "--data-root", str(root), "--method", "priortr", "--method-config", str(options["method_config"]),
                    "--roi-mode", "anyres_max_9", "--no-sample", "--no-plots", "--benchmark-metrics", "--warmup", "2",
                    "--seed", "42", "--output-dir", str(options["output_dir"])]
            with fixture.fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(fixture, calls, 2)), \
                    patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(len(calls), 24)
            self.assertTrue(all(kwargs["benchmark_metrics"] for _, _, _, kwargs in calls))
            for rate in (0, 50, 70, 90):
                metrics = json.loads((options["output_dir"] / f"prune_{rate:02d}/metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["evaluated_samples"], 4)
                self.assertEqual(metrics["warmup_samples"], 2)
                self.assertEqual(metrics["accuracy"], 1)
                self.assertAlmostEqual(metrics["lat_mean_ms"], 150)
                self.assertAlmostEqual(metrics["mem_peak_allocated_gib"], 1 + rate / 100)
            metadata = json.loads((options["output_dir"] / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["method_details"]["source"]["commit"], "e442a4c14af1b333067a6b8310c9111cad922033")
            self.assertIn("native QK", metadata["inference"]["score_dtype"])

    def test_layer2_average_targets_resolve_and_report_rounding_in_all_outputs(self):
        fixture = common.RunBenchmarkTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as temporary:
            root, calls = Path(temporary), []
            options = fixture.fixture(root, count=4, rates=(50, 70, 90), method="priortr", warmup=2)
            with fixture.fake_gpu(), patch("llava_pruning.runner.LlavaBackend", self.backend(fixture, calls, 2)), \
                    contextlib.redirect_stdout(io.StringIO()):
                output = run(**options)
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                summaries = list(csv.DictReader(stream))
            self.assertEqual([float(row["prune_rate"]) for row in summaries], [50, 70, 90])
            for target, row in zip((50, 70, 90), summaries):
                folder = output / f"prune_{target:02d}"
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                expected_actual = []
                for index, record in enumerate(records):
                    count = 100 + index
                    retained = max(1, math.floor(count * (1 - target / 100 * 28 / 26)))
                    actual = 100 * (1 - (2 * count + 26 * retained) / (28 * count))
                    expected_actual.append(actual)
                    self.assertAlmostEqual(record["actual_average_prune_rate"], actual)
                    self.assertAlmostEqual(record["requested_layer_prune_rate"], target * 28 / 26)
                    self.assertEqual(record["target_average_prune_rate"], target)
                self.assertEqual(metrics["rate_semantics"], "layer_average")
                self.assertEqual(metrics["pruning_layer"], 2)
                self.assertEqual(metrics["evaluated_samples"], 4)
                self.assertEqual(metrics["warmup_samples"], 2)
                self.assertAlmostEqual(metrics["actual_average_prune_rate"], sum(expected_actual) / 4)
                self.assertAlmostEqual(float(row["actual_average_prune_rate"]), sum(expected_actual) / 4)
                self.assertAlmostEqual(metrics["lat_mean_ms"], 150)
                with (folder / "layer_tokens.csv").open(encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 4 * 28)
            run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(run_metadata["resolved_budgets"]), 3)

    def test_average_budget_rejects_unreachable_target_at_actual_depth(self):
        method = load_method("priortr")
        self.assertEqual(method.layer, 2)
        self.assertEqual(method.rates, (50, 70, 90))
        self.assertEqual(method.rate_semantics, "layer_average")
        for target in method.rates:
            budget = method.resolve_budget(target, 28)
            self.assertAlmostEqual(budget["requested_layer_prune_rate"] * 26 / 28, target)
        with self.assertRaisesRegex(ValueError, "unreachable"):
            replace(method, layer=3).resolve_budgets(28)

    def test_signed_score_display_keeps_all_negative_scores_distinguishable(self):
        image = Image.new("RGB", (8, 8), "white")
        metadata = {"original_size": [8, 8], "roi_boxes": [], "processed_view_sizes": [[8, 8]]}
        mask = {"span": [1, 6], "keep": [True] * 5}
        prepared = prepare_image_masks(image.size, metadata, mask, 2, 4)
        attention = {"span": [1, 6], "scores": [-4, -3, -2, -1, 0], "score_kind": "priortr"}
        corrected = np.asarray(build_attention_overlay(image, attention, mask, prepared))
        ordinary = np.asarray(build_attention_overlay(image, dict(attention, score_kind="attention"), mask, prepared))
        self.assertFalse(np.array_equal(corrected, ordinary))
        self.assertGreater(len(np.unique(corrected[-8:].reshape(-1, 3), axis=0)), 2)

    def test_real_stage_visualization_outputs_two_pngs_and_scalar_counts(self):
        from test_fastv_attention import tiny_config
        from llava.model.language_model.llava_qwen import LlavaQwenModel
        from llava_pruning.methods import load_method
        method = replace(load_method("priortr"), layer=3, rate_semantics="final_cumulative")
        core = LlavaQwenModel(tiny_config("sdpa")).eval()
        method.configure(core, 70, capture_attention=True, capture_visualization=True)
        core.set_priortr_image_spans([[(1, 15)]], 18)
        with torch.no_grad():
            core(input_ids=torch.arange(1, 19)[None], use_cache=True)
        result = dict(image=Image.new("RGB", (16, 8), "white"), answer="B", stats=method.stats(core, 70),
                      stages=core.get_priortr_stages(),
                      crop_metadata=[dict(mode="anyres_max_9", original_size=[16, 8],
                                          grid_patches=[2, 1], roi_boxes=[], final_newline=False)])
        tower = types.SimpleNamespace(num_patches_per_side=2, config=types.SimpleNamespace(patch_size=4))
        with tempfile.TemporaryDirectory() as temporary:
            paths = method.visualize(result=result, vision_tower=tower, output_dir=temporary,
                                     sample_id="000000108", save_prune=True, save_attention=True, ground_truth=0)
            folder = Path(paths[0]["directory"])
            self.assertEqual({path.name for path in folder.glob("*.png")}, {"comparison.png", "attention_overlay.png"})
            saved = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["method"], "priortr")
            self.assertEqual(saved["stages"][0]["after_layer"], 3)
            self.assertNotIn("scores", saved["stages"][0])


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flash_attn"),
                     "CUDA + FlashAttention2 required; run on the inference server")
class PriorTRFlashTests(unittest.TestCase):
    def test_native_fp16_flash_zero_identity_and_physical_cached_decode(self):
        from test_fastv_attention import tiny_config
        from llava.model.language_model.llava_qwen import LlavaQwenModel
        torch.manual_seed(680)
        config = tiny_config("flash_attention_2")
        config.torch_dtype = torch.float16
        model = LlavaQwenModel(config).cuda().half().eval()
        tokens = torch.arange(1, 13, device="cuda")[None]
        with torch.no_grad():
            baseline = model(input_ids=tokens, use_cache=True)
            model.configure_priortr(enabled=False, layer=2, keep_ratio=1)
            zero = model(input_ids=tokens, use_cache=True)
            self.assertTrue(torch.equal(baseline.last_hidden_state, zero.last_hidden_state))
            model.configure_priortr(enabled=True, layer=2, keep_ratio=.4)
            model.set_priortr_image_spans([[(2, 9)]], 12)
            output = model(input_ids=tokens, use_cache=True)
            self.assertEqual(output.past_key_values.get_seq_length(), 7)
            model(input_ids=torch.tensor([[13]], device="cuda"), past_key_values=output.past_key_values, use_cache=True)
            self.assertEqual(model.dtype, torch.float16)
            self.assertEqual(model.config._attn_implementation, "flash_attention_2")


if __name__ == "__main__":
    unittest.main()
