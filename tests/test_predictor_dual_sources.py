"""Two ranking sources, identical Base-full budgets; six isolated experiments."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image

import compare_attention_sources as compare
from test_attention_predictor import make_checkpoint
from test_clse import layout, toy_model
from llava_pruning.methods import load_method


class DualSourceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_only_ranking_source_changes_while_budgets_base_and_newlines_match(self):
        model, _, _ = toy_model()
        core, geo = model.get_model(), layout()
        spatial = [i for grid in geo["grids"] for i in grid["indices"]]
        hidden, ids = torch.randn(1, 19, 32), torch.arange(19)[None]
        scores = torch.tensor([4., 3., 2., 1.] + [-10.] * 7 + [10.])
        sources, choices, plans = {}, {}, {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            make_checkpoint(path, core)
            for mode in ("base_full", "anyres_self"):
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path,
                                                   selection_mode=mode, capture_attention=True)
                core.set_clse_image_layout([[(2, 16)]], 19, geo)
                def forward(h2, positions, indices):
                    self.assertIs(h2, hidden)
                    sources[mode] = list(indices)
                    return scores[[spatial.index(index - 2) for index in indices]].log_softmax(0)
                with patch.object(core.attention_predictor, "forward", side_effect=forward):
                    ranking = core._clse_rank(hidden, 2, 16, ids, None)
                chosen, target, protected = core._clse_select(ranking, spatial, 14)
                choices[mode], plans[mode] = chosen.tolist(), list(core._clse_stats["selection_groups"])
                self.assertEqual(target, 7)
                self.assertEqual(protected, 2)
                self.assertTrue({0, 1, 2, 3, 8, 13}.issubset(set(chosen.tolist())))
                self.assertNotIn("crop_budgets", core._clse_stats)
            self.assertEqual(sources["base_full"], [2, 3, 4, 5])
            self.assertEqual(sources["anyres_self"], [i + 2 for i in spatial])
            self.assertEqual(plans["base_full"], plans["anyres_self"])
            self.assertEqual(choices["base_full"], [0, 1, 2, 3, 4, 8, 13])
            self.assertEqual(choices["anyres_self"], [0, 1, 2, 3, 8, 12, 13])

    def test_real_own_attention_generate_preserves_base_both_orders_and_zero_identity(self):
        for order in ("base_first", "anyres_first"):
            model, features, options = toy_model(order=order)
            core = model.get_model()
            with tempfile.TemporaryDirectory() as directory, patch.object(model, "encode_images", return_value=features), torch.no_grad():
                path = Path(directory) / "best.pt"
                make_checkpoint(path, core)
                baseline = model.generate(**options)
                core.configure_attention_predictor(enabled=False, keep_ratio=1, selection_mode="anyres_self")
                self.assertTrue(torch.equal(baseline, model.generate(**options)))
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path,
                                                   selection_mode="anyres_self", capture_attention=True, capture_visualization=True)
                first = model.generate(**options)
                self.assertTrue(torch.equal(first, model.generate(**options)))
                stats, stage = core.get_clse_stats(), core.get_clse_stages()[0]
                self.assertEqual(stats["selection_mode"], "anyres_self")
                self.assertEqual(stage["attention"]["score_kind"], "attention_predictor_anyres_self")
                self.assertEqual(stats["selection_groups"][0]["actual_prune_rate"], 0)
                self.assertEqual(stats["selection_groups"][0]["kept_tokens"], 4)
                self.assertEqual(stats["selection_groups"][1]["kept_tokens"], 1)
                self.assertAlmostEqual(sum(stage["attention"]["scores"]), 1, places=5)
                self.assertEqual(core.config._attn_implementation, "sdpa")

    def test_config_accepts_self_and_protects_identical_layer_average_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best.pt").write_bytes(b"config parsing only")
            for mode in ("anyres_self", "base_full"):
                path = root / f"{mode}.json"
                path.write_text(json.dumps(dict(checkpoint="best.pt", selection_mode=mode,
                                                prune_rates=[30, 50, 70], visualize_rates=[70])), encoding="utf-8")
                method = load_method("attention_predictor", path)
                self.assertEqual(method.selection_mode, mode)
                self.assertEqual(method.visualize_rates, frozenset({70}))
                self.assertEqual(method.rates, (30, 50, 70))
                self.assertAlmostEqual(method.resolve_budget(70, 28)["requested_layer_prune_rate"], 70 * 28 / 26)
                self.assertIn("Base fully retained", method.details()["budget_scope"])


class ComparisonDriverTests(unittest.TestCase):
    def fixture(self, root, *extra):
        (root / "model").mkdir()
        (root / "data").mkdir()
        (root / "input.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "best.pt").write_bytes(b"mock child checkpoint")
        return compare.build_parser().parse_args([
            "--gpu", "4", "--model-path", str(root / "model"), "--input-json", str(root / "input.jsonl"),
            "--data-root", str(root / "data"), "--checkpoint", str(root / "best.pt"),
            "--output-dir", str(root / "output"), *extra])

    def process(self, calls, *, fail=False, wrong_mode=False):
        def invoke(command, *, cwd, env, check):
            calls.append((command, cwd, env, check))
            if fail:
                return types.SimpleNamespace(returncode=2)
            config = json.loads(Path(command[command.index("--method-config") + 1]).read_text(encoding="utf-8"))
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            mode = config["selection_mode"]
            (output / "run.json").write_text(json.dumps(dict(method_details=dict(selection_mode="WRONG" if wrong_mode else mode))), encoding="utf-8")
            fields = ("prune_rate", "complete", "accuracy", "precision", "recall", "tnr", "lat_mean_ms", "mem_peak_allocated_gib")
            with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for rate in config["prune_rates"]:
                    writer.writerow(dict(prune_rate=rate, complete=True, accuracy=.9, precision=.91, recall=.92,
                                         tnr=.88, lat_mean_ms=100 + rate, mem_peak_allocated_gib=20))
                    folder = output / f"prune_{rate}"
                    folder.mkdir()
                    if rate in config["visualize_rates"]:
                        folder = folder / "visualizations"
                        folder.mkdir()
                        for name in ("comparison.png", "attention_overlay.png"):
                            Image.new("RGB", (1, 1)).save(folder / name)
            return types.SimpleNamespace(returncode=0)
        return invoke

    def test_one_command_two_isolated_children_six_rows_and_only70_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            args = self.fixture(root, "--visualize-samples", "sample_000000000", "000000270")
            with contextlib.redirect_stdout(io.StringIO()):
                output = compare.run_comparison(args, run_process=self.process(calls))
            self.assertEqual(len(calls), 2)
            for (mode, _), (command, cwd, env, check) in zip(compare.TRIALS, calls):
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4")
                self.assertFalse(check)
                self.assertEqual(cwd, str(compare.PROJECT))
                self.assertEqual(command[0], compare.sys.executable)
                self.assertIn("--benchmark-metrics", command)
                self.assertIn("--no-metric-plot", command)
                self.assertIn("--save-prune-vis", command)
                self.assertIn("--save-attention-vis", command)
                config = json.loads((output / "configs" / f"{mode}.json").read_text(encoding="utf-8"))
                self.assertEqual(config["checkpoint"], str((root / "best.pt").resolve()))
                self.assertEqual(config["prune_rates"], [30, 50, 70])
                self.assertEqual(config["visualize_rates"], [70])
                self.assertEqual(config["selection_mode"], mode)
                for rate in (30, 50):
                    self.assertFalse(list((output / mode / f"prune_{rate}").rglob("*.png")))
                self.assertEqual(len(list((output / mode / "prune_70").rglob("*.png"))), 2)
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["selection_mode"] for r in rows], ["anyres_self"] * 3 + ["base_full"] * 3)
            self.assertEqual([r["prune_rate"] for r in rows], ["30", "50", "70"] * 2)
            self.assertTrue(all(r["lat_mean_ms"] and r["mem_peak_allocated_gib"] for r in rows))
            manifest = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual([t["status"] for t in manifest["trials"]], ["complete", "complete"])

    def test_metrics_only_disables_capture_and_stops_first_failed_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            args = self.fixture(root, "--no-visualizations")
            with contextlib.redirect_stdout(io.StringIO()):
                output = compare.run_comparison(args, run_process=self.process(calls))
            self.assertFalse(list(output.rglob("*.png")))
            self.assertTrue(all("--no-plots" in cmd and "--save-prune-vis" not in cmd for cmd, *_ in calls))
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            args = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exited with code"):
                compare.run_comparison(args, run_process=self.process(calls, fail=True))
            self.assertEqual(len(calls), 1)
            manifest = json.loads((root / "output/comparison.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["trials"][0]["status"], "failed")

    def test_paths_and_recorded_mode_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            args = self.fixture(root)
            args.checkpoint = root / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                compare.run_comparison(args, run_process=self.process(calls))
            self.assertFalse(calls)
            self.assertFalse((root / "output").exists())
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            args = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Wrong selection policy"):
                compare.run_comparison(args, run_process=self.process(calls, wrong_mode=True))


if __name__ == "__main__":
    unittest.main()
