"""Two-stage conservation, actual MyPool merging, sparse scores, native decode."""

import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import runMyPruningPoolSpatial as pool
import run_pool_distill as entry
from llava_pruning.pool_distill_budget import hybrid_budget, pre_view_budget
from llava_pruning.pool_distill import (PoolDistillAdapter, PoolDistillMethod, source_records,
                                      weighted_sources, mapped_scores)
from test_pool_spatial import _Core
from test_clse import toy_model
from test_attention_predictor import make_checkpoint


class PoolDistillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_budget_is_additive_contributions_not_naive_sequential_rates(self):
        for n in (1485, 7371, 9001):
            for share in (0, .25, .5, .75, 1):
                p = hybrid_budget(n, 30, share)
                actual = 100 * (1 - (2*p["pre_llm_tokens"] + 26*p["final_llm_tokens"]) / (28*n))
                self.assertAlmostEqual(actual, p["actual_average_prune_rate"])
                self.assertGreaterEqual(actual + 1e-12, 30)
                self.assertLess(actual - 30, 200/n)
                if share == 0:
                    self.assertEqual(p["pre_llm_tokens"], p["final_llm_tokens"])
                if share == 1:
                    self.assertEqual(p["pre_llm_tokens"], n)
        p = hybrid_budget(260000, 30, .5)
        self.assertAlmostEqual(p["actual_pre_contribution"], 15)
        self.assertAlmostEqual(p["stage2_remaining_prune_rate"], 19.0045, places=3)
        for value in (-.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                hybrid_budget(7371, 30, value)
        with self.assertRaises(ValueError):
            pre_view_budget(dict(hybrid_budget(1485, 90, .99), raw_grid_height=1), 1)

    def test_real_pool_adapter_preserves_exact_budgets_hooks_and_source_coefficients(self):
        for share in (0, .25, .5, .75):
            core = _Core()
            core.layers = [None] * 28
            adapter = PoolDistillAdapter(core, pool.MethodSettings(prune_rate=0), rate=30, share=share, mode="anyres_self")
            adapter.configure(0, capture_visualization=False)
            adapter.set_image_context((1024, 1024))
            packed = adapter.encode_images(torch.zeros(10, 3, 1, 1))
            self.assertEqual(len(packed), adapter.plan["pre_llm_tokens"])
            self.assertEqual(len(adapter.records), len(packed))
            self.assertEqual(sum(r is None for r in adapter.records), 81)
            self.assertEqual(adapter.allocation["target_anyres_keep"], adapter.view_plan["anyres_keep"])
            self.assertLess(adapter.view_plan["base_keep"], 729)
            for view in range(10):
                records = [r for r in adapter.records if r is not None and r["view"] == view]
                sources = [s for r in records for s in r["sources"]]
                self.assertEqual(sorted(sources), list(range(729)))
                self.assertEqual(len(sources), len(set(sources)))
                for r in records:
                    self.assertEqual(r["weights"][0], 1.)
                    if len(r["sources"]) > 1:
                        self.assertAlmostEqual(sum(r["weights"][1:]), 1.)
            for block in core.tower.blocks:
                self.assertFalse(block._forward_hooks)
                self.assertFalse(block.self_attn._forward_hooks)

    def test_source_order_and_values_match_actual_sparse_packer(self):
        torch.manual_seed(1811)
        features = torch.randn(3, 729, 6)
        scores = torch.rand(3, 729)
        metric = torch.randn(3, 729, 4)
        newline = torch.ones(6) * -99
        for extra in (False, True):
            packed, decisions = pool.compress_variable_views(features, scores, metric, [500, 490, 510], (2, 1),
                torch.nn.Identity(), pool.compress_tokens, pool.token_budget, image_newline=newline,
                merge_type="spatial_unpad_add_newl" if extra else "spatial_unpad")
            records = source_records(decisions, (2, 1), final_newline=extra)
            expected = []
            for r in records:
                expected.append(newline if r is None else (features[r["view"], r["sources"]] * torch.tensor(r["weights"])[:, None]).sum(0))
            torch.testing.assert_close(packed, torch.stack(expected), rtol=1e-5, atol=1e-6)

    def test_base_mapping_uses_all_sources_and_merge_weights_not_anchor_only(self):
        base = [dict(view=0, anchor=i, sources=[i], weights=[1.]) for i in range(729)]
        anyres = [dict(view=1, anchor=0, sources=[0, 1, 728], weights=[1., .5, .5]),
                  dict(view=1, anchor=10, sources=[10], weights=[1.])]
        geo = pool.ScoreGeometry.create((384, 384), (1, 1), 384, 14)
        logits = torch.arange(729).float()/100
        output = mapped_scores(logits, base, anyres, geo).exp()
        prob = logits.softmax(0)
        expected = (prob[0] + .5*prob[1] + .5*prob[728])/2
        torch.testing.assert_close(output[729]/output[730], expected/prob[10], rtol=1e-6, atol=1e-5)
        self.assertGreater(float(output[729]), float(output[0]))
        # Compressed Base sources jointly cover the original729 positions.
        compressed = [base[0], dict(view=0, anchor=1, sources=list(range(1, 729)), weights=[1.] + [1/727]*727)]
        output = mapped_scores(torch.tensor([.3, .7]).log(), compressed, anyres, geo)
        self.assertTrue(torch.isfinite(output).all())
        self.assertAlmostEqual(float(output.exp().sum()), 1., places=6)

    def test_native_multimodal_generation_sparse_scores_positions_and_cache(self):
        # Exercise the REAL model.generate + prefill override + predictor + KV
        # compression, replacing only expensive image encoder with synthetic rows.
        for share in (0, .5):
            outputs = []
            for mode in ("anyres_self", "base_full"):
                torch.manual_seed(1812)
                model, _, options = toy_model()
                core = model.get_model()
                plan = hybrid_budget(40, 30, share, len(core.layers))
                pre = plan["pre_llm_tokens"]
                records = [dict(view=0, anchor=i, sources=[i], weights=[1.]) for i in range(4)]
                records += [dict(view=1, anchor=i, sources=[i], weights=[1.]) for i in range(pre - 6)]
                records += [None, None]
                packed = torch.randn(pre, 32)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory)/"best.pt"
                    make_checkpoint(checkpoint, core)
                    method = PoolDistillMethod(core, checkpoint, rate=30, share=share, mode=mode, settings=pool.MethodSettings(prune_rate=0))
                    adapter = method.adapter
                    adapter.plan, adapter.records = plan, records
                    adapter.geometry = pool.ScoreGeometry.create((384,384), (1,1),384,14)
                    adapter.view_plan = dict(base_keep=4, protected_newlines=2)
                    def encode(images):
                        core.clse_keep_ratio = plan["final_llm_tokens"]/pre
                        return packed
                    with patch.object(adapter, "configure"), patch.object(adapter, "set_image_context"), \
                            patch.object(adapter, "encode_images", side_effect=encode), patch.object(adapter, "pack", side_effect=lambda f,*a:f), torch.no_grad():
                        method.configure(core, 30, capture_attention=False, capture_visualization=False)
                        prepare = model.prepare_inputs_labels_for_multimodal
                        with method.generation_context(model):
                            generated = model.generate(**options)
                        self.assertNotIn("_pool_distill_adapter", core.__dict__)
                        self.assertEqual(model.prepare_inputs_labels_for_multimodal, prepare)
                        stats = method.stats(core, 30)
                        self.assertEqual(stats["final_llm_tokens"], plan["final_llm_tokens"])
                        self.assertEqual(stats["final_base_tokens"], 4)
                        if share:
                            self.assertEqual(core.get_clse_stats()["position_policy"], "preserve_original")
                        else:
                            self.assertIsNone(core.attention_predictor)
                        outputs.append(generated)
            if not share:
                self.assertTrue(torch.equal(*outputs))

    def test_lambda1_is_exact_existing_predictor_path_and_never_calls_pool(self):
        for mode in ("anyres_self", "base_full"):
            model, features, options = toy_model()
            core = model.get_model()
            with tempfile.TemporaryDirectory() as directory, torch.no_grad(), patch.object(model, "encode_images", return_value=features):
                checkpoint = Path(directory)/"best.pt"
                make_checkpoint(checkpoint, core)
                method = PoolDistillMethod(core, checkpoint, rate=20, share=1, mode=mode, settings=pool.MethodSettings(prune_rate=0))
                method.native.configure(core, 20)
                expected = model.generate(**options)
                method.configure(core, 20)
                with patch.object(method.adapter, "encode_images", side_effect=AssertionError("MyPool must be disabled")), method.generation_context(model):
                    actual = model.generate(**options)
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(method.stats(core, 20)["actual_pre_contribution"], 0.)

    def test_launcher_ten_isolated_jobs_summaries_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"model").mkdir()
            (root/"data").mkdir()
            (root/"input.json").write_text("[]")
            (root/"best.pt").write_bytes(b"mock")
            args = entry.build_parser().parse_args(["--model-path", str(root/"model"), "--data-root", str(root/"data"),
                "--input-json", str(root/"input.json"), "--checkpoint", str(root/"best.pt"), "--prune-rates", "30", "--output-dir", str(root/"out")])
            calls=[]
            def child(command, **kwargs):
                trial = entry.build_parser().parse_args(command[2:])
                calls.append((trial, kwargs))
                self.assertTrue(trial.worker)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "5")
                trial.output_dir.mkdir()
                entry.write_csv(trial.output_dir/"summary.csv", [dict(complete=True, prune_rate=30, distill_share=trial.lambdas[0],
                    selection_mode=trial.selection_modes[0], lat_mean_ms=12., mem_peak_allocated_gib=3., actual_average_prune_rate=30.001,
                    f1=.8, bacc=.8, baseline_accuracy=trial.baseline_acc/100, relative_accuracy_pct=90.)])
                return types.SimpleNamespace(returncode=0)
            entry.run_sweep(args, run_process=child)
            self.assertEqual(len(calls), 10)
            with (root/"out/summary.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),10)
            self.assertTrue(json.loads((root/"out/comparison.json").read_text(encoding="utf-8"))["complete"])
            args.output_dir=root/"fail"
            with self.assertRaises(RuntimeError):
                entry.run_sweep(args, run_process=lambda *a,**k: types.SimpleNamespace(returncode=2))
            manifest = json.loads((root/"fail/comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["trials"]),1)
            self.assertEqual(manifest["trials"][0]["status"],"failed")


if __name__ == "__main__":
    unittest.main()
