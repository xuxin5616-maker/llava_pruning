"""Classification additions, all30 trials, measured-worker/report-only exports."""

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import run_pool_distill as entry
import ablate_attention_predictor as ablation
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.pool_distill_budget import hybrid_budget
from llava_pruning.prompts import resolve_prompt


class PoolDistillMetricTests(unittest.TestCase):
    def fixture(self, root, *extra):
        (root / "model").mkdir()
        (root / "data").mkdir()
        (root / "input.json").write_text("[]", encoding="utf-8")
        (root / "best.pt").write_bytes(b"mock")
        return entry.build_parser().parse_args(["--model-path", str(root/"model"), "--data-root", str(root/"data"),
            "--input-json", str(root/"input.json"), "--checkpoint", str(root/"best.pt"),
            "--output-dir", str(root/"output"), *extra])

    def read_rows(self, path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_metric_math_matches_existing_ablation_and_undefined_values(self):
        self.assertIs(ablation.extended_metrics, extended_metrics)
        row = extended_metrics(dict(tp=2, fp=1, tn=2, fn=1, unparsed=2, accuracy=.5), .9118)
        self.assertAlmostEqual(row["f1"], 2/3)
        self.assertAlmostEqual(row["bacc"], 2/3)
        self.assertAlmostEqual(row["relative_accuracy_pct"], 100*.5/.9118)
        self.assertGreater(extended_metrics(dict(tp=1,fp=0,tn=1,fn=0,accuracy=1), .9118)["relative_accuracy_pct"], 100)
        empty = extended_metrics(dict(tp=0,fp=0,tn=0,fn=0,accuracy=None), .9118)
        self.assertIsNone(empty["f1"])
        self.assertIsNone(empty["bacc"])
        self.assertIsNone(empty["relative_accuracy_pct"])
        for baseline in (0, 101, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                entry.baseline_reference(baseline)

    def test_defaults_run_all30_jobs_and_forward_reference_without_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root, "--baseline-acc", "90")
            self.assertEqual(args.prune_rates, [30,50,70])
            self.assertEqual(args.lambdas, [.01,.25,.5,.75,.99])
            calls = []
            def child(command, **kwargs):
                trial = entry.build_parser().parse_args(command[2:])
                calls.append(trial)
                self.assertTrue(trial.worker)
                self.assertNotIn("--save-prune-vis", command)
                self.assertNotIn("--save-attention-vis", command)
                self.assertEqual(trial.baseline_acc, 90)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "5")
                trial.output_dir.mkdir()
                metrics = extended_metrics(dict(prune_rate=trial.prune_rates[0], complete=True, tp=2,fp=1,tn=2,fn=1,
                    accuracy=.5, distill_share=trial.lambdas[0], selection_mode=trial.selection_modes[0],
                    lat_mean_ms=100., mem_peak_allocated_gib=20., actual_average_prune_rate=trial.prune_rates[0]+.001), .9)
                entry.write_csv(trial.output_dir / "summary.csv", [metrics])
                return SimpleNamespace(returncode=0)
            entry.run_sweep(args, run_process=child)
            self.assertEqual(len(calls),30)
            self.assertEqual({trial.lambdas[0] for trial in calls}, {.01,.25,.5,.75,.99})
            rows = self.read_rows(root / "output/summary.csv")
            self.assertEqual(len(rows),30)
            self.assertEqual({float(r["prune_rate"]) for r in rows}, {30,50,70})
            self.assertEqual({float(r["distill_share"]) for r in rows}, {.01,.25,.5,.75,.99})
            self.assertTrue(all(float(r["baseline_accuracy"])==.9 for r in rows))
            self.assertTrue(all(abs(float(r["relative_accuracy_pct"])-100*.5/.9)<1e-10 for r in rows))

    def make_old_report(self, root, name="trial", *, bad_counts=False):
        folder = root / name
        folder.mkdir(parents=True)
        metrics = dict(prune_rate=30, complete=True, tp=2.,fp=1,tn=2,fn=1, accuracy=.5,
                       distill_share=.5, selection_mode="anyres_self", lat_mean_ms=100., mem_peak_allocated_gib=20.)
        if not bad_counts:
            metrics["tp"] = 2
        entry.write_json(folder / "metrics.json", metrics)
        entry.write_csv(folder / "summary.csv", [metrics])
        entry.write_json(folder / "run.json", dict(complete=True, definitions=dict(lat="unchanged")))
        (folder / "predictions.jsonl").write_text("preserve predictions", encoding="utf-8")
        (folder / "budgets.csv").write_text("preserve budgets", encoding="utf-8")
        return dict(status="complete", rate=30, distill_share=.5, selection_mode="anyres_self",
                    directory="Z:\\old-server-output\\" + name)

    def test_report_only_updates_existing_moved_results_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = self.make_old_report(root)
            entry.write_json(root / "comparison.json", dict(complete=False, trials=[trial,
                dict(status="failed", directory="failed_trial")]))
            with patch.object(entry, "run_worker", side_effect=AssertionError("No inference")), \
                    patch.object(entry, "run_sweep", side_effect=AssertionError("No sweep")), \
                    patch("sys.argv", ["run_pool_distill.py", "--report-only", str(root), "--baseline-acc", "90"]):
                entry.main()
            row = self.read_rows(root / "summary.csv")[0]
            self.assertAlmostEqual(float(row["f1"]), 2/3)
            self.assertAlmostEqual(float(row["bacc"]), 2/3)
            self.assertAlmostEqual(float(row["relative_accuracy_pct"]), 100*.5/.9)
            self.assertEqual(row, self.read_rows(root / "trial/summary.csv")[0])
            self.assertFalse(json.loads((root/"comparison.json").read_text(encoding="utf-8"))["complete"])
            metadata = json.loads((root/"trial/run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["definitions"]["lat"], "unchanged")
            self.assertEqual(metadata["baseline_accuracy_reference"]["accuracy"], .9)
            self.assertEqual((root/"trial/predictions.jsonl").read_text(), "preserve predictions")
            self.assertEqual((root/"trial/budgets.csv").read_text(), "preserve budgets")
            entry.rebuild_reports(root, 91.18)
            self.assertEqual(float(self.read_rows(root/"summary.csv")[0]["baseline_accuracy"]), .9118)

    def test_report_validates_every_completed_trial_before_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = [self.make_old_report(root, "good"), self.make_old_report(root,"bad",bad_counts=True)]
            entry.write_json(root/"comparison.json",dict(complete=True,trials=trials))
            before = (root/"good/metrics.json").read_bytes()
            with self.assertRaises(ValueError):
                entry.rebuild_reports(root)
            self.assertEqual((root/"good/metrics.json").read_bytes(), before)
            self.assertFalse((root/"summary.csv").exists())

    def test_worker_exports_metrics_for_all_images_and_excludes_warmups(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.fixture(root, "--worker", "--prune-rates", "30", "--lambdas", ".5",
                              "--selection-modes", "anyres_self", "--warmup", "2")
            image=root/"data/image.png"
            Image.new("RGB",(10,10)).save(image)
            answers=["A","A","B","B","unparsed"]
            samples=[SimpleNamespace(sample_id=str(i),image=image,gt=gt,category="bottle",text="test")
                     for i,gt in enumerate([1,0,0,1,1])]
            core=SimpleNamespace(layers=[None]*28)
            model=SimpleNamespace(get_model=lambda:core)
            calls=[]
            class Method:
                def __init__(self, *a, **k):
                    self.adapter=SimpleNamespace(set_image_context=lambda size:None)
                    self.active=False
                @contextmanager
                def generation_context(self, current):
                    self.active=True
                    try: yield
                    finally: self.active=False
            class Backend:
                def __init__(self, *a, **k):
                    self.model=model
                    self.inference_config={"attention":"mock"}
                def generate(self, sample, prompt, **kwargs):
                    # Assert the real worker-to-backend interface, not only
                    # metrics or direct model.generate pruning equivalence.
                    assert isinstance(prompt, str)
                    assert prompt == resolve_prompt(args.prompt_version, sample.category, sample.text)[0]
                    assert "mvtec_v0" not in prompt and "\\n" not in prompt
                    calls.append(kwargs)
                    assert kwargs["method"].active and kwargs["benchmark_metrics"]
                    assert not kwargs["capture_visualization"] and not kwargs["capture_attention"] and not kwargs["do_sample"]
                    plan=hybrid_budget(7371,30,.5)
                    return dict(answer=answers[int(sample.sample_id)], lat_seconds=.1,generation_seconds=.2,
                                generated_tokens=1, peak_allocated_bytes=1024,peak_reserved_bytes=2048,
                                stats=dict(plan, pre_base_tokens=400,final_anyres_tokens=plan["final_llm_tokens"]-481))
            with patch("llava_pruning.backend.LlavaBackend",Backend), patch("llava_pruning.pool_distill.PoolDistillMethod",Method), \
                    patch("llava_pruning.data.load_samples",return_value=samples), \
                    patch("benchmark_fastv.validate_single_gpu"), patch.object(torch.cuda,"is_available",return_value=True), \
                    patch.object(torch.cuda,"device_count",return_value=1), \
                    patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(total_memory=4*1024**3)):
                entry.run_worker(args)
            self.assertEqual(len(calls),7)
            rows=self.read_rows(root/"output/summary.csv")
            self.assertEqual(len(rows),1)
            row=rows[0]
            self.assertEqual(int(row["evaluated_samples"]),5)
            self.assertEqual(int(row["unparsed"]),1)
            self.assertAlmostEqual(float(row["accuracy"]),.4)
            self.assertAlmostEqual(float(row["f1"]),.5)
            self.assertAlmostEqual(float(row["bacc"]),.5)
            self.assertAlmostEqual(float(row["relative_accuracy_pct"]),100*.4/.9118)
            self.assertEqual(len((root/"output/predictions.jsonl").read_text().splitlines()),5)
            metrics=json.loads((root/"output/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["f1"],.5)
            metadata=json.loads((root/"output/run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["prompt_format"],entry.PROMPT_FORMAT)
            self.assertFalse(list((root/"output").rglob("*.png")))


if __name__=="__main__":
    unittest.main()
