"""Check quota ablations through the existing baseline generation paths."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import benchmark_prefill_kv as entry
from llava_pruning import raba_ablation as raba
import runMyPruningPoolSpatial as pool
from test_attention_predictor import make_checkpoint
from test_mmad_binary import fixture
from test_pool_ablations import model_and_options
from test_pool_spatial import _Tower


class RabaTests(unittest.TestCase):
    def test_grid_scores_are_the_existing_components(self):
        features = torch.rand(5, 729, 8, generator=torch.Generator().manual_seed(36))
        full = pool.pooled_crop_redundancy(features, (2, 2))
        for key in ("redundancy", "base_redundancy", "anyres_redundancy"):
            lean = pool.pooled_crop_redundancy(features, (2, 2), audit=False, component=key)
            torch.testing.assert_close(lean["redundancy"], full[key], atol=0, rtol=0)

    def test_schedules_run_only_hybrid_for_all_rates_and_three_shuffle_seeds(self):
        for mode in ("mvtecmode", "mmadmode"):
            for next_model in (False, True):
                jobs = entry.schedule(mode, [0, 1], next_model=next_model, raba_ablation=True)
                self.assertEqual(len({job["job_id"] for job in jobs}), 21)
                self.assertEqual({job["method"] for job in jobs}, {"hybrid_baseline"})
                for rate in (50, 70, 90):
                    selected = [job for job in jobs if job["prune_rate"] == rate]
                    self.assertEqual({job["raba_strategy"] for job in selected}, set(raba.STRATEGIES))
                    self.assertEqual([job["raba_seed"] for job in selected if job["raba_strategy"] == "shuffled"], [0, 1, 2])
                    self.assertTrue(all("tail=8(extra)" in job["config"] for job in selected))

    @torch.inference_mode()
    def test_90_percent_matches_baseline_and_keeps_both_stages_fixed(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(32)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for mode in ("mvtecmode", "mmadmode"):
                model, options = model_and_options(28)
                core = model.get_model()
                core.config.image_grid_pinpoints = [[768, 768]]
                core.vision_tower = _Tower()
                core.vision_tower.features = core.vision_tower.features[:5]
                size = (768, 768)
                options.update(images=[torch.zeros(5, 3, 384, 384)], image_sizes=[size], max_new_tokens=1)
                path = folder / "best.pt"
                make_checkpoint(path, core)
                spec = dict(mode=mode, checkpoint={"path": str(path)})
                original, = [job for job in entry.schedule(mode, [0])
                             if job["method"] == "hybrid_baseline" and job["prune_rate"] == 90]
                jobs = [job for job in entry.schedule(mode, [0], raba_ablation=True) if job["prune_rate"] == 90]
                events = {}
                for job in (original, *jobs):
                    strategy = job.get("raba_strategy", "baseline")
                    with self.subTest(mode=mode, strategy=strategy, seed=job.get("raba_seed")), \
                            patch.object(entry, "student_identity", return_value=spec["checkpoint"]):
                        method = entry.build_method(SimpleNamespace(model=model), [], spec, job, folder)
                        quota = getattr(method, "quota_ablation", None)
                        if quota is not None:
                            quota.image_key = "test.png"
                        method.configure(core, 90)
                        method.prepare_inputs(size, options["inputs"])
                        with method.generation_context(model):
                            generation = model.generate(**options)
                        budget = method.stats(core, 90)
                        counts = tuple(budget[key] for key in ("pre_base_tokens", "pre_llm_tokens", "stage2_llm_visual_tokens"))
                        keeps = [crop["keep"] for crop in method.adapter.allocation["crops"]]
                        if strategy == "baseline":
                            events.update(counts=counts, keeps=keeps, sequences=generation.sequences)
                        else:
                            self.assertEqual(counts, events["counts"])
                            self.assertEqual(sum(keeps), sum(events["keeps"]))
                            if strategy == "global_local":
                                self.assertEqual(keeps, events["keeps"])
                                self.assertTrue(torch.equal(generation.sequences, events["sequences"]))
                            elif strategy == "uniform":
                                self.assertLessEqual(max(keeps) - min(keeps), 1)
                            elif strategy == "shuffled":
                                self.assertCountEqual(keeps, events["keeps"])

    def test_foreground_metric_counts_retained_anchors_and_excludes_padding(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bottle/ground_truth/crack/000_mask.png"
            path.parent.mkdir(parents=True)
            mask = np.zeros((4, 8), dtype=np.uint8)
            mask[0, 0], mask[0, 2] = 255, 255
            Image.fromarray(mask).save(path)
            row = dict(origin_path="bottle/test/crack/000.png", resolved_image=str(root / "input.png"),
                       image_size=[8, 4], gt=1)
            raba.freeze_masks([row], root)
            geometry = pool.ScoreGeometry.create((8, 4), (2, 2), 4, 2)
            decisions = [dict(dominant_indices=torch.tensor([2]), contextual_indices=torch.tensor([0])) for _ in range(4)]
            stats = raba.measure_defect_tokens(row, geometry, decisions)
            self.assertEqual(stats, dict(defect_mask_status="covered", defect_tokens=2, kept_defect_tokens=1))
            records = [dict(gt=1, raba=dict(stats, crop_keeps=[5, 7], unchanged=None)),
                       dict(gt=1, raba=dict(defect_mask_status="missing", crop_keeps=[6, 6], unchanged=None))]
            summary = raba.summarize(records)
            self.assertEqual(summary["annotated_samples"], 1)
            self.assertEqual(summary["defect_token_retention_pct"], 50)

    def test_summary_waits_for_all_shuffle_runs(self):
        rows = [dict(prune_rate=90, raba_strategy="shuffled", status="success", acc=90 + seed,
                     **dict.fromkeys(raba.SAMPLE_FIELDS, 1)) for seed in (0, 1, 2)]
        row, = raba.summarize_trials(rows)
        self.assertEqual(row["acc_mean"], 91)
        self.assertGreater(row["acc_std"], 0)
        rows[-1]["status"] = "pending"
        row, = raba.summarize_trials(rows)
        self.assertEqual(row["completed_runs"], 2)
        self.assertNotIn("acc_mean", row)

    def test_frozen_runs_and_pending_reports_for_both_datasets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = fixture(root)
            model = Path(paths.model_path)
            entry.common.save_json(model / "config.json", dict(num_hidden_layers=28, image_grid_pinpoints=[[1152, 1152]]))
            questions = root / "mvtec.jsonl"
            Image.new("RGB", (8, 4)).save(root / "image.png")
            questions.write_text('{"question_id":"1","image":"image.png","category":"bottle","gt":0}\n', encoding="utf-8")
            for mode, data_root in (("mvtecmode", root), ("mmadmode", paths.data_root)):
                output = root / mode
                args = entry.parser().parse_args([
                    mode, "--raba-ablation", "--output-dir", str(output), "--model-path", str(model),
                    "--checkpoint", paths.checkpoint, "--data-root", str(data_root), "--input-json", str(questions),
                    "--mvtec-root", str(Path(paths.data_root) / "MVTec-AD"),
                ])
                spec = entry.setup(args, output)
                self.assertEqual(entry.load_run(output, mode), spec)
                entry.write_reports(output, spec)
                with (output / "raba_summary.csv").open(encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 15)
                self.assertTrue(all(row["completed_runs"] == "0" and row["acc_mean"] == "" for row in rows))
