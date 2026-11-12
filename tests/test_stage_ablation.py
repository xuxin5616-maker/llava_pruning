"""Exercise all stage combinations with real generation and matched token budgets."""

from contextlib import nullcontext
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

import benchmark_stage_ablation as entry
from llava.model.language_model import fastv_attention
from llava_pruning.pool_distill_budget import hybrid_budget
from llava_pruning.stage_ablation import StageAblationMethod
from test_attention_predictor import make_checkpoint
from test_mmad_binary import fixture
from test_pool_ablations import model_and_options
from test_pool_spatial import _Tower


class StageAblationTests(unittest.TestCase):
    def test_mode_inputs_reports_and_resume_use_original_labels(self):
        for mode in ("mvtecmode", "mmadmode"):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = fixture(root)
                config = dict(num_hidden_layers=28, mm_patch_merge_type="spatial_unpad",
                              image_grid_pinpoints=[[384, 384], [1152, 1152]])
                entry.common.save_json(Path(paths.model_path) / "config.json", config)
                raw = root / "mvtec.jsonl"
                Image.new("RGB", (384, 384)).save(root / "01.png")
                raw.write_text(json.dumps(dict(question_id="1", image="01.png", category="bottle", gt=0)),
                               encoding="utf-8")
                is_mmad = mode == "mmadmode"
                args = entry.parser().parse_args([mode, "--output-dir", str(paths.output_dir),
                    "--model-path", paths.model_path, "--checkpoint", paths.checkpoint,
                    "--data-root", paths.data_root if is_mmad else str(root), "--input-json", str(raw),
                    "--mvtec-root", str(Path(paths.data_root) / "MVTec-AD")])
                spec = entry.setup(args, paths.output_dir)
                self.assertEqual(entry.load_run(paths.output_dir, mode), spec)
                self.assertEqual(len(spec["jobs"]), 13 if is_mmad else 12)
                if is_mmad:
                    self.assertEqual(spec["dataset_info"]["excluded_sources"], ["VisA"])
                    samples = entry.mm.load_frozen(spec["input_json"], exclude_sources=("VisA",))
                    self.assertTrue(all(s.source["image_size"] == [1152, 1152] for s in samples))
                else:
                    samples = entry.samples_for(spec)
                job = next(j for j in spec["jobs"] if j["prune_rate"])
                trial = paths.output_dir / job["job_id"]
                trial.mkdir()
                model = SimpleNamespace(get_model=lambda: SimpleNamespace(layers=[None] * 28))
                original = 7371 if is_mmad else 1485
                plan = hybrid_budget(original, 50, .7)
                pre, final = plan["pre_llm_tokens"], plan["final_llm_tokens"]
                lengths = dict(enumerate([pre + 4] * 2 + [final + 4] * 18 + [4] * 8))

                def generate(sample, prompt, *_args, **_kwargs):
                    self.assertIsInstance(prompt, str)
                    answer = sample.source["correct_option"] if is_mmad else "B"
                    return dict(answer=answer, stats=plan, lat_seconds=.2, generation_seconds=.3,
                                generated_tokens=2, peak_allocated_bytes=1024, peak_reserved_bytes=2048)

                backend = SimpleNamespace(model=model, generate=generate)
                with patch("llava_pruning.visa_comparison_methods.observe_prefill", return_value=nullcontext(lengths)), \
                        patch("llava_pruning.visa_comparison_methods.native_counts", return_value=(None, (original,))):
                    result = entry.evaluate(backend, None, samples, spec, job, trial, 4096)
                self.assertEqual(result["accuracy"], 1)
                journal = trial / "predictions.jsonl"
                journal.write_bytes(journal.read_bytes().rstrip(b"\n"))
                self.assertEqual(entry.evaluate(backend, None, samples, spec, job, trial, 4096), result)
                entry.common.save_json(trial / "metrics.json", dict(result, status="success", job_id=job["job_id"],
                                                                      input_sha256=spec["input_sha256"]))
                entry.write_reports(paths.output_dir, spec)
                with (paths.output_dir / "summary.csv").open(encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(sum(row["acc"] == "100.0" for row in rows), 1)

    @torch.inference_mode()
    def test_all_pairs_match_budget_and_remove_image_from_tail(self):
        torch.manual_seed(42)
        model, options = model_and_options(28)
        core = model.get_model()
        core.config.image_grid_pinpoints = [[768, 384]]
        core.vision_tower = _Tower()
        core.vision_tower.features = core.vision_tower.features[:3]
        size = (768, 384)
        options.update(images=[torch.zeros(3, 3, 384, 384)], image_sizes=[size], max_new_tokens=2)
        original = 3 * 729 + 27
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "student.pt"
            make_checkpoint(checkpoint, core)
            for mode in ("mvtecmode", "mmadmode"):
                for job in entry.schedule(mode, 3):
                    rate = job["prune_rate"]
                    if not rate:
                        continue
                    with self.subTest(mode=mode, job=job["job_id"]):
                        method = StageAblationMethod(core, checkpoint, job, mode)
                        method.configure(core, rate)
                        method.prepare_inputs(size, options["inputs"])
                        with patch.object(fastv_attention, "last_prompt_attention",
                                          wraps=fastv_attention.last_prompt_attention) as score:
                            with method.generation_context(model):
                                model.generate(**options)
                        raw = core.get_clse_stats()
                        stats = method.stats(core, rate)
                        plan = hybrid_budget(original, rate, .7)
                        pre, final = plan["pre_llm_tokens"], plan["final_llm_tokens"]
                        self.assertEqual([row["image_tokens_in"] for row in raw["layers"]],
                                         [pre] * 2 + [final] * 18 + [0] * 8)
                        self.assertEqual(stats["stage2_llm_visual_tokens"], final)
                        self.assertEqual(stats["tail_drop_first_layer"], 21)
                        self.assertEqual(stats["final_llm_visual_tokens"], 0)
                        self.assertEqual(stats["stage2_base_policy"], "joint" if mode == "mmadmode" else "anyres_first")
                        if method.fastv:
                            self.assertFalse(core.attention_predictor_enabled)
                            score.assert_called_once()
                            self.assertIs(score.call_args.args[0], core.layers[1])
                            self.assertEqual(raw["score"], "fastv_last_prompt_attention")
                        else:
                            self.assertEqual(score.call_count, 0)
                        self.assertFalse(hasattr(core, "_pool_fastv_adapter"))
                        if method.pre_stage == "visionzip":
                            keeps = [decision["retained_token_count"] for decision in method.adapter.device_decisions]
                            self.assertLessEqual(max(keeps[1:]) - min(keeps[1:]), 1)
                            if mode == "mvtecmode" and not method.fastv:
                                self.assertTrue(all(d["cluster_assignments"] is not None
                                                    for d in method.adapter.device_decisions))
