"""Check prefill-only measurement against the decoder's actual retained K/V."""

from collections import Counter
from contextlib import nullcontext
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention, SigLipVisionConfig

import benchmark_prefill_kv as entry
from llava_pruning.prefill_kv import KV_MIB, observe_post_prefill_kv
from llava_pruning.sparsevlm import SparseVLMMethod
from llava_pruning.visa_comparison_methods import observed_budget, observe_prefill
from test_attention_predictor import make_checkpoint
from test_fastv_attention import tiny_config
from test_mmad_binary import fixture
from test_pool_ablations import model_and_options
from test_pool_spatial import _Tower


class PrefillKVTests(unittest.TestCase):
    @torch.inference_mode()
    def test_next_32_block_fixed_grid_and_high_rate_hybrid(self):
        from llava_pruning.next_interleave import INPUT_SIZE, NATIVE_PINPOINTS

        def config(implementation):
            result = tiny_config(implementation)
            result.num_key_value_heads = result.num_attention_heads
            result.max_position_embeddings = 32768
            return result

        with TemporaryDirectory() as tmp, patch("test_pool_ablations.tiny_config", side_effect=config):
            folder = Path(tmp)
            trials = [("mvtecmode", "llava_unpruned", 0)]
            trials.extend(("mvtecmode", name, 50)
                          for name in ("fastv", "visionzip", "anchorprune", "clse", "priortr", "sparsevlm"))
            trials.extend((("mvtecmode", "pyramiddrop", 70),
                           ("mvtecmode", "hybrid_baseline", 90),
                           ("mmadmode", "hybrid_baseline", 90)))
            for mode, name, rate in trials:
                with self.subTest(mode=mode, method=name):
                    model, options = model_and_options(32)
                    core = model.get_model()
                    core.config.image_grid_pinpoints = sorted(NATIVE_PINPOINTS)
                    core.vision_tower = _Tower()
                    core.vision_tower.features = core.vision_tower.features[:5]
                    if name == "anchorprune":
                        core.vision_tower.blocks[-1].self_attn = SigLipAttention(
                            SigLipVisionConfig(hidden_size=8, num_attention_heads=2))
                    options.update(images=[torch.zeros(5, 3, 384, 384)], image_sizes=[INPUT_SIZE])
                    checkpoint = folder / "best.pt"
                    if name == "hybrid_baseline":
                        make_checkpoint(checkpoint, core)
                    job = next(job for job in entry.schedule(mode, [1], next_model=True)
                               if job["method"] == name and job["prune_rate"] == rate)
                    student = {"path": str(checkpoint)}
                    if name == "sparsevlm":
                        method = SparseVLMMethod(core, .7, job["after_blocks"])
                    else:
                        with patch.object(entry, "student_identity", return_value=student):
                            method = entry.build_method(SimpleNamespace(model=model), [],
                                                        dict(mode=mode, checkpoint=student), job, folder)
                    method.configure(core, rate, capture_visualization=False, capture_attention=False)
                    if hasattr(method, "prepare_inputs"):
                        method.prepare_inputs(INPUT_SIZE, options["inputs"])
                    if name == "anchorprune":
                        method.query_ids = torch.tensor([[1, 2]])
                    outer = method.generation_context(model) if hasattr(method, "generation_context") else nullcontext()
                    inner = method.inference_context(model) if hasattr(method, "inference_context") else nullcontext()
                    with outer, inner, observe_prefill(core) as lengths, observe_post_prefill_kv(core) as size:
                        model.generate(**options)
                    budget = observed_budget(name, rate, method.stats(core, rate), lengths, 3699, 32)
                    self.assertGreater(size[0], 0)
                    if name == "llava_unpruned":
                        self.assertEqual(lengths[0], options["inputs"].shape[-1] - 1 + 3699)
                        self.assertEqual(budget["physical_average_prune_rate"], 0)
                    elif name == "pyramiddrop":
                        self.assertEqual(tuple(core.pyramiddrop_layers), (8, 16, 24))
                        self.assertAlmostEqual(budget["actual_average_prune_rate"], 70, delta=.05)
                    elif name == "hybrid_baseline":
                        self.assertAlmostEqual(budget["two_stage_average_prune_rate"], 90, delta=.05)
                        self.assertEqual(budget["final_layer_visual_tokens"], 0)
                    elif name == "sparsevlm":
                        self.assertGreater(budget["actual_average_prune_rate"], 0)
                    else:
                        self.assertAlmostEqual(budget["actual_average_prune_rate"], 50, delta=.05)

    def test_full_data_setup_freezes_mmad_stretch_and_mvtec_native_size(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = fixture(root)
            config_path = Path(paths.model_path) / "config.json"
            entry.common.save_json(config_path, dict(num_hidden_layers=28, image_grid_pinpoints=[[1152, 1152]]))
            mvtec_input = root / "input.jsonl"
            Image.new("RGB", (128, 384)).save(root / "01.png")
            mvtec_input.write_text('{"question_id":"1","image":"01.png","category":"bottle","gt":0}\n', encoding="utf-8")
            for mode in ("mvtecmode", "mmadmode"):
                with self.subTest(mode=mode):
                    is_mmad = mode == "mmadmode"
                    output = root / mode
                    args = entry.parser().parse_args([mode, "--output-dir", str(output), "--model-path", paths.model_path,
                        "--checkpoint", paths.checkpoint, "--data-root", paths.data_root if is_mmad else str(root),
                        "--input-json", str(mvtec_input), "--mvtec-root", str(Path(paths.data_root) / "MVTec-AD")])
                    spec = entry.setup(args, output)
                    self.assertEqual(entry.load_run(output, mode), spec)
                    if is_mmad:
                        samples = entry.mm.load_frozen(spec["input_json"], exclude_sources=("VisA",))
                        self.assertEqual(len(samples), 8)
                        for sample in samples:
                            with Image.open(sample.image) as image:
                                self.assertEqual(image.size, (1152, 1152))
                    else:
                        sample, = entry.samples_for(spec)
                        self.assertEqual(sample.source["image_size"], [128, 384])

    @torch.inference_mode()
    def test_cached_generation_uses_only_prefill_bytes_for_all_methods(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            for name in ("llava_unpruned", *entry.common.METHODS, "sparsevlm", "mmadmode"):
                with self.subTest(method=name):
                    torch.manual_seed(2)
                    model, options = model_and_options(28)
                    core = model.get_model()
                    if name == "anchorprune":
                        core.vision_tower.blocks[-1].self_attn = SigLipAttention(
                            SigLipVisionConfig(hidden_size=8, num_attention_heads=2))
                    rate = 0 if name == "llava_unpruned" else 50
                    if name == "sparsevlm":
                        method = SparseVLMMethod(core, .7)
                    else:
                        mode = "mmadmode" if name == "mmadmode" else "mvtecmode"
                        method_name = "hybrid_baseline" if name == "mmadmode" else name
                        job = next(j for j in entry.schedule(mode, [2]) if j["method"] == method_name and j["prune_rate"] == rate)
                        if method_name == "hybrid_baseline":
                            make_checkpoint(checkpoint, core)
                        student = {"path": str(checkpoint)}
                        with patch.object(entry, "student_identity", return_value=student):
                            method = entry.build_method(SimpleNamespace(model=model), [], dict(mode=mode, checkpoint=student),
                                                        job, Path(tmp))
                    method.configure(core, rate, capture_visualization=False, capture_attention=False)
                    if hasattr(method, "prepare_inputs"):
                        method.prepare_inputs((384, 384), options["inputs"])
                    if name == "anchorprune":
                        method.query_ids = torch.tensor([[1, 2]])
                    outer = method.generation_context(model) if hasattr(method, "generation_context") else nullcontext()
                    inner = method.inference_context(model) if hasattr(method, "inference_context") else nullcontext()
                    with outer, inner, observe_prefill(core) as lengths, observe_post_prefill_kv(core) as size:
                        output = model.generate(**options)
                    pairs = list(output.past_key_values)
                    final_bytes = sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in pairs)
                    bytes_per_token = sum((key.shape[1] * key.shape[-1] * key.element_size()
                                           + value.shape[1] * value.shape[-1] * value.element_size()) for key, value in pairs)
                    self.assertEqual(size, [final_bytes - bytes_per_token * (output.sequences.shape[1] - 1)])
                    self.assertFalse(core._forward_hooks)
                    if name == "llava_unpruned":
                        baseline_size = size[0]
                    elif name == "fastv":
                        self.assertEqual(size[0], baseline_size)
                    else:
                        self.assertLess(size[0], baseline_size)
                    if name in {"hybrid_baseline", "mmadmode"}:
                        self.assertEqual([int(key.shape[2]) - 2 for key, _ in pairs],
                                         [lengths[2]] * 20 + [lengths[20]] * 8)

    def test_schedules_and_reports_leave_unreachable_metrics_blank(self):
        for mode in ("mvtecmode", "mmadmode"):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                jobs = entry.schedule(mode, [3])
                self.assertEqual(len(jobs), 25)
                self.assertEqual([j["job_id"] for j in jobs if j["status"] == "unreachable"], ["pyramiddrop_90"])
                self.assertEqual(jobs[-1]["after_blocks"], [2, 7, 16])
                if mode == "mmadmode":
                    self.assertTrue(all("resize=stretch;canvas=1152;grid=3x3" in j["config"] for j in jobs))
                    self.assertTrue(all(j["variant"] == "joint_self" for j in jobs if j["method"] == "hybrid_baseline"))
                root = Path(tmp)
                reference = root / jobs[0]["job_id"]
                reference.mkdir()
                test_result = dict(tp=3, fp=1, tn=2, fn=0, accuracy=5/6, precision=.75, recall=1., tnr=2/3,
                                   bacc=5/6, f1=6/7, unparsed=0, nonbinary=0, mmad_score=None,
                                   lat_mean_ms=25., mem_peak_allocated_gib=16., status="success", complete=True,
                                   input_sha256="sha", job_id=jobs[0]["job_id"])
                test_result[KV_MIB] = 78.5
                entry.common.save_json(reference / "metrics.json", test_result)
                entry.write_reports(root, dict(jobs=jobs, mode=mode, input_sha256="sha", dataset_info={}))
                with (root / "summary.csv").open(encoding="utf-8-sig") as stream:
                    table = csv.DictReader(stream)
                    self.assertEqual(table.fieldnames[table.fieldnames.index("mem") + 1], KV_MIB)
                    rows = list(table)
                self.assertEqual(rows[0][KV_MIB], "78.5")
                self.assertEqual(next(row[KV_MIB] for row in rows if row["status"] == "unreachable"), "")

    def test_combined_launcher_uses_each_gpu_once_at_a_time_and_keeps_separate_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            args = entry.parser().parse_args(["both", "--gpus", "0", "1", "2", "3", "4", "5", "--output-dir", str(root)])
            active, launched = set(), Counter()
            peak = 0

            def fake_setup(child, folder):
                folder.mkdir()
                return dict(mode=child.mode, gpus=child.gpus, jobs=entry.schedule(child.mode, child.gpus), input_sha256="sha")

            def fake_reports(output, spec):
                return {job["job_id"]: {} for job in spec["jobs"]}

            def start(command, *, env, **kwargs):
                nonlocal peak
                gpu = int(env["CUDA_VISIBLE_DEVICES"])
                self.assertNotIn(gpu, active)
                mode, output, job_id = command[3], Path(command[5]), command[-1]
                self.assertEqual(output, root / mode)
                self.assertEqual(gpu < 3, mode == "mvtecmode")
                self.assertNotEqual(job_id, "pyramiddrop_90")
                active.add(gpu)
                peak = max(peak, len(active))
                launched[gpu] += 1

                def finish():
                    active.remove(gpu)
                    entry.common.save_json(output / job_id / "metrics.json", dict(status="success", complete=True))

                return SimpleNamespace(poll=lambda: 0, wait=finish)

            real_run_trials = entry.run_trials
            with patch.object(entry, "setup", side_effect=fake_setup), patch.object(entry, "write_reports", side_effect=fake_reports), \
                    patch.object(entry, "run_trials", side_effect=lambda runs: real_run_trials(runs, process_factory=start)):
                self.assertEqual(entry.run(args), 0)
            self.assertEqual(peak, 6)
            self.assertEqual(launched, Counter({gpu: 8 for gpu in range(6)}))
            self.assertFalse(active)
