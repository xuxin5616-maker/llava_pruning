"""Check prefill-only measurement against the decoder's actual retained K/V."""

from collections import Counter
from contextlib import nullcontext
import csv
from pathlib import Path
from statistics import pstdev
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
    def test_tau_sweep_changes_only_crop_quota_distribution(self):
        for mode in ("mvtecmode", "mmadmode"):
            model, _ = model_and_options(28)
            core = model.get_model()
            core.config.image_grid_pinpoints = [[1152, 1152]]
            size = (1152, 1152)
            scores = torch.linspace(.9, 1.2, 9).numpy()
            spec = dict(mode=mode, checkpoint={"path": "best.pt"})
            jobs = entry.schedule(mode, [0, 1], tau_sweep=True)
            controls = entry.schedule(mode, [0, 1])
            self.assertEqual(len(jobs), 15)
            for rate in entry.RATES:
                group = [job for job in jobs if job["prune_rate"] == rate]
                original = next(job for job in controls if job["method"] == "hybrid_baseline" and job["prune_rate"] == rate)
                self.assertEqual(tuple(job["tau"] for job in group), entry.TAU_VALUES)
                values = []
                for job in (original, *group):
                    with self.subTest(mode=mode, rate=rate, tau=job.get("tau")), \
                            patch.object(entry, "student_identity", return_value=spec["checkpoint"]):
                        method = entry.build_method(SimpleNamespace(model=model), [], spec, job, Path("."))
                        adapter = method.adapter
                        adapter.set_image_context(size)
                        allocation, _, dominant, contextual = adapter.allocate_view_budgets(
                            scores, adapter.prepared, dict(spatial_prior=None))
                        quotas = [crop["keep"] for crop in allocation["crops"]]
                        self.assertGreaterEqual(min(quotas), 4)
                        self.assertEqual(adapter.settings.allocation_temperature, job.get("tau", .05))
                        values.append((dominant + contextual, sum(quotas), adapter.plan["final_llm_tokens"], quotas))
                self.assertEqual(len({value[:3] for value in values}), 1)
                self.assertEqual(values[0], values[3])  # tau=.05
                self.assertGreater(pstdev(values[1][3]), pstdev(values[-1][3]))

    def test_all_ablations_share_dataset_manifests_and_keep_separate_tables(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = fixture(root)
            model = Path(paths.model_path)
            entry.common.save_json(model / "config.json", dict(num_hidden_layers=28, image_grid_pinpoints=[[1152, 1152]]))
            mvtec = root / "input.jsonl"
            Image.new("RGB", (128, 384)).save(root / "01.png")
            mvtec.write_text('{"question_id":"1","image":"01.png","category":"bottle","gt":0}\n', encoding="utf-8")
            output = root / "all"
            command = ["both", "--all-ablations", "--gpus", *map(str, range(6)), "--output-dir", str(output),
                       "--model-path", str(model), "--checkpoint", paths.checkpoint, "--data-root", str(root),
                       "--mmad-data-root", paths.data_root, "--input-json", str(mvtec),
                       "--mvtec-root", str(Path(paths.data_root) / "MVTec-AD")]

            def inspect(runs):
                self.assertEqual(len(runs), 2)
                for folder, spec in runs:
                    self.assertEqual(spec["gpus"], list(range(6)))
                    self.assertEqual({job["gpu"] for job in spec["jobs"]}, set(range(6)))
                    self.assertEqual(len({job["job_id"] for job in spec["jobs"]}), 81)
                    entry.write_reports(folder, spec)
                    for index, count, key in ((1, 21, "raba_strategy"), (2, 9, "adtp_score"),
                                              (3, 21, "distill_share"), (4, 15, "kmin"), (5, 15, "tau")):
                        task = f"task{index}"
                        with (folder / task / "summary.csv").open(encoding="utf-8-sig") as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual(len(rows), count)
                        self.assertTrue(all(row[key] for row in rows))
                        for rate in entry.RATES:
                            self.assertEqual(len({job["gpu"] for job in spec["jobs"]
                                                  if job["task"] == task and job["prune_rate"] == rate}), 1)
                    self.assertTrue((folder / "task1" / "raba_summary.csv").is_file())
                return 0

            with patch.object(entry, "run_trials", side_effect=inspect):
                self.assertEqual(entry.run(entry.parser().parse_args(command)), 0)
                self.assertEqual(entry.run(entry.parser().parse_args([*command, "--resume"])), 0)
            self.assertEqual(list(output.rglob("prepared_images")), [output / "mmadmode" / "prepared_images"])

    @torch.inference_mode()
    def test_kmin_sweep_respects_each_crop_floor_with_fixed_hybrid_budgets(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = folder / "best.pt"
            for mode in ("mvtecmode", "mmadmode"):
                jobs = entry.schedule(mode, [1, 2], kmin_sweep=True)
                self.assertEqual(len({job["job_id"] for job in jobs}), 15)
                for rate in (50, 70, 90):
                    self.assertEqual(tuple(job["kmin"] for job in jobs if job["prune_rate"] == rate), entry.KMIN_VALUES)
                model, options = model_and_options(28)
                core = model.get_model()
                core.config.image_grid_pinpoints = [[768, 768]]
                core.vision_tower = _Tower()
                core.vision_tower.features = core.vision_tower.features[:5]
                size = (768, 768)
                options.update(images=[torch.zeros(5, 3, 384, 384)], image_sizes=[size], max_new_tokens=1)
                make_checkpoint(checkpoint, core)
                spec = dict(mode=mode, checkpoint={"path": str(checkpoint)})
                original = next(job for job in entry.schedule(mode, [1])
                                if job["method"] == "hybrid_baseline" and job["prune_rate"] == 90)
                expected = None
                for job in (original, *jobs[-5:]):
                    with self.subTest(mode=mode, kmin=job.get("kmin")), \
                            patch.object(entry, "student_identity", return_value=spec["checkpoint"]):
                        method = entry.build_method(SimpleNamespace(model=model), [], spec, job, folder)
                        method.configure(core, 90)
                        method.prepare_inputs(size, options["inputs"])
                        allocate = method.adapter.allocate_view_budgets
                        # Three crops can absorb the budget; almost no mass goes to the fourth.
                        scores = torch.tensor([0., 0., 0., 2.]).numpy()
                        with method.generation_context(model), patch.object(
                                method.adapter, "allocate_view_budgets", side_effect=lambda raw, first, spatial: allocate(scores, first, spatial)):
                            result = model.generate(**options)
                        keeps = [crop["keep"] for crop in method.adapter.allocation["crops"]]
                        self.assertEqual(keeps[-1], job.get("kmin", 4))
                        stats = method.stats(core, 90)
                        counts = tuple(stats[key] for key in ("pre_base_tokens", "pre_anyres_tokens", "stage2_llm_visual_tokens"))
                        if expected is None:
                            expected = counts, keeps, result.sequences
                        else:
                            self.assertEqual(counts, expected[0])
                            if job["kmin"] == 4:
                                self.assertEqual(keeps, expected[1])
                                torch.testing.assert_close(result.sequences, expected[2], rtol=0, atol=0)

    @torch.inference_mode()
    def test_lambda_sweep_endpoints_disable_the_correct_stage_and_keep_tail(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = folder / "best.pt"
            for mode in ("mvtecmode", "mmadmode"):
                jobs = entry.schedule(mode, [1, 2], lambda_sweep=True)
                self.assertEqual(len({job["job_id"] for job in jobs}), 21)
                for rate in (50, 70, 90):
                    self.assertEqual(tuple(j["distill_share"] for j in jobs if j["prune_rate"] == rate), entry.LAMBDA_SHARES)
                model, options = model_and_options(28)
                core = model.get_model()
                core.config.image_grid_pinpoints = [[768, 384]]
                core.vision_tower = _Tower()
                core.vision_tower.features = core.vision_tower.features[:3]
                size = (700, 300)
                options.update(images=[torch.zeros(3, 3, 384, 384)], image_sizes=[size])
                make_checkpoint(checkpoint, core)
                spec = dict(mode=mode, checkpoint={"path": str(checkpoint)})
                for job in jobs[-7:]:
                    share = job["distill_share"]
                    with self.subTest(mode=mode, share=share), patch.object(entry, "student_identity", return_value=spec["checkpoint"]):
                        method = entry.build_method(SimpleNamespace(model=model), [], spec, job, folder)
                        method.configure(core, 90)
                        self.assertEqual(core.attention_predictor is None, share == 0)
                        method.prepare_inputs(size, options["inputs"])
                        with method.generation_context(model), patch.object(
                                method.adapter, "encode_images", wraps=method.adapter.encode_images) as pre_stage, \
                                patch.object(method.adapter, "rank", wraps=method.adapter.rank) as post_stage:
                            model.generate(**options)
                        self.assertEqual(pre_stage.call_count, int(share != 1))
                        self.assertEqual(post_stage.call_count, int(share != 0))
                        stats = method.stats(core, 90)
                        pre, final = stats["pre_llm_tokens"], stats["stage2_llm_visual_tokens"]
                        self.assertAlmostEqual(stats["two_stage_average_prune_rate"], 90, delta=.08)
                        self.assertAlmostEqual(stats["actual_pre_contribution"], (1-share) * 90, delta=.08)
                        self.assertAlmostEqual(stats["actual_distill_contribution"], share * 90, delta=.08)
                        rows = core.get_clse_stats()["layers"]
                        self.assertEqual([row["image_tokens_in"] for row in rows], [pre] * 2 + [final] * 18 + [0] * 8)
                        if share == 1 and mode == "mvtecmode":
                            self.assertLess(stats["stage2_base_tokens"], stats["pre_base_tokens"])

    @torch.inference_mode()
    def test_adtp_sources_share_the_hybrid_budget_and_match_copied_initialization(self):
        from llava.model.language_model.attention_predictor import CopiedAttentionPredictor, backbone_signature
        from llava.model.language_model.fastv_attention import last_prompt_attention
        from llava_pruning.predictor_training import save_student

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = folder / "best.pt"
            for mode in ("mvtecmode", "mmadmode"):
                model, options = model_and_options(28)
                core = model.get_model()
                student = CopiedAttentionPredictor.from_backbone(core)
                student.layers[-1].self_attn.q_proj.weight.add_(1.)
                distilled = student.layers[-1].self_attn.q_proj.weight.clone()
                save_student(checkpoint, student, student_precision="float32", backbone_depth=28,
                             teacher_layer=22, signature=backbone_signature(core.config))
                spec = dict(mode=mode, checkpoint={"path": str(checkpoint)})
                jobs = entry.schedule(mode, [1, 2], adtp_ablation=True)
                self.assertEqual(len(jobs), 9)
                for rate in (50, 70, 90):
                    selected = [job for job in jobs if job["prune_rate"] == rate]
                    self.assertEqual(tuple(job["adtp_score"] for job in selected), entry.ADTP_SOURCES)
                    self.assertEqual(len({job["gpu"] for job in selected}), 1)
                budgets = []
                for job in jobs[-3:]:
                    source = job["adtp_score"]
                    with self.subTest(mode=mode, source=source), patch.object(entry, "student_identity", return_value=spec["checkpoint"]):
                        method = entry.build_method(SimpleNamespace(model=model), [], spec, job, folder)
                        method.configure(core, 90)
                        if source == "native_shallow":
                            self.assertIsNone(core.attention_predictor)
                        else:
                            branch = core.attention_predictor
                            expected = (core.layers[4].self_attn.q_proj.weight if source == "copied_initial" else distilled)
                            torch.testing.assert_close(branch.layers[-1].self_attn.q_proj.weight, expected, atol=0, rtol=0)
                            if source == "copied_initial":
                                method.configure(core, 90)
                                self.assertIs(core.attention_predictor, branch)
                        method.prepare_inputs((384, 384), options["inputs"])
                        with method.generation_context(model), patch(
                                "llava.model.language_model.fastv_attention.last_prompt_attention", wraps=last_prompt_attention) as native, \
                                patch.object(method.adapter, "map_ranking", wraps=method.adapter.map_ranking) as mapping:
                            model.generate(**options)
                        app = method.adapter
                        mapped_input = mapping.call_args.args[0]
                        self.assertEqual(len(mapped_input), len(app.base if mode == "mvtecmode" else app.spatial))
                        if source == "native_shallow":
                            native.assert_called_once()
                            self.assertIs(native.call_args.args[0], core.layers[1])
                        else:
                            native.assert_not_called()
                        stats = method.stats(core, 90)
                        budgets.append(tuple(stats[key] for key in ("pre_base_tokens", "pre_llm_tokens", "stage2_llm_visual_tokens")))
                self.assertEqual(len(set(budgets)), 1)

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
