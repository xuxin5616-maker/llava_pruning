"""Check prefill-only measurement against the decoder's actual retained K/V."""

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
from llava_pruning.visa_comparison_methods import observe_prefill
from test_attention_predictor import make_checkpoint
from test_mmad_binary import fixture
from test_pool_ablations import model_and_options


class PrefillKVTests(unittest.TestCase):
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
                        job = next(j for j in entry.schedule(mode, 2) if j["method"] == method_name and j["prune_rate"] == rate)
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
                jobs = entry.schedule(mode, 3)
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
