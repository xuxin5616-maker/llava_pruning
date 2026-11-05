"""Letterbox geometry, real native packing, isolated GPU scheduling and reports."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import benchmark_visa_grid3 as entry
import benchmark_visa_methods as common
from llava_pruning.prompts import GENERIC_PROMPT_V0


class Grid3Tests(unittest.TestCase):
    def fixture(self, root):
        model = root / "model"; model.mkdir()
        common.save_json(model / "config.json", dict(num_hidden_layers=28))
        student = root / "student"; student.mkdir()
        checkpoint = student / "last.pt"; checkpoint.write_bytes(b"weights")
        common.save_json(student / "train_run.json", dict(original_visa_test_is_training_source=False))
        data = root / "VisA"; (data / "split_csv").mkdir(parents=True)
        rows = []
        for category in ("candle", "pcb3"):
            for label in ("Normal", "Anomaly"):
                relative = f"{category}/Data/Images/{label}/000.png"
                path = data / relative; path.parent.mkdir(parents=True)
                Image.new("RGB", (37, 21), "red").save(path)
                rows.append((category, "test", label.lower(), relative))
        with (data / "split_csv/1cls.csv").open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["object", "split", "label", "image"]); writer.writerows(rows)
        args = entry.parser().parse_args(["--model-path", str(model), "--data-root", str(data),
            "--checkpoint", str(checkpoint), "--resize-modes", "letterbox", "--output-dir", str(root / "output")])
        return args

    def test_letterbox_preserves_aspect_and_content_in_all_eight_visa_shapes(self):
        sizes = [(1274,1176),(1284,1168),(1300,1154),(1342,1118),(1358,1104),
                 (1404,1070),(1500,1000),(1562,960),(800,1200)]
        for size in sizes:
            source = Image.new("RGB", size, (200, 50, 20))
            output, geometry = entry.letterbox(source)
            self.assertEqual(output.size, (1152,1152))
            w,h = geometry["resized_size"]; x,y = geometry["paste_xy"]
            self.assertEqual(max(w,h),1152)
            self.assertLess(abs(w / h - size[0] / size[1]), .002)
            self.assertEqual(output.getpixel((x+w//2,y+h//2)),(200,50,20))
            self.assertEqual(source.size,size)
            if x: self.assertEqual(output.getpixel((0,0)),(0,0,0))
            if y: self.assertEqual(output.getpixel((0,0)),(0,0,0))

    def test_actual_siglip_pixels_share_base_and_exactly_nine_tiles(self):
        from llava.mm_utils import process_images
        from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
        processor = SigLipImageProcessor(size=(384,384), crop_size={"height":384,"width":384})
        source = Image.fromarray(np.random.RandomState(3).randint(0,256,(91,143,3),dtype=np.uint8))
        configs = [SimpleNamespace(image_aspect_ratio=mode, image_grid_pinpoints=[[1152,1152]],
                                   mm_patch_merge_type="spatial_unpad")
                   for mode in ("base_only","anyres_only","anyres_max_9")]
        for mode in ("letterbox","stretch"):
            image, _ = entry.resize_image(source,mode)
            base, anyres, full = [process_images([image],processor,c)[0] for c in configs]
            self.assertEqual(tuple(base.shape),(3,384,384))
            self.assertEqual(tuple(anyres.shape),(9,3,384,384))
            self.assertEqual(tuple(full.shape),(10,3,384,384))
            self.assertTrue(torch.equal(base,full[0]))
            self.assertTrue(torch.equal(anyres,full[1:]))

    def test_stretch_is_direct_bicubic_without_padding_and_differs_from_letterbox(self):
        source = Image.new("RGB",(150,100),(200,50,20))
        stretched, geometry = entry.resize_image(source,"stretch")
        padded, _ = entry.resize_image(source,"letterbox")
        self.assertEqual(stretched.size,(1152,1152))
        self.assertEqual(stretched.getpixel((0,0)),(200,50,20))
        self.assertEqual(padded.getpixel((0,0)),(0,0,0))
        np.testing.assert_array_equal(np.array(stretched),np.array(source.resize((1152,1152),Image.Resampling.BICUBIC)))
        self.assertEqual(geometry["resize_mode"],"stretch")
        self.assertEqual(geometry["resized_size"],[1152,1152])
        self.assertEqual(geometry["paste_xy"],[0,0])
        self.assertEqual(source.size,(150,100))
        with self.assertRaises(ValueError): entry.resize_image(source,"other")

    def test_native_llava_packing_lengths_and_base_identity(self):
        from test_reference import ToyPacking
        from llava.constants import IMAGE_TOKEN_INDEX
        packed = {}
        for mode, views, expected in (("base_only",1,729),("anyres_only",9,6642),("anyres_max_9",10,7371)):
            toy = ToyPacking(torch.float32)
            toy.config.image_aspect_ratio = mode
            toy.config.image_grid_pinpoints = [[1152,1152]]
            toy.config.tokenizer_model_max_length = 32768
            toy.tower.num_patches_per_side = 27; toy.tower.image_size = 384
            # A constant feature per crop makes Base duplication/omission visible.
            toy.encode_images = lambda pixels: pixels[:,0,0,0,None,None].expand(-1,729,4).clone()
            values = torch.arange(10 if mode != "base_only" else 1,dtype=torch.float32)
            if mode == "anyres_only": values = values[1:]
            pixels = values[:,None,None,None].expand(views,3,384,384)
            images = pixels[None] if mode != "base_only" else pixels
            result = toy.prepare_inputs_labels_for_multimodal(
                torch.tensor([[1,IMAGE_TOKEN_INDEX,2]]),None,None,None,None,images,image_sizes=[(1152,1152)])
            hidden = result[4]
            self.assertEqual(hidden.shape[1]-2,expected)
            packed[mode] = hidden[0,1:-1]
        self.assertTrue(torch.equal(packed["base_only"],packed["anyres_max_9"][:729]))
        self.assertTrue(torch.equal(packed["anyres_only"],packed["anyres_max_9"][729:]))

    def test_schedule_is_five_trials_and_tail_pair_share_one_gpu(self):
        jobs = entry.schedule([70.],[0,9],.7,[3,4])
        self.assertEqual(len(jobs),5)
        self.assertEqual([j["roi_mode"] for j in jobs[:3]],["base_only","anyres_only","anyres_max_9"])
        self.assertEqual([j["tail"] for j in jobs[3:]],[0,9])
        self.assertEqual({j["gpu"] for j in jobs[:3]},{3})
        self.assertEqual({j["gpu"] for j in jobs[3:]},{4})
        self.assertTrue(all(j["prune_rate"]==70 for j in jobs[3:]))

    def test_setup_uses_all_test_images_original_files_unchanged_and_resume_checks_cache(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.fixture(Path(temporary))
            original = {str(p):p.read_bytes() for p in Path(args.data_root).rglob("*.png")}
            output,spec = entry.setup(args)
            self.assertEqual(spec["sample_count"],4)
            self.assertEqual(spec["dataset_info"]["abnormal"],2)
            records = [json.loads(line) for line in Path(spec["input_json"]).read_text().splitlines()]
            self.assertTrue(all(r["text"]==GENERIC_PROMPT_V0 and r["image_size"]==[1152,1152] for r in records))
            self.assertEqual({str(p):p.read_bytes() for p in Path(args.data_root).rglob("*.png")},original)
            args.resume=True
            self.assertEqual(entry.setup(args),(output,spec))
            image = Path(spec["prepared_root"]) / records[0]["image"]
            Image.new("RGB",(1152,1152),"blue").save(image)
            with self.assertRaisesRegex(ValueError,"prepared image changed"):
                entry.setup(args)

    def test_two_resize_caches_have_same_ids_labels_but_different_pixels(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=self.fixture(root);specs={}
            for mode in ("letterbox","stretch"):
                args.resize_modes=[mode];args.output_dir=root/mode
                _,specs[mode]=entry.setup(args)
            records={mode:[json.loads(line) for line in Path(spec["input_json"]).read_text().splitlines()]
                     for mode,spec in specs.items()}
            keys=lambda rows:[(r["question_id"],r["gt"],r["text"]) for r in rows]
            self.assertEqual(keys(records["letterbox"]),keys(records["stretch"]))
            self.assertNotEqual(specs["letterbox"]["prepared_images_sha256"],specs["stretch"]["prepared_images_sha256"])
            for mode,spec in specs.items():
                entry.verify(spec)
                self.assertTrue(all(j["config"].startswith(f"resize={mode};") for j in spec["jobs"]))
                self.assertEqual(spec["transform"]["mode"],mode)

    def test_shared_measurement_checks_views_excludes_warmup_and_resumes(self):
        from llava_pruning.data import Sample
        layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(28)])
        tower = SimpleNamespace(num_patches_per_side=27,image_size=384,vision_tower=torch.nn.Identity())
        core = SimpleNamespace(layers=layers,get_vision_tower=lambda:tower,
            config=SimpleNamespace(image_grid_pinpoints=[[1152,1152]],mm_patch_merge_type="spatial_unpad"))
        calls = []
        class Backend:
            model=SimpleNamespace(get_model=lambda:core)
            def generate(self,sample,prompt,method,rate,mode,**options):
                calls.append((sample.sample_id,mode,options))
                tower.vision_tower(torch.empty((9,3,384,384)))
                for layer in layers: layer(torch.empty((1,6644,1)))
                return dict(answer="B",stats={},lat_seconds=.01,generation_seconds=.02,
                    generated_tokens=2,peak_allocated_bytes=100,peak_reserved_bytes=200)
        samples=[Sample(str(i),Path("unused"),"candle",str(i),GENERIC_PROMPT_V0,0,None,None,{}) for i in range(3)]
        job=entry.schedule([70],[0,9],.7,[3,4])[1]
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            output=Path(temporary)
            result=entry.evaluate(Backend(),object(),samples,dict(seed=42,warmup=1),job,output,1000)
            self.assertEqual(len(calls),4)
            self.assertEqual(result["evaluated_samples"],3)
            self.assertEqual(result["budget"]["original_image_tokens"],6642)
            self.assertEqual(result["budget"]["encoded_views"],9)
            self.assertTrue(all(c[1]=="anyres_only" and c[2]["benchmark_metrics"] for c in calls))
            self.assertTrue(all(not c[2]["do_sample"] and c[2]["include_pruning_time"] for c in calls))
            entry.evaluate(Backend(),object(),samples,dict(seed=42,warmup=1),job,output,1000)
            self.assertEqual(len(calls),4)
            self.assertFalse(tower.vision_tower._forward_pre_hooks)
            self.assertTrue(all(not layer._forward_pre_hooks for layer in layers))

    def test_summary_uses_new_square_full_baseline_and_eleven_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);jobs=entry.schedule([70],[0,9],.7,[3,4])
            spec=dict(jobs=jobs,input_sha256="test")
            for job in jobs:
                folder=output/job["job_id"];folder.mkdir()
                common.save_json(folder/"metrics.json",dict(job_id=job["job_id"],input_sha256="test",
                    status="success",complete=True,accuracy=.8 if job["job_id"]=="base_anyres" else .6,
                    tp=3,tn=3,fp=2,fn=2,lat_mean_ms=20,mem_peak_allocated_gib=2,budget={}))
            entry.write_reports(output,spec)
            with (output/"summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(tuple(reader.fieldnames),common.FIELDS)
            self.assertEqual(len(rows),5)
            self.assertEqual(float(rows[2]["relative_accuracy_pct"]),100)
            self.assertAlmostEqual(float(rows[0]["relative_accuracy_pct"]),75)

    def test_pair_merged_ratios_keep_independent_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for mode,baseline in (("letterbox",.8),("stretch",.5)):
                output=root/mode;output.mkdir();jobs=entry.schedule([70],[0,9],.7,[3])
                for job in jobs:
                    job["config"]=f"resize={mode};"+job["config"]
                    folder=output/job["job_id"];folder.mkdir()
                    common.save_json(folder/"metrics.json",dict(job_id=job["job_id"],input_sha256="test",
                        status="success",complete=True,accuracy=baseline if job["job_id"]=="base_anyres" else .6,
                        tp=3,tn=3,fp=2,fn=2,lat_mean_ms=20,mem_peak_allocated_gib=2,budget={}))
                entry.write_reports(output,dict(jobs=jobs,input_sha256="test"))
            entry.merge_resize_reports(root,["letterbox","stretch"])
            with (root/"summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),10)
            self.assertAlmostEqual(float(rows[0]["relative_accuracy_pct"]),75)
            self.assertAlmostEqual(float(rows[5]["relative_accuracy_pct"]),120)
            self.assertEqual(float(rows[2]["relative_accuracy_pct"]),100)
            self.assertEqual(float(rows[7]["relative_accuracy_pct"]),100)

    def test_pair_starts_two_single_gpu_controllers_and_preserves_all_settings(self):
        events=[]
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=self.fixture(Path(temporary));args.resize_modes=["letterbox","stretch"]
            output=args.output_dir
            class Process:
                def __init__(self,command,**options):
                    self.mode=command[command.index("--resize-modes")+1]
                    self.folder=Path(command[command.index("--output-dir")+1])
                    events.append(("spawn",self.mode,command,options));self.polled=False
                def poll(self):
                    events.append(("poll",self.mode))
                    if not self.polled:
                        self.polled=True;return None
                    self.folder.mkdir(exist_ok=True)
                    rows=[dict(config=f"resize={self.mode};trial={i}",acc=50,relative_accuracy_pct=100) for i in range(5)]
                    common.save_csv(self.folder/"summary.csv",rows,common.FIELDS)
                    common.save_csv(self.folder/"budget_summary.csv",[],common.BUDGET_FIELDS)
                    common.save_json(self.folder/"run.json",dict(resize_mode=self.mode))
                    return 0
            with patch.object(entry.time,"sleep"):
                self.assertEqual(entry.run(args,process_factory=Process),0)
            self.assertEqual([e[0] for e in events[:2]],["spawn","spawn"])
            starts=[e for e in events if e[0]=="spawn"]
            self.assertEqual(len(starts),2)
            for start,gpu in zip(starts,("3","4")):
                command,options=start[2:]
                self.assertEqual(command[command.index("--gpus")+1],gpu)
                self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"],gpu)
                self.assertEqual(command[command.index("--prune-rates")+1],"70.0")
                self.assertEqual(command[command.index("--distill-share")+1],"0.7")
                index=command.index("--tails");self.assertEqual(command[index+1:index+3],["0","9"])
            with (output/"summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),10)
            self.assertTrue(json.loads((output/"run.json").read_text(encoding="utf-8"))["complete"])
            args.resume=True;args.gpus=[7];events.clear()
            with patch.object(entry.time,"sleep"):
                self.assertEqual(entry.run(args,process_factory=Process),0)
            resumed=[e for e in events if e[0]=="spawn"]
            self.assertEqual(len(resumed),2)
            self.assertTrue(all("--resume" in e[2] for e in resumed))
            self.assertEqual([e[3]["env"]["CUDA_VISIBLE_DEVICES"] for e in resumed],["3","4"])

    def test_two_workers_start_before_poll_and_never_share_a_gpu(self):
        events=[]
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            output=Path(temporary)
            spec=dict(gpus=[3,4],sample_count=2,input_sha256="test",jobs=entry.schedule([70],[0,9],.7,[3,4]))
            class Process:
                def __init__(self,command,**options):
                    self.job=next(j for j in spec["jobs"] if j["job_id"]==command[command.index("--worker")+1])
                    events.append(("spawn",self.job["job_id"],options["env"]["CUDA_VISIBLE_DEVICES"]))
                    self.polled=False
                def poll(self):
                    events.append(("poll",self.job["job_id"]))
                    if not self.polled:
                        self.polled=True;return None
                    folder=output/self.job["job_id"];folder.mkdir(exist_ok=True)
                    common.save_json(folder/"metrics.json",dict(job_id=self.job["job_id"],input_sha256="test",
                        status="success",complete=True,accuracy=.5,tp=1,tn=0,fp=1,fn=0,
                        lat_mean_ms=10,mem_peak_allocated_gib=1,budget={}))
                    return 0
            with patch.object(entry,"setup",return_value=(output,spec)),patch.object(entry.time,"sleep"):
                self.assertEqual(entry.run(object(),process_factory=Process),0)
            self.assertEqual([e[0] for e in events[:2]],["spawn","spawn"])
            starts=[e for e in events if e[0]=="spawn"]
            self.assertEqual(len(starts),5)
            self.assertTrue(all(e[2]=="4" for e in starts if e[1].startswith("hybrid")))


if __name__ == "__main__": unittest.main()
