"""Matched stretched inputs, six-method schedule, original workers and resume."""

from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import benchmark_visa_stretch as entry
import benchmark_visa_methods as common
from llava_pruning.prompts import GENERIC_PROMPT_V0


class StretchComparisonTests(unittest.TestCase):
    def fixture(self, root):
        model=root/"model";model.mkdir()
        common.save_json(model/"config.json",dict(num_hidden_layers=28,
            image_grid_pinpoints=[[384*w,384*h] for w in range(1,7) for h in range(1,7)]))
        raw=root/"VisA";(raw/"split_csv").mkdir(parents=True)
        rows=[]
        for category in ("candle","pcb3"):
            for label in ("Normal","Anomaly"):
                relative=f"{category}/Data/Images/{label}/000.png"
                path=raw/relative;path.parent.mkdir(parents=True)
                Image.new("RGB",(39,21),(200,50,20)).save(path)
                rows.append([category,"test",label.lower(),relative])
        rows.append(["candle","train","normal","candle/Data/Images/Normal/not_used.png"])
        with (raw/"split_csv/1cls.csv").open("w",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["object","split","label","image"]);writer.writerows(rows)
        return entry.parser().parse_args(["--model-path",str(model),"--data-root",str(raw),
                                          "--output-dir",str(root/"output")])

    def test_six_methods_only_18_valid_trials_and_same_gpu_for_each_rate_curve(self):
        jobs=entry.schedule([50,70,90],[3,4])
        self.assertEqual(len(jobs),19)
        self.assertEqual(sum(j["status"]=="pending" for j in jobs),18)
        self.assertEqual({j["method"] for j in jobs},{"llava_unpruned",*entry.METHODS})
        self.assertEqual([j["job_id"] for j in jobs if j["status"]=="unreachable"],["pyramiddrop_90"])
        for method in entry.METHODS:
            self.assertEqual(len({j["gpu"] for j in jobs if j["method"]==method}),1)
        self.assertEqual([sum(j["status"]=="pending" and j["gpu"]==gpu for j in jobs) for gpu in (3,4)],[9,9])
        self.assertTrue(all("lambda" not in j["config"] and "tail=" not in j["config"] for j in jobs))

    def test_all_test_images_are_stretched_without_padding_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=self.fixture(Path(temporary))
            raw={str(p):p.read_bytes() for p in Path(args.data_root).rglob("*.png")}
            config=(Path(args.model_path)/"config.json").read_bytes()
            output,spec=entry.setup(args)
            self.assertEqual(spec["sample_count"],4)
            self.assertIsNone(spec["checkpoint"])
            self.assertNotIn("runtime_source",spec)
            self.assertFalse((output/"_runtime").exists())
            self.assertEqual(spec["source_policy"],"current_directory")
            self.assertEqual(spec["source_identity"],common.source_identity())
            self.assertEqual(spec["dataset_info"]["normal"],2)
            self.assertEqual(spec["dataset_info"]["abnormal"],2)
            rows=[json.loads(line) for line in Path(spec["input_json"]).read_text(encoding="utf-8").splitlines()]
            for row in rows:
                self.assertEqual(row["text"],GENERIC_PROMPT_V0)
                self.assertEqual(row["image_size"],[1152,1152])
                self.assertEqual(row["transform"]["resize_mode"],"stretch")
                with Image.open(Path(spec["data_root"])/row["image"]) as image:
                    self.assertEqual(image.size,(1152,1152))
                    self.assertEqual(image.getpixel((0,0)),(200,50,20))
                    np.testing.assert_array_equal(np.array(image),np.full((1152,1152,3),(200,50,20),dtype=np.uint8))
            self.assertEqual((Path(args.model_path)/"config.json").read_bytes(),config)
            self.assertEqual({str(p):p.read_bytes() for p in Path(args.data_root).rglob("*.png")},raw)
            args.resume=True
            self.assertEqual(entry.setup(args),(output,spec))
            Image.new("RGB",(1152,1152),"blue").save(Path(spec["data_root"])/rows[0]["image"])
            with self.assertRaisesRegex(ValueError,"pixels changed"): entry.setup(args)

    def test_native_counts_and_budgets_use_7371_not_old_original_image_geometry(self):
        from llava_pruning.visa_comparison_methods import native_counts,single_drop_keep
        core=SimpleNamespace(config=SimpleNamespace(image_grid_pinpoints=[[384*w,384*h] for w in range(1,7) for h in range(1,7)],
            mm_patch_merge_type="spatial_unpad"),get_vision_tower=lambda:SimpleNamespace(image_size=384,num_patches_per_side=27))
        grid,counts=native_counts(core,(1152,1152))
        self.assertEqual(tuple(grid),(3,3))
        self.assertEqual(counts,(729,6561,81))
        for rate in (50,70,90):
            keep=single_drop_keep(sum(counts),rate,28,2)
            actual=100*(1-(2*7371+26*keep)/(28*7371))
            self.assertAlmostEqual(actual,rate,delta=.02)

    def test_reports_use_new_measured_stretch_reference_and_keep_unreachable_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);jobs=entry.schedule([50,70,90],[3,4]);spec=dict(jobs=jobs)
            for job in jobs:
                if job["status"]!="pending":continue
                folder=output/job["job_id"];folder.mkdir()
                common.save_json(folder/"metrics.json",dict(job_id=job["job_id"],status="success",complete=True,
                    accuracy=.8 if job["method"]=="llava_unpruned" else .6,tp=3,tn=3,fp=2,fn=2,
                    lat_mean_ms=10,mem_peak_allocated_gib=2,budget={}))
            common.write_reports(output,spec)
            with (output/"summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(tuple(reader.fieldnames),common.FIELDS)
            self.assertEqual(len(rows),19)
            self.assertEqual(float(rows[0]["relative_accuracy_pct"]),100)
            self.assertAlmostEqual(float(rows[1]["relative_accuracy_pct"]),75)
            unavailable=next(r for r in rows if "method=pyramiddrop;prune=90%" in r["config"])
            self.assertEqual(unavailable["acc"],"")
            self.assertIn("unreachable",unavailable["config"])

    def test_parallel_scheduler_uses_original_workers_and_resumes_without_relaunch(self):
        events=[]
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            output=Path(temporary)
            spec=dict(gpus=[3,4],sample_count=2,checkpoint=None,input_sha256="test",jobs=entry.schedule([50,70,90],[3,4]))
            class Process:
                def __init__(self,command,**options):
                    self.job=command[command.index("--worker")+1];self.polled=False
                    events.append(("spawn",self.job,command,options))
                def poll(self):
                    events.append(("poll",self.job))
                    if not self.polled:self.polled=True;return None
                    folder=output/self.job;folder.mkdir()
                    common.save_json(folder/"metrics.json",dict(job_id=self.job,status="success",complete=True,
                        input_sha256="test",accuracy=1,tp=1,tn=1,fp=0,fn=0,lat_mean_ms=10,mem_peak_allocated_gib=2))
                    return 0
            with patch.object(entry,"setup",return_value=(output,spec)),patch.object(entry.time,"sleep"):
                self.assertEqual(entry.run(None,process_factory=Process),0)
                self.assertEqual(entry.run(None,process_factory=Process),0)
            self.assertEqual([e[0] for e in events[:2]],["spawn","spawn"])
            starts=[e for e in events if e[0]=="spawn"]
            self.assertEqual(len(starts),18)
            self.assertTrue(all(e[2][2].endswith("benchmark_visa_methods.py") for e in starts))
            self.assertFalse(any("pyramiddrop_90" in e[1] for e in starts))
            self.assertFalse(any("hybrid" in e[1] for e in starts))
            self.assertEqual({e[3]["env"]["CUDA_VISIBLE_DEVICES"] for e in starts},{"3","4"})

    def test_new_manifest_runs_through_original_worker_for_all_six_methods(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=self.fixture(Path(temporary));args.prune_rates=[50]
            output,spec=entry.setup(args)
            config=json.loads((Path(args.model_path)/"config.json").read_text(encoding="utf-8"))
            core=SimpleNamespace(layers=[None]*28,config=SimpleNamespace(**config,mm_patch_merge_type="spatial_unpad"),
                get_vision_tower=lambda:SimpleNamespace(image_size=384,num_patches_per_side=27))
            calls=[]
            class Backend:
                def __init__(self,*args,**kwargs):
                    self.model=SimpleNamespace(get_model=lambda:core)
                    self.inference_config=dict(model_dtype="torch.float16",attention="flash_attention_2")
            def evaluate(backend,method,samples,saved,job,folder,total_memory):
                calls.append(job["method"])
                self.assertEqual(len(samples),4)
                self.assertTrue(all(s.source["image_size"]==[1152,1152] and s.text==GENERIC_PROMPT_V0 for s in samples))
                self.assertTrue(all(s.image.is_relative_to(Path(spec["data_root"])) for s in samples))
                return dict(complete=True,evaluated_samples=4,accuracy=.75,tp=1,tn=2,fp=0,fn=1,
                            lat_mean_ms=10,mem_peak_allocated_gib=1,budget={})
            with patch("llava_pruning.backend.LlavaBackend",Backend),\
                 patch.object(common,"validate_single_gpu"),\
                 patch("llava_pruning.visa_comparison_methods.make_method",return_value=object()),\
                 patch.object(common,"evaluate_samples",side_effect=evaluate),\
                 patch("torch.cuda.get_device_properties",return_value=SimpleNamespace(name="CPU-test",total_memory=2**30)):
                for job in spec["jobs"]:
                    self.assertEqual(common.worker(output/"run.json",job["job_id"]),0)
                    saved=json.loads((output/job["job_id"]/"metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(saved["source_policy"],"current_directory")
                    self.assertEqual(saved["source_identity"],common.source_identity())
            self.assertEqual(set(calls),{"llava_unpruned",*entry.METHODS})

    def test_failure_is_printed_and_saved_without_discarding_other_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);captured=io.StringIO()
            spec=dict(gpus=[3],sample_count=2,checkpoint=None,input_sha256="test",jobs=entry.schedule([50],[3]))
            class Process:
                def __init__(self,command,**options):
                    self.job=command[command.index("--worker")+1]
                    if self.job=="visionzip_50":
                        options["stdout"].write("Error: test worker initialization failed\n")
                def poll(self):
                    if self.job=="visionzip_50":return 1
                    folder=output/self.job;folder.mkdir()
                    common.save_json(folder/"metrics.json",dict(job_id=self.job,status="success",complete=True,
                        input_sha256="test",accuracy=1,tp=1,tn=1,fp=0,fn=0,lat_mean_ms=10,mem_peak_allocated_gib=2))
                    return 0
            with patch.object(entry,"setup",return_value=(output,spec)),patch.object(entry.time,"sleep"),redirect_stdout(captured):
                self.assertEqual(entry.run(None,process_factory=Process),1)
            self.assertIn("FAILED visionzip_50",captured.getvalue())
            self.assertIn("test worker initialization failed",captured.getvalue())
            saved=json.loads((output/"run.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["complete"])
            self.assertEqual(saved["completed_trials"],6)
            self.assertEqual(saved["failed_trials"],["visionzip_50"])

    def test_fresh_process_setup_needs_no_vendor_pythonpath_or_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary))
            code="""
import importlib.util,sys
assert importlib.util.find_spec('llava') is None
import benchmark_visa_stretch as entry
args=entry.parser().parse_args(sys.argv[1:])
output,spec=entry.setup(args)
entry.verify(spec)
import torch
assert not torch.cuda.is_initialized()
assert spec['sample_count']==4 and spec['checkpoint'] is None
print('STRETCH_SETUP_OK')
"""
            env=dict(os.environ,PYTHONPATH=str(entry.PROJECT/".test-deps"),PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8")
            result=subprocess.run([sys.executable,"-B","-c",code,"--model-path",args.model_path,
                "--data-root",args.data_root,"--output-dir",str(args.output_dir)],cwd=entry.PROJECT,
                env=env,capture_output=True,text=True,encoding="utf-8",timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("STRETCH_SETUP_OK",result.stdout)


if __name__=="__main__": unittest.main()
