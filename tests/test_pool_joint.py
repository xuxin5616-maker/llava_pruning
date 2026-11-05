"""Joint spatial ranking, real decode/cache, observed groups and isolated VisA jobs."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import runMyPruningPoolSpatial as pool
import run_pool_distill as cli
import benchmark_visa_joint as entry
import benchmark_visa_methods as common
import benchmark_visa_grid3 as grid3
from llava_pruning.pool_ablation_report import compact_row
from llava_pruning.pool_distill import PoolDistillAdapter,PoolDistillMethod
from llava_pruning.pool_distill_budget import hybrid_budget,pre_view_budget,stage2_view_budget
from llava_pruning.pool_distill_parallel import worker_command
from test_attention_predictor import make_checkpoint
from test_pool_ablations import model_and_options
from test_visa_grid3 import Grid3Tests


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_joint_topk_has_no_base_quota_and_preserves_newlines_and_order(self):
        adapter=object.__new__(PoolDistillAdapter)
        adapter.spatial=list(range(6));adapter.base=[0,1,2];adapter.anyres=[3,4,5]
        adapter.protected=[6,7];adapter.records=[{}]*8;adapter.plan=dict(final_llm_tokens=5)
        adapter.base_policy="joint"
        selected,_,_=adapter.select(torch.tensor([.1,.9,.2,.8,.3,.7]),adapter.spatial,8)
        self.assertEqual(selected.tolist(),[1,3,5,6,7])
        self.assertEqual(int(adapter.stage2_counts["base_keep"]),1)
        self.assertEqual(int(adapter.stage2_counts["anyres_keep"]),2)
        self.assertTrue(torch.is_tensor(adapter.stage2_counts["base_keep"]))
        selected,_,_=adapter.select(torch.tensor([0.,0.,0.,3.,2.,1.]),adapter.spatial,8)
        self.assertEqual(selected.tolist(),[3,4,5,6,7])
        self.assertEqual(int(adapter.stage2_counts["base_keep"]),0)

    def test_all_nine_packed_budgets_stay_exact_without_fabricated_group_counts(self):
        for rate in (50,70,90):
            for share in (.7,.9,.99):
                plan=dict(hybrid_budget(7371,rate,share),raw_grid_height=3)
                pre=pre_view_budget(plan,9,base_policy="joint")
                post=stage2_view_budget(plan["final_llm_tokens"],pre["base_keep"],81,"joint")
                self.assertIsNone(post["base_keep"])
                self.assertIsNone(post["anyres_keep"])
                self.assertEqual(post["spatial_keep"]+81,plan["final_llm_tokens"])
                self.assertGreater(post["spatial_keep"],0)
                self.assertAlmostEqual(plan["actual_average_prune_rate"],rate,delta=.03)

    def test_real_joint_generation_and_decode_keeps_exact_topk_and_observed_counts(self):
        cases=[(50,.9,0,False),(70,.7,9,False),(90,.99,0,False),(90,.99,9,False),(90,.9,0,True)]
        for rate,share,tail,force_anyres in cases:
            torch.manual_seed(1801)
            model,options=model_and_options(28);core=model.get_model();lengths=[]
            handles=[layer.register_forward_pre_hook(lambda m,a,k:lengths.append((a[0].shape[1],k["position_ids"].clone())),
                                                     with_kwargs=True) for layer in core.layers]
            with tempfile.TemporaryDirectory() as temporary,torch.no_grad():
                checkpoint=Path(temporary)/"best.pt";make_checkpoint(checkpoint,core)
                method=PoolDistillMethod(core,checkpoint,rate=rate,share=share,mode="base_full",
                    settings=pool.MethodSettings(prune_rate=0),predictor_input="base_text",base_mapping="fast",
                    tail_drop_layers=tail,base_policy="joint")
                method.configure(core,rate);method.prepare_inputs((384,384),options["inputs"])
                if force_anyres:
                    method.adapter.rank=lambda hidden,start,ids:torch.arange(len(method.adapter.spatial),dtype=torch.float32)
                original=method.adapter.select;observed={}
                def checked(ranking,spatial,total):
                    selected=original(ranking,spatial,total)
                    keep=method.adapter.plan["final_llm_tokens"]-len(method.adapter.protected)
                    order=torch.argsort(ranking,descending=True,stable=True)[:keep]
                    expected=torch.cat((torch.tensor(spatial)[order],torch.tensor(method.adapter.protected))).sort().values
                    self.assertTrue(torch.equal(selected[0],expected))
                    observed["base"]=len(set(expected.tolist())&set(method.adapter.base))
                    observed["anyres"]=len(set(expected.tolist())&set(method.adapter.anyres))
                    return selected
                with patch.object(method.adapter,"select",side_effect=checked),method.generation_context(model):
                    generated=model.generate(**options)
                stats=method.stats(core,rate)
                pre,final=stats["pre_llm_tokens"],stats["final_llm_tokens"]
                self.assertEqual(generated.sequences.shape[-1],3)
                self.assertEqual(stats["stage2_base_policy"],"joint")
                self.assertEqual(stats["stage2_base_tokens"],observed["base"])
                self.assertEqual(stats["stage2_anyres_tokens"],observed["anyres"])
                self.assertEqual(stats["stage2_base_tokens"]+stats["stage2_anyres_tokens"]+stats["protected_newlines"],final)
                self.assertEqual(stats["observed_layer_token_sum"],2*pre+(26-tail)*final)
                self.assertEqual([n for n,_ in lengths[:28]],[pre+4]*2+[final+4]*(26-tail)+[4]*tail)
                self.assertEqual(lengths[27][1][0,-1].item(),pre+3)
                self.assertEqual(stats["final_base_tokens"],0 if tail else observed["base"])
                if force_anyres: self.assertEqual(observed["base"],0)
                with self.assertRaisesRegex(ValueError,"cannot reconstruct"):
                    method.recover_legacy_budget((384,384))
            for handle in handles: handle.remove()

    def test_public_cli_and_compact_rows_record_joint_policy(self):
        args=cli.build_parser().parse_args(["--base-policy","joint","--lambdas","0.7","0.9","0.99"])
        command=worker_command(args,cli,90,.99,"base_full",5,Path("unused"))
        self.assertEqual(cli.build_parser().parse_args(command[2:]).base_policy,"joint")
        row=dict(prune_rate=70,distill_share=.7,selection_mode="base_full",predictor_input="base_text",tail_drop_layers=9,
            base_mapping="fast",stage2_base_policy="joint",accuracy=.8,tp=8,tn=8,fp=2,fn=2,
            lat_mean_ms=10,mem_peak_allocated_gib=1)
        self.assertIn("base_policy=joint",compact_row(row)["config"])

    def test_visa_schedule_is_baseline_plus_18_joint_trials_with_paired_tails(self):
        jobs=entry.schedule([50,70,90],[.7,.9,.99],[0,9],[5,6])
        self.assertEqual(len(jobs),19)
        for rate in (50,70,90):
            for share in (.7,.9,.99):
                pair=[j for j in jobs if j["prune_rate"]==rate and j["distill_share"]==share]
                self.assertEqual([j["tail"] for j in pair],[0,9])
                self.assertEqual(len({j["gpu"] for j in pair}),1)
                self.assertTrue(all(j["base_policy"]=="joint" for j in pair))
        self.assertEqual({j["gpu"] for j in jobs},{5,6})

    def test_visa_manifests_worker_factory_and_resume_use_same_checkpoint_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            fixture=Grid3Tests().fixture(Path(temporary))
            args=entry.parser().parse_args(["--model-path",fixture.model_path,"--data-root",fixture.data_root,
                "--checkpoint",str(fixture.checkpoint),"--output-dir",str(fixture.output_dir)])
            output,spec=entry.setup(args)
            self.assertEqual(spec["sample_count"],4)
            self.assertEqual(len(spec["jobs"]),19)
            child=json.loads((output/"worker.json").read_text(encoding="utf-8"))
            grid3.verify(child,checkpoint_hash=True)
            calls=[]
            def factory(*a,**options): calls.append(options);return object()
            with patch("llava_pruning.pool_distill.PoolDistillMethod",side_effect=factory):
                for job in child["jobs"][1:]: grid3.make_method(object(),job,child)
            self.assertEqual({c["share"] for c in calls},{.7,.9,.99})
            self.assertEqual({c["base_policy"] for c in calls},{"joint"})
            args.resume=True
            self.assertEqual(entry.setup(args),(output,spec))

    def test_joint_reports_include_actual_base_anyres_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);jobs=entry.schedule([70],[.7],[0,9],[5,6]);spec=dict(jobs=jobs)
            for job in jobs:
                folder=output/job["job_id"];folder.mkdir()
                common.save_json(folder/"metrics.json",dict(job_id=job["job_id"],status="success",complete=True,
                    accuracy=.8,tp=8,tn=8,fp=2,fn=2,lat_mean_ms=10,mem_peak_allocated_gib=1,
                    budget=dict(stage2_base_tokens=123,stage2_anyres_tokens=456,stage2_base_policy="joint")))
            entry.write_reports(output,spec)
            with (output/"budget_summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(all(row["stage2_base_tokens"]=="123" and row["stage2_anyres_tokens"]=="456" for row in rows))

    def test_scheduler_launches_joint_manifest_on_only_gpu5_6_and_reuses_complete_trials(self):
        events=[]
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            output=Path(temporary)
            spec=dict(gpus=[5,6],sample_count=2,input_sha256="test",
                      jobs=entry.schedule([50,70,90],[.7,.9,.99],[0,9],[5,6]))
            class Process:
                def __init__(self,command,**options):
                    self.job=command[command.index("--worker")+1];self.polled=False
                    events.append((command,options))
                def poll(self):
                    if not self.polled: self.polled=True;return None
                    folder=output/self.job;folder.mkdir()
                    common.save_json(folder/"metrics.json",dict(job_id=self.job,status="success",complete=True,
                        input_sha256="test",accuracy=.8,tp=1,tn=1,fp=0,fn=0,lat_mean_ms=10,
                        mem_peak_allocated_gib=2,budget=dict(stage2_base_tokens=1,stage2_anyres_tokens=2)))
                    return 0
            with patch.object(entry,"setup",return_value=(output,spec)),patch.object(entry.stretch.time,"sleep"):
                self.assertEqual(entry.run(None,process_factory=Process),0)
                self.assertEqual(entry.run(None,process_factory=Process),0)
            self.assertEqual(len(events),19)
            self.assertTrue(all(command[2].endswith("benchmark_visa_grid3.py") and command[-1]==str(output/"worker.json")
                                for command,_ in events))
            self.assertEqual({options["env"]["CUDA_VISIBLE_DEVICES"] for _,options in events},{"5","6"})
            saved=json.loads((output/"run.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["complete"])
            self.assertEqual(saved["completed_trials"],19)
            with (output/"summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),19)
            self.assertTrue(all(float(row["relative_accuracy_pct"])==100 for row in rows))


if __name__=="__main__": unittest.main()
