"""Exact fallback budgets, actual autoregressive pruning, CLI and resume guards."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import runMyPruningPoolSpatial as pool
import run_pool_distill as entry
from llava_pruning.pool_ablation_report import FIELDS
from llava_pruning.pool_distill import PoolDistillMethod
from llava_pruning.pool_distill_budget import hybrid_budget, pre_view_budget, stage2_view_budget
from llava_pruning.pool_distill_parallel import worker_command
from llava_pruning.pool_distill_resume import validate_configuration
from test_attention_predictor import make_checkpoint
from test_pool_ablations import model_and_options


class AnyResFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact90_budget_preserves_newlines_and_falls_back_to_base(self):
        for share,expected in ((.7,(5380,379,530,298)),(.9,(6707,277,663,196))):
            plan=hybrid_budget(7371,90,share);plan["raw_grid_height"]=3
            with self.assertRaisesRegex(ValueError,"cannot protect"):
                pre_view_budget(plan,9)
            pre=pre_view_budget(plan,9,base_policy="anyres_first")
            post=stage2_view_budget(plan["final_llm_tokens"],pre["base_keep"],81,"anyres_first")
            self.assertEqual((plan["pre_llm_tokens"],plan["final_llm_tokens"],pre["base_keep"],post["base_keep"]),expected)
            self.assertEqual(post["anyres_keep"],0)
            self.assertEqual(post["base_keep"]+post["protected_newlines"],plan["final_llm_tokens"])
        # The one-crop layout that previously failed is feasible too.
        plan=dict(hybrid_budget(1485,90,.9),raw_grid_height=1)
        self.assertGreater(pre_view_budget(plan,1,base_policy="anyres_first")["base_keep"],0)
        with self.assertRaisesRegex(ValueError,"at least one spatial"):
            stage2_view_budget(81,530,81,"anyres_first")

    def test_real90_base_text_generation_keeps_highest_base_original_ids_and_tail_cache(self):
        for share,tail,mapping in ((.7,0,"fast"),(.7,9,"fast"),(.9,0,"fast"),(.9,9,"fast"),(.7,9,"reference")):
            torch.manual_seed(971)
            model,options=model_and_options(depth=28);core=model.get_model()
            observed=[]
            def record(module,args,kwargs):
                observed.append((args[0].shape[1],kwargs["position_ids"].clone()))
            handles=[layer.register_forward_pre_hook(record,with_kwargs=True) for layer in core.layers]
            with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                checkpoint=Path(directory)/"best.pt";make_checkpoint(checkpoint,core)
                method=PoolDistillMethod(core,checkpoint,rate=90,share=share,mode="base_full",
                    settings=pool.MethodSettings(prune_rate=0),predictor_input="base_text",base_mapping=mapping,
                    tail_drop_layers=tail,base_policy="anyres_first")
                method.configure(core,90);method.prepare_inputs((384,384),options["inputs"])
                captured={};select=method.adapter.select
                def checked_select(ranking,spatial,original):
                    adapter=method.adapter
                    result=select(ranking,spatial,original)
                    keep=adapter.plan["final_llm_tokens"]-len(adapter.protected)
                    # Scores correspond to Base followed by AnyRes in packed spatial order.
                    best=torch.argsort(ranking[:len(adapter.base)],descending=True,stable=True)[:keep]
                    base=torch.tensor(adapter.base)[best]
                    expected=torch.cat((base,torch.tensor(adapter.protected))).sort().values
                    self.assertTrue(torch.equal(result[0],expected))
                    self.assertFalse(set(result[0].tolist()) & set(adapter.anyres))
                    captured["selected"]=result[0]
                    return result
                with patch.object(method.adapter,"select",side_effect=checked_select),method.generation_context(model):
                    result=model.generate(**options)
                stats=method.stats(core,90)
                pre,final=stats["pre_llm_tokens"],stats["final_llm_tokens"]
                self.assertEqual(result.sequences.shape[-1],3)
                self.assertEqual(stats["stage2_anyres_tokens"],0)
                self.assertEqual(stats["stage2_base_tokens"]+stats["protected_newlines"],final)
                self.assertGreater(stats["stage2_base_pruned_tokens"],0)
                self.assertEqual(stats["final_base_tokens"],0 if tail else stats["stage2_base_tokens"])
                self.assertEqual(stats["observed_layer_token_sum"],2*pre+(26-tail)*final)
                self.assertEqual([n for n,_ in observed[:28]],[pre+4]*2+[final+4]*(26-tail)+[4]*tail)
                self.assertEqual(observed[27][1][0,-1].item(),pre+3)
                cache=result.past_key_values
                length=lambda i:cache.get_seq_length(i) if hasattr(cache,"get_seq_length") else cache[i][0].shape[2]
                self.assertEqual(length(0),final+6)
                self.assertEqual(length(27),6+(0 if tail else final))
                recovered=method.recover_legacy_budget((384,384))
                for key in ("stage2_base_tokens","stage2_anyres_tokens","final_base_tokens","actual_average_prune_rate"):
                    self.assertEqual(recovered[key],stats[key])
                self.assertIn("selected",captured)
            for handle in handles: handle.remove()

    def test_feasible_old_policy_retains_identical_selected_tokens_and_answers(self):
        sequences,selections=[],[]
        for policy in ("protect","anyres_first"):
            torch.manual_seed(972)
            model,options=model_and_options();core=model.get_model()
            with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                checkpoint=Path(directory)/"best.pt";make_checkpoint(checkpoint,core)
                method=PoolDistillMethod(core,checkpoint,rate=30,share=.8,mode="base_full",
                    settings=pool.MethodSettings(prune_rate=0),predictor_input="base_text",base_mapping="fast",base_policy=policy)
                method.configure(core,30);method.prepare_inputs((384,384),options["inputs"])
                select=method.adapter.select
                def capture(*args):
                    result=select(*args);selections.append(result[0].clone());return result
                with patch.object(method.adapter,"select",side_effect=capture),method.generation_context(model):
                    sequences.append(model.generate(**options).sequences)
                self.assertEqual(method.stats(core,30)["stage2_base_pruned_tokens"],0)
        self.assertTrue(torch.equal(*selections));self.assertTrue(torch.equal(*sequences))

    def test_cli_forwards_policy_and_reports_both_single_gpu_trials_without_visualizations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"model").mkdir();(root/"data").mkdir()
            (root/"best.pt").write_bytes(b"mock");entry.write_json(root/"input.json",[])
            args=entry.build_parser().parse_args(["--gpu","5","--model-path",str(root/"model"),
                "--data-root",str(root/"data"),"--input-json",str(root/"input.json"),"--checkpoint",str(root/"best.pt"),
                "--prune-rates","90","--lambdas",".7",".9","--selection-modes","base_full",
                "--predictor-input","base_text","--base-mapping","fast","--base-policy","anyres_first",
                "--tail-drop-layers","9","--summary-format","compact","--output-dir",str(root/"out")])
            launched=[]
            def child(command,**options):
                worker=entry.build_parser().parse_args(command[2:]);launched.append(worker)
                self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"],"5")
                self.assertEqual((worker.base_policy,worker.tail_drop_layers,worker.predictor_input),("anyres_first",9,"base_text"))
                worker.output_dir.mkdir()
                row=dict(prune_rate=90,distill_share=worker.lambdas[0],selection_mode="base_full",complete=True,
                    tp=1,fp=0,tn=1,fn=0,accuracy=1.,lat_mean_ms=100.,mem_peak_allocated_gib=20.,
                    actual_average_prune_rate=92.,stage2_base_policy=worker.base_policy,
                    base_mapping=worker.base_mapping,predictor_input=worker.predictor_input,tail_drop_layers=9)
                entry.write_json(worker.output_dir/"metrics.json",row)
                entry.write_json(worker.output_dir/"run.json",dict(configuration={"summary_format":"compact"}))
                return SimpleNamespace(returncode=0)
            entry.run_sweep(args,run_process=child)
            self.assertEqual([w.lambdas for w in launched],[[.7],[.9]])
            with (root/"out/summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(tuple(reader.fieldnames),FIELDS)
            self.assertEqual(len(rows),2)
            self.assertTrue(all("base_policy=anyres_first" in row["config"] for row in rows))
            self.assertFalse(list((root/"out").rglob("*.png")))
            command=worker_command(args,entry,90,.7,"base_full",5,root/"parallel")
            self.assertEqual(entry.build_parser().parse_args(command[2:]).base_policy,"anyres_first")

    def test_resume_does_not_mix_policies_and_old_prose_manifest_stays_protected(self):
        args=entry.build_parser().parse_args([]);saved=vars(args).copy()
        saved.pop("base_policy");saved.update(prune_rates=[90],lambdas=[.7],selection_modes=["base_full"])
        validate_configuration(args,saved,90,.7,"base_full")
        args.base_policy="anyres_first"
        with self.assertRaisesRegex(ValueError,"base_policy"):
            validate_configuration(args,saved,90,.7,"base_full")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            saved={k:str(v) if isinstance(v,Path) else v for k,v in saved.items()}
            entry.write_json(root/"comparison.json",dict(configuration=saved,prompt_format=entry.PROMPT_FORMAT,
                base_policy="MyPool compresses Base + AnyRes; block2 protects remaining Base"))
            restored=entry.restore_sweep_args(entry.build_parser().parse_args(["--output-dir",str(root)]),[])
            self.assertEqual(restored.base_policy,"protect")


if __name__=="__main__": unittest.main()
