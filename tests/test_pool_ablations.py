"""Real autoregressive tail/cache/side-input checks and the 63-trial launcher."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.language_model.attention_predictor import base_text_predictor_inputs, CopiedAttentionPredictor
from llava.constants import IMAGE_TOKEN_INDEX
from llava_pruning.pool_distill import PoolDistillMethod
from llava_pruning.pool_distill_resume import validate_configuration
from llava_pruning.pool_ablation_report import FIELDS, compact_row
from test_fastv_attention import tiny_config
from test_attention_predictor import make_checkpoint
from test_pool_spatial import _Tower
import runMyPruningPoolSpatial as pool
import run_pool_distill as entry
import run_pool_distill_ablations as suite
from test_pool_distill_parallel import Processes, Process
from llava_pruning.pool_distill_parallel import run_parallel_sweep


def model_and_options():
    config=tiny_config("sdpa")
    config.num_hidden_layers=8
    config.eos_token_id,config.pad_token_id=None,0
    config.image_aspect_ratio,config.mm_patch_merge_type="anyres_max_9","spatial_unpad"
    config.image_grid_pinpoints,config.image_token_order=[[384,384]],"base_first"
    config.tokenizer_model_max_length,config.tokenizer_padding_side=4096,"right"
    config.use_pos_skipping=False
    model=LlavaQwenForCausalLM(config).eval()
    core=model.get_model()
    tower=_Tower();tower.features=tower.features[:2]
    core.vision_tower=tower
    core.mm_projector=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.GELU())
    core.image_newline=torch.nn.Parameter(torch.full((32,),-7.))
    return model,dict(inputs=torch.tensor([[1,IMAGE_TOKEN_INDEX,2,3,4]]),
        images=[torch.zeros(2,3,384,384)],image_sizes=[(384,384)],
        do_sample=False,max_new_tokens=3,use_cache=True,return_dict_in_generate=True)


class PoolAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_base_text_input_selects_only_base_and_text_preserving_position_gaps(self):
        hidden=torch.arange(40.).reshape(1,10,4)
        ids=torch.arange(10)[None]*3
        copy=hidden.clone()
        h,p,k=base_text_predictor_inputs(hidden,ids,2,8,[0,2])
        selected=torch.tensor([0,1,2,4,8,9])
        self.assertTrue(torch.equal(h,hidden[:,selected]))
        self.assertTrue(torch.equal(p,ids[:,selected]))
        self.assertEqual(k.tolist(),[2,3])
        self.assertTrue(torch.equal(hidden,copy))
        with self.assertRaises(ValueError): base_text_predictor_inputs(hidden,ids,2,8,[2,0])

    def test_tail_two_three_four_real_generation_earlier_kv_survives_and_budgets_unchanged(self):
        baseline_prefix = None
        for tail in (0,2,3,4):
            torch.manual_seed(920)
            model,options=model_and_options();core=model.get_model()
            observed=[]
            def hook(index):
                def record(module,args,kwargs):
                    observed.append((index,args[0].shape[1],kwargs["position_ids"].clone()))
                return record
            handles=[layer.register_forward_pre_hook(hook(i),with_kwargs=True) for i,layer in enumerate(core.layers)]
            with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                checkpoint=Path(directory)/"best.pt";make_checkpoint(checkpoint,core)
                method=PoolDistillMethod(core,checkpoint,rate=30,share=.7,mode="anyres_self",
                    settings=pool.MethodSettings(prune_rate=0),tail_drop_layers=tail)
                method.configure(core,30)
                method.prepare_inputs((384,384),options["inputs"])
                with method.generation_context(model): result=model.generate(**options)
                stats=method.stats(core,30)
                self.assertFalse(hasattr(core,"_pool_tail_drop_layers"))
                self.assertFalse(hasattr(core,"_pool_predictor_input"))
                final=method.adapter.plan["final_llm_tokens"]
                pre=method.adapter.plan["pre_llm_tokens"]
                self.assertEqual(stats["observed_layer_token_sum"],2*pre+(6-tail)*final)
                self.assertEqual(stats["stage2_llm_visual_tokens"],final)
                self.assertEqual(stats["final_llm_visual_tokens"],0 if tail else final)
                self.assertAlmostEqual(stats["actual_average_prune_rate"],
                    stats["two_stage_average_prune_rate"]+stats["actual_tail_contribution"])
                prefill=observed[:8]
                text_tokens=4
                self.assertEqual([length for _,length,_ in prefill],
                    [pre+text_tokens]*2+[final+text_tokens]*(6-tail)+[text_tokens]*tail)
                cache=result.past_key_values
                length=lambda layer: cache.get_seq_length(layer) if hasattr(cache,"get_seq_length") else cache[layer][0].shape[2]
                self.assertEqual(length(0),final+text_tokens+2)
                self.assertEqual(length(7),text_tokens+2+(0 if tail else final))
                key=cache.key_cache[0] if hasattr(cache,"key_cache") else cache[0][0]
                prefix=key[:,:,:final+text_tokens].clone()
                if baseline_prefix is None: baseline_prefix=prefix
                else: self.assertTrue(torch.equal(prefix,baseline_prefix))
                # Last-layer text IDs are original positions, not renumbered.
                self.assertEqual(prefill[-1][2][0,-1].item(),pre+text_tokens-1)
                for _,length,pos in observed[8:]:
                    self.assertEqual(length,1)
                    self.assertGreaterEqual(pos.item(),pre+text_tokens)
                self.assertEqual(core.config._attn_implementation,"sdpa")
            for handle in handles: handle.remove()

    def test_copied_qwen_side_branch_accepts_selected_rows_and_original_rope_ids(self):
        torch.manual_seed(922)
        model,_=model_and_options();core=model.get_model()
        predictor=CopiedAttentionPredictor.from_backbone(core).eval()
        hidden=torch.randn(1,12,32)
        ids=torch.arange(12)[None]
        short,positions,spatial=base_text_predictor_inputs(hidden,ids,2,10,[0,1,4])
        with torch.no_grad(): result=predictor(short,positions,spatial)
        self.assertEqual(positions.tolist(),[[0,1,2,3,6,10,11]])
        self.assertEqual(result.shape,(3,))
        self.assertTrue(torch.isfinite(result).all())
        self.assertAlmostEqual(result.exp().sum().item(),1.,places=6)

    def test_base_text_real_side_input_keeps_backbone_h2_and_only_changes_branch_rows(self):
        torch.manual_seed(921)
        model,options=model_and_options();core=model.get_model()
        captured={}
        with tempfile.TemporaryDirectory() as directory,torch.no_grad():
            checkpoint=Path(directory)/"best.pt";make_checkpoint(checkpoint,core)
            method=PoolDistillMethod(core,checkpoint,rate=30,share=.8,mode="base_full",
                settings=pool.MethodSettings(prune_rate=0),predictor_input="base_text")
            method.configure(core,30);method.prepare_inputs((384,384),options["inputs"])
            def main_hook(module,args,output):
                if output[0].shape[1]>1: captured["h2"]=output[0].clone()
            def side_hook(module,args):
                captured.update(hidden=args[0].clone(),ids=args[1].clone(),indices=torch.as_tensor(args[2]).clone())
            h=core.layers[1].register_forward_hook(main_hook)
            s=core.attention_predictor.register_forward_pre_hook(side_hook)
            with method.generation_context(model): result=model.generate(**options)
            h.remove();s.remove()
            pre=method.adapter.plan["pre_llm_tokens"]
            base=method.adapter.base
            expected=torch.tensor([0]+[1+i for i in base]+list(range(1+pre,pre+4)))
            self.assertTrue(torch.equal(captured["hidden"],captured["h2"][:,expected]))
            self.assertTrue(torch.equal(captured["ids"],expected[None]))
            self.assertEqual(len(captured["indices"]),len(base))
            self.assertEqual(captured["hidden"].shape[1],len(base)+4)
            self.assertEqual(method.stats(core,30)["final_base_tokens"],len(base))
            self.assertEqual(result.sequences.shape[-1],3)

    def test_compact_csv_units_and_no_extra_columns(self):
        row=compact_row(dict(prune_rate=50,distill_share=.8,selection_mode="base_full",predictor_input="base_text",
            tail_drop_layers=0,tp=12,fp=3,tn=5,fn=2,accuracy=17/22,lat_mean_ms=123,mem_peak_allocated_gib=20.5))
        self.assertEqual(tuple(row),FIELDS)
        self.assertAlmostEqual(row["acc"],100*17/22)
        self.assertEqual(row["Lat"],123);self.assertEqual(row["mem"],20.5)
        self.assertIn("input=base_text;tail=0",row["config"])

    def fixture(self,root):
        (root/"model").mkdir();(root/"data").mkdir()
        (root/"input.json").write_text("[]");(root/"best.pt").write_bytes(b"mock")
        return suite.build_parser().parse_args(["--gpus","5","6","7","--model-path",str(root/"model"),
            "--data-root",str(root/"data"),"--input-json",str(root/"input.json"),
            "--checkpoint",str(root/"best.pt"),"--output-dir",str(root/"out")])

    def test_single_command_builds_all63_non_crossed_trials_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root);launched=[]
            def factory(command,**options):
                a=entry.build_parser().parse_args(command[2:]);launched.append(a)
                self.assertEqual(a.gpus,[5,6,7]);self.assertEqual(a.summary_format,"compact")
                a.output_dir.mkdir(exist_ok=True)
                rows=[dict(prune_rate=r,distill_share=w,selection_mode=m,predictor_input=a.predictor_input,
                    tail_drop_layers=a.tail_drop_layers,tp=1,fp=0,tn=1,fn=0,accuracy=1.,lat_mean_ms=100.,mem_peak_allocated_gib=20.)
                    for r in a.prune_rates for w in a.lambdas for m in a.selection_modes]
                entry.write_csv(a.output_dir/"summary.csv",entry.summary_rows(rows,"compact"))
                entry.write_json(a.output_dir/"comparison.json",dict(complete=True))
                return SimpleNamespace(poll=lambda:0,wait=lambda:0)
            suite.run(args,process_factory=factory,poll_interval=0)
            self.assertEqual(len(launched),4)
            self.assertEqual([a.tail_drop_layers for a in launched],[2,3,4,0])
            self.assertEqual(launched[-1].selection_modes,["base_full"])
            self.assertEqual(launched[-1].predictor_input,"base_text")
            with (root/"out/summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(tuple(reader.fieldnames),FIELDS)
                self.assertEqual(len(rows),63)
                self.assertEqual(len({r["config"] for r in rows}),63)
            argv=["--resume","--output-dir",str(root/"out")]
            restored=suite.restore_args(suite.build_parser().parse_args(argv),argv)
            self.assertEqual(restored.gpus,[5,6,7]);self.assertEqual(restored.lambdas,[.7,.8,.9])
            command=suite.group_command(restored,suite.GROUPS[0],root/"out")
            self.assertIn("--resume",command)
            restored.tail_drop_layers=2
            with self.assertRaises(ValueError): suite.run(restored,process_factory=factory)

    def test_resume_rejects_new_ablation_for_old_results(self):
        args=entry.build_parser().parse_args([])
        saved=vars(args).copy()
        saved.update(prune_rates=[30],lambdas=[.7],selection_modes=["anyres_self"])
        saved.pop("tail_drop_layers");saved.pop("predictor_input")
        validate_configuration(args,saved,30,.7,"anyres_self")
        args.tail_drop_layers=2
        with self.assertRaisesRegex(ValueError,"tail_drop_layers"):
            validate_configuration(args,saved,30,.7,"anyres_self")

    def test_parallel_workers_receive_ablation_flags_and_export_compact_results_on_resume(self):
        native_complete=Process.complete
        def complete(process):
            native_complete(process)
            a=process.trial
            path=a.output_dir/"metrics.json"
            row=json.loads(path.read_text(encoding="utf-8"))
            row.update(tail_drop_layers=a.tail_drop_layers,predictor_input=a.predictor_input)
            entry.write_json(path,row)
        for tail,branch,modes in ((2,"full",["anyres_self","base_full"]),(0,"base_text",["base_full"])):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);args=self.fixture(root)
                args.prune_rates=[30];args.lambdas=[.7,.8]
                args.tail_drop_layers=tail;args.predictor_input=branch;args.selection_modes=modes
                entry.write_json(args.input_json,[dict(question_id="1",gt=1,image="image.png")])
                processes=Processes()
                with patch.object(Process,"complete",complete):
                    run_parallel_sweep(args,api=entry,process_factory=processes,poll_interval=0)
                self.assertTrue(all(a.tail_drop_layers==tail and a.predictor_input==branch
                                    and a.summary_format=="compact" for a in processes.launches))
                with (root/"out/summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                    reader=csv.DictReader(stream);rows=list(reader)
                    self.assertEqual(tuple(reader.fieldnames),FIELDS)
                self.assertEqual(len(rows),2*len(modes))
                self.assertTrue(all(f"input={branch};tail={tail}" in row["config"] for row in rows))
                args.resume=True
                def forbidden(*a,**k): raise AssertionError("Verified complete trials must be reused")
                run_parallel_sweep(args,api=entry,process_factory=forbidden,poll_interval=0)


if __name__=="__main__": unittest.main()
