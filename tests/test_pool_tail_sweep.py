"""192 fixed-GPU jobs and actual 28-block decoding for tails0..15."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import runMyPruningPoolSpatial as pool
import run_pool_distill as entry
import run_pool_tail_sweep as sweep
from llava_pruning.pool_distill import PoolDistillMethod
from llava_pruning.pool_ablation_report import FIELDS
from test_pool_distill_parallel import Processes
from test_clse import toy_model
from test_pool_ablations import model_and_options
from test_attention_predictor import make_checkpoint


class TailSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def fixture(self,root):
        (root/"model").mkdir();(root/"data").mkdir();(root/"best.pt").write_bytes(b"mock")
        entry.write_json(root/"input.json",[dict(question_id="1",gt=1,image="image.png")])
        return sweep.build_parser().parse_args(["--gpus","7","6","5","4","3","2",
            "--model-path",str(root/"model"),"--checkpoint",str(root/"best.pt"),
            "--data-root",str(root/"data"),"--input-json",str(root/"input.json"),"--output-dir",str(root/"out")])

    def test_fraction_units_and_global_cli_compatibility(self):
        self.assertEqual(sweep.normalize_rates([.3,.5,.7]),[30,50,70])
        self.assertEqual(sweep.normalize_rates([30,50,70]),[30,50,70])
        self.assertEqual(sweep.normalize_rates([.29]),[29])
        self.assertEqual(entry.build_parser().parse_args(["--prune-rates","0.3"]).prune_rates,[.3])
        self.assertEqual(entry.build_parser().parse_args(["--tail-drop-layers","15"]).tail_drop_layers,15)
        for rates in ([.3,50],[],[0],[100],[float("nan")]):
            with self.assertRaises(ValueError): sweep.normalize_rates(rates)

    def test_all192_configs_six_gpus_fixed_across_tails_and_compact_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root);processes=Processes()
            sweep.run(args,process_factory=processes,poll_interval=0)
            self.assertEqual(processes.max_live,6)
            self.assertEqual(len(processes.launches),192)
            combinations={(a.prune_rates[0],a.lambdas[0],a.tail_drop_layers) for a in processes.launches}
            self.assertEqual(combinations,{(r,s,t) for r in (30,50,70) for s in (.6,.7,.8,.9) for t in range(16)})
            for index,(rate,share) in enumerate([(r,s) for r in (30,50,70) for s in (.6,.7,.8,.9)]):
                jobs=[a for a in processes.launches if a.prune_rates==[rate] and a.lambdas==[share]]
                self.assertEqual([a.tail_drop_layers for a in jobs],list(range(16)))
                self.assertEqual({a.gpu for a in jobs},{[7,6,5,4,3,2][index%6]})
            self.assertTrue(all(a.base_mapping=="fast" and a.predictor_input=="base_text" and
                a.selection_modes==["base_full"] and a.pruning_audit=="lean" for a in processes.launches))
            with (root/"out/summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(tuple(reader.fieldnames),FIELDS)
            self.assertEqual(len(rows),192);self.assertEqual(len({r["config"] for r in rows}),192)
            self.assertTrue(all("base_full;input=base_text;tail=" in row["config"] and
                                row["config"].endswith("mapping=fast") for row in rows))
            self.assertEqual(len(list((root/"out/logs").glob("*.log"))),192)
            manifest=json.loads((root/"out/comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"]);self.assertEqual(manifest["completed_trials"],192)
            argv=["--resume","--output-dir",str(root/"out")]
            resumed=sweep.restore_args(sweep.build_parser().parse_args(argv),argv)
            def forbidden(*a,**k): raise AssertionError("Complete jobs must be reused")
            sweep.run(resumed,process_factory=forbidden,poll_interval=0)
            self.assertEqual(resumed.prune_rates,[30,50,70])
            changed=argv+["--tail-end","14"]
            with self.assertRaisesRegex(ValueError,"tail-end"):
                sweep.restore_args(sweep.build_parser().parse_args(changed),changed)

    def test_failed_sweep_resumes_pending_tails_and_keeps_fast_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root)
            args.tail_end=1;args.prune_rates=[.3];args.lambdas=[.6,.7]
            failed=Processes(fail_gpu=6)
            with self.assertRaises(RuntimeError): sweep.run(args,process_factory=failed,poll_interval=0)
            self.assertFalse(failed.live)
            committed=len(list((root/"out").glob("*/metrics.json")))
            argv=["--resume","--output-dir",str(root/"out")]
            resumed=sweep.restore_args(sweep.build_parser().parse_args(argv),argv)
            done=Processes();sweep.run(resumed,process_factory=done,poll_interval=0)
            self.assertEqual(len(done.launches),4-committed)
            self.assertTrue(all(a.base_mapping=="fast" and a.tail_drop_layers in (0,1) for a in done.launches))

    def test_actual28_block_generation_every_tail_preserves_early_kv_and_ids(self):
        torch.manual_seed(950)
        model,features,options=toy_model();core=model.get_model()
        core.config.num_hidden_layers=28
        core.layers=torch.nn.ModuleList([Qwen2DecoderLayer(core.config,layer_idx=i) for i in range(28)])
        options.update(max_new_tokens=3,return_dict_in_generate=True)
        with tempfile.TemporaryDirectory() as directory,torch.no_grad(),patch.object(model,"encode_images",return_value=features):
            checkpoint=Path(directory)/"best.pt";make_checkpoint(checkpoint,core)
            for tail in (*range(16),26):
                observed=[]
                def hook(module,args,kwargs):
                    if args[0].shape[1]>1: observed.append((args[0].shape[1],kwargs["position_ids"].clone()))
                handles=[layer.register_forward_pre_hook(hook,with_kwargs=True) for layer in core.layers]
                method=PoolDistillMethod(core,checkpoint,rate=20,share=1,mode="base_full",
                    settings=pool.MethodSettings(prune_rate=0),tail_drop_layers=tail,predictor_input="base_text")
                method.configure(core,20)
                with method.generation_context(model): result=model.generate(**options)
                for handle in handles: handle.remove()
                stats=method.stats(core,20);pre=stats["pre_llm_tokens"];final=stats["stage2_llm_visual_tokens"]
                self.assertEqual(len(observed),28)
                self.assertEqual([n for n,_ in observed],[pre+4]*2+[final+4]*(26-tail)+[4]*tail)
                self.assertEqual(observed[-1][1][0,-1].item(),pre+3)
                cache=result.past_key_values
                length=lambda i:cache.get_seq_length(i) if hasattr(cache,"get_seq_length") else cache[i][0].shape[2]
                self.assertEqual(length(0),final+6)
                self.assertEqual(length(27),6+(0 if tail else final))
                self.assertEqual(stats["observed_layer_token_sum"],2*pre+(26-tail)*final)
                self.assertEqual(stats["final_llm_visual_tokens"],0 if tail else final)
            with self.assertRaises(ValueError):
                PoolDistillMethod(core,checkpoint,rate=20,share=1,mode="base_full",
                    settings=pool.MethodSettings(prune_rate=0),tail_drop_layers=27)

    def test_real_hybrid_base_text_fast_with_tail15_on28_blocks(self):
        torch.manual_seed(951)
        model,options=model_and_options(depth=28);core=model.get_model()
        with tempfile.TemporaryDirectory() as directory,torch.no_grad():
            checkpoint=Path(directory)/"best.pt";make_checkpoint(checkpoint,core)
            method=PoolDistillMethod(core,checkpoint,rate=30,share=.6,mode="base_full",
                settings=pool.MethodSettings(prune_rate=0),tail_drop_layers=15,predictor_input="base_text",base_mapping="fast")
            method.configure(core,30);method.prepare_inputs((384,384),options["inputs"])
            with method.generation_context(model): result=model.generate(**options)
            stats=method.stats(core,30)
            self.assertEqual(stats["tail_drop_first_layer"],14)
            self.assertEqual(stats["final_llm_visual_tokens"],0)
            self.assertEqual(result.sequences.shape[-1],3)
            self.assertAlmostEqual(stats["actual_average_prune_rate"],
                stats["two_stage_average_prune_rate"]+stats["actual_tail_contribution"])


if __name__=="__main__": unittest.main()
