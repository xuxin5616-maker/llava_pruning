"""SIGINT recovery, legacy migration, prefix integrity and completed-job reuse."""

from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import run_pool_distill as entry
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.metrics import Accuracy
from llava_pruning.pool_distill import PoolDistillMethod
from llava_pruning.pool_distill_budget import hybrid_budget
from llava_pruning.pool_distill_resume import load_records, read_prefix
from test_pool_spatial import _Core
from llava_pruning.prompts import resolve_prompt


class ResumeTests(unittest.TestCase):
    def fixture(self, root, *extra):
        (root/"model").mkdir()
        (root/"data").mkdir()
        image = root/"data/image.png"
        Image.new("RGB", (10,10)).save(image)
        (root/"best.pt").write_bytes(b"mock")
        items = [dict(question_id=str(i),image="image.png",category="bottle",gt=gt,text="test")
                 for i,gt in enumerate([1,0,0,1,1])]
        entry.write_json(root/"input.json", items)
        args = entry.build_parser().parse_args(["--model-path",str(root/"model"),"--data-root",str(root/"data"),
            "--input-json",str(root/"input.json"),"--checkpoint",str(root/"best.pt"),"--output-dir",str(root/"output"),
            "--prune-rates","30","--lambdas",".5","--selection-modes","base_full","--warmup","1",*extra])
        samples = [SimpleNamespace(sample_id=item["question_id"],gt=item["gt"],image=image,category="bottle",text="test") for item in items]
        return args, samples

    @contextmanager
    def fake_worker(self, samples, *, interrupt=True):
        core = SimpleNamespace(layers=[None]*28)
        state = dict(interrupt=interrupt, calls=[], recover_calls=0)
        plan = dict(hybrid_budget(7371,30,.5),selection_mode="base_full",pre_base_tokens=400,final_anyres_tokens=4609)
        class Method:
            def __init__(self,*a,**kw):
                self.adapter=SimpleNamespace(set_image_context=lambda size:None)
            @contextmanager
            def generation_context(self, model): yield
            def recover_legacy_budget(self, size):
                state["recover_calls"]+=1
                return dict(plan,budget_source="legacy_reconstructed_counts",observed_layer_token_sum=None)
        class Backend:
            def __init__(self,*a,**kw):
                self.model=SimpleNamespace(get_model=lambda:core)
                self.inference_config=dict(attention="mock")
            def generate(self,sample,prompt,**kw):
                assert isinstance(prompt,str)
                assert prompt == resolve_prompt("v0",sample.category,sample.text)[0]
                index=int(sample.sample_id)
                state["calls"].append(index)
                if index==2 and state["interrupt"]:
                    raise KeyboardInterrupt
                return dict(answer="A" if sample.gt else "B",lat_seconds=.1+index*.01,generation_seconds=.3,
                    generated_tokens=2,peak_allocated_bytes=1024+index,peak_reserved_bytes=2048,
                    stats=dict(plan))
        with ExitStack() as stack:
            for target,value in (("llava_pruning.backend.LlavaBackend",Backend),
                                 ("llava_pruning.pool_distill.PoolDistillMethod",Method)):
                stack.enter_context(patch(target,value))
            stack.enter_context(patch("llava_pruning.data.load_samples",return_value=samples))
            stack.enter_context(patch("benchmark_fastv.validate_single_gpu"))
            stack.enter_context(patch.object(torch.cuda,"is_available",return_value=True))
            stack.enter_context(patch.object(torch.cuda,"device_count",return_value=1))
            stack.enter_context(patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(total_memory=4*1024**3)))
            yield state

    def test_worker_resumes_modern_journal_and_torn_tail_without_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args,samples=self.fixture(root)
            with self.fake_worker(samples) as state:
                with self.assertRaises(KeyboardInterrupt): entry.run_worker(args)
                original,_=read_prefix(root/"output/predictions.jsonl")
                self.assertEqual(len(original),2)
                self.assertFalse((root/"output/metrics.json").exists())
                run=json.loads((root/"output/run.json").read_text(encoding="utf-8"))
                self.assertEqual(run["status"],"interrupted")
                with (root/"output/progress.jsonl").open("ab") as stream: stream.write(b'{"index":2,')
                args.resume=True
                state["interrupt"]=False
                state["calls"].clear()
                entry.run_worker(args)
                self.assertEqual(state["calls"],[0,2,3,4])  # one excluded warmup; no repeated measured0/1
                predictions,_=read_prefix(root/"output/predictions.jsonl")
                self.assertEqual(predictions[:2],original)
                self.assertEqual([p["question_id"] for p in predictions],[str(i) for i in range(5)])
                self.assertEqual(len(read_prefix(root/"output/progress.jsonl")[0]),5)
                metrics=json.loads((root/"output/metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["evaluated_samples"],5)
                self.assertEqual(metrics["resumed_samples"],2)
                self.assertEqual(metrics["fresh_samples"],3)
                self.assertEqual(metrics["budget_reconstructed_samples"],0)
                self.assertEqual(metrics["measurement_segments"],2)
                self.assertAlmostEqual(metrics["lat_mean_ms"],120)
                self.assertEqual(metrics["accuracy"],1.)
                state["calls"].clear()
                entry.run_worker(args)
                self.assertFalse(state["calls"])  # all samples committed; no warmup or generation

    def test_verified_text_predictions_without_journal_reconstruct_missing_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args,samples=self.fixture(root)
            with self.fake_worker(samples) as state:
                with self.assertRaises(KeyboardInterrupt): entry.run_worker(args)
                (root/"output/progress.jsonl").unlink()  # missing journal, but prompt format is verified
                metadata=json.loads((root/"output/run.json").read_text(encoding="utf-8"))
                metadata.pop("input_json_sha256")
                metadata.pop("checkpoint_identity")
                entry.write_json(root/"output/run.json",metadata)
                old,_=read_prefix(root/"output/predictions.jsonl")
                args.resume=True
                state["interrupt"]=False
                state["calls"].clear()
                entry.run_worker(args)
                self.assertEqual(state["calls"],[0,2,3,4])
                self.assertEqual(state["recover_calls"],1)
                rows,_=read_prefix(root/"output/progress.jsonl")
                self.assertEqual([r["prediction"] for r in rows[:2]],old)
                for row in rows[:2]:
                    self.assertEqual(row["budget"]["budget_source"],"legacy_reconstructed_counts")
                    self.assertIsNone(row["budget"]["observed_layer_token_sum"])
                metrics=json.loads((root/"output/metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["budget_reconstructed_samples"],2)
                self.assertEqual(metrics["budget_observed_samples"],3)

    def test_recovery_rejects_changed_input_checkpoint_ids_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args,samples=self.fixture(root)
            with self.fake_worker(samples) as state:
                with self.assertRaises(KeyboardInterrupt): entry.run_worker(args)
                args.resume=True
                input_bytes=args.input_json.read_bytes()
                args.input_json.write_bytes(input_bytes+b" ")
                with self.assertRaisesRegex(ValueError,"Input JSON"): entry.run_worker(args)
                args.input_json.write_bytes(input_bytes)
                (root/"best.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"checkpoint changed"): entry.run_worker(args)
            rows,_=read_prefix(root/"output/progress.jsonl")
            rows[0]["prediction"]["question_id"]="wrong"
            entry.write_text_atomic(root/"output/progress.jsonl","".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError,"ID/GT/order"):
                load_records(root/"output",samples,lambda s:None,rate=30,share=.5,mode="base_full")
            with (root/"output/progress.jsonl").open("ab") as stream: stream.write(b'broken\n')
            with self.assertRaisesRegex(ValueError,"Corrupt checkpoint"): read_prefix(root/"output/progress.jsonl")

    def complete_mock(self, trial, samples):
        folder=trial.output_dir
        folder.mkdir(parents=True,exist_ok=True)
        accuracy=Accuracy(len(samples))
        predictions=[]
        for sample in samples:
            answer="A" if sample.gt else "B"
            accuracy.add(sample.gt,answer)
            predictions.append(dict(question_id=sample.sample_id,gt=sample.gt,answer=answer,lat_seconds=.1,
                generation_seconds=.2,generated_tokens=1,peak_allocated_bytes=1024,peak_reserved_bytes=2048))
        metrics=extended_metrics(dict(accuracy.result(30,complete=True),distill_share=trial.lambdas[0],
            selection_mode="base_full",lat_mean_ms=100.,mem_peak_allocated_gib=1.,actual_average_prune_rate=30.001),trial.baseline_acc/100)
        entry.write_json(folder/"metrics.json",metrics)
        entry.write_csv(folder/"summary.csv",[metrics])
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(trial).items()}
        entry.write_json(folder/"run.json",dict(complete=True,prompt_format=entry.PROMPT_FORMAT,configuration=configuration,inference={}))
        entry.write_text_atomic(folder/"predictions.jsonl","".join(json.dumps(p)+"\n" for p in predictions))

    def test_parent_records_sigint_restores_old_args_and_skips_complete_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args,samples=self.fixture(root,"--lambdas",".25",".75","--gpu","4","--baseline-acc","90")
            calls=[]
            def first(command,**kwargs):
                trial=entry.build_parser().parse_args(command[2:])
                calls.append(trial)
                if trial.lambdas==[.75]: raise KeyboardInterrupt
                self.complete_mock(trial,samples)
                return SimpleNamespace(returncode=0)
            with self.assertRaises(KeyboardInterrupt): entry.run_sweep(args,run_process=first)
            manifest=json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["trials"][-1]["status"],"interrupted")
            short=["--resume","--output-dir",str(root/"output")]
            restored=entry.restore_sweep_args(entry.build_parser().parse_args(short),short)
            self.assertEqual(restored.gpu,4)
            self.assertEqual(restored.baseline_acc,90)
            self.assertEqual(restored.prune_rates,[30])
            self.assertEqual(restored.lambdas,[.25,.75])
            resumed=[]
            def second(command,**kwargs):
                trial=entry.build_parser().parse_args(command[2:])
                self.assertTrue(trial.resume)
                resumed.append(trial.lambdas[0])
                self.complete_mock(trial,samples)
                return SimpleNamespace(returncode=0)
            entry.run_sweep(restored,run_process=second)
            self.assertEqual(resumed,[.75])
            manifest=json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(len(manifest["trials"]),2)
            self.assertTrue(manifest["trials"][0]["reused"])
            changed=short+["--seed","43"]
            with self.assertRaisesRegex(ValueError,"changed --seed"):
                entry.restore_sweep_args(entry.build_parser().parse_args(changed),changed)
            # A verified prompt-text run can use the command-only root schema.
            manifest.pop("configuration")
            entry.write_json(root/"output/comparison.json",manifest)
            recovered=entry.restore_sweep_args(entry.build_parser().parse_args(short),short)
            self.assertEqual(recovered.gpu,4)
            self.assertEqual(recovered.lambdas,[.25,.75])

    def test_legacy_tuple_outputs_cannot_be_resumed_or_reused_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args,samples=self.fixture(root)
            with self.fake_worker(samples,interrupt=False) as state:
                entry.run_worker(args)
                args.resume=True
                worker_path=root/"output/run.json"
                metadata=json.loads(worker_path.read_text(encoding="utf-8"))
                metadata.pop("prompt_format")  # actual older outputs have no format marker
                entry.write_json(worker_path,metadata)
                before=worker_path.read_bytes()
                state["calls"].clear()
                with self.assertRaisesRegex(ValueError,"NEW --output-dir"):
                    entry.run_worker(args)
                self.assertFalse(state["calls"])
                with self.assertRaisesRegex(ValueError,"NEW --output-dir"):
                    entry.verified_complete(root/"output",args,30,.5,"base_full")
                self.assertEqual(worker_path.read_bytes(),before)
            # Parent refuses old roots before rewriting status or spawning.
            parent_path=root/"output/comparison.json"
            entry.write_json(parent_path,dict(complete=True,trials=[]))
            before=parent_path.read_bytes()
            with patch.object(entry.subprocess,"run",side_effect=AssertionError("No child execution")):
                with self.assertRaisesRegex(ValueError,"NEW --output-dir"):
                    entry.run_sweep(args)
                with self.assertRaisesRegex(ValueError,"NEW --output-dir"):
                    entry.restore_sweep_args(args,[])
            self.assertEqual(parent_path.read_bytes(),before)

    def test_real_count_reconstruction_does_not_encode_or_invent_observed_rows(self):
        core=_Core()
        core.layers=[None]*28
        for share in (0,.5,1):
            method=PoolDistillMethod(core,Path("unused.pt"),rate=30,share=share,mode="base_full",
                                    settings=__import__("runMyPruningPoolSpatial").MethodSettings(prune_rate=0))
            with patch.object(method.adapter,"encode_images",side_effect=AssertionError("No model execution")):
                result=method.recover_legacy_budget((1024,1024))
            self.assertIsNone(result["observed_layer_token_sum"])
            self.assertEqual(result["budget_source"],"legacy_reconstructed_counts")
            self.assertEqual(result["final_base_tokens"],result["pre_base_tokens"])
            self.assertEqual(result["final_llm_tokens"],result["final_base_tokens"]+result["final_anyres_tokens"]+result["protected_newlines"])


if __name__=="__main__": unittest.main()
