"""CPU-only independent-trial scheduling on GPU5/6/7, failure and resume."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import run_pool_distill as entry
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.pool_distill_parallel import run_parallel_sweep


class Processes:
    def __init__(self, *, fail_gpu=None, interrupt_gpu=None):
        self.live={}
        self.launches=[]
        self.processes=[]
        self.max_live=0
        self.stopped=[]
        self.fail_gpu=fail_gpu
        self.interrupt_gpu=interrupt_gpu

    def __call__(self, command, **options):
        trial=entry.build_parser().parse_args(command[2:])
        gpu=trial.gpu
        if gpu in self.live:
            raise AssertionError("Two active models on one GPU")
        if options["env"]["CUDA_VISIBLE_DEVICES"] != str(gpu):
            raise AssertionError("Worker must see exactly its assigned GPU")
        if trial.gpus is not None or not trial.worker:
            raise AssertionError("Parent pool leaked into child CLI")
        self.launches.append(trial)
        process=Process(self,trial,options)
        self.processes.append(process)
        self.live[gpu]=process
        self.max_live=max(self.max_live,len(self.live))
        return process


class Process:
    def __init__(self, owner, trial, options):
        self.owner,self.trial,self.options=owner,trial,options
        self.returncode=None
        self.polls=0
        self.threshold=7 if trial.gpu==5 else 1

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.owner.interrupt_gpu==self.trial.gpu:
            self.owner.interrupt_gpu=None
            raise KeyboardInterrupt
        self.polls+=1
        if self.polls >= self.threshold:
            self.returncode=2 if self.owner.fail_gpu==self.trial.gpu else 0
            if self.returncode==0:
                self.complete()
            self.owner.live.pop(self.trial.gpu,None)
        return self.returncode

    def complete(self):
        a=self.trial
        a.output_dir.mkdir(parents=True,exist_ok=True)
        metrics=extended_metrics(dict(prune_rate=a.prune_rates[0],distill_share=a.lambdas[0],
            selection_mode=a.selection_modes[0],complete=True,tp=1,fp=0,tn=0,fn=0,accuracy=1.,
            evaluated_samples=1,lat_mean_ms=100.,mem_peak_allocated_gib=20.,
            actual_average_prune_rate=a.prune_rates[0]+.001,gpu_id=a.gpu),a.baseline_acc/100)
        entry.write_json(a.output_dir/"metrics.json",metrics)
        entry.write_csv(a.output_dir/"summary.csv",[metrics])
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        entry.write_json(a.output_dir/"run.json",dict(complete=True,configuration=config,prompt_format=entry.PROMPT_FORMAT))
        prediction=dict(question_id="1",gt=1,answer="A",lat_seconds=.1,generation_seconds=.2,
            generated_tokens=1,peak_allocated_bytes=100,peak_reserved_bytes=200)
        entry.write_text_atomic(a.output_dir/"predictions.jsonl",json.dumps(prediction)+"\n")
        self.options["stdout"].write("mock per-image progress\n")

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode=130
            self.owner.live.pop(self.trial.gpu,None)
        return self.returncode

    def send_signal(self, value):
        self.owner.stopped.append(self.trial.gpu)
        self.returncode=130
        self.owner.live.pop(self.trial.gpu,None)

    def terminate(self): self.send_signal(None)
    def kill(self): self.send_signal(None)


class ParallelTests(unittest.TestCase):
    def fixture(self, root, *extra):
        (root/"model").mkdir()
        (root/"data").mkdir()
        (root/"best.pt").write_bytes(b"mock")
        entry.write_json(root/"input.json",[dict(question_id="1",gt=1,image="image.png")])
        return entry.build_parser().parse_args(["--gpus","5","6","7","--model-path",str(root/"model"),
            "--data-root",str(root/"data"),"--input-json",str(root/"input.json"),
            "--checkpoint",str(root/"best.pt"),"--output-dir",str(root/"output"),*extra])

    def rows(self, root):
        with (root/"output/summary.csv").open(encoding="utf-8-sig",newline="") as stream:
            return list(csv.DictReader(stream))

    def test_three_gpu_dynamic_queue_all30_unique_jobs_and_same_worker_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.fixture(root)
            processes=Processes()
            run_parallel_sweep(args,api=entry,process_factory=processes,poll_interval=0)
            self.assertEqual(processes.max_live,3)
            self.assertFalse(processes.live)
            self.assertEqual(len(processes.launches),30)
            self.assertEqual([p.gpu for p in processes.launches[:3]],[5,6,7])
            self.assertNotEqual(processes.launches[3].gpu,5)  # fast GPU starts another before slowGPU5 finishes
            combinations={(p.prune_rates[0],p.lambdas[0],p.selection_modes[0]) for p in processes.launches}
            self.assertEqual(len(combinations),30)
            self.assertEqual({p.lambdas[0] for p in processes.launches},{.01,.25,.5,.75,.99})
            self.assertTrue(all(p.warmup==10 and p.seed==42 and p.prompt_version=="v0" for p in processes.launches))
            rows=self.rows(root)
            self.assertEqual(len(rows),30)
            self.assertEqual({int(r["gpu_id"]) for r in rows},{5,6,7})
            self.assertEqual([float(r["prune_rate"]) for r in rows],[30.]*10+[50.]*10+[70.]*10)
            self.assertTrue(all("f1" in r and "bacc" in r and "relative_accuracy_pct" in r for r in rows))
            manifest=json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertFalse(manifest["isolated_sequential_processes"])
            self.assertEqual(manifest["gpus"],[5,6,7])
            self.assertTrue(all(t["status"]=="complete" and Path(t["log"]).is_file() for t in manifest["trials"]))
            self.assertEqual(len(list((root/"output/logs").glob("*.log"))),30)

    def test_public_dispatch_and_resume_restore_pool_without_new_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.fixture(root,"--prune-rates","30","--lambdas",".01",".99")
            processes=Processes()
            entry.run_sweep(args,process_factory=processes)
            argv=["--resume","--output-dir",str(root/"output")]
            resume=entry.restore_sweep_args(entry.build_parser().parse_args(argv),argv)
            self.assertEqual(resume.gpus,[5,6,7])
            self.assertEqual(resume.prune_rates,[30])
            self.assertEqual(resume.lambdas,[.01,.99])
            def forbidden(*a,**k): raise AssertionError("Complete trials should be reused")
            entry.run_sweep(resume,process_factory=forbidden)
            self.assertEqual(len(self.rows(root)),4)
            self.assertTrue(all(t["reused"] for t in json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))["trials"]))
            argv += ["--gpu","6"]
            single=entry.restore_sweep_args(entry.build_parser().parse_args(argv),argv)
            self.assertIsNone(single.gpus)
            self.assertEqual(single.gpu,6)

    def test_worker_failure_stops_and_reaps_other_gpu_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.fixture(root)
            processes=Processes(fail_gpu=6)
            with self.assertRaisesRegex(RuntimeError,"Worker exited2"):
                run_parallel_sweep(args,api=entry,process_factory=processes,poll_interval=0)
            self.assertFalse(processes.live)
            self.assertTrue(processes.stopped)
            manifest=json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["status"],"failed")
            self.assertTrue(any(t["status"]=="failed" for t in manifest["trials"]))
            self.assertTrue(any(t["status"]=="interrupted" for t in manifest["trials"]))
            self.assertTrue(all(p.options["stdout"].closed for p in processes.processes))

    def test_sigint_stops_all_workers_and_preserves_resumable_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.fixture(root)
            processes=Processes(interrupt_gpu=6)
            with self.assertRaises(KeyboardInterrupt):
                run_parallel_sweep(args,api=entry,process_factory=processes,poll_interval=0)
            self.assertFalse(processes.live)
            manifest=json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"],"interrupted")
            self.assertFalse(manifest["complete"])
            self.assertFalse(any(t["status"]=="running" for t in manifest["trials"]))

    def test_duplicate_negative_and_worker_gpu_pool_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.fixture(root)
            for ids in ([5,5,7],[-1,6,7],[]):
                args.gpus=ids
                with self.assertRaises(ValueError): entry.validate(args)
            args.gpus=[5,6,7]
            args.worker=True
            with self.assertRaisesRegex(ValueError,"Each worker"):
                entry.validate(args)

    def test_real_cpu_child_processes_receive_one_gpu_and_produce_logs_and_summary(self):
        # Real Popen/stdout/env/process reaping, without importing Torch or using
        # actual GPUs in children. Scheduler tests above cover nonoverlap and FIFO.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.fixture(root,"--prune-rates","30","--lambdas",".01",".99")
            source="""import json,os,sys,time
from pathlib import Path
spec=json.loads(sys.argv[1])
assert os.environ['CUDA_VISIBLE_DEVICES']==str(spec['gpu'])
time.sleep(.05)
folder=Path(spec['folder']); folder.mkdir()
row=dict(complete=True,prune_rate=30,distill_share=spec['share'],selection_mode=spec['mode'],
         tp=1,fp=0,tn=0,fn=0,accuracy=1.,lat_mean_ms=100.,mem_peak_allocated_gib=20.,actual_average_prune_rate=30.001)
(folder/'metrics.json').write_text(json.dumps(row),encoding='utf-8')
print('CPU process received physical GPU'+str(spec['gpu']),flush=True)
"""
            children=[]
            def actual(command,**options):
                trial=entry.build_parser().parse_args(command[2:])
                spec=dict(gpu=trial.gpu,folder=str(trial.output_dir),share=trial.lambdas[0],mode=trial.selection_modes[0])
                child=subprocess.Popen([sys.executable,"-c",source,json.dumps(spec)],**options)
                children.append(child)
                return child
            run_parallel_sweep(args,api=entry,process_factory=actual,poll_interval=.01)
            self.assertEqual(len(children),4)
            self.assertTrue(all(p.poll()==0 for p in children))
            self.assertEqual(len(self.rows(root)),4)
            text="\n".join(p.read_text(encoding="utf-8") for p in (root/"output/logs").glob("*.log"))
            for gpu in (5,6,7): self.assertIn(f"physical GPU{gpu}",text)


if __name__=="__main__": unittest.main()
