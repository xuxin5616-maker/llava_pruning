"""Two GPUs: MVTec student training alongside independent VisA methods, then hybrid.

The evaluation GPU stays fixed for every measured method. Completed independent
trials are copied with their measurements to the final manifest, never rerun.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import benchmark_visa_methods as benchmark


PROJECT = Path(__file__).resolve().parent
MATCHED_FIELDS = ("gpu","model_path","model_config_sha256","data_root","dataset_info",
    "input_sha256","sample_count","prompt","seed","warmup","source_identity")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-gpu", required=True)
    p.add_argument("--eval-gpu", required=True)
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--mvtec-root", default="/home/yz/xxy/data/datasets/MVTec-AD")
    p.add_argument("--visa-root", default="/home/yz/xxy/data/datasets/VisA")
    p.add_argument("--visa-split-csv")
    p.add_argument("--prune-rates", nargs="+",type=int,choices=(50,70,90),default=[50,70,90])
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output-dir",type=Path,default=Path("output/mvtec_train_visa_two_gpu_01"))
    return p


def eval_options(args, output, checkpoint=None):
    options = ["--gpu",str(args.eval_gpu),"--model-path",args.model_path,"--data-root",args.visa_root,
        "--prune-rates",*(str(r) for r in args.prune_rates),"--warmup",str(args.warmup),
        "--seed",str(args.seed),"--output-dir",str(output)]
    if args.visa_split_csv: options += ["--visa-split-csv",args.visa_split_csv]
    if checkpoint is not None: options += ["--checkpoint",str(checkpoint)]
    return options


def commands(args, output):
    training = [sys.executable,"-u",str(PROJECT/"train_attention_predictor_mvtec.py"),
        "--model-path",args.model_path,"--data-root",args.mvtec_root,"--validation-fraction","0",
        "--epochs",str(args.epochs),"--seed",str(args.seed),"--output-dir",str(output/"student")]
    evaluation = [sys.executable,"-u",str(PROJECT/"benchmark_visa_methods.py"),
        *eval_options(args,output/"visa_independent")]
    return training,evaluation


def progress_line(path, training=False):
    if not path.is_file(): return "starting"
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-16384))
        lines = stream.read().decode("utf-8",errors="replace").splitlines()
    meaningful = [line for line in lines if ("total_ETA=" in line if training else
                 line.startswith(("Start ","Running ","Finished ","Summary:")))]
    return meaningful[-1] if meaningful else "loading/setup; details in log"


def interrupt(process):
    if process.poll() is not None: return
    # SIGINT lets the existing benchmark parent clean up its GPU worker.
    if os.name=="posix": process.send_signal(signal.SIGINT)
    else: process.terminate()
    try: process.wait(timeout=25)
    except subprocess.TimeoutExpired: process.kill();process.wait()


def launch_parallel(args, output):
    jobs = commands(args,output)
    logs = [output/"training.log",output/"visa_independent.log"]
    processes = []
    started = time.perf_counter()
    with ExitStack() as stack:
        try:
            # Spawn BOTH before waiting for either; each sees exactly one GPU.
            for name,gpu,command,path in zip(("training","VisA independent"),
                    (args.train_gpu,args.eval_gpu),jobs,logs):
                log = stack.enter_context(path.open("w",encoding="utf-8"))
                environment = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED="1")
                processes.append(subprocess.Popen(command,cwd=PROJECT,env=environment,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!="nt"))
                print(f"Started {name}: GPU {gpu}; log={path}",flush=True)
            next_update = started
            while any(p.poll() is None for p in processes):
                now = time.perf_counter()
                if now>=next_update:
                    print(f"Elapsed {now-started:.0f}s | train: {progress_line(logs[0],True)}",flush=True)
                    print(f"eval: {progress_line(logs[1])}",flush=True)
                    next_update = now+30
                time.sleep(2)
            return [p.wait() for p in processes]
        except BaseException:
            for process in processes: interrupt(process)
            raise


def prepare_final(args, output):
    checkpoint = output/"student/last.pt"
    status = json.loads((output/"student/training_status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or not checkpoint.is_file():
        raise ValueError("Training has not completed; cannot evaluate an intermediate last.pt")
    prior_dir = output/"visa_independent"
    prior = json.loads((prior_dir/"run.json").read_text(encoding="utf-8"))
    options = benchmark.parser().parse_args(eval_options(args,output/"visa_all_methods",checkpoint))
    final_dir,spec = benchmark.setup(options)
    for name in MATCHED_FIELDS:
        if prior[name]!=spec[name]: raise ValueError(f"Cannot reuse VisA measurements: {name} differs")
    expected = {job["job_id"]:job for job in spec["jobs"]}
    copied = []
    for job in prior["jobs"]:
        if job["status"]!="pending": continue  # no student: hybrid was skipped
        target = expected.get(job["job_id"])
        if target is None or target["config"]!=job["config"] or target["status"]!="pending":
            raise ValueError("Independent and final method configurations differ")
        source = prior_dir/job["job_id"]
        report = json.loads((source/"metrics.json").read_text(encoding="utf-8"))
        if (report.get("status")!="success" or not report.get("complete") or
                report.get("job_id")!=job["job_id"] or report.get("input_sha256")!=spec["input_sha256"] or
                report.get("evaluated_samples")!=spec["sample_count"]):
            raise ValueError(f"Cannot reuse an incomplete/mismatched trial: {job['job_id']}")
        shutil.copytree(source,final_dir/job["job_id"])
        copied.append(job["job_id"])
    if len(copied)!=sum(job["status"]=="pending" for job in prior["jobs"]):
        raise RuntimeError("Incomplete independent-result transfer")
    spec["reused_trials"] = dict(source=str(prior_dir),jobs=copied,
        timing_context="Independent trials ran on the fixed evaluation GPU while student trained on the other GPU; hybrid runs after training.")
    benchmark.save_json(final_dir/"run.json",spec)
    benchmark.write_reports(final_dir,spec)
    options.resume = True
    return options


def run(args):
    if (not str(args.train_gpu).isdigit() or not str(args.eval_gpu).isdigit() or
            int(args.train_gpu)==int(args.eval_gpu)):
        raise ValueError("Provide two DIFFERENT physical GPU numbers")
    if args.epochs<1 or args.warmup<0 or len(set(args.prune_rates))!=len(args.prune_rates):
        raise ValueError("Invalid epoch/warmup/rate configuration")
    for name in ("model_path","mvtec_root","visa_root"):
        path = Path(getattr(args,name)).expanduser().resolve()
        if not path.is_dir(): raise FileNotFoundError(f"Directory not found: {path}")
        setattr(args,name,str(path))
    if args.visa_split_csv: args.visa_split_csv = str(Path(args.visa_split_csv).expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Use a new/empty pipeline output directory: {output}")
    output.mkdir(parents=True,exist_ok=True)
    benchmark.save_json(output/"pipeline.json",dict(configuration={k:str(v) if isinstance(v,Path) else v
        for k,v in vars(args).items()},train_command=commands(args,output)[0],eval_command=commands(args,output)[1]))
    codes = launch_parallel(args,output)
    if any(codes):
        print(f"Training exit={codes[0]}; independent evaluation exit={codes[1]}. Saved outputs/logs: {output}",flush=True)
        return 1
    options = prepare_final(args,output)
    print("Training complete. Reusing independent results; running the three hybrid trials on the evaluation GPU.",flush=True)
    code = benchmark.run(options)
    print(f"Final CSV: {output/'visa_all_methods/summary.csv'}",flush=True)
    return code


def main():
    try: return run(parser().parse_args())
    except KeyboardInterrupt:
        print("Interrupted; outputs and logs are preserved.",flush=True);return 130
    except (ValueError,OSError) as error:
        print(f"Error: {error}",file=sys.stderr);return 1


if __name__=="__main__": sys.exit(main())
