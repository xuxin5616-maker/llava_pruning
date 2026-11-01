"""Full-data Base+Text / fast Base-mapping tail sweep, default 192 trials.

Fractions0.3/0.5/0.7 mean rates30/50/70 percent in THIS launcher. Each rate/lambda
stays on one GPU for all tails. Existing model worker and sample resume reused.
"""

from collections import deque
from copy import copy
from datetime import datetime,timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import run_pool_distill as entry
from llava_pruning.pool_ablation_report import compact_row
from llava_pruning.pool_distill_parallel import worker_command,stop_active


def build_parser():
    parser=entry.build_parser()
    parser.description=__doc__
    parser.add_argument("--tail-start",type=int,default=0)
    parser.add_argument("--tail-end",type=int,default=15,help="Inclusive: 0..15 runs sixteen tails")
    parser.set_defaults(gpu=7,prune_rates=[.3,.5,.7],lambdas=[.6,.7,.8,.9],selection_modes=["base_full"],
        predictor_input="base_text",base_mapping="fast",pruning_audit="lean",summary_format="compact",
        output_dir=entry.PROJECT/"output/base_text_fast_tail_0_15_01")
    return parser


def normalize_rates(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Pruning rates must be finite and nonempty")
    if all(0<v<1 for v in values):
        return [float(Decimal(str(v))*100) for v in values]
    if all(1<=v<100 for v in values):
        return list(values)
    raise ValueError("Use all fractions (0.3 0.5 0.7) or all percentages (30 50 70); do not mix units")


def restore_args(args,argv):
    data=json.loads((args.output_dir.resolve()/"comparison.json").read_text(encoding="utf-8"))
    if data.get("kind")!="base_text_fast_tail_sweep":
        raise ValueError("Not a Base+Text fast tail-sweep directory")
    explicit={arg.split("=",1)[0] for arg in argv if arg.startswith("--")}
    args.prune_rates=normalize_rates(args.prune_rates)
    for key,value in data["configuration"].items():
        if key in {"output_dir","resume"}: continue
        flag="--"+key.replace("_","-")
        if flag in explicit and not entry.same_setting(key,getattr(args,key),value):
            raise ValueError(f"Cannot change {flag} when resuming this fixed-GPU tail sweep")
        if flag not in explicit:
            setattr(args,key,Path(value) if key in {"model_path","checkpoint","input_json","data_root"} else value)
    return args


def run(args,*,process_factory=None,poll_interval=.2):
    args.prune_rates=normalize_rates(args.prune_rates)
    if (args.worker or args.report_only is not None or args.selection_modes!=["base_full"]
            or args.predictor_input!="base_text" or args.base_mapping!="fast" or args.pruning_audit!="lean"
            or args.summary_format!="compact" or args.tail_drop_layers!=0):
        raise ValueError("This sweep fixes base_full, base_text, fast, lean and compact; tail-start/end control the tails")
    if not 0<=args.tail_start<=args.tail_end<=26:
        raise ValueError("Tail bounds must satisfy 0 <= start <= end <= 26 for this 28-block model")
    if any(not 0<s<1 for s in args.lambdas):
        raise ValueError("The hybrid fast sweep requires lambda strictly between 0 and 1")
    args.gpus=args.gpus or [args.gpu]
    root=entry.validate(args);root.mkdir(parents=True,exist_ok=True)
    path=root/"comparison.json"
    manifest=json.loads(path.read_text(encoding="utf-8")) if args.resume else dict(
        kind="base_text_fast_tail_sweep",created_utc=datetime.now(timezone.utc).isoformat(),
        prompt_format=entry.PROMPT_FORMAT,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        trials=[],definitions=dict(acc="percent",Lat="mean first-token ms including all actual pruning",
            mem="peak PyTorch allocated GiB over full generation including resident weights/inputs",
            rate="original two-stage layer-average target; extra tail removal is not deducted from that budget",
            tail="remove Base/AnyRes/image-newlines before the last N blocks; text position IDs preserved",
            execution="one model per GPU; every rate/lambda stays on the same physical GPU across all tails"))
    entry.require_current_prompt_format(manifest)
    if manifest.get("kind")!="base_text_fast_tail_sweep": raise ValueError("Wrong sweep directory kind")
    tails=range(args.tail_start,args.tail_end+1)
    manifest.update(complete=False,status="running",expected_trials=len(args.prune_rates)*len(args.lambdas)*len(tails))
    previous={(t["rate"],t["distill_share"],t["tail_drop_layers"]):t for t in manifest["trials"]}
    if len(previous)!=len(manifest["trials"]): raise ValueError("Duplicate saved tail configurations")
    order,completed=[],{}
    queues={gpu:deque() for gpu in args.gpus}
    pairs=[(rate,share) for rate in args.prune_rates for share in args.lambdas]
    for pair_index,(rate,share) in enumerate(pairs):
        gpu=args.gpus[pair_index%len(args.gpus)]
        for tail in tails:
            key=(rate,share,tail);order.append(key)
            tag=f"prune_{rate:g}_lambda_{share:g}_tail_{tail:02d}"
            folder=root/tag
            trial=previous.get(key)
            if trial is None:
                trial=dict(rate=rate,distill_share=share,tail_drop_layers=tail,selection_mode="base_full",
                           gpu_id=gpu,directory=str(folder));manifest["trials"].append(trial)
            if trial["gpu_id"]!=gpu: raise ValueError("Saved tail-sweep GPU assignment changed")
            worker=copy(args);worker.gpu=gpu;worker.tail_drop_layers=tail
            job=dict(key=key,tag=tag,folder=folder,trial=trial,args=worker)
            if args.resume and entry.verified_complete(folder,worker,rate,share,"base_full"):
                completed[key]=entry.read_trial_rows(folder,rate,share,"base_full",args.baseline_acc)[0]
                trial.update(status="complete",reused=True)
            else:
                trial.update(status="pending");queues[gpu].append(job)
    logs=root/"logs";logs.mkdir(exist_ok=True)

    def save():
        rows=[]
        for key in order:
            if key not in completed: continue
            row=dict(completed[key])
            for field,value in (("predictor_input","base_text"),("base_mapping","fast"),("tail_drop_layers",key[2])):
                if field in row and row[field]!=value: raise ValueError(f"Wrong worker result for {field}")
                row[field]=value
            rows.append(compact_row(row))
        entry.write_csv(root/"summary.csv",rows)
        manifest["completed_trials"]=len(rows)
        entry.write_json(path,manifest)

    active={};factory=process_factory or subprocess.Popen
    started=last_status=time.monotonic();durations=[]
    save()
    print(f"Base+Text fast tail sweep: {manifest['expected_trials']} trials, tails {args.tail_start}..{args.tail_end}, "
          f"GPUs {args.gpus}; reused {len(completed)}",flush=True)
    try:
        while any(queues.values()) or active:
            for gpu,queue in queues.items():
                if gpu in active or not queue: continue
                job=queue.popleft();rate,share,_=job["key"]
                command=worker_command(job["args"],entry,rate,share,"base_full",gpu,job["folder"])
                log=logs/(job["tag"]+".log")
                stream=log.open("a" if args.resume else "w",encoding="utf-8")
                job["trial"].update(status="running",command=command,log=str(log))
                save()
                options=dict(cwd=str(entry.PROJECT),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED="1"),
                             stdout=stream,stderr=subprocess.STDOUT)
                if os.name=="posix": options["start_new_session"]=True
                try: process=factory(command,**options)
                except BaseException:
                    stream.close();job["trial"]["status"]="failed";raise
                active[gpu]=dict(job,process=process,stream=stream,started=time.monotonic())
                print(f"Started GPU{gpu}: {job['tag']}; log={log}",flush=True)
            for gpu,job in list(active.items()):
                code=job["process"].poll()
                if code is None: continue
                job["process"].wait();job["stream"].close();del active[gpu]
                if code:
                    job["trial"].update(status="interrupted" if code in (130,-2) else "failed",exit_code=code)
                    if code in (130,-2): raise KeyboardInterrupt
                    raise RuntimeError(f"{job['tag']} exited {code}; see {job['trial']['log']}")
                rate,share,_=job["key"]
                completed[job["key"]]=entry.read_trial_rows(job["folder"],rate,share,"base_full",args.baseline_acc)[0]
                job["trial"].update(status="complete",exit_code=0)
                durations.append(time.monotonic()-job["started"]);save()
                print(f"Finished GPU{gpu}: {job['tag']}; {len(completed)}/{len(order)} complete",flush=True)
            now=time.monotonic()
            if now-last_status>=30:
                remaining=max((len(queues[g])+int(g in active) for g in queues),default=0)
                eta="estimating" if not durations else f"~{remaining*sum(durations)/len(durations):.0f}s"
                print(f"Complete {len(completed)}/{len(order)}, elapsed={now-started:.0f}s ETA={eta}; per-image progress: logs/",flush=True)
                last_status=now
            if active: time.sleep(poll_interval)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        for job in active.values(): job["trial"]["status"]="interrupted"
        try: save()
        finally: stop_active(active)
        raise
    manifest.update(complete=True,status="complete");manifest.pop("error",None);save()
    print(f"Completed {len(completed)} full-data trials: {root/'summary.csv'}",flush=True)
    return root


def main():
    args=build_parser().parse_args()
    if args.resume: args=restore_args(args,sys.argv[1:])
    run(args)


if __name__=="__main__":
    try: main()
    except KeyboardInterrupt:
        print("Interrupted; resume with run_pool_tail_sweep.py --resume and the same output directory",flush=True)
        raise SystemExit(130)
