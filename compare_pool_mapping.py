"""Paired full-data Base+Text runs before/after Base mapping optimization.

Each rate/lambda pair stays on one physical GPU, reference then fast, one model
at a time. Reuses the existing inference worker, metrics and sample journals.
"""

from collections import deque
from copy import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_pool_distill as entry
from llava_pruning.pool_distill_parallel import worker_command,stop_active
from llava_pruning.pool_ablation_report import compact_row
from llava_pruning.metrics import OPTION


def build_parser():
    parser=entry.build_parser()
    parser.description=__doc__
    parser.set_defaults(gpu=7,prune_rates=[30.,70.],lambdas=[.7,.8,.9],selection_modes=["base_full"],
        predictor_input="base_text",tail_drop_layers=0,summary_format="compact",pruning_audit="lean",
        output_dir=entry.PROJECT/"output/base_text_mapping_compare_01")
    return parser


def restore_args(args,argv):
    manifest=json.loads((args.output_dir.resolve()/"comparison.json").read_text(encoding="utf-8"))
    if manifest.get("kind")!="paired_base_mapping":
        raise ValueError("This is not a paired Base-mapping comparison directory")
    explicit={value.split("=",1)[0] for value in argv if value.startswith("--")}
    for key,value in manifest["configuration"].items():
        if key in {"output_dir","resume"}: continue
        flag="--"+key.replace("_","-")
        if flag in explicit and not entry.same_setting(key,getattr(args,key),value):
            raise ValueError(f"Cannot change {flag} during paired resume; keep the original GPU assignments and settings")
        if flag not in explicit:
            setattr(args,key,Path(value) if key in {"model_path","checkpoint","input_json","data_root"} else value)
    return args


def pair_check(reference,fast):
    """Compare all saved answers by ID without running inference or changing CSVs."""
    def predictions(folder):
        rows,_=entry.read_prefix(folder/"predictions.jsonl")
        result={str(row["question_id"]):row for row in rows}
        if len(result)!=len(rows): raise ValueError("Duplicate prediction IDs in paired comparison")
        return result
    a,b=predictions(reference),predictions(fast)
    if set(a)!=set(b): raise ValueError("Paired runs do not have the same prediction IDs")
    raw,parsed=0,0
    for key,left in a.items():
        right=b[key]
        if left["gt"]!=right["gt"]: raise ValueError("Paired prediction GT differs")
        raw+=left["answer"]!=right["answer"]
        def label(answer):
            match=OPTION.match(answer)
            return match.group(1).upper() if match else None
        parsed+=label(left["answer"])!=label(right["answer"])
    return dict(samples=len(a),raw_answer_differences=raw,parsed_answer_differences=parsed)


def run(args,*,process_factory=None,poll_interval=.2):
    if (args.worker or args.report_only is not None or args.selection_modes!=["base_full"]
            or args.predictor_input!="base_text" or args.tail_drop_layers!=0 or args.summary_format!="compact"
            or args.pruning_audit!="lean" or args.base_mapping!="reference"):
        raise ValueError("This comparison fixes base_full, Base+Text, tail=0, lean and compact; it runs both mapping implementations itself")
    if any(not 0<s<1 for s in args.lambdas):
        raise ValueError("Compare the hybrid Base mapping with lambda strictly between 0 and 1")
    args.gpus=args.gpus or [args.gpu]
    root=entry.validate(args);root.mkdir(parents=True,exist_ok=True)
    path=root/"comparison.json"
    manifest=json.loads(path.read_text(encoding="utf-8")) if args.resume else dict(
        kind="paired_base_mapping",created_utc=datetime.now(timezone.utc).isoformat(),prompt_format=entry.PROMPT_FORMAT,
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},trials=[],pair_checks={},
        definitions=dict(acc="percent",Lat="mean first-token latency in ms; all actual pruning included",
            mem="peak PyTorch allocated GiB over full generation; resident weights/inputs included",
            execution="fixed GPU per rate/lambda; reference then fast; one active model per GPU"))
    entry.require_current_prompt_format(manifest)
    if manifest.get("kind")!="paired_base_mapping": raise ValueError("Wrong comparison directory kind")
    manifest.update(complete=False,status="running",expected_trials=2*len(args.prune_rates)*len(args.lambdas))
    existing={(t["rate"],t["distill_share"],t["mapping"]):t for t in manifest["trials"]}
    if len(existing)!=len(manifest["trials"]): raise ValueError("Duplicate saved comparison trials")
    order,completed,jobs=[],{},{}
    queues={gpu:deque() for gpu in args.gpus}
    pairs=[(rate,share) for rate in args.prune_rates for share in args.lambdas]
    for index,(rate,share) in enumerate(pairs):
        gpu=args.gpus[index%len(args.gpus)]
        for mapping in ("reference","fast"):
            key=(rate,share,mapping)
            tag=f"prune_{rate:g}_lambda_{share:g}_{mapping}"
            folder=root/tag
            trial=existing.get(key)
            if trial is None:
                trial=dict(rate=rate,distill_share=share,mapping=mapping,selection_mode="base_full",
                           gpu_id=gpu,directory=str(folder));manifest["trials"].append(trial)
            if trial["gpu_id"]!=gpu: raise ValueError("Saved paired GPU assignment changed")
            worker=copy(args);worker.base_mapping=mapping;worker.gpu=gpu
            job=dict(key=key,tag=tag,folder=folder,trial=trial,args=worker)
            jobs[key]=job;order.append(key)
            if args.resume and entry.verified_complete(folder,worker,rate,share,"base_full"):
                completed[key]=entry.read_trial_rows(folder,rate,share,"base_full",args.baseline_acc)[0]
                trial.update(status="complete",reused=True)
                print(f"Reusing GPU{gpu}: {tag}",flush=True)
            else:
                trial.update(status="pending");queues[gpu].append(job)
    logs=root/"logs";logs.mkdir(exist_ok=True)

    def save():
        rows=[]
        for key in order:
            if key not in completed: continue
            row=dict(completed[key]);row.setdefault("predictor_input","base_text");row.setdefault("tail_drop_layers",0)
            row["base_mapping"]=key[2]
            row=compact_row(row)
            if key[2]=="reference": row["config"]+=";mapping=reference"
            rows.append(row)
        entry.write_csv(root/"summary.csv",rows)
        for rate,share in pairs:
            name=f"prune_{rate:g}_lambda_{share:g}"
            ref,fast=(rate,share,"reference"),(rate,share,"fast")
            if name in manifest["pair_checks"] or ref not in completed or fast not in completed: continue
            check=pair_check(jobs[ref]["folder"],jobs[fast]["folder"])
            a,b=completed[ref],completed[fast]
            check.update(gpu_id=jobs[ref]["trial"]["gpu_id"],
                lat_reference_ms=float(a["lat_mean_ms"]),lat_fast_ms=float(b["lat_mean_ms"]),
                lat_speedup=float(a["lat_mean_ms"])/float(b["lat_mean_ms"]),
                acc_delta_percentage_points=100*(float(b["accuracy"])-float(a["accuracy"])))
            manifest["pair_checks"][name]=check
            print(f"Pair {name}: differing parsed answers={check['parsed_answer_differences']}; "
                  f"Lat speedup={check['lat_speedup']:.3f}x",flush=True)
        manifest["completed_trials"]=len(rows)
        entry.write_json(path,manifest)

    active={};factory=process_factory or subprocess.Popen
    started=last_status=time.monotonic();durations=[]
    save()
    print(f"Paired comparison: {len(pairs)} pairs, GPUs {args.gpus}; reference/fast stay on the same GPU",flush=True)
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
                durations.append(time.monotonic()-job["started"])
                save()
                print(f"Finished GPU{gpu}: {job['tag']}; {len(completed)}/{len(order)} complete",flush=True)
            now=time.monotonic()
            if now-last_status>=30:
                # Independent per-GPU queues: slowest outstanding queue sets the ETA.
                remaining=max((len(queues[gpu])+int(gpu in active) for gpu in queues),default=0)
                eta="estimating" if not durations else f"~{remaining*sum(durations)/len(durations):.0f}s"
                print(f"Complete {len(completed)}/{len(order)}, elapsed={now-started:.0f}s ETA={eta}; per-image progress in logs/",flush=True)
                last_status=now
            if active: time.sleep(poll_interval)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        for job in active.values(): job["trial"]["status"]="interrupted"
        try: save()
        finally: stop_active(active)
        raise
    manifest.update(complete=True,status="complete");manifest.pop("error",None);save()
    print(f"Completed {len(completed)} real-model trials: {root/'summary.csv'}",flush=True)
    return root


def main():
    args=build_parser().parse_args()
    if args.resume: args=restore_args(args,sys.argv[1:])
    run(args)


if __name__=="__main__":
    try: main()
    except KeyboardInterrupt:
        print("Interrupted; use compare_pool_mapping.py --resume with the same output directory",flush=True)
        raise SystemExit(130)
