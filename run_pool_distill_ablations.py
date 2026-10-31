"""One command: tail-image removal and Base+Text-only side-input ablations.

Four separate sweeps reuse the original worker, GPU scheduler and resume
journals. Defaults: 3 rates x 3 lambdas x (3 tails x 2 sources + 1 Base+Text)
= 63 trials. No images, no training, no dataset truncation.
"""

import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_pool_distill as entry
from llava_pruning.pool_ablation_report import FIELDS


GROUPS = (("tail_2", 2, "full", ("anyres_self", "base_full")),
          ("tail_3", 3, "full", ("anyres_self", "base_full")),
          ("tail_4", 4, "full", ("anyres_self", "base_full")),
          ("base_text", 0, "base_text", ("base_full",)))


def build_parser():
    parser = entry.build_parser()
    parser.description = __doc__
    parser.set_defaults(prune_rates=[30.,50.,70.], lambdas=[.7,.8,.9],
                        summary_format="compact", output_dir=entry.PROJECT/"output/pool_distill_ablations_01")
    return parser


def group_command(args, group, root):
    name, tail, branch, modes = group
    command = [sys.executable, str(entry.PROJECT/"run_pool_distill.py")]
    command += (["--gpus", *map(str,args.gpus)] if args.gpus else ["--gpu",str(args.gpu)])
    for name_arg in ("model_path","checkpoint","input_json","data_root","prompt_version",
                     "pruning_audit","seed","warmup","baseline_acc","allocation_temperature","spatial_mix_weight"):
        command += ["--"+name_arg.replace("_","-"),str(getattr(args,name_arg))]
    command += ["--prune-rates",*map(str,args.prune_rates),"--lambdas",*map(str,args.lambdas),
                "--selection-modes",*modes,"--redundancy-layers",*map(str,args.redundancy_layers),
                "--visionzip-layers",*map(str,args.visionzip_layers),"--tail-drop-layers",str(tail),
                "--predictor-input",branch,"--summary-format","compact","--output-dir",str(root/name)]
    if args.resume and (root/name/"comparison.json").is_file():
        command.append("--resume")
    return command


def collect_summary(root):
    """Only read completed/committed child summaries; never infer missing rows."""
    rows = []
    for name,_,_,_ in GROUPS:
        path = root/name/"summary.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig",newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected ablation summary columns: {path}")
            rows.extend(reader)
    if len({row["config"] for row in rows}) != len(rows):
        raise ValueError("Duplicate ablation configurations")
    entry.write_csv(root/"summary.csv",rows)
    return rows


def restore_args(args, argv):
    path = args.output_dir.expanduser().resolve()/"suite.json"
    saved = json.loads(path.read_text(encoding="utf-8"))["configuration"]
    explicit = {arg.split("=",1)[0] for arg in argv if arg.startswith("--")}
    changeable = {"gpu","gpus","output_dir","resume"}
    for key,value in saved.items():
        flag = "--"+key.replace("_","-")
        if key in changeable:
            continue
        if flag in explicit and not entry.same_setting(key,getattr(args,key),value):
            raise ValueError(f"Cannot resume with changed {flag}")
        if flag not in explicit:
            setattr(args,key,Path(value) if key in {"model_path","checkpoint","input_json","data_root"} else value)
    if "--gpus" not in explicit and "--gpu" not in explicit:
        args.gpus, args.gpu = saved.get("gpus"),saved["gpu"]
    return args


def run(args, *, process_factory=None, poll_interval=.2):
    if args.worker or args.report_only is not None or args.tail_drop_layers or args.predictor_input!="full":
        raise ValueError("The suite supplies its own worker/tail/input settings; use run_pool_distill.py for an individual trial")
    if args.summary_format != "compact" or args.selection_modes != ["anyres_self","base_full"]:
        raise ValueError("This suite always runs both tail sources and Base-mapped Base+Text, with compact summaries")
    if any(s<=0 for s in args.lambdas):
        raise ValueError("Ablations require lambda > 0")
    root = entry.validate(args)
    root.mkdir(parents=True,exist_ok=True)
    path = root/"suite.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if args.resume else dict(
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        prompt_format=entry.PROMPT_FORMAT, groups={g[0]:dict(status="pending") for g in GROUPS},
        definitions=dict(acc="percent",Lat="mean first-token latency in ms; actual pruning included",
            mem="peak PyTorch allocated GiB during full generation; includes resident weights and inputs",
            tail="extra removal; original two-stage average budgets unchanged",
            base_text="only side input rows are selected from H2; original position IDs preserved"))
    entry.require_current_prompt_format(manifest)
    manifest.update(complete=False,status="running",expected_trials=len(args.prune_rates)*len(args.lambdas)*7)
    entry.write_json(path,manifest)
    factory = process_factory or subprocess.Popen
    try:
        for group in GROUPS:
            name = group[0]
            command = group_command(args,group,root)
            manifest["groups"][name].update(status="running",command=command)
            entry.write_json(path,manifest)
            print(f"Starting group {name}; total requested trials={manifest['expected_trials']}",flush=True)
            options = dict(cwd=str(entry.PROJECT),env=dict(os.environ,PYTHONUNBUFFERED="1"))
            # Same foreground process group: Ctrl+C also reaches the child
            # scheduler, which owns and checkpoints its GPU workers.
            process = factory(command,**options)
            try:
                while process.poll() is None:
                    collect_summary(root)
                    time.sleep(poll_interval)
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    if os.name=="posix": process.send_signal(signal.SIGINT)
                    else: process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            if code in (130,-2): raise KeyboardInterrupt
            if code: raise RuntimeError(f"Group {name} failed with exit code {code}; see its terminal/logs")
            manifest["groups"][name]["status"]="complete"
            manifest["completed_trials"]=len(collect_summary(root))
            entry.write_json(path,manifest)
        if manifest["completed_trials"] != manifest["expected_trials"]:
            raise RuntimeError("Suite finished without all expected result rows")
    except BaseException as error:
        collect_summary(root)
        manifest.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        entry.write_json(path,manifest)
        raise
    manifest.update(complete=True,status="complete")
    manifest.pop("error",None)
    entry.write_json(path,manifest)
    print(f"Done {manifest['completed_trials']} trials: {root/'summary.csv'}",flush=True)
    return root


def main():
    args = build_parser().parse_args()
    if args.resume: args=restore_args(args,sys.argv[1:])
    run(args)


if __name__=="__main__":
    try: main()
    except KeyboardInterrupt:
        print("Interrupted; resume this suite with --resume and the same --output-dir",flush=True)
        raise SystemExit(130)
