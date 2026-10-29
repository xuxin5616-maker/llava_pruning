"""Single-GPU hybrid trials scheduled over a physical GPU pool.

Parent uses stdlib only. Every child sees ONE CUDA device and writes its own
output/log. Only the parent edits the root manifest/CSV. Existing worker math,
resolved prompts, measurement, sample checkpoints and resume checks are reused.
"""

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker_command(args, api, rate, share, mode, gpu, folder):
    command = [sys.executable, str(Path(api.__file__).resolve()), "--worker", "--gpu", str(gpu),
        "--model-path", str(args.model_path), "--input-json", str(args.input_json),
        "--data-root", str(args.data_root), "--checkpoint", str(args.checkpoint),
        "--prune-rates", str(rate), "--lambdas", str(share), "--selection-modes", mode,
        "--prompt-version", args.prompt_version, "--warmup", str(args.warmup),
        "--seed", str(args.seed), "--baseline-acc", str(args.baseline_acc),
        "--allocation-temperature", str(args.allocation_temperature),
        "--spatial-mix-weight", str(args.spatial_mix_weight),
        "--redundancy-layers", *map(str, args.redundancy_layers),
        "--visionzip-layers", *map(str, args.visionzip_layers), "--output-dir", str(folder)]
    if args.resume:
        command.append("--resume")
    return command


def stop_active(active, *, grace=20.):
    """Give workers time to close their sample journals; reap every child."""
    for item in active.values():
        process = item["process"]
        if process.poll() is None:
            try:
                if os.name == "posix":
                    process.send_signal(signal.SIGINT)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    for item in active.values():
        process = item["process"]
        try:
            process.wait(timeout=max(.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        finally:
            item["stream"].close()


def run_parallel_sweep(args, *, api, process_factory=None, poll_interval=.2):
    output = api.validate(args)
    if args.worker or not args.gpus or len(args.gpus) < 2:
        raise ValueError("Parallel parent requires at least two --gpus")
    factory = process_factory or subprocess.Popen
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "comparison.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        api.require_current_prompt_format(manifest)
    else:
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), trials=[],
            prune_rates=args.prune_rates, lambdas=args.lambdas, selection_modes=args.selection_modes,
            prompt_format=api.PROMPT_FORMAT, visualization=False,
            lambda_definition="distillation contribution / target WHOLE native-packed layer-average removal",
            base_policy="MyPool compresses Base + AnyRes; block2 protects remaining Base",
            checkpoint_policy="reuse existing predictor; no training")
    manifest.update(complete=False, gpus=list(args.gpus), isolated_sequential_processes=False,
        execution_mode="independent_single_gpu_trials", max_concurrent_trials=len(args.gpus),
        baseline_accuracy_reference=api.baseline_reference(args.baseline_acc),
        metric_definitions=api.METRIC_DEFINITIONS,
        configuration={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()})
    if args.resume:
        manifest["last_resumed_utc"] = datetime.now(timezone.utc).isoformat()
    existing = {(t["rate"],t["selection_mode"],t["distill_share"]): t for t in manifest["trials"]}
    if len(existing) != len(manifest["trials"]):
        raise ValueError("Duplicate saved trial configurations")
    order, ready, completed = [], deque(), {}
    for rate in args.prune_rates:
        for mode in args.selection_modes:
            for share in args.lambdas:
                key = (rate, mode, share)
                tag = f"prune_{rate:g}_{mode}_lambda_{share:g}"
                folder = output / tag
                order.append(key)
                trial = existing.get(key)
                if trial is None:
                    trial = dict(rate=rate,distill_share=share,selection_mode=mode,directory=str(folder))
                    manifest["trials"].append(trial)
                if args.resume and api.verified_complete(folder,args,rate,share,mode):
                    rows = api.read_trial_rows(folder,rate,share,mode,args.baseline_acc)
                    # Old complete single-GPU outputs may lack this CSV column.
                    if "gpu_id" not in rows[0]:
                        run = json.loads((folder/"run.json").read_text(encoding="utf-8"))
                        rows[0]["gpu_id"] = run["configuration"]["gpu"]
                    completed[key] = rows
                    trial.update(status="complete",reused=True)
                    trial.pop("error",None)
                    print(f"Skipping verified complete trial: {tag}",flush=True)
                else:
                    trial.update(status="pending")
                    ready.append(dict(key=key,tag=tag,folder=folder,trial=trial))
    logs = output / "logs"
    logs.mkdir(exist_ok=True)

    def save():
        rows = [row for key in order if key in completed for row in completed[key]]
        api.write_csv(output/"summary.csv",rows)
        api.write_json(manifest_path,manifest)

    save()
    active, free = {}, deque(args.gpus)
    started, last_status, durations = time.monotonic(), time.monotonic(), []
    print(f"GPU pool {args.gpus}: at most one trial per GPU; {len(ready)} pending, {len(completed)} reused",flush=True)
    try:
        while ready or active:
            while ready and free:
                gpu, job = free.popleft(), ready.popleft()
                rate, mode, share = job["key"]
                command = worker_command(args,api,rate,share,mode,gpu,job["folder"])
                log_path = logs / (job["tag"] + ".log")
                stream = log_path.open("a" if args.resume else "w",encoding="utf-8")
                stream.write(f"\nGPU {gpu}; started {datetime.now(timezone.utc).isoformat()}\n")
                stream.flush()
                job["trial"].update(status="running",gpu_id=gpu,command=command,log=str(log_path),
                    attempts=job["trial"].get("attempts",0)+1)
                job["trial"].pop("error",None)
                options = dict(cwd=str(api.PROJECT),
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED="1"),
                    stdout=stream,stderr=subprocess.STDOUT)
                if os.name == "posix":
                    options["start_new_session"] = True
                try:
                    save()
                    process = factory(command,**options)
                except BaseException as error:
                    stream.close()
                    job["trial"].update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
                    raise
                active[gpu] = dict(job,process=process,stream=stream,started=time.monotonic())
                print(f"Started GPU {gpu}: {job['tag']}; log={log_path}",flush=True)
            finished = False
            for gpu, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                finished = True
                item["process"].wait()
                item["stream"].close()
                del active[gpu]
                if code:
                    item["trial"].update(status="interrupted" if code in (130,-2) else "failed",
                        exit_code=code,error=f"Worker exited{code}; see {item['trial']['log']}")
                    if code in (130,-2):
                        raise KeyboardInterrupt
                    raise RuntimeError(item["trial"]["error"])
                try:
                    rate,mode,share = item["key"]
                    rows = api.read_trial_rows(item["folder"],rate,share,mode,args.baseline_acc)
                    for row in rows:
                        row["gpu_id"] = gpu
                    completed[item["key"]] = rows
                except Exception as error:
                    item["trial"].update(status="failed",error=str(error))
                    raise
                duration = time.monotonic()-item["started"]
                durations.append(duration)
                item["trial"].update(status="complete",exit_code=0,worker_wall_seconds=duration)
                item["trial"].pop("error",None)
                free.append(gpu)
                save()
                print(f"Finished GPU {gpu}: {item['tag']}; complete {len(completed)}/{len(order)}",flush=True)
            now = time.monotonic()
            if now-last_status >= 30:
                estimate = sum(durations)/len(durations)*(len(ready)+len(active))/len(args.gpus) if durations else None
                eta = "estimating" if estimate is None else f"~{estimate:.0f}s"
                jobs = ", ".join(f"GPU{gpu}:{item['tag']}" for gpu,item in active.items())
                print(f"Completed {len(completed)}/{len(order)}, elapsed={now-started:.0f}s, ETA={eta}; {jobs}. Per-image progress: logs/",flush=True)
                last_status = now
            if active and not finished:
                time.sleep(poll_interval)
    except BaseException as error:
        for item in active.values():
            item["trial"].update(status="interrupted",error="Scheduler stopped; partial sample journal can be resumed")
        manifest["status"] = "interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
        try:
            save()
        finally:
            stop_active(active)
        raise
    manifest.update(complete=True,status="complete")
    save()
    print(f"Completed {len(completed)} configurations on GPUs{args.gpus}: {output/'summary.csv'}",flush=True)
    return output
