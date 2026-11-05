"""Rerun the SIX existing VisA pruning baselines on identical stretched 3x3 input.

Reuses benchmark_visa_methods.py workers, algorithms, budgets and metric output.
No student checkpoint, hybrid method, lambda, tail or model-config override.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_visa_methods as common
from benchmark_visa_grid3 import cache_fingerprint, input_fingerprint, resize_image, validate_grid


PROJECT = Path(__file__).resolve().parent
KIND = "visa-six-methods-stretch-grid3-v1"
METHODS = ("fastv", "visionzip", "pyramiddrop", "anchorprune", "clse", "priortr")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+", type=int, default=[3, 4])
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--data-root", default="/home/yz/xxy/data/datasets/VisA")
    p.add_argument("--visa-split-csv", help="Default: DATA_ROOT/split_csv/1cls.csv; test rows only")
    p.add_argument("--prune-rates", nargs="+", type=int, choices=(50, 70, 90), default=[50, 70, 90])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("output/visa_stretch_methods_50_70_90_03"))
    p.add_argument("--resume", action="store_true")
    return p


def schedule(rates, gpus):
    jobs = [job for job in common.schedule(rates, include_hybrid=False) if job["method"] != "hybrid_baseline"]
    for job in jobs:
        index = METHODS.index(job["method"]) if job["method"] in METHODS else 0
        job["gpu"] = gpus[index % len(gpus)]
        job["config"] = "resize=stretch;canvas=1152;grid=3x3;" + job["config"] + f";gpu={job['gpu']}"
    return jobs


def verify(spec):
    if spec.get("kind") != KIND or spec["version"] != common.VERSION:
        raise ValueError("Not this stretched six-method benchmark")
    if spec.get("source_policy") != "current_directory" and spec["source_identity"] != common.source_identity():
        raise ValueError("Inference source changed; use a new output directory")
    common.verify_inputs(spec)
    records = [json.loads(line) for line in Path(spec["input_json"]).read_text(encoding="utf-8").splitlines()]
    if len(records) != spec["sample_count"] or any(row["image_size"] != [1152, 1152] for row in records):
        raise ValueError("Frozen square test input changed")
    if cache_fingerprint(records, Path(spec["data_root"])) != spec["prepared_images_sha256"]:
        raise ValueError("Prepared image pixels changed")
    from llava_pruning.visa_data import load_visa_test
    samples, info = load_visa_test(spec["raw_data_root"], spec["dataset_info"]["split_csv"])
    if info != spec["dataset_info"] or input_fingerprint(samples) != spec["raw_inputs_sha256"]:
        raise ValueError("Original images or official test split changed")


def setup(args, *, source_policy="current_directory"):
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        spec = json.loads((output / "run.json").read_text(encoding="utf-8"))
        verify(spec)
        return output, spec
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0
            or not args.prune_rates or len(set(args.prune_rates)) != len(args.prune_rates)
            or args.warmup < 0 or not 0 <= args.seed < 2**32):
        raise ValueError("Invalid GPUs, rates, warmup or seed")
    if source_policy not in {"current_directory", "strict"}:
        raise ValueError("Unknown benchmark source policy")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty directory or --resume: {output}")
    model = Path(args.model_path).expanduser().resolve()
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != 28:
        raise ValueError("Expected the existing 28-layer Qwen2 model")
    validate_grid(SimpleNamespace(**config))  # Original candidates; no mutation.
    from PIL import Image
    from llava_pruning.visa_data import load_visa_test
    raw = Path(args.data_root).expanduser().resolve()
    split = Path(args.visa_split_csv).expanduser().resolve() if args.visa_split_csv else raw / "split_csv/1cls.csv"
    samples, info = load_visa_test(raw, split)
    raw_identity = input_fingerprint(samples)
    prepared = output / "prepared"
    (prepared / "images").mkdir(parents=True)
    rows = []
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        with Image.open(sample.image) as original:
            image, geometry = resize_image(original, "stretch")
        relative = f"images/{index:06d}.png"
        image.save(prepared / relative, compress_level=1)
        rows.append(dict(sample.source, image=relative, image_size=[1152, 1152], transform=geometry))
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(samples):
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(samples) - index - 1) / (index + 1)
            print(f"Prepare stretch {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    if input_fingerprint(samples) != raw_identity:
        raise ValueError("Original images changed during preparation")
    input_path = output / "visa_test_stretch.jsonl"
    input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    spec = dict(version=common.VERSION, kind=KIND, created_utc=datetime.now(timezone.utc).isoformat(),
        model_path=str(model), model_config_sha256=common.digest(model / "config.json"),
        raw_data_root=str(raw), data_root=str(prepared), dataset_info=info, raw_inputs_sha256=raw_identity,
        input_json=str(input_path), input_sha256=common.digest(input_path),
        prepared_images_sha256=cache_fingerprint(rows, prepared), sample_count=len(samples),
        source_identity=common.source_identity(), source_policy=source_policy, checkpoint=None, gpus=args.gpus,
        seed=args.seed, warmup=args.warmup, jobs=schedule(args.prune_rates, args.gpus),
        preprocessing=dict(mode="stretch", size=[1152,1152], resample="BICUBIC", padding=False,
            candidate_policy="Preserve checkpoint grid candidates; natural 3x3 selection verified"),
        definitions=dict(common.DEFINITIONS,
            relative_accuracy="100*ACC / newly measured stretched Base+AnyRes unpruned ACC in THIS run",
            rate="Mean removed image-token fraction across 28 LLM layers; native stretched Base+AnyRes+newlines denominator",
            timing_policy="Existing resident-input TTFT and full-answer peak memory. ALL online pruning included; CPU resize/PNG/cache I/O/H2D/model load/warmups/saving excluded.",
            methods="Six existing methods only; no hybrid, student training, lambda or tail",
            parallelism="Separate GPU workers; all rates of a method stay on the same GPU, saved in config"),
        preparation_seconds=time.perf_counter()-started)
    spec["definitions"]["source_policy"] = (
        "Each worker uses current project code at startup and records its source identity; no source copy or source-change blocking. Input, model and inference identities remain checked."
        if source_policy == "current_directory" else "Source identity must match experiment setup.")
    common.save_json(output / "run.json", spec)
    return output, spec


def run(args, *, process_factory=None, setup_function=None, worker_script=None,
        manifest_name="run.json", report_writer=None):
    output, spec = (setup_function or setup)(args)
    save_reports = report_writer or common.write_reports
    save_reports(output, spec)
    queues = {gpu:deque() for gpu in spec["gpus"]}
    for job in spec["jobs"]:
        if job["status"] == "pending":
            queues[job["gpu"]].append(job)
        else:
            print(f"SKIP {job['job_id']}: {job['reason']}", flush=True)
    active, failures = {}, []
    factory = process_factory or subprocess.Popen
    total = sum(len(queue) for queue in queues.values())
    completed = 0
    last_progress = time.perf_counter()
    print(f"{spec['sample_count']} official VisA test images; stretch 1152; 3x3; "
          f"{total} valid trials; GPUs={spec['gpus']}; no visualizations", flush=True)
    try:
        while active or any(queues.values()):
            for gpu, queue in queues.items():
                while gpu not in active and queue:
                    job = queue.popleft()
                    report_path = output / job["job_id"] / "metrics.json"
                    old = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
                    if old and old.get("input_sha256") != spec["input_sha256"]:
                        raise ValueError("Saved report input identity changed")
                    if old.get("status") == "success" and old.get("complete"):
                        completed += 1
                        print(f"Reuse completed {job['job_id']}", flush=True); continue
                    command = [sys.executable, "-u", str(worker_script or PROJECT / "benchmark_visa_methods.py"),
                               "--worker", job["job_id"], "--manifest", str(output / manifest_name)]
                    logfile = output / (job["job_id"] + ".log")
                    log = logfile.open("a", encoding="utf-8")
                    try:
                        process = factory(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=os.name != "nt",
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1"))
                    except BaseException:
                        log.close(); raise
                    active[gpu] = (process, log, job, time.perf_counter())
                    print(f"Start GPU {gpu}: {job['job_id']}; log={logfile}", flush=True)
            for gpu, (process, log, job, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del active[gpu]
                report_path = output / job["job_id"] / "metrics.json"
                result = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
                if code or result.get("status") != "success" or not result.get("complete"):
                    failures.append(job["job_id"])
                    report_path.parent.mkdir(exist_ok=True)
                    error = result.get("error")
                    if not error:
                        lines = (output / (job["job_id"] + ".log")).read_text(encoding="utf-8", errors="replace").splitlines()
                        detail = next((line.strip() for line in reversed(lines) if line.strip()), "no complete result")
                        error = f"Worker exit={code}; {detail}"
                    common.save_json(report_path, dict(result, job_id=job["job_id"], status="failed", complete=False,
                                                      input_sha256=spec["input_sha256"], error=error))
                    print(f"FAILED {job['job_id']}: {error}\nLog: {output/(job['job_id']+'.log')}", flush=True)
                else:
                    completed += 1
                print(f"Finished GPU {gpu}: {job['job_id']}; exit={code}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
                save_reports(output, spec)
            if time.perf_counter() - last_progress >= 30:
                for gpu, (_, _, job, started) in active.items():
                    count = len(common.read_prefix(output / job["job_id"] / "predictions.jsonl")[0])
                    print(f"GPU {gpu}: {job['job_id']} {count}/{spec['sample_count']} saved; "
                          f"elapsed={time.perf_counter()-started:.0f}s; sample ETA in log", flush=True)
                last_progress = time.perf_counter()
            if active:
                time.sleep(.25)
    except BaseException:
        for process, log, _, _ in active.values():
            if process.poll() is None:
                if os.name == "posix": process.send_signal(signal.SIGINT)
                else: process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            log.close()
        raise
    finally:
        save_reports(output, spec)
    spec.update(complete=not failures, completed_trials=completed, failed_trials=failures)
    common.save_json(output / "run.json", spec)
    print(f"Completed {completed}/{total} valid trials; failed={len(failures)}; "
          f"summary={output/'summary.csv'}; budgets={output/'budget_summary.csv'}", flush=True)
    return int(bool(failures))


def main():
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
