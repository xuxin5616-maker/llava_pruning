"""Benchmark 0--90% on the first 100 samples; export curves and an Excel workbook.

Only the worker imports Torch, after CUDA_VISIBLE_DEVICES has been set by the
launcher. Peak memory covers model loading and the selected samples, not just
the last image. These are PyTorch allocator peaks, not whole-board nvidia-smi
usage (CUDA contexts and allocations outside PyTorch are not included).
"""

import argparse
import csv
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from benchmark_report import check_report_dependencies, generate_reports


PROJECT = Path(__file__).resolve().parent
PRUNE_RATES = tuple(range(0, 100, 10))
SAMPLE_LIMIT = 100
GIB = 1024 ** 3


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    parser.add_argument("--input-json", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl")
    parser.add_argument("--data-root", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/")
    parser.add_argument("--image-token-order", choices=("base_first", "anyres_first"),
                        default="base_first", help="Visual token block order (default: base_first)")
    parser.add_argument("--output-dir", type=Path,
                        help="New/empty experiment directory; default output/ex_YYYYMMDD_HHMMSS")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shared seed for all jobs; generated once if omitted")
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5],
                        help="GPU IDs, one job per GPU at a time (default: 4 5); use --gpus 5 for a serial sweep")
    parser.add_argument("--report-only", type=Path,
                        help="Regenerate charts/Excel/CSV from an existing experiment directory, without inference")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser


def read_first_samples(path):
    """Keep input order; do not even parse JSONL records beyond the prefix."""
    with Path(path).open(encoding="utf-8") as stream:
        if Path(path).suffix.lower() == ".jsonl":
            records = []
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
                    if len(records) == SAMPLE_LIMIT:
                        break
        else:
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError("Input JSON must be an array of samples")
            records = records[:SAMPLE_LIMIT]
    if not records:
        raise ValueError("Input contains no samples")
    return records


def prepare_jobs(args, output):
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("--gpus must contain distinct nonnegative GPU IDs")
    base = json.loads((PROJECT / "configs" / "fastv.json").read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else secrets.randbelow(2 ** 31)
    source_input = Path(args.input_json).expanduser().resolve()
    records = read_first_samples(source_input)
    # Both workers receive exactly the same prefix. The original input is untouched.
    subset_path = (output / "input_first_100.json").resolve()
    write_json(subset_path, records)
    print(f"Selected {len(records)} samples (first {SAMPLE_LIMIT} in input order); "
          "sample visualization disabled for all jobs", flush=True)
    jobs = []
    for index, rate in enumerate(PRUNE_RATES):
        gpu = args.gpus[index % len(args.gpus)]
        name = f"gpu{gpu}_prune_{rate:02d}"
        # Benchmark mode: disable both visualization selection and save flags.
        # The independent attention scores needed for pruning still run.
        config = dict(base, prune_rates=[rate], visualize_rates=[])
        config_path = output / f"{name}_config.json"
        write_json(config_path, config)
        job = {
            "gpu": gpu, "prune_rate": rate,
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "input_json": str(subset_path), "source_input_json": str(source_input),
            "sample_limit": SAMPLE_LIMIT, "selected_samples": len(records),
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "method_config": str(config_path),
            "output_dir": str(output / name),
            "resource_report": str(output / f"{name}_resources.json"),
            "log": str(output / f"{name}.log"),
            "seed": seed,
            "image_token_order": args.image_token_order,
        }
        spec_path = output / f"{name}_job.json"
        write_json(spec_path, job)
        jobs.append(dict(job, spec_path=str(spec_path)))
    return jobs


def worker_environment(gpu):
    environment = os.environ.copy()
    # Override an inherited CUDA_VISIBLE_DEVICES=5 rather than nesting indices.
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def run_worker(spec_path):
    started = time.perf_counter()
    job = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    report = {
        "gpu": job["gpu"], "prune_rate": job["prune_rate"], "pid": os.getpid(),
        "selected_samples": job["selected_samples"],
        "image_token_order": job.get("image_token_order", "base_first"),
        "status": "failed", "peak_allocated_bytes": None, "peak_reserved_bytes": None,
        "peak_allocated_gib": None, "peak_reserved_gib": None,
        "memory_scope": "PyTorch allocator on logical cuda:0; includes model load; excludes non-PyTorch allocations",
        "worker_wall_seconds": None, "generation_seconds_sum": None,
        "accuracy": None, "evaluated_samples": 0, "complete": False,
    }
    torch = None
    exit_code = 1
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(job["gpu"]):
            raise RuntimeError("GPU visibility must be configured before starting the worker")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each worker requires exactly one visible CUDA GPU")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        print(f"GPU {job['gpu']} -> cuda:0; pruning {job['prune_rate']}%; {report['gpu_name']}", flush=True)
        from llava_pruning.runner import run
        run(
            model_path=job["model_path"], input_json=job["input_json"],
            data_root=job["data_root"], prompt_version="v0", method_name="fastv",
            method_config=job["method_config"], roi_mode="anyres_max_9",
            save_prune_vis=False, save_attention_vis=False,
            output_dir=job["output_dir"], seed=job["seed"], no_sample=True,
            image_token_order=job.get("image_token_order", "base_first"),
        )
        torch.cuda.synchronize(0)
        report["status"] = "success"
        exit_code = 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "KeyboardInterrupt"
        exit_code = 130
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        report["worker_wall_seconds"] = time.perf_counter() - started
        if torch is not None and torch.cuda.is_initialized():
            try:
                for name, value in (
                    ("allocated", torch.cuda.max_memory_allocated(0)),
                    ("reserved", torch.cuda.max_memory_reserved(0)),
                ):
                    report[f"peak_{name}_bytes"] = int(value)
                    report[f"peak_{name}_gib"] = value / GIB
            except Exception as error:
                report["memory_error"] = str(error)
        rate_dir = Path(job["output_dir"]) / f"prune_{job['prune_rate']:02d}"
        try:
            metrics = rate_dir / "metrics.json"
            if metrics.is_file():
                values = json.loads(metrics.read_text(encoding="utf-8"))
                for name in ("accuracy", "evaluated_samples", "complete", "labeled_samples",
                             "correct", "incorrect", "unparsed"):
                    report[name] = values.get(name)
            predictions = rate_dir / "predictions.jsonl"
            if predictions.is_file():
                with predictions.open(encoding="utf-8") as stream:
                    report["generation_seconds_sum"] = sum(
                        float(json.loads(line)["generation_seconds"]) for line in stream if line.strip()
                    )
        except Exception as error:
            report["result_read_error"] = str(error)
        if exit_code == 0 and (not report["complete"] or
                               report["evaluated_samples"] != job["selected_samples"]):
            report.update(status="failed", error="Evaluation did not complete the selected sample count")
            exit_code = 1
        write_json(job["resource_report"], report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return exit_code


def launch_jobs(jobs, output):
    started = time.perf_counter()
    active = []
    pending = list(jobs)
    launch_error = None
    try:
        while pending or any("total_seconds" not in item for item in active):
            busy = {item["job"]["gpu"] for item in active if "total_seconds" not in item}
            for job in pending[:]:
                if job["gpu"] in busy:
                    continue
                log = Path(job["log"]).open("w", encoding="utf-8")
                try:
                    job_started = time.perf_counter()
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(PROJECT / "ex.py"), "--worker-spec", job["spec_path"]],
                        cwd=PROJECT, env=worker_environment(job["gpu"]),
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                except BaseException:
                    log.close()
                    raise
                active.append({"job": job, "process": process, "log": log, "started": job_started})
                pending.remove(job)
                busy.add(job["gpu"])
                print(f"Started GPU {job['gpu']}, prune={job['prune_rate']}%, log={job['log']}", flush=True)
            for item in active:
                if "total_seconds" not in item and item["process"].poll() is not None:
                    item["total_seconds"] = time.perf_counter() - item["started"]
                    item["log"].close()
                    print(f"Finished GPU {item['job']['gpu']}, prune={item['job']['prune_rate']}%, "
                          f"exit={item['process'].returncode}", flush=True)
            if pending or any("total_seconds" not in item for item in active):
                time.sleep(0.2)
    except (Exception, KeyboardInterrupt) as error:
        launch_error = f"{type(error).__name__}: {error}"
        print(launch_error, file=sys.stderr, flush=True)
    finally:
        # Only terminate processes created by this launcher; never other GPU jobs.
        for item in active:
            process = item["process"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            item.setdefault("total_seconds", time.perf_counter() - item["started"])
            item["log"].close()

    experiment_seconds = time.perf_counter() - started
    rows = []
    for job in jobs:
        item = next((entry for entry in active if entry["job"] is job), None)
        resource = Path(job["resource_report"])
        try:
            report = json.loads(resource.read_text(encoding="utf-8")) if resource.is_file() else {}
        except (OSError, ValueError) as error:
            report = {"status": "failed", "error": f"Cannot read worker resource report: {error}"}
        report.update({"gpu": job["gpu"], "prune_rate": job["prune_rate"],
                       "selected_samples": job["selected_samples"],
                       "image_token_order": job.get("image_token_order", "base_first"),
                       "output_dir": job["output_dir"], "log": job["log"],
                       "exit_code": item["process"].returncode if item else None,
                       "total_seconds": item["total_seconds"] if item else None})
        if (report["exit_code"] != 0 or not report.get("complete") or
                report.get("evaluated_samples") != job["selected_samples"]):
            report["status"] = "failed"
            report.setdefault("error", launch_error or "Worker failed; see its log (possibly killed/OOM)")
        write_json(resource, report)
        rows.append(report)
        allocated = report.get("peak_allocated_gib")
        reserved = report.get("peak_reserved_gib")
        peak = f"allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB" if allocated is not None and reserved is not None else "unavailable"
        print(f"GPU {job['gpu']} prune={job['prune_rate']}% status={report['status']} "
              f"peak VRAM: {peak}; total={report['total_seconds']} s", flush=True)
    summary = {
        "experiment_wall_seconds": experiment_seconds,
        "time_scope": "launch to exit, including interpreter/model loading, inference and JSON/log saving; no sample visualization; excludes queue wait and final chart/Excel export; polling resolution 0.2 s",
        "generation_time_scope": "sum of existing per-image synchronized generate() durations, excludes preprocessing/model loading/result saving; includes pruning-score computation; no warmup exclusion",
        "memory_scope": "PyTorch peak allocated/reserved; GiB=1024^3 bytes, not whole-board usage",
        "settings": {"prompt_version": "v0", "roi_mode": "anyres_max_9", "do_sample": False,
                     "image_token_order": jobs[0].get("image_token_order", "base_first"),
                     "save_prune_vis": False, "save_attention_vis": False,
                     "model_path": jobs[0]["model_path"], "input_json": jobs[0]["input_json"],
                     "source_input_json": jobs[0]["source_input_json"],
                     "sample_limit": jobs[0]["sample_limit"],
                     "selected_samples": jobs[0]["selected_samples"],
                     "prune_rates": [job["prune_rate"] for job in jobs],
                     "gpus": sorted({job["gpu"] for job in jobs}),
                     "gpu_assignment": [{"prune_rate": job["prune_rate"], "gpu": job["gpu"]} for job in jobs],
                     "method_config": json.loads((PROJECT / "configs" / "fastv.json").read_text(encoding="utf-8")),
                     "sample_selection": "first records in input order; no shuffling",
                     "data_root": jobs[0]["data_root"], "seed": jobs[0]["seed"]},
        "jobs": rows,
    }
    write_json(output / "benchmark.json", summary)
    fields = ("gpu", "prune_rate", "status", "exit_code", "total_seconds", "worker_wall_seconds",
              "generation_seconds_sum", "peak_allocated_gib", "peak_reserved_gib",
              "selected_samples", "evaluated_samples", "accuracy", "complete")
    with (output / "benchmark.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"All {len(jobs)} jobs wall time: {experiment_seconds:.2f} s; summary: {output / 'benchmark.csv'}", flush=True)
    report_ok = True
    try:
        generate_reports(output / "benchmark.json")
        print(f"Saved charts: {output / 'benchmark_curves.png'} and .pdf; "
              f"Excel: {output / 'benchmark.xlsx'}", flush=True)
    except Exception as error:
        report_ok = False
        print(f"Report export failed: {error}. Raw JSON/CSV are saved. "
              f"Retry without inference: python ex.py --report-only {output}", file=sys.stderr, flush=True)
    return 0 if report_ok and launch_error is None and all(row.get("status") == "success" for row in rows) else 1


def main():
    args = build_parser().parse_args()
    if args.worker_spec is not None:
        return run_worker(args.worker_spec)
    if args.report_only is not None:
        generate_reports(args.report_only.expanduser().resolve() / "benchmark.json")
        print(f"Regenerated charts, Excel and CSV in {args.report_only}")
        return 0
    for name in ("model_path", "data_root", "input_json"):
        path = Path(getattr(args, name)).expanduser()
        if not (path.is_file() if name == "input_json" else path.is_dir()):
            raise FileNotFoundError(f"Invalid {name}: {path}")
    output = (args.output_dir or PROJECT / "output" / datetime.now().strftime("ex_%Y%m%d_%H%M%S")).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {output}")
    # Fail before loading any checkpoint if reporting dependencies are missing.
    check_report_dependencies()
    output.mkdir(parents=True, exist_ok=True)
    return launch_jobs(prepare_jobs(args, output), output)


if __name__ == "__main__":
    raise SystemExit(main())
