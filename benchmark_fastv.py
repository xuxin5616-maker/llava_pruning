"""Sequential full-dataset LLaVA/FastV-mask benchmark. No plots or 100-image cap.

Run on the inference server: python benchmark_fastv.py --gpu 5
Each configuration uses a fresh process, the same GPU, and warmup before timing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime


ROOT = Path(__file__).resolve().parent
RATES = (0, 50, 70, 90)
GIB = 1024 ** 3
DEFINITIONS = {
    "baseline": "Same LLaVA checkpoint/backend, pruning disabled; not a separate checkpoint.",
    "fastv": "Existing attention-mask implementation; no physical token/KV sequence compaction.",
    "lat": "Seconds from synchronized resident inputs entering model.generate to first selected "
           "token appended and synchronized. Includes vision/projector/packing/prefill/pruning. "
           "Excludes model loading, file I/O, CPU preprocessing, H2D input transfer and saving.",
    "memory": "Maximum over measured samples of PyTorch peak allocated/reserved bytes during "
              "FULL answer generation, including resident model weights and inputs. Reset after "
              "input preparation each call; loading/preprocessing/warmup transients excluded. "
              "Reserved includes allocator cache. Not nvidia-smi whole-board usage.",
    "memory_percent": "100 * peak PyTorch allocated bytes / device total bytes; NOT GPU compute utilization.",
    "generation": "Full synchronized generate duration; includes first-token measurement sync overhead.",
    "accuracy": "Full generated answer, not a forced one-token answer. A=defect, B=normal. "
                "ACC includes unparsed labeled answers as errors; PRE/Recall/TNR use parsed "
                "labeled answers only. Zero denominator=null. Metrics pooled across images.",
    "units": "accuracy/precision/recall/tnr are fractions [0,1]; lat_*_ms are milliseconds; "
             "mem_peak_*_gib are GiB; mem_peak_allocated_pct is percent [0,100].",
}
CSV_FIELDS = (
    "model", "prune_rate", "status", "complete", "expected_samples", "evaluated_samples",
    "labeled_samples", "correct", "unparsed", "tp", "fp", "tn", "fn",
    "accuracy", "precision", "recall", "tnr", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms",
    "lat_std_ms", "lat_speedup_vs_baseline", "mem_peak_allocated_gib", "mem_peak_reserved_gib",
    "mem_peak_allocated_pct", "device_total_gib", "generation_mean_ms", "mean_generated_tokens",
    "warmup_samples", "worker_wall_seconds", "error",
)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=5, help="Physical GPU ID; all four jobs run sequentially")
    parser.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    parser.add_argument("--input-json", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl")
    parser.add_argument("--data-root", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/")
    parser.add_argument("--method-config", default=str(ROOT / "configs" / "fastv.json"),
                        help="Reuse layer/min_tokens/newline settings; rates fixed to 0,50,70,90")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup images per job, excluded from metrics (default 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", help="New or empty directory; default output/benchmark_fastv_<timestamp>")
    parser.add_argument("--worker-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--worker-rate", type=int, choices=RATES, help=argparse.SUPPRESS)
    return parser


def prepare(args):
    from llava_pruning.data import load_samples
    from llava_pruning.methods import FastVMethod
    from llava_pruning.prompts import resolve_prompt

    if args.gpu < 0 or args.warmup < 1 or not 0 <= args.seed < 2 ** 32:
        raise ValueError("gpu must be nonnegative, warmup >= 1, seed in [0, 2**32)")
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model checkpoint not found: {model_path}")
    source = Path(args.input_json).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    method = FastVMethod.from_config(args.method_config)
    samples = load_samples(source, data_root)
    if not samples:
        raise ValueError("Input dataset is empty")
    if any(sample.gt is None for sample in samples):
        raise ValueError("This classification benchmark requires gt=0/1 on every sample")
    for sample in samples:
        resolve_prompt("v0", sample.category, sample.text)
    effective = {"layer": method.layer, "prune_rates": list(RATES), "visualize_rates": [],
                 "min_tokens": method.min_tokens, "preserve_image_newline": method.preserve_image_newline}
    output = Path(args.output_dir or f"output/benchmark_fastv_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "fastv_effective.json", effective)
    spec = {"model_path": str(model_path), "input_json": str(source), "data_root": str(data_root),
            "input_sha256": file_sha256(source), "sample_count": len(samples), "gpu": args.gpu,
            "seed": args.seed, "warmup": args.warmup, "prune_rates": list(RATES),
            "prompt_version": "v0", "roi_mode": "anyres_max_9", "image_token_order": "base_first",
            "batch_size": 1, "decoding": "greedy, max_new_tokens<=512 (existing LLaVA context budget)",
            "do_sample": False, "include_pruning_time": True, "save_visualizations": False,
            "method_config": str(output / "fastv_effective.json"), "fastv": effective,
            "definitions": DEFINITIONS}
    save_json(output / "run.json", spec)
    return output, spec


def warmup_indices(count, requested):
    size = min(count, requested)
    return [round(i * (count - 1) / max(size - 1, 1)) for i in range(size)]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measurement_summary(rows, total_memory):
    if not rows:
        return {}
    latencies = [row["lat_seconds"] * 1000 for row in rows]
    allocated = max(row["peak_allocated_bytes"] for row in rows)
    reserved = max(row["peak_reserved_bytes"] for row in rows)
    return {
        "lat_mean_ms": statistics.mean(latencies), "lat_p50_ms": percentile(latencies, 0.5),
        "lat_p95_ms": percentile(latencies, 0.95), "lat_std_ms": statistics.pstdev(latencies),
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "mem_peak_allocated_gib": allocated / GIB, "mem_peak_reserved_gib": reserved / GIB,
        "mem_peak_allocated_pct": 100 * allocated / total_memory,
        "device_total_gib": total_memory / GIB,
        "generation_mean_ms": statistics.mean(row["generation_seconds"] for row in rows) * 1000,
        "mean_generated_tokens": statistics.mean(row["generated_tokens"] for row in rows),
    }


def validate_single_gpu(model, torch):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Benchmark requires exactly one visible GPU")
    if any(parameter.device != torch.device("cuda:0") for parameter in model.parameters()):
        raise RuntimeError("Model must fit entirely on cuda:0; CPU/disk offloading is not comparable")


def run_worker(manifest, rate):
    # Parent sets CUDA_VISIBLE_DEVICES before this process imports Torch.
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
    output = Path(manifest).parent / f"prune_{rate:02d}"
    output.mkdir(exist_ok=False)
    from llava_pruning.metrics import Accuracy
    accuracy = Accuracy(spec["sample_count"])
    report = {"model": "LLaVA (no pruning)" if rate == 0 else f"FastV-mask {rate}%",
              "prune_rate": rate, "status": "failed", "warmup_samples": 0}
    measurements = []
    total_memory = None
    try:
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.data import load_samples
        from llava_pruning.methods import FastVMethod
        from llava_pruning.prompts import resolve_prompt

        if file_sha256(spec["input_json"]) != spec["input_sha256"]:
            raise ValueError("Input JSON changed after benchmark setup")
        samples = load_samples(spec["input_json"], spec["data_root"])
        if len(samples) != spec["sample_count"]:
            raise ValueError("Dataset size changed after benchmark setup")
        prompts = [resolve_prompt("v0", sample.category, sample.text)[0] for sample in samples]
        method = FastVMethod.from_config(spec["method_config"])
        if json.loads(Path(spec["method_config"]).read_text(encoding="utf-8")) != spec["fastv"]:
            raise ValueError("Effective FastV configuration changed")
        print(f"Loading {report['model']}; {len(samples)} evaluation images, GPU {spec['gpu']}", flush=True)
        torch.manual_seed(spec["seed"])
        backend = LlavaBackend(spec["model_path"], "anyres_max_9", image_token_order="base_first")
        validate_single_gpu(backend.model, torch)
        if method.layer >= len(backend.model.get_model().layers):
            raise ValueError("FastV layer must be below the model's decoder layer count")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        report["inference_config"] = backend.inference_config

        def generate(index):
            return backend.generate(
                samples[index], prompts[index], method, rate, "anyres_max_9",
                capture_visualization=False, capture_attention=False,
                random_seed=(spec["seed"] + index) % (2 ** 32), do_sample=False,
                include_pruning_time=True, benchmark_metrics=True,
            )

        indices = warmup_indices(len(samples), spec["warmup"])
        for position, index in enumerate(indices, 1):
            generate(index)  # Identical instrumented path; discard ALL warmup metrics.
            report["warmup_samples"] += 1
            print(f"prune={rate}% warmup {position}/{len(indices)}", flush=True)
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for index, sample in enumerate(samples):
                result = generate(index)
                measurement = {key: result[key] for key in (
                    "lat_seconds", "generation_seconds", "generated_tokens",
                    "peak_allocated_bytes", "peak_reserved_bytes")}
                row = {"question_id": sample.sample_id, "origin_path": sample.origin_path,
                       "gt": sample.gt, "answer": result["answer"], **measurement}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                accuracy.add(sample.gt, result["answer"])
                measurements.append(measurement)
                del result
                if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(samples):
                    print(f"prune={rate}% evaluated {index + 1}/{len(samples)}; "
                          f"last Lat={measurement['lat_seconds'] * 1000:.2f} ms", flush=True)
        report["status"] = "success"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        report.update(accuracy.result(rate, complete=report["status"] == "success"))
        report.update(measurement_summary(measurements, total_memory))
        save_json(output / "metrics.json", report)
    return 0 if report["status"] == "success" else 1


def write_summary(output, reports):
    baseline = next((row for row in reports if row["prune_rate"] == 0
                     and row.get("status") == "success" and row.get("complete")), None)
    for row in reports:
        row["lat_speedup_vs_baseline"] = (
            baseline["lat_mean_ms"] / row["lat_mean_ms"]
            if baseline and row.get("status") == "success" and row.get("complete")
            and row.get("lat_mean_ms", 0) > 0 else None)
    save_json(output / "summary.json", {"definitions": DEFINITIONS, "results": reports})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)


def run_child(manifest, rate, gpu, log_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker-manifest", str(manifest), "--worker-rate", str(rate)]
    process = None
    with Path(log_path).open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()


def launch(args):
    output, spec = prepare(args)
    print(f"Full dataset: {spec['sample_count']} images; rates={RATES}; "
          f"GPU={spec['gpu']}; FastV layer={spec['fastv']['layer']}; output={output}", flush=True)
    print("FP16/FlashAttention2 unchanged; greedy; warmup per job; no visualization. "
          "Lat includes pruning. Mem is PyTorch allocated/reserved, not whole-board usage.", flush=True)
    reports = []
    for rate in RATES:
        start = time.perf_counter()
        print(f"\nStarting prune={rate}% on GPU {spec['gpu']} (new process)", flush=True)
        exit_code = run_child(output / "run.json", rate, spec["gpu"], output / f"prune_{rate:02d}.log")
        metrics_path = output / f"prune_{rate:02d}" / "metrics.json"
        if metrics_path.is_file():
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            report = {"model": "LLaVA (no pruning)" if rate == 0 else f"FastV-mask {rate}%",
                      "prune_rate": rate, "status": "failed", "complete": False,
                      "error": f"Worker exited {exit_code} without metrics; inspect its log"}
        if exit_code != 0:
            report.update(status="failed", complete=False)
            report.setdefault("error", f"Worker exit={exit_code}; inspect its log")
        report["worker_wall_seconds"] = time.perf_counter() - start
        reports.append(report)
        write_summary(output, reports)
        if report["status"] == "success":
            scores = " ".join(f"{name}={report[key] * 100:.2f}%" if report[key] is not None
                              else f"{name}=N/A" for name, key in (
                                  ("ACC", "accuracy"), ("PRE", "precision"),
                                  ("Recall", "recall"), ("TNR", "tnr")))
            print(f"Finished prune={rate}%: {scores}; Lat(mean)={report['lat_mean_ms']:.2f} ms; "
                  f"Mem(allocated)={report['mem_peak_allocated_gib']:.3f} GiB "
                  f"({report['mem_peak_allocated_pct']:.2f}%); unparsed={report['unparsed']}", flush=True)
        else:
            print(f"FAILED prune={rate}%: {report.get('error')}; continuing remaining jobs", flush=True)
    print(f"Summary: {output / 'summary.csv'}", flush=True)
    return 0 if all(row["status"] == "success" for row in reports) else 1


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_manifest:
        if args.worker_rate is None:
            parser.error("--worker-manifest requires --worker-rate")
        return run_worker(args.worker_manifest, args.worker_rate)
    if args.worker_rate is not None:
        parser.error("--worker-rate is an internal option")
    try:
        return launch(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted; completed reports and partial predictions are preserved.\n")


if __name__ == "__main__":
    raise SystemExit(main())
