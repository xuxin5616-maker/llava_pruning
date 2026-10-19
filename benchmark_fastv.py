"""Sequential full-dataset LLaVA/FastV-mask benchmark. No plots or 100-image cap.

Run on the inference server: python benchmark_fastv.py --gpu 5
Each configuration uses a fresh process, the same GPU, and warmup before timing.
One baseline plus FastV layers 2/3 independently at 50/70/90%: seven jobs.
Use --dataset visa for raw/prepared VisA test data with category-free Triad v0.
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
LAYERS = (2, 3)
DEFAULT_MVTEC_JSON = "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"
DEFAULT_DATA_ROOTS = {
    "mvtec": "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/",
    "visa": "/home/yz/xxy/data/datasets/VisA",
}
GIB = 1024 ** 3
DEFINITIONS = {
    "baseline": "Same LLaVA checkpoint/backend, pruning disabled; not a separate checkpoint.",
    "fastv": "Existing attention-mask implementation; no physical token/KV sequence compaction.",
    "layers": "Independent layer=2 and layer=3 experiments, each at 50/70/90%; "
              "not two pruning stages in one model. One shared no-pruning baseline (fastv_layer=null).",
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
    "dataset", "model", "fastv_layer", "prune_rate", "status", "complete", "expected_samples", "evaluated_samples",
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


def job_info(rate, layer=None):
    if rate == 0 and layer is None:
        return {"job_id": "prune_00", "output_subdir": "prune_00",
                "model": "LLaVA (no pruning)", "fastv_layer": None, "prune_rate": 0}
    if rate not in RATES[1:] or layer not in LAYERS:
        raise ValueError("Use baseline rate=0 without a layer, or FastV layer=2/3 with rate=50/70/90")
    return {"job_id": f"layer_{layer}_prune_{rate:02d}",
            "output_subdir": f"layer_{layer}/prune_{rate:02d}",
            "model": f"FastV-mask layer={layer} {rate}%", "fastv_layer": layer, "prune_rate": rate}


def experiment_jobs():
    return [job_info(0)] + [job_info(rate, layer) for layer in LAYERS for rate in RATES[1:]]


def benchmark_prompt(sample, version):
    from llava_pruning.prompts import GENERIC_PROMPT_V0, resolve_prompt
    if version == "v0_generic":
        return GENERIC_PROMPT_V0  # Deliberately ignore category and dataset text.
    return resolve_prompt(version, sample.category, sample.text)[0]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=5, help="Physical GPU ID; all seven jobs run sequentially")
    parser.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    parser.add_argument("--dataset", choices=("mvtec", "visa"), default="mvtec",
                        help="mvtec: original JSON benchmark; visa: test-only raw/prepared images, no input JSON needed")
    parser.add_argument("--input-json", help=f"MVTec only; default {DEFAULT_MVTEC_JSON}")
    parser.add_argument("--data-root", help="Dataset root; default follows --dataset (VisA: /home/yz/xxy/data/datasets/VisA)")
    parser.add_argument("--visa-split-csv", help="VisA split CSV (or its directory containing 1cls.csv); "
                        "default: data-root/split_csv/1cls.csv, then data-root/1cls.csv, "
                        "then test folders, then bundled split")
    parser.add_argument("--method-config", default=str(ROOT / "configs" / "fastv.json"),
                        help="Reuse min_tokens/newline settings; override layer with 2 and 3; rates 0,50,70,90")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup images per job, excluded from metrics (default 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", help="New or empty directory; default output/benchmark_fastv_<timestamp>")
    parser.add_argument("--worker-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--worker-rate", type=int, choices=RATES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-layer", type=int, choices=LAYERS, help=argparse.SUPPRESS)
    return parser


def prepare(args):
    from llava_pruning.data import load_samples
    from llava_pruning.methods import FastVMethod

    if args.gpu < 0 or args.warmup < 1 or not 0 <= args.seed < 2 ** 32:
        raise ValueError("gpu must be nonnegative, warmup >= 1, seed in [0, 2**32)")
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model checkpoint not found: {model_path}")
    data_root = Path(args.data_root or DEFAULT_DATA_ROOTS[args.dataset]).expanduser().resolve()
    method = FastVMethod.from_config(args.method_config)
    if args.dataset == "visa":
        if args.input_json is not None:
            raise ValueError("--dataset visa reads test images directly; do not supply --input-json")
        from llava_pruning.visa_data import load_visa_test
        samples, dataset_info = load_visa_test(data_root, args.visa_split_csv)
        if not dataset_info["normal"] or not dataset_info["abnormal"]:
            raise ValueError(
                f"VisA benchmark requires both normal and abnormal test images; selected "
                f"normal={dataset_info['normal']}, abnormal={dataset_info['abnormal']} "
                f"from {dataset_info['split_csv'] or dataset_info['category_root']}. "
                "Check split_csv/1cls.csv (split=test, label=normal/anomaly) and its image paths. "
                "An anomaly-only test/ folder cannot measure TNR. "
                "Training images will not be added automatically. No benchmark jobs started."
            )
        source = None
        prompt_version = "v0_generic"
    else:
        if args.visa_split_csv is not None:
            raise ValueError("--visa-split-csv requires --dataset visa")
        source = Path(args.input_json or DEFAULT_MVTEC_JSON).expanduser().resolve()
        samples = load_samples(source, data_root)
        dataset_info = {"layout": "question_json", "source": str(source)}
        prompt_version = "v0"
    if not samples:
        raise ValueError("Input dataset is empty")
    if any(sample.gt is None for sample in samples):
        raise ValueError("This classification benchmark requires gt=0/1 on every sample")
    for sample in samples:
        benchmark_prompt(sample, prompt_version)
    output = Path(args.output_dir or f"output/benchmark_fastv_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if source is None:
        # Freeze the exact test selection for all seven jobs; never modify VisA.
        source = output / "input_visa_test.jsonl"
        with source.open("x", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample.source, ensure_ascii=False, allow_nan=False) + "\n")
    configs = {}
    for layer in LAYERS:
        effective = {"layer": layer, "prune_rates": list(RATES), "visualize_rates": [],
                     "min_tokens": method.min_tokens, "preserve_image_newline": method.preserve_image_newline}
        path = output / f"fastv_layer_{layer}.json"
        save_json(path, effective)
        configs[str(layer)] = {"method_config": str(path), "fastv": effective}
    spec = {"model_path": str(model_path), "input_json": str(source), "data_root": str(data_root),
            "dataset": args.dataset, "dataset_info": dataset_info,
            "input_sha256": file_sha256(source), "sample_count": len(samples), "gpu": args.gpu,
            "seed": args.seed, "warmup": args.warmup, "prune_rates": list(RATES),
            "fastv_layers": list(LAYERS), "jobs": experiment_jobs(),
            "prompt_version": prompt_version, "roi_mode": "anyres_max_9", "image_token_order": "base_first",
            "batch_size": 1, "decoding": "greedy, max_new_tokens<=512 (existing LLaVA context budget)",
            "do_sample": False, "include_pruning_time": True, "save_visualizations": False,
            "fastv_configs": configs,
            "definitions": DEFINITIONS}
    if args.dataset == "visa":
        spec["prompt_text"] = benchmark_prompt(samples[0], prompt_version)
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


def run_worker(manifest, rate, layer=None):
    # Parent sets CUDA_VISIBLE_DEVICES before this process imports Torch.
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
    job = job_info(rate, layer)
    output = Path(manifest).parent / job["output_subdir"]
    output.mkdir(parents=True, exist_ok=False)
    from llava_pruning.metrics import Accuracy
    accuracy = Accuracy(spec["sample_count"])
    report = {**job, "dataset": spec.get("dataset", "mvtec"), "status": "failed", "warmup_samples": 0}
    measurements = []
    total_memory = None
    try:
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.data import load_samples
        from llava_pruning.methods import FastVMethod

        if file_sha256(spec["input_json"]) != spec["input_sha256"]:
            raise ValueError("Input JSON changed after benchmark setup")
        samples = load_samples(spec["input_json"], spec["data_root"])
        if len(samples) != spec["sample_count"]:
            raise ValueError("Dataset size changed after benchmark setup")
        prompts = [benchmark_prompt(sample, spec["prompt_version"]) for sample in samples]
        if job not in spec["jobs"]:
            raise ValueError("Worker configuration is not in the experiment manifest")
        # The baseline's layer is unused because pruning is disabled; report null.
        config_layer = LAYERS[0] if layer is None else layer
        configured = spec["fastv_configs"][str(config_layer)]
        method = FastVMethod.from_config(configured["method_config"])
        if (method.layer != config_layer or
                json.loads(Path(configured["method_config"]).read_text(encoding="utf-8")) != configured["fastv"]):
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
            print(f"{job['job_id']} warmup {position}/{len(indices)}", flush=True)
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
                    print(f"{job['job_id']} evaluated {index + 1}/{len(samples)}; "
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


def run_child(manifest, rate, gpu, log_path, *, layer=None):
    job_info(rate, layer)  # Validate before starting a GPU process.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker-manifest", str(manifest), "--worker-rate", str(rate)]
    if layer is not None:
        command.extend(["--worker-layer", str(layer)])
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
    print(f"Dataset={spec['dataset']}: {spec['sample_count']} images; rates={RATES}; "
          f"GPU={spec['gpu']}; FastV layers={LAYERS}; {len(spec['jobs'])} jobs; output={output}", flush=True)
    if spec["dataset"] == "visa":
        info = spec["dataset_info"]
        print(f"VisA TEST only: normal={info['normal']}, abnormal={info['abnormal']}; "
              f"layout={info['layout']}; per-category={info['categories']}", flush=True)
        if info["split_csv"]:
            print(f"VisA split CSV: {info['split_csv']} (only split=test rows)", flush=True)
        print(f"Prompt for every image:\n{spec['prompt_text']}", flush=True)
    print("FP16/FlashAttention2 unchanged; greedy; warmup per job; no visualization. "
          "Lat includes pruning. Mem is PyTorch allocated/reserved, not whole-board usage.", flush=True)
    reports = []
    for job in spec["jobs"]:
        rate, layer = job["prune_rate"], job["fastv_layer"]
        start = time.perf_counter()
        print(f"\nStarting {job['model']} on GPU {spec['gpu']} (new process)", flush=True)
        exit_code = run_child(output / "run.json", rate, spec["gpu"],
                              output / f"{job['job_id']}.log", layer=layer)
        metrics_path = output / job["output_subdir"] / "metrics.json"
        if metrics_path.is_file():
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            report = {**job, "dataset": spec["dataset"], "status": "failed", "complete": False,
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
            print(f"Finished {job['model']}: {scores}; Lat(mean)={report['lat_mean_ms']:.2f} ms; "
                  f"Mem(allocated)={report['mem_peak_allocated_gib']:.3f} GiB "
                  f"({report['mem_peak_allocated_pct']:.2f}%); unparsed={report['unparsed']}", flush=True)
        else:
            print(f"FAILED {job['model']}: {report.get('error')}; continuing remaining jobs", flush=True)
    print(f"Summary: {output / 'summary.csv'}", flush=True)
    return 0 if all(row["status"] == "success" for row in reports) else 1


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_manifest:
        if args.worker_rate is None:
            parser.error("--worker-manifest requires --worker-rate")
        try:
            job_info(args.worker_rate, args.worker_layer)
        except ValueError as error:
            parser.error(str(error))
        return run_worker(args.worker_manifest, args.worker_rate, args.worker_layer)
    if args.worker_rate is not None or args.worker_layer is not None:
        parser.error("--worker-rate and --worker-layer are internal options")
    try:
        return launch(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted; completed reports and partial predictions are preserved.\n")


if __name__ == "__main__":
    raise SystemExit(main())
