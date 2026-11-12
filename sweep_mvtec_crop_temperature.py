"""Compare four MyPool crop allocation temperatures on selected MVTec images."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import benchmark_visa_methods as common


TEMPERATURES = (0.05, 0.10, 0.15, 0.20)
RATES = (50, 70)
MODE = "crop_quota"


def prepare(results_dir, indices, gpus):
    if len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("Sample filename indices must be distinct and nonnegative")
    if len(set(gpus)) != len(TEMPERATURES) or min(gpus) < 0:
        raise ValueError("Provide four distinct nonnegative GPU IDs")
    source = json.loads((results_dir / "run.json").read_text(encoding="utf-8"))
    if source.get("experiment") != "mypool_pre_llm_only" or not set(RATES).issubset(source["rates"]):
        raise ValueError("Expected the original MyPool pre-LLM 50% and 70% run")
    source_input = Path(source["input_json"])
    if common.digest(source_input) != source["input_sha256"]:
        raise ValueError("The original frozen sample list changed")
    if common.digest(Path(source["model_path"]) / "config.json") != source["model_config_sha256"]:
        raise ValueError("The original model configuration changed")
    samples, _ = common.read_prefix(source_input)
    if len(samples) != source["sample_count"] or max(indices) >= len(samples):
        raise ValueError("Requested sample indices exceed the original frozen input")
    rows = [samples[index] for index in indices]

    output = results_dir / "temperature_sweep"
    output.mkdir(exist_ok=True)
    manifest = output / "run.json"
    frozen = output / "input_frozen.jsonl"
    input_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    spec = {name: source[name] for name in ("model_path", "model_config_sha256", "data_root", "seed", "panel_height")}
    spec.update(sample_indices=indices, sample_count=len(rows), input_json=str(frozen),
                rates=list(RATES), source_input_sha256=source["input_sha256"],
                groups=[dict(allocation_temperature=tau, folder=f"tau_{tau:.2f}", gpu=gpu)
                        for tau, gpu in zip(TEMPERATURES, gpus)])
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (dict(spec, input_sha256=common.digest(frozen)) != saved
                or frozen.read_text(encoding="utf-8") != input_text):
            raise ValueError("The saved temperature sweep uses different inputs or GPU assignments")
    else:
        if any(output.iterdir()):
            raise ValueError("The temperature sweep directory has files but no run manifest")
        frozen.write_text(input_text, encoding="utf-8")
        spec["input_sha256"] = common.digest(frozen)
        common.save_json(manifest, spec)
    return manifest


def worker(manifest, group_index):
    import torch
    import visualize_mvtec_baseline as baseline
    from llava_pruning.backend import LlavaBackend
    from visualize_mvtec_pre_llm import run_trial

    spec = json.loads(manifest.read_text(encoding="utf-8"))
    group = spec["groups"][group_index]
    spec.update(group)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(group["gpu"]):
        raise ValueError("Worker GPU does not match the saved assignment")
    if common.digest(spec["input_json"]) != spec["input_sha256"]:
        raise ValueError("Frozen sample list changed")
    samples = baseline.samples_for(spec)
    root = manifest.parent / group["folder"]
    jobs = [(rate, root / f"prune_{rate}") for rate in spec["rates"]]
    if all(len(baseline.restored(folder, samples, MODE, sample_indices=spec["sample_indices"])) == len(samples)
           and (folder / "metrics.json").is_file() for _, folder in jobs):
        print(f"Reusing completed {group['folder']}", flush=True)
        return
    backend = LlavaBackend(spec["model_path"], "anyres_max_9", image_token_order="base_first")
    common.validate_single_gpu(backend.model, torch)
    device = torch.cuda.get_device_properties(0)
    with torch.inference_mode():
        for rate, folder in jobs:
            folder.mkdir(parents=True, exist_ok=True)
            runtime = dict(backend.inference_config, allocation_temperature=group["allocation_temperature"])
            common.verify_runtime(folder, runtime, device.name, device.total_memory)
            run_trial(backend, samples, spec, folder, MODE, rate)


def write_spread(manifest):
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    rows = []
    for group in spec["groups"]:
        for rate in spec["rates"]:
            folder = manifest.parent / group["folder"] / f"prune_{rate}"
            records, _ = common.read_prefix(folder / "predictions.jsonl")
            if len(records) != spec["sample_count"]:
                raise ValueError(f"Incomplete temperature result: {folder}")
            for index, record in zip(spec["sample_indices"], records):
                kept = record["anyres_kept_per_crop"]
                rows.append(dict(sample_index=index, question_id=record["question_id"], prune_rate=rate,
                                 tau=group["allocation_temperature"], anyres_kept=sum(kept),
                                 min_crop_kept=min(kept), max_crop_kept=max(kept),
                                 crop_prune_gap_pp=100 * (max(kept) - min(kept)) / 729,
                                 crop_kept=json.dumps(kept)))
    path = manifest.parent / "crop_spread.csv"
    common.save_csv(path, rows, tuple(rows[0]))
    print(f"Crop spread | {path}", flush=True)


def launch(manifest):
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    processes = []
    try:
        for index, group in enumerate(spec["groups"]):
            folder = manifest.parent / group["folder"]
            folder.mkdir(exist_ok=True)
            log = folder / "worker.log"
            with log.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-u", __file__, "--worker", str(index), "--manifest", str(manifest)],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(group["gpu"])),
                    stdout=stream, stderr=subprocess.STDOUT,
                )
            processes.append((process, group, log))
            print(f"GPU {group['gpu']} | {group['folder']} | {log}", flush=True)
        failures = []
        for process, group, log in processes:
            code = process.wait()
            print(f"{group['folder']} finished | exit={code}", flush=True)
            if code:
                failures.append(str(log))
        if failures:
            raise RuntimeError("Failed temperature workers. See: " + ", ".join(failures))
    finally:
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _, _ in processes:
            process.wait()
    write_spread(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[2, 71, 294, 594, 1207])
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--worker", type=int, choices=range(4), help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.manifest, args.worker)
    else:
        if args.results_dir is None:
            parser.error("--results-dir is required")
        launch(prepare(args.results_dir.expanduser().resolve(), args.sample_indices, args.gpus))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
