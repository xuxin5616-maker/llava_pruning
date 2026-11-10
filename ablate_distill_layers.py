"""Retrain four VisA side-branch placements and compare them on MVTec AD."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_pool_distill as evaluation
import train_attention_predictor as training
from llava_pruning.pool_ablation_report import FIELDS, compact_result_row
from llava_pruning.pool_distill_budget import hybrid_budget, pre_view_budget
from llava_pruning.pool_distill_resume import file_hash


PROJECT = Path(__file__).resolve().parent
LAYERS = (1, 2, 3, 4)
RATES = (50, 70, 90)
SHARE, TAIL, MODE = .7, 8, "base_full"


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def previous_recipe(path):
    saved = read_json(path)
    config = saved["configuration"]
    if (config.get("dataset", "visa") != "visa" or not saved["source"].get("source_csv")
            or saved["student_source_layer"] != 2
            or tuple(saved["architecture"]["source_layers"]) != (3, 4, 5)):
        raise ValueError("Reference must be the VisA run trained at block 2 with copied blocks 3/4/5")
    required = ("model_path", "data_root", "seed", "epochs", "validation_fraction", "learning_rate",
                "weight_decay", "adam_beta1", "adam_beta2", "warmup_ratio", "gradient_accumulation",
                "max_grad_norm", "gradient_checkpointing", "student_precision", "teacher_layer")
    missing = sorted(set(required) - config.keys())
    if missing:
        raise ValueError(f"Training manifest has no saved settings for {', '.join(missing)}")
    if config["teacher_layer"] <= max(LAYERS):
        raise ValueError("The reference teacher must follow all student source layers")
    split = Path(saved["source"]["source_csv"]).expanduser().resolve()
    if file_hash(split) != saved["source"]["source_sha256"]:
        raise ValueError(f"Reference VisA split CSV changed: {split}")
    recipe = {key: config[key] for key in required}
    recipe.update(dataset="visa", visa_split_csv=str(split), log_every=config.get("log_every", 20),
                  cache_dir=Path(config["cache_dir"]) if config.get("cache_dir") else None)
    return saved, recipe


def preflight(model_config, input_json, data_root):
    from PIL import Image
    from llava_pruning.data import load_samples
    from llava_pruning.predictor_training import vendor_imports
    vendor_imports()
    from llava.mm_utils import get_anyres_image_grid_shape
    from runMyPruningPoolSpatial import baseline_packed_count

    samples = load_samples(input_json, data_root)
    if not samples:
        raise ValueError("MVTec test set is empty")
    depth = model_config["num_hidden_layers"]
    merge = model_config["mm_patch_merge_type"]
    if depth != 28 or merge not in {"spatial_unpad", "spatial_unpad_add_newl"}:
        raise ValueError("Expected the 28-block LLaVA-OneVision model with spatial_unpad packing")
    jobs = {f"{layer}/{rate}": dict(layer=layer, rate=rate, infeasible_samples=0, reason=None)
            for layer in LAYERS for rate in RATES}
    checked = {}
    for sample in samples:
        with Image.open(sample.image) as image:
            size = image.size
        if size not in checked:
            width, height = get_anyres_image_grid_shape(size, model_config["image_grid_pinpoints"], 384)
            original = baseline_packed_count(size, (width, height), 27, merge)
            failures = {}
            for key, job in jobs.items():
                try:
                    plan = hybrid_budget(original, job["rate"], SHARE, depth, job["layer"])
                    plan["raw_grid_height"] = height
                    pre_view_budget(plan, width * height, final_newline=merge.endswith("_add_newl"),
                                    base_policy="anyres_first")
                except ValueError as error:
                    failures[key] = str(error)
            checked[size] = failures
        for key, reason in checked[size].items():
            jobs[key]["infeasible_samples"] += 1
            if jobs[key]["reason"] is None:
                jobs[key]["reason"] = f"id={sample.sample_id}, size={size}: {reason}"
    return jobs


def evaluate_layer(root, spec, layer, gpu, checkpoint):
    failed = False
    for rate in RATES:
        if spec["jobs"][f"{layer}/{rate}"]["infeasible_samples"]:
            continue
        folder = root / f"layer_{layer}" / f"prune_{rate}"
        args = evaluation.build_parser().parse_args([
            "--worker", "--gpu", str(gpu), "--output-dir", str(folder), "--resume",
            "--model-path", spec["model_path"], "--input-json", spec["input_json"],
            "--data-root", spec["data_root"], "--checkpoint", str(checkpoint),
            "--prune-rates", str(rate), "--lambdas", str(SHARE), "--selection-modes", MODE,
            "--predictor-input", "base_text", "--base-mapping", "fast", "--base-policy", "anyres_first",
            "--tail-drop-layers", str(TAIL), "--pruning-audit", "lean", "--summary-format", "compact",
        ])
        print(f"MVTec layer {layer}, {rate}%", flush=True)
        try:
            if evaluation.verified_complete(folder, args, rate, SHARE, MODE):
                saved = read_json(folder / "run.json")
                stat = checkpoint.stat()
                if (saved["configuration"].get("student_layer", 2) != layer
                        or saved["checkpoint_identity"] != dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)):
                    raise ValueError("Completed trial uses a different checkpoint or student layer")
                print("Reusing completed trial", flush=True)
            else:
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                evaluation.run_worker(args, student_layer=layer)
        except Exception:
            traceback.print_exc()
            failed = True
    return int(failed)


def run_layer(root, spec, layer, gpu):
    manifest = Path(spec["previous_training"])
    if file_hash(manifest) != spec["previous_training_sha256"]:
        raise ValueError("Reference training manifest changed")
    saved, recipe = previous_recipe(manifest)
    folder = root / f"layer_{layer}" / "student"
    status_path = folder / "training_status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    if status.get("complete"):
        metadata = read_json(folder / "train_run.json")
        if (metadata["student_source_layer"] != layer or metadata["split_sha256"] != saved["split_sha256"]
                or metadata["source"]["source_sha256"] != saved["source"]["source_sha256"]):
            raise ValueError(f"Existing training run does not match layer {layer} or the reference split")
        print(f"Reusing completed VisA layer {layer} training", flush=True)
    else:
        if folder.exists() and any(folder.iterdir()):
            archive = folder.with_name(f"student_interrupted_{time.time_ns()}")
            folder.rename(archive)
            print(f"Archived incomplete training to {archive}; restarting layer {layer}", flush=True)
        training.main(["--student-layer", str(layer), "--output-dir", str(folder)],
                      defaults=recipe, reference=saved)
        status = read_json(status_path)
    checkpoint = folder / status["evaluation_checkpoint"]
    if not status["complete"] or not checkpoint.is_file():
        raise ValueError(f"Training did not produce its selected checkpoint: {checkpoint}")
    return evaluate_layer(root, spec, layer, gpu, checkpoint)


def write_report(root, spec, exit_codes):
    rows = []
    for job in spec["jobs"].values():
        layer, rate = job["layer"], job["rate"]
        config = (f"layer={layer};prune={rate}%;lambda={SHARE:g};{MODE};input=base_text;"
                  f"tail={TAIL};mapping=fast;base_policy=anyres_first")
        row = dict.fromkeys((*FIELDS, "status", "reason", "infeasible_samples", "source_blocks",
                             "validation_kl", "gpu_id", "actual_two_stage_pct", "actual_with_tail_pct"))
        row["config"] = config
        folder = root / f"layer_{layer}"
        metric_path = folder / f"prune_{rate}" / "metrics.json"
        trial_run = metric_path.with_name("run.json")
        if job["infeasible_samples"]:
            status, reason = "unreachable", job["reason"]
        elif metric_path.is_file() and trial_run.is_file() and read_json(trial_run).get("complete"):
            metrics = read_json(metric_path)
            row.update(compact_result_row(metrics, config),
                       gpu_id=metrics["gpu_id"],
                       actual_two_stage_pct=metrics["two_stage_average_prune_rate"],
                       actual_with_tail_pct=metrics["actual_average_prune_rate"])
            status, reason = "complete", ""
        else:
            status = "failed" if exit_codes.get(layer) else "pending"
            reason = str(folder / "worker.log")
        status_path = folder / "student" / "training_status.json"
        row.update(status=status, reason=reason, infeasible_samples=job["infeasible_samples"],
                   source_blocks="/".join(map(str, range(layer + 1, layer + 4))),
                   validation_kl=read_json(status_path).get("best_validation_kl") if status_path.is_file() else None)
        rows.append(row)
    evaluation.write_csv(root / "summary.csv", rows)


def run_all(root, spec, gpus):
    waiting, active, exit_codes = list(LAYERS), {}, {}
    try:
        while waiting or active:
            for gpu in gpus:
                if gpu in active or not waiting:
                    continue
                layer = waiting.pop(0)
                folder = root / f"layer_{layer}"
                folder.mkdir(exist_ok=True)
                log = folder / "worker.log"
                with log.open("a", encoding="utf-8") as stream:
                    active[gpu] = (layer, subprocess.Popen(
                        [sys.executable, "-u", __file__, "--output-dir", str(root),
                         "--worker-layer", str(layer), "--gpus", str(gpu)],
                        cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdout=stream, stderr=subprocess.STDOUT))
                print(f"Layer {layer} on GPU {gpu} | {log}", flush=True)
            for gpu, (layer, child) in list(active.items()):
                code = child.poll()
                if code is not None:
                    exit_codes[layer] = code
                    del active[gpu]
                    print(f"Layer {layer} finished | exit={code}", flush=True)
            write_report(root, spec, exit_codes)
            if active:
                time.sleep(5)
    finally:
        for _, child in active.values():
            if child.poll() is None:
                child.terminate()
        for _, child in active.values():
            child.wait()
        write_report(root, spec, exit_codes)
    return int(any(exit_codes.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = evaluation.build_parser().parse_args([])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--previous-training", type=Path, default=defaults.checkpoint.parent / "train_run.json")
    parser.add_argument("--input-json", type=Path, default=defaults.input_json)
    parser.add_argument("--data-root", type=Path, default=defaults.data_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-layer", type=int, choices=LAYERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    manifest = root / "run.json"
    if args.worker_layer:
        return run_layer(root, read_json(manifest), args.worker_layer, args.gpus[0])
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("--gpus requires distinct nonnegative device IDs")
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error(f"Use a new output directory or --resume: {root}")
    previous = args.previous_training.resolve()
    _, recipe = previous_recipe(previous)
    model = Path(recipe["model_path"]).resolve()
    model_config = model / "config.json"
    spec = dict(version=1, previous_training=str(previous), previous_training_sha256=file_hash(previous),
                model_path=str(model), model_config_sha256=file_hash(model_config),
                input_json=str(args.input_json.resolve()), input_json_sha256=file_hash(args.input_json),
                data_root=str(args.data_root.resolve()), lambda_share=SHARE, tail_drop_layers=TAIL,
                jobs=preflight(read_json(model_config), args.input_json, args.data_root))
    if args.resume and (not manifest.is_file() or read_json(manifest) != spec):
        parser.error("Resume requires the same recipe, model configuration, evaluation input and preflight jobs")
    root.mkdir(parents=True, exist_ok=True)
    evaluation.write_json(manifest, spec)
    for job in spec["jobs"].values():
        if job["infeasible_samples"]:
            print(f"Skipping layer {job['layer']}, {job['rate']}% | {job['infeasible_samples']} infeasible images | {job['reason']}", flush=True)
    code = run_all(root, spec, args.gpus)
    print(f"Summary | {root / 'summary.csv'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
