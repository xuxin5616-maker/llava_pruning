"""MyPool before LLM + distilled block2 pruning; isolated lambda/source trials.

Defaults: average30/50/70%, lambda.01/.25/.5/.75/.99, BOTH scoring sources, ALL images.
Results-only: ACC/PRE/Recall/TNR/BACC/F1/relative ACC, TTFT Lat, peak Mem, budgets.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from llava_pruning.extended_metrics import EXTRA_FIELDS, METRIC_DEFINITIONS, extended_metrics
from llava_pruning.pool_distill_resume import (COMMON_FIELDS, TIMING_FIELDS, file_hash,
    load_records, read_prefix, same_setting, validate_configuration, validate_prediction)

PROJECT = Path(__file__).resolve().parent
PROMPT_FORMAT = "resolved_text_v1"


def require_current_prompt_format(metadata):
    if metadata.get("prompt_format") != PROMPT_FORMAT:
        raise ValueError("This output predates the resolved prompt-text fix or has an incompatible prompt format. "
                         "Use a NEW --output-dir and rerun; do not --resume results generated from the prompt tuple.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", type=int, default=5, help="One physical GPU; sequential trials")
    gpu_group.add_argument("--gpus", type=int, nargs="+", default=None,
                           help="Physical GPUs for parallel independent trials, e.g. --gpus 5 6 7")
    parser.add_argument("--model-path", type=Path, default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov"))
    parser.add_argument("--input-json", type=Path, default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec"))
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "output/attention_predictor_twig_train_01/best.pt")
    parser.add_argument("--prune-rates", type=float, nargs="+", default=[30., 50., 70.])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[.01, .25, .5, .75, .99])
    parser.add_argument("--selection-modes", nargs="+", choices=("anyres_self", "base_full"), default=["anyres_self", "base_full"])
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    parser.add_argument("--redundancy-layers", type=int, nargs="+", default=[26])
    parser.add_argument("--visionzip-layers", type=int, nargs="+", default=[26])
    parser.add_argument("--allocation-temperature", type=float, default=.05)
    parser.add_argument("--spatial-mix-weight", type=float, default=0., help="MyPool spatial prior weight; independent of --lambdas")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-acc", type=float, default=91.18,
                        help="Unpruned reference ACC in PERCENT (default91.18); no baseline rerun")
    parser.add_argument("--report-only", type=Path, metavar="EXISTING_OUTPUT",
                        help="Add/recompute BACC/F1/relative ACC in existing reports; no inference/GPU")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed trials and continue interrupted samples in the SAME output directory")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output/pool_distill_30_50_70_01")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def write_json(path, data):
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_text_atomic(path, text):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path, rows):
    if not rows:
        return
    import io
    with io.StringIO(newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        write_text_atomic(path, "\ufeff" + stream.getvalue())


def validate(args):
    baseline_reference(args.baseline_acc)
    if args.gpus is not None:
        if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
            raise ValueError("--gpus needs distinct nonnegative physical GPU IDs")
        if args.worker:
            raise ValueError("Each worker uses exactly one --gpu; --gpus belongs to the parent scheduler")
        args.gpu = args.gpus[0]
    for values, low, high in ((args.prune_rates, 0, 100), (args.lambdas, 0, 1)):
        if not values or len(set(values)) != len(values) or any(not math.isfinite(v) or v < low or v > high for v in values):
            raise ValueError("Rates/lambdas must be finite, distinct and in range")
    if any(not 0 < r < 100 for r in args.prune_rates):
        raise ValueError("This sweep requires average rates strictly between0 and100")
    if args.gpu < 0 or args.warmup < 0 or not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("Invalid GPU / warmup / seed")
    if len(set(args.selection_modes)) != len(args.selection_modes):
        raise ValueError("Duplicate selection modes")
    if not math.isfinite(args.allocation_temperature) or args.allocation_temperature <= 0 or not 0 <= args.spatial_mix_weight <= 1:
        raise ValueError("Invalid MyPool temperature / spatial weight")
    for layers in (args.redundancy_layers, args.visionzip_layers):
        if sorted(set(layers)) != layers or any(not 1 <= i <= 26 for i in layers):
            raise ValueError("SigLIP layer lists must be increasing, distinct, within1..26")
    for name, is_dir in (("model_path", True), ("input_json", False), ("data_root", True),
                         ("checkpoint", False)):
        path = getattr(args, name).expanduser().resolve()
        if name == "checkpoint" and not any(args.lambdas):
            continue
        if not (path.is_dir() if is_dir else path.is_file()):
            raise FileNotFoundError(f"{name}: {path}")
        setattr(args, name, path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    return output


def restore_sweep_args(args, argv):
    """A short --resume command inherits the original sweep, not new defaults."""
    path = args.output_dir.expanduser().resolve() / "comparison.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require_current_prompt_format(manifest)
    saved = manifest.get("configuration")
    if saved is None:
        if not manifest.get("trials"):
            raise ValueError("No saved sweep configuration/command to resume")
        command = manifest["trials"][0]["command"]
        saved = vars(build_parser().parse_args(command[command.index("--worker"):]))
    explicit = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    for name in (*COMMON_FIELDS, "gpu", "baseline_acc", "prune_rates", "lambdas", "selection_modes"):
        value = manifest.get(name, saved.get(name))
        if value is None:
            # Earliest legacy manifests/commands predate this reporting option.
            if name == "baseline_acc":
                value = 91.18
            else:
                raise ValueError(f"Saved sweep is missing {name}")
        flag = "--" + name.replace("_", "-")
        if flag in explicit:
            if name not in {"gpu", "baseline_acc"} and not same_setting(name, getattr(args, name), value):
                raise ValueError(f"Cannot resume with changed {flag}; keep original settings")
        else:
            setattr(args, name, Path(value) if name in {"model_path","input_json","data_root","checkpoint"} else value)
    if "--gpus" in explicit:
        pass  # Reassign unfinished trials to an explicitly requested GPU pool.
    elif "--gpu" in explicit:
        args.gpus = None
    else:
        args.gpus = manifest.get("gpus", saved.get("gpus"))
    return args


def read_trial_rows(folder, rate, share, mode, baseline_acc):
    metrics_path = folder / "metrics.json"
    if metrics_path.is_file():
        row = extended_metrics(json.loads(metrics_path.read_text(encoding="utf-8")), baseline_acc / 100)
        write_json(metrics_path, row)
        write_csv(folder / "summary.csv", [row])
    else:
        with (folder / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise RuntimeError(f"Unexpected trial summary: {folder}")
        row = rows[0]
    if (str(row.get("complete")).lower() != "true" or float(row["prune_rate"]) != rate
            or float(row["distill_share"]) != share or row["selection_mode"] != mode):
        raise RuntimeError(f"Incomplete or wrong trial results: {folder}")
    for key in ("lat_mean_ms", "mem_peak_allocated_gib", "actual_average_prune_rate"):
        if not math.isfinite(float(row[key])):
            raise RuntimeError(f"Invalid {key} for {folder}")
    if not set(EXTRA_FIELDS).issubset(row) or not math.isclose(float(row["baseline_accuracy"]), baseline_acc/100):
        raise RuntimeError(f"Missing extended metrics / wrong baseline for {folder}")
    return [row]


def verified_complete(folder, args, rate, share, mode):
    paths = [folder / name for name in ("run.json", "metrics.json", "predictions.jsonl")]
    if not all(path.is_file() for path in paths):
        return False
    run, metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths[:2]]
    if not run.get("complete") or not metrics.get("complete"):
        return False
    require_current_prompt_format(run)
    validate_configuration(args, run["configuration"], rate, share, mode)
    if run.get("input_json_sha256") not in (None, file_hash(args.input_json)):
        raise ValueError("Input JSON content changed; cannot reuse completed trials")
    from llava_pruning.data import _records
    source = _records(args.input_json)
    predictions, _ = read_prefix(paths[2])
    if len(predictions) != len(source) or metrics.get("evaluated_samples") != len(source):
        raise ValueError(f"Completed trial has inconsistent sample counts: {folder}")
    for index, (row, item) in enumerate(zip(predictions, source)):
        validate_prediction(row, str(item.get("question_id", item.get("id", ""))), item.get("gt"), index)
    return True


def baseline_reference(percent):
    if isinstance(percent, bool) or not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("--baseline-acc must be an ACC percentage in (0,100]")
    return dict(accuracy=percent / 100, accuracy_percent=percent,
                source="user-supplied unpruned reference; not rerun", default_reference_percent=91.18)


def rebuild_reports(directory, baseline_acc=91.18):
    """Enrich completed trial reports from exact JSON counts; never call a model.

    A moved/copied output folder is supported: child basenames are resolved
    within the requested root, not at stale absolute paths in comparison.json.
    Validate every completed trial before rewriting any report. Failed/running
    trials remain excluded, and manifest completion/status is not changed.
    """
    reference = baseline_reference(baseline_acc)
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "comparison.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Hybrid comparison.json not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending, seen = [], set()
    for trial in manifest["trials"]:
        if trial["status"] != "complete":
            continue
        name = trial["directory"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        folder = (root / name).resolve()
        if folder.parent != root or folder in seen:
            raise ValueError("Invalid or duplicate trial report directory")
        seen.add(folder)
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if (not metrics.get("complete") or metrics["prune_rate"] != trial["rate"]
                or metrics["distill_share"] != trial["distill_share"]
                or metrics["selection_mode"] != trial["selection_mode"]):
            raise ValueError(f"Incomplete or mismatched metrics: {folder}")
        enriched = extended_metrics(metrics, reference["accuracy"])
        run_path = folder / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["baseline_accuracy_reference"] = reference
        run["definitions"] = dict(run.get("definitions", {}), **METRIC_DEFINITIONS)
        pending.append((folder, enriched, run))
    if not pending:
        raise ValueError("No verified complete hybrid trials to update")
    for folder, metrics, run in pending:
        write_csv(folder / "summary.csv", [metrics])
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "run.json", run)
    write_csv(root / "summary.csv", [metrics for _, metrics, _ in pending])
    manifest["baseline_accuracy_reference"] = reference
    manifest["metric_definitions"] = METRIC_DEFINITIONS
    manifest["report_updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"Updated {len(pending)} completed reports without inference: {root / 'summary.csv'}", flush=True)
    return root


def run_sweep(args, *, run_process=None, process_factory=None):
    if args.gpus is not None and len(args.gpus) > 1:
        if run_process is not None:
            raise ValueError("Parallel trials use a nonblocking process_factory")
        from llava_pruning.pool_distill_parallel import run_parallel_sweep
        return run_parallel_sweep(args, api=sys.modules[__name__], process_factory=process_factory)
    output = validate(args)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "comparison.json"
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if args.resume else None
    if saved_manifest is not None:
        require_current_prompt_format(saved_manifest)
    manifest = saved_manifest or dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                    prune_rates=args.prune_rates, lambdas=args.lambdas, selection_modes=args.selection_modes,
                    isolated_sequential_processes=True,
                    lambda_definition="distillation contribution / target WHOLE native-packed layer-average removal",
                    base_policy="MyPool compresses Base + AnyRes; block2 protects remaining Base",
                    checkpoint_policy="reuse existing dense-trained predictor; no training in this sweep",
                    sparse_distribution_shift="side predictor now receives MyPool-compressed/merged inputs; performance requires empirical verification",
                    visualization=False, baseline_accuracy_reference=baseline_reference(args.baseline_acc),
                    metric_definitions=METRIC_DEFINITIONS, prompt_format=PROMPT_FORMAT, trials=[])
    manifest.update(complete=False, baseline_accuracy_reference=baseline_reference(args.baseline_acc),
                    gpus=[args.gpu], isolated_sequential_processes=True,
                    execution_mode="independent_single_gpu_trials", max_concurrent_trials=1)
    manifest["configuration"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.resume:
        manifest["last_resumed_utc"] = datetime.now(timezone.utc).isoformat()
    existing = {(item["rate"], item["selection_mode"], item["distill_share"]): item for item in manifest["trials"]}
    if len(existing) != len(manifest["trials"]):
        raise ValueError("Duplicate saved trial configurations")
    combined = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    for rate in args.prune_rates:
        for mode in args.selection_modes:
            for share in args.lambdas:
                tag = f"prune_{rate:g}_{mode}_lambda_{share:g}"
                trial_dir = output / tag
                key = (rate, mode, share)
                trial = existing.get(key)
                if args.resume and verified_complete(trial_dir, args, rate, share, mode):
                    combined.extend(read_trial_rows(trial_dir, rate, share, mode, args.baseline_acc))
                    if trial is None:
                        trial = dict(rate=rate, distill_share=share, selection_mode=mode, directory=str(trial_dir))
                        manifest["trials"].append(trial)
                    trial.update(status="complete", reused=True)
                    write_csv(output / "summary.csv", combined)
                    write_json(manifest_path, manifest)
                    print(f"Skipping verified complete trial: {tag}", flush=True)
                    continue
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--gpu", str(args.gpu),
                           "--model-path", str(args.model_path), "--input-json", str(args.input_json),
                           "--data-root", str(args.data_root), "--checkpoint", str(args.checkpoint),
                           "--prune-rates", str(rate), "--lambdas", str(share), "--selection-modes", mode,
                           "--prompt-version", args.prompt_version, "--warmup", str(args.warmup),
                           "--seed", str(args.seed), "--baseline-acc", str(args.baseline_acc),
                           "--allocation-temperature", str(args.allocation_temperature),
                           "--spatial-mix-weight", str(args.spatial_mix_weight),
                           "--redundancy-layers", *map(str, args.redundancy_layers),
                           "--visionzip-layers", *map(str, args.visionzip_layers), "--output-dir", str(trial_dir)]
                if args.resume:
                    command.append("--resume")
                if trial is None:
                    trial = dict(rate=rate, distill_share=share, selection_mode=mode, directory=str(trial_dir))
                    manifest["trials"].append(trial)
                trial.update(command=command, status="running", attempts=trial.get("attempts", 0)+1)
                write_json(manifest_path, manifest)
                print(f"Starting {tag}; GPU{args.gpu}; isolated sequential job", flush=True)
                try:
                    completed = (run_process or subprocess.run)(command, cwd=str(PROJECT), env=environment, check=False)
                    if completed.returncode in (130, -2):
                        raise KeyboardInterrupt
                    if completed.returncode:
                        raise RuntimeError(f"{tag} exited with code{completed.returncode}; see traceback above")
                    combined.extend(read_trial_rows(trial_dir, rate, share, mode, args.baseline_acc))
                    write_csv(output / "summary.csv", combined)
                except KeyboardInterrupt:
                    trial.update(status="interrupted", error="SIGINT / KeyboardInterrupt; resume the same output directory")
                    write_json(manifest_path, manifest)
                    raise
                except Exception as error:
                    trial.update(status="failed", error=str(error))
                    write_json(manifest_path, manifest)
                    raise
                trial["status"] = "complete"
                write_json(manifest_path, manifest)
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    print(f"Completed {len(combined)} trials: {output / 'summary.csv'}", flush=True)
    return output


def run_worker(args):
    output = validate(args)
    if len(args.prune_rates) != 1 or len(args.lambdas) != 1 or len(args.selection_modes) != 1:
        raise ValueError("Worker requires one rate/lambda/mode")
    import statistics
    import torch
    from PIL import Image
    from benchmark_fastv import DEFINITIONS, measurement_summary, validate_single_gpu, warmup_indices
    from llava_pruning.backend import LlavaBackend
    from llava_pruning.data import load_samples
    from llava_pruning.metrics import Accuracy
    from llava_pruning.prompts import resolve_prompt
    from llava_pruning.pool_distill import PoolDistillMethod
    from runMyPruningPoolSpatial import MethodSettings
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one visible CUDA GPU")
    rate, share, mode = args.prune_rates[0], args.lambdas[0], args.selection_modes[0]
    previous = None
    if args.resume and output.exists() and any(output.iterdir()):
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        require_current_prompt_format(previous)
        validate_configuration(args, previous["configuration"], rate, share, mode)
    input_digest = file_hash(args.input_json)
    checkpoint = args.checkpoint.stat() if args.checkpoint.is_file() else None
    checkpoint_identity = dict(size=checkpoint.st_size, mtime_ns=checkpoint.st_mtime_ns) if checkpoint else None
    if previous:
        if previous.get("input_json_sha256") not in (None, input_digest):
            raise ValueError("Input JSON content changed; refusing mixed results")
        if previous.get("checkpoint_identity") not in (None, checkpoint_identity):
            raise ValueError("Predictor checkpoint changed; refusing mixed results")
    samples = load_samples(args.input_json, args.data_root)
    if not samples or args.seed + len(samples) > 2**32:
        raise ValueError("Empty samples / invalid per-image seed range")
    # resolve_prompt returns (text, source). Match runner.py: only the text is
    # passed to generate, never the Python tuple repr with escaped newlines.
    prompts = [resolve_prompt(args.prompt_version, s.category, s.text)[0] for s in samples]
    backend = LlavaBackend(args.model_path, roi_mode="anyres_max_9", image_token_order="base_first")
    if previous:
        for key in ("model_dtype", "vision_dtype", "attention", "matmul_tf32", "cudnn_tf32",
                    "image_aspect_ratio", "mm_patch_merge_type", "image_token_order", "gpu", "torch",
                    "cuda", "transformers", "flash_attn"):
            if key in previous["inference"] and previous["inference"][key] != backend.inference_config.get(key):
                raise ValueError(f"Inference environment changed for {key}; do not mix Lat/Mem from different environments")
    validate_single_gpu(backend.model, torch)
    core = backend.model.get_model()
    settings = MethodSettings(prune_rate=0, redundancy_layers=tuple(args.redundancy_layers),
        visionzip_layers=tuple(args.visionzip_layers), allocation_temperature=args.allocation_temperature,
        spatial_strength=args.spatial_mix_weight)
    method = PoolDistillMethod(core, args.checkpoint, rate=rate, share=share, mode=mode, settings=settings)
    # Feasibility checks do not score images, skip records, or loosen budgets.
    if share == 1:
        method.native.validate_input_budgets(backend, samples, {rate: method.native.resolve_budget(rate, len(core.layers))})
    else:
        seen = set()
        for sample in samples:
            with Image.open(sample.image) as opened:
                size = opened.size
            if size not in seen:
                try:
                    method.adapter.set_image_context(size)
                except ValueError as error:
                    raise ValueError(f"id={sample.sample_id}, size={size}: {error}") from error
                seen.add(size)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_metadata = dict(configuration=metadata, inference=backend.inference_config,
        definitions=dict(DEFINITIONS, **METRIC_DEFINITIONS), baseline_accuracy_reference=baseline_reference(args.baseline_acc),
        method="pool_distill", prompt_format=PROMPT_FORMAT, predictor_training="unchanged dense-input checkpoint",
        budget_reference="native unpruned packed Base+AnyRes+newlines; per-image baseline",
        actual_budget="recorded per image; integer rounding <200/N percentage points",
        mapping="Base mass-preserving backprojection by normalized actual merge coefficients; bilinear image-coordinate sampling; source-weighted AnyRes average",
        position_policy="compressed MyPool prefill positions; second-stage pruning preserves those positions",
        no_visualizations=True, complete=False, input_json_sha256=input_digest,
        checkpoint_identity=checkpoint_identity,
        resume_history=list((previous or {}).get("resume_history", [])))
    geometry_cache = {}
    def recover_budget(sample):
        with Image.open(sample.image) as opened:
            size = opened.size
        if size not in geometry_cache:
            geometry_cache[size] = method.recover_legacy_budget(size)
        return dict(geometry_cache[size])
    records = load_records(output, samples, recover_budget, rate=rate, share=share, mode=mode) if args.resume else []
    resumed_samples = len(records)
    run_metadata["resume_history"].append(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_id=args.gpu, resumed_samples=resumed_samples,
        legacy_without_content_fingerprint=bool(previous and "input_json_sha256" not in previous)))
    run_metadata["completed_samples"] = resumed_samples
    write_json(output / "run.json", run_metadata)
    # Rewrite the validated prefix atomically: removes only a torn final journal
    # fragment, and migrates legacy predictions without changing their timings.
    write_text_atomic(output / "progress.jsonl", "".join(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n" for record in records))
    if args.resume:
        print(f"Recovered {resumed_samples}/{len(samples)} samples; continuing at sample {resumed_samples+1}. "
              "Saved answers/Lat/Mem are reused; legacy budgets are labeled reconstructed.", flush=True)
    def generate(index):
        with method.generation_context(backend.model):
            return backend.generate(samples[index], prompts[index], method=method, rate=rate,
                roi_mode="anyres_max_9", capture_visualization=False, capture_attention=False,
                random_seed=args.seed + index, do_sample=False, include_pruning_time=True, benchmark_metrics=True)
    warmups = warmup_indices(len(samples), args.warmup) if resumed_samples < len(samples) else []
    for i, index in enumerate(warmups):
        generate(index)
        print(f"Warmup {i + 1}/{len(warmups)}", flush=True)
    validate_single_gpu(backend.model, torch)  # includes the newly loaded side.
    total_memory = torch.cuda.get_device_properties(0).total_memory
    accuracy, timings, budgets = Accuracy(len(samples)), [], []
    for record in records:
        row = record["prediction"]
        accuracy.add(row["gt"], row["answer"])
        timings.append({key: row[key] for key in TIMING_FIELDS})
        budgets.append(record["budget"])
    started = time.perf_counter()
    try:
        with (output / "progress.jsonl").open("a", encoding="utf-8") as journal, \
                (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record["prediction"], ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            for i in range(resumed_samples, len(samples)):
                sample = samples[i]
                result = generate(i)
                timing = {key: result[key] for key in TIMING_FIELDS}
                budget = dict(question_id=sample.sample_id, **result["stats"])
                budget.setdefault("selection_mode", mode)
                budget.setdefault("budget_source", "observed_generation")
                prediction = dict(question_id=sample.sample_id, gt=sample.gt, answer=result["answer"], **timing)
                record = dict(index=i, prediction=prediction, budget=budget)
                journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                journal.flush()  # sample commit, outside measured generate
                records.append(record)
                accuracy.add(sample.gt, result["answer"])
                timings.append(timing)
                budgets.append(budget)
                stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                done = i+1-resumed_samples
                if done == 1 or (i+1) % 20 == 0 or i+1 == len(samples):
                    elapsed = time.perf_counter() - started
                    print(f"{mode} rate={rate:g} lambda={share:g} {i+1}/{len(samples)} "
                          f"elapsed={elapsed:.1f}s ETA={elapsed/done*(len(samples)-i-1):.1f}s", flush=True)
    except BaseException as error:
        run_metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            completed_samples=len(records), complete=False)
        write_json(output / "run.json", run_metadata)
        raise
    metrics = accuracy.result(rate, complete=True)
    metrics.pop("rule", None)
    metrics.update(selection_mode=mode, distill_share=share, gpu_id=args.gpu, warmup_samples=len(warmups),
                   resumed_samples=resumed_samples, fresh_samples=len(samples)-resumed_samples,
                   budget_reconstructed_samples=sum(row.get("budget_source")=="legacy_reconstructed_counts" for row in budgets),
                   budget_observed_samples=sum(row.get("budget_source")!="legacy_reconstructed_counts" for row in budgets),
                   measurement_segments=len(run_metadata["resume_history"]),
                   **measurement_summary(timings, total_memory))
    for key in ("actual_average_prune_rate", "actual_pre_contribution", "actual_distill_contribution",
                "baseline_packed_tokens", "pre_llm_tokens", "final_llm_tokens", "pre_base_tokens", "final_anyres_tokens"):
        metrics[key] = statistics.mean(row[key] for row in budgets)
    metrics = extended_metrics(metrics, args.baseline_acc / 100)
    write_csv(output / "budgets.csv", budgets)
    write_csv(output / "summary.csv", [metrics])
    write_json(output / "metrics.json", metrics)
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    manifest.update(complete=True, status="complete", completed_samples=len(samples))
    write_json(output / "run.json", manifest)
    print(f"Done {mode} lambda={share:g}; ACC={metrics['accuracy']}; Lat={metrics['lat_mean_ms']:.2f}ms; Mem={metrics['mem_peak_allocated_gib']:.3f}GiB", flush=True)


def main():
    args = build_parser().parse_args()
    if args.report_only is not None:
        if args.worker:
            raise ValueError("--report-only cannot be combined with --worker")
        rebuild_reports(args.report_only, args.baseline_acc)
    elif args.worker:
        run_worker(args)
    else:
        if args.resume:
            args = restore_sweep_args(args, sys.argv[1:])
        run_sweep(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by SIGINT. Completed samples were checkpointed; resume with --resume and the SAME --output-dir.", flush=True)
        raise SystemExit(130)
