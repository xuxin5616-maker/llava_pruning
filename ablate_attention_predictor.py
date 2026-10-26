"""Four independent VisA distillations, then full MVTec evaluation.

Two single-GPU training workers, NOT DDP: batch={32,64}, peak LR={2e-5,5e-5}.
Evaluation is serialized on one chosen GPU AFTER all training workers exit.
F1/BACC use parsed A/B answers; ACC counts unparsed answers as incorrect.
Only this launcher's output files are enriched; existing entrypoints are unchanged.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from llava_pruning.data import load_samples
from llava_pruning.predictor_data import load_visa_distillation, split_samples
from llava_pruning.prompts import GENERIC_PROMPT_V0, resolve_prompt
from llava_pruning.rates import format_prune_rate, is_prune_rate
from llava_pruning.training_schedule import training_step_plan


PROJECT = Path(__file__).resolve().parent
BATCHES = (32, 64)
LEARNING_RATES = (2e-5, 5e-5)
EXTRA_FIELDS = ("f1", "bacc", "baseline_accuracy", "relative_accuracy_pct")
SUMMARY_FIELDS = (
    "experiment", "effective_batch_size", "learning_rate", "epochs", "best_epoch", "validation_kl",
    "train_gpu", "test_gpu", "training_seconds", "testing_seconds", "prune_rate", "complete",
    "evaluated_samples", "unparsed", "accuracy", "precision", "recall", "tnr",
    *EXTRA_FIELDS, "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "mem_peak_allocated_gib",
    "mem_peak_reserved_gib", "actual_average_prune_rate", "status", "error",
)
METRIC_DEFINITIONS = {
    "positive_class": "A=defect=1; B=normal=0",
    "accuracy": "All labeled answers; unparsed answers count as incorrect.",
    "f1": "2*TP/(2*TP+FP+FN), parsed labeled A/B answers only; zero denominator=null.",
    "bacc": "0.5*(TP/(TP+FN)+TN/(TN+FP)), parsed answers only; either class absent=null.",
    "relative_accuracy_pct": "100*accuracy/baseline_accuracy; may exceed 100. Reference supplied by user, not rerun.",
    "units": "ACC/PRE/Recall/TNR/F1/BACC/baseline_accuracy are fractions; relative_accuracy_pct is percent.",
    "timing": "Reuse run.py Lat/Mem, including pruning. No sample PNGs. All tests on the same GPU after training exits.",
    "stage_seconds": "training_seconds/testing_seconds are whole child-process wall times, including loading/I/O; NOT Lat.",
    "selection": "Each run selects best.pt by VisA validation KL; MVTec is not used to pick checkpoints.",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs=2, type=int, help="Two distinct physical GPU IDs for independent training")
    parser.add_argument("--eval-gpu", type=int, default=None, help="One of --gpus; defaults to the first GPU")
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/"))
    parser.add_argument("--visa-root", type=Path, default=Path("/home/yz/xxy/data/datasets/VisA"))
    parser.add_argument("--visa-split-csv", type=Path, default=None, help="FULL CSV; default VisA/split_csv/1cls.csv")
    parser.add_argument("--input-json", type=Path,
                        default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/"))
    parser.add_argument("--epochs", type=int, default=5, help="Same complete epoch budget for all four runs (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="Same split, initialization and shuffle seed for all runs")
    parser.add_argument("--prune-rates", nargs="+", type=float, default=[50, 70, 90], help="LLM-layer-average percentages")
    parser.add_argument("--baseline-acc", type=float, default=91.18, help="Reference ACC in PERCENT, not fraction")
    parser.add_argument("--warmup", type=int, default=10, help="Test warmup images per rate, then evaluate ALL images")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Optional shared atomic teacher cache; H2 may need hundreds of GB. Default: no cache")
    parser.add_argument("--log-every", type=int, default=20, help="Training child progress frequency in images")
    parser.add_argument("--status-every", type=float, default=30, help="Console interval for forwarding worker progress")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output" / "predictor_ablation_32_64")
    parser.add_argument("--dry-run", action="store_true", help="Check all input paths, show commands; no GPU or output writes")
    parser.add_argument("--resume", action="store_true",
                        help="Skip VERIFIED complete stages only. NOT optimizer resume; incomplete outputs are never overwritten")
    parser.add_argument("--report-only", type=Path, default=None, metavar="EXISTING_OUTPUT",
                        help="Rebuild enhanced metrics and summary.csv without training, inference or GPUs")
    return parser


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomic replacement prevents reports/cache readers from seeing half JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_hash(samples):
    # Exact same serialization as train_attention_predictor.py, including order.
    records = (dict(question_id=s.sample_id, image=s.origin_path, category=s.category,
                    text=GENERIC_PROMPT_V0) for s in samples)
    # The training writer uses platform-native newlines, then hashes raw bytes.
    content = "".join(json.dumps(row, ensure_ascii=False) + os.linesep for row in records)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def prepare(args):
    """CPU-only, non-mutating preflight; NEVER truncate the training/test sets."""
    if not args.gpus or len(set(args.gpus)) != 2 or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("--gpus needs two distinct nonnegative GPU IDs, for example --gpus 4 5")
    eval_gpu = args.gpus[0] if args.eval_gpu is None else args.eval_gpu
    if eval_gpu not in args.gpus:
        raise ValueError("--eval-gpu must be one of --gpus")
    if (args.epochs < 1 or args.log_every < 1 or args.warmup < 0 or not 0 <= args.seed < 2**32
            or not math.isfinite(args.status_every) or args.status_every < 1
            or not math.isfinite(args.baseline_acc) or not 0 < args.baseline_acc <= 100):
        raise ValueError("Invalid epochs, logging interval, seed, warmup or baseline ACC")
    rates = [int(r) if r == int(r) else r for r in args.prune_rates if is_prune_rate(r)]
    if len(rates) != len(args.prune_rates) or not rates or len(set(rates)) != len(rates):
        raise ValueError("--prune-rates must be distinct finite percentages in [0,100)")
    model = args.model_path.expanduser().resolve()
    if not model.is_dir():
        raise NotADirectoryError(f"Model checkpoint not found: {model}")
    samples, source = load_visa_distillation(args.visa_root, args.visa_split_csv)
    train, validation = split_samples(samples, .1, args.seed)
    test = load_samples(args.input_json, args.data_root)
    if not test or {s.gt for s in test} != {0, 1}:
        raise ValueError("Test input needs labeled normal AND defect images for F1/BACC; unlabeled samples are not allowed")
    if args.seed > 2**32 - len(test):
        raise ValueError("Test seed + sample index exceeds NumPy's 32-bit range")
    if {s.image.resolve() for s in samples} & {s.image.resolve() for s in test}:
        raise ValueError("VisA distillation pool and test input contain the same resolved image paths")
    for sample in test:
        resolve_prompt("v0", sample.category, sample.text)
    settings = dict(model_path=str(model), visa_root=str(args.visa_root.expanduser().resolve()),
                    visa_split_csv=source["source_csv"], input_json=str(args.input_json.resolve()),
                    data_root=str(args.data_root.resolve()), epochs=args.epochs, seed=args.seed,
                    gpus=args.gpus, eval_gpu=eval_gpu, baseline_accuracy=args.baseline_acc / 100,
                    prune_rates=rates, warmup=args.warmup, log_every=args.log_every,
                    cache_dir=str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None,
                    roi_mode="anyres_max_9", image_token_order="base_first", prompt_version="v0",
                    selection_mode="base_guided", teacher_layer=22, validation_fraction=.1,
                    weight_decay=0., adam_beta1=.9, adam_beta2=.98, warmup_ratio=.03, max_grad_norm=1.,
                    student_precision="bfloat16", micro_batch_size=1, save_visualizations=False)
    output = args.output_dir.expanduser().resolve()
    jobs = []
    for batch in BATCHES:
        for lr in LEARNING_RATES:
            name = f"b{batch}_lr{lr:.0e}".replace("e-0", "e-")
            folder = output / name
            jobs.append(dict(name=name, batch_size=batch, learning_rate=lr, directory=str(folder),
                             schedule=training_step_plan(len(train), args.epochs, batch, .03),
                             train=dict(status="pending"), test=dict(status="pending")))
    return dict(version=1, output_dir=str(output), settings=settings, metric_definitions=METRIC_DEFINITIONS,
                data=dict(training_samples=len(train), validation_samples=len(validation), test_samples=len(test),
                          visa_csv_sha256=source["source_sha256"], test_input_sha256=file_hash(args.input_json),
                          split_sha256=dict(train=split_hash(train), validation=split_hash(validation))),
                experiments=jobs)


def worker_environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    return env


def command_for(plan, job, phase):
    settings, folder = plan["settings"], Path(job["directory"])
    if phase == "train":
        command = [sys.executable, "-u", str(PROJECT / "train_attention_predictor.py"),
                   "--model-path", settings["model_path"], "--data-root", settings["visa_root"],
                   "--visa-split-csv", settings["visa_split_csv"], "--epochs", str(settings["epochs"]),
                   "--gradient-accumulation", str(job["batch_size"]), "--learning-rate", str(job["learning_rate"]),
                   "--seed", str(settings["seed"]), "--validation-fraction", ".1", "--teacher-layer", "22",
                   "--weight-decay", "0", "--adam-beta1", ".9", "--adam-beta2", ".98",
                   "--warmup-ratio", ".03", "--max-grad-norm", "1", "--student-precision", "bfloat16",
                   "--log-every", str(settings["log_every"]), "--output-dir", str(folder / "train")]
        if settings["cache_dir"]:
            command += ["--cache-dir", settings["cache_dir"]]
        return command
    return [sys.executable, "-u", str(PROJECT / "run.py"), "--model-path", settings["model_path"],
            "--input-json", settings["input_json"], "--data-root", settings["data_root"],
            "--prompt-version", "v0", "--roi-mode", "anyres_max_9", "--image-token-order", "base_first",
            "--method", "attention_predictor", "--method-config", str(folder / "test_method.json"),
            "--no-sample", "--no-plots", "--include-pruning-time", "--benchmark-metrics",
            "--warmup", str(settings["warmup"]), "--seed", str(settings["seed"]),
            "--output-dir", str(folder / "test")]


def extended_metrics(metrics, baseline):
    """Undefined denominators stay null; do not derive F1 from rounded PRE/Recall."""
    if not math.isfinite(baseline) or not 0 < baseline <= 1:
        raise ValueError("Baseline must be a fraction in (0,1]")
    counts = [metrics[key] for key in ("tp", "fp", "tn", "fn")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, fp, tn, fn = counts
    accuracy = metrics.get("accuracy")
    if accuracy is not None and (type(accuracy) not in (int, float) or not math.isfinite(accuracy)
                                 or not 0 <= accuracy <= 1):
        raise ValueError("ACC must be a fraction in [0,1], not a percentage")
    recall = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return dict(metrics, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                bacc=(recall + tnr) / 2 if recall is not None and tnr is not None else None,
                baseline_accuracy=baseline,
                relative_accuracy_pct=100 * accuracy / baseline if accuracy is not None else None)


def validate_training(plan, job):
    folder = Path(job["directory"]) / "train"
    status, metadata = read_json(folder / "training_status.json"), read_json(folder / "train_run.json")
    settings, data, configuration = plan["settings"], plan["data"], metadata["configuration"]
    if status.get("complete") is not True or status.get("completed_epochs") != settings["epochs"]:
        raise ValueError("Training did not complete the planned epoch budget")
    expected_configuration = dict(
        epochs=settings["epochs"], seed=settings["seed"], teacher_layer=22,
        gradient_accumulation=job["batch_size"], learning_rate=job["learning_rate"],
        model_path=settings["model_path"], data_root=settings["visa_root"], visa_split_csv=settings["visa_split_csv"],
        validation_fraction=.1, weight_decay=0., adam_beta1=.9, adam_beta2=.98, warmup_ratio=.03,
        max_grad_norm=1., student_precision="bfloat16")
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"Training configuration mismatch: {key}")
    if (metadata.get("version") != 2 or metadata.get("student_source_layer") != 2
            or metadata.get("teacher_layer") != 22 or metadata.get("backbone_frozen") is not True
            or metadata.get("architecture", {}).get("source_layers") != [3, 4, 5]):
        raise ValueError("Training is not the planned frozen-backbone, copied-block3/4/5 distillation")
    if (metadata.get("training_samples") != data["training_samples"]
            or metadata.get("validation_samples") != data["validation_samples"]
            or metadata.get("source", {}).get("source_sha256") != data["visa_csv_sha256"]
            or metadata.get("split_sha256") != data["split_sha256"]):
        raise ValueError("Training source or train/validation split mismatch between experiments")
    for name, expected in data["split_sha256"].items():
        if file_hash(folder / f"{name}.jsonl") != expected:
            raise ValueError(f"Saved {name} image list does not match the fixed split")
    if not (folder / "best.pt").is_file() or (folder / "best.pt").stat().st_size == 0:
        raise ValueError("Missing or empty best.pt")
    with (folder / "history.csv").open(encoding="utf-8", newline="") as stream:
        validation = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    if len(validation) != settings["epochs"]:
        raise ValueError("Incomplete validation history")
    best = min(validation, key=lambda row: float(row["kl"]))
    kl = float(best["kl"])
    if not math.isfinite(kl) or not math.isclose(kl, status["best_validation_kl"], abs_tol=1e-10):
        raise ValueError("Best validation KL disagrees with history")
    return dict(best_epoch=int(best["epoch"]), validation_kl=kl)


def test_metrics(plan, job, *, strict=True):
    rows, settings = [], plan["settings"]
    for rate in settings["prune_rates"]:
        path = Path(job["directory"]) / "test" / f"prune_{format_prune_rate(rate)}" / "metrics.json"
        metrics = read_json(path)
        if strict:
            count = plan["data"]["test_samples"]
            if (metrics.get("complete") is not True or metrics.get("prune_rate") != rate
                    or any(metrics.get(key) != count for key in ("expected_samples", "evaluated_samples", "labeled_samples"))):
                raise ValueError(f"Incomplete/full-test count mismatch at rate {rate}")
            counts = [metrics.get(key) for key in ("tp", "fp", "tn", "fn", "unparsed")]
            if any(type(value) is not int or value < 0 for value in counts) or sum(counts) != count:
                raise ValueError(f"Confusion/unparsed counts disagree at rate {rate}")
            if not math.isclose(metrics["accuracy"], (counts[0] + counts[2]) / count, abs_tol=1e-12):
                raise ValueError(f"ACC disagrees with confusion counts at rate {rate}")
            for field in ("lat_mean_ms", "mem_peak_allocated_gib", "mem_peak_reserved_gib"):
                value = metrics.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Missing or invalid {field} at rate {rate}")
        rows.append(extended_metrics(metrics, settings["baseline_accuracy"]))
    return rows


def validate_test(plan, job):
    folder = Path(job["directory"]) / "test"
    run = read_json(folder / "run.json")
    settings = plan["settings"]
    for key, expected in dict(model_path=settings["model_path"], input_json=settings["input_json"],
                              data_root=settings["data_root"], method="attention_predictor",
                              roi_mode="anyres_max_9", prompt_version="v0", save_prune_vis=False,
                              save_attention_vis=False, warmup_requested=settings["warmup"],
                              image_token_order="base_first", do_sample=False, include_pruning_time=True,
                              benchmark_metrics=True, seed=settings["seed"], samples=plan["data"]["test_samples"]).items():
        if run.get(key) != expected:
            raise ValueError(f"Test configuration mismatch: {key}")
    expected_config = read_json(Path(job["directory"]) / "test_method.json")
    if run.get("method_config") != expected_config:
        raise ValueError("Test did not use the planned checkpoint/rates/selection mode")
    rows = test_metrics(plan, job)
    # Enrich only successfully completed run.py outputs; no source/CLI changes.
    for metrics in rows:
        write_json(folder / f"prune_{format_prune_rate(metrics['prune_rate'])}" / "metrics.json", metrics)
    with (folder / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        fields = csv.DictReader(stream).fieldnames
    if not fields:
        raise ValueError("Missing test summary header")
    fields = fields + [key for key in EXTRA_FIELDS if key not in fields]
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    run["ablation_metric_definitions"] = METRIC_DEFINITIONS
    run["baseline_accuracy_reference"] = dict(accuracy=settings["baseline_accuracy"], source="user supplied; not rerun")
    write_json(folder / "run.json", run)
    return {}


def save_plan(plan):
    write_json(Path(plan["output_dir"]) / "ablation.json", plan)


def latest_progress(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 16384))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        return next((line for line in reversed(lines) if any(token in line for token in
                    ("loss=", "mean_loss=", "rate=", "Done.", "AdamW:", "Loading checkpoint shards:"))),
                    "initializing; full output is in the log")
    except OSError:
        return "waiting for log output"


def write_report(plan):
    rows = []
    for job in plan["experiments"]:
        for rate in plan["settings"]["prune_rates"]:
            row = dict(experiment=job["name"], effective_batch_size=job["batch_size"],
                       learning_rate=job["learning_rate"], epochs=plan["settings"]["epochs"],
                       best_epoch=job["train"].get("best_epoch"), validation_kl=job["train"].get("validation_kl"),
                       train_gpu=job["train"].get("gpu"), test_gpu=job["test"].get("gpu"),
                       training_seconds=job["train"].get("seconds"), testing_seconds=job["test"].get("seconds"),
                       prune_rate=rate, status=job["test"]["status"] if job["train"]["status"] == "success" else
                       f"train_{job['train']['status']}", error=job["test"].get("error") or job["train"].get("error"))
            path = Path(job["directory"]) / "test" / f"prune_{format_prune_rate(rate)}" / "metrics.json"
            if path.is_file():
                try:
                    metrics = extended_metrics(read_json(path), plan["settings"]["baseline_accuracy"])
                    row.update({key: value for key, value in metrics.items() if key in SUMMARY_FIELDS})
                    if metrics.get("complete") is not True and row["status"] == "success":
                        row["status"] = "incomplete"
                except (ValueError, KeyError, TypeError) as error:
                    row.update(status="invalid_metrics", error=str(error))
            elif row["status"] == "success":
                row.update(status="missing_metrics", error=str(path))
            rows.append(row)
    path = Path(plan["output_dir"]) / "summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def stop_workers(active, phase):
    for process, log, job, started in active.values():
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            log.close()
            job[phase].update(status="interrupted", seconds=time.perf_counter() - started,
                              error="Launcher interrupted; no automatic restart of partial outputs")


def run_phase(plan, phase, gpus, *, status_every=30):
    """Bounded slots: never overlap two owned workers on one GPU."""
    pending, active = deque(), {}
    for job in plan["experiments"]:
        if phase == "test" and job["train"]["status"] != "success":
            job[phase].update(status="skipped", error="Training is not verified complete")
            continue
        if job[phase]["status"] == "success":
            try:
                job[phase].update(validate_training(plan, job) if phase == "train" else validate_test(plan, job))
                print(f"Verified completed {phase}: {job['name']}; skipping", flush=True)
                continue
            except (OSError, ValueError, KeyError, TypeError) as error:
                job[phase].update(status="failed", error=f"Saved stage validation failed: {error}")
        output = Path(job["directory"]) / phase
        if output.exists() and any(output.iterdir()):
            job[phase].update(status="failed", error=f"Partial/nonverified output is not overwritten: {output}. "
                              "Optimizer resume is unsupported; move that stage directory or use a new output root.")
            print(f"Cannot restart {phase} {job['name']}: {job[phase]['error']}", flush=True)
        else:
            pending.append(job)
    save_plan(plan)
    next_status = time.monotonic() + status_every
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job = pending.popleft()
                folder = Path(job["directory"])
                folder.mkdir(parents=True, exist_ok=True)
                if phase == "test":
                    write_json(folder / "test_method.json", dict(
                        checkpoint=str(folder / "train" / "best.pt"), layer=2, rate_semantics="layer_average",
                        selection_mode="base_guided", prune_rates=plan["settings"]["prune_rates"], visualize_rates=[]))
                log_path = folder / f"{phase}.log"
                log = log_path.open("ab")
                job[phase] = dict(status="running", gpu=gpu, log=str(log_path), command=command_for(plan, job, phase))
                started = time.perf_counter()
                try:
                    process = subprocess.Popen(job[phase]["command"], cwd=PROJECT, env=worker_environment(gpu),
                                               stdout=log, stderr=subprocess.STDOUT)
                except OSError as error:
                    log.close()
                    job[phase].update(status="failed", error=str(error), seconds=time.perf_counter() - started)
                    print(f"Failed to start {phase} {job['name']}: {error}", flush=True)
                else:
                    active[gpu] = (process, log, job, started)
                    print(f"Started {phase} {job['name']} on GPU {gpu}; log={log_path}", flush=True)
                save_plan(plan)
            for gpu, (process, log, job, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                job[phase].update(seconds=time.perf_counter() - started, exit_code=code)
                try:
                    if code != 0:
                        raise ValueError(f"Child exited {code}; inspect {job[phase]['log']}")
                    result = validate_training(plan, job) if phase == "train" else validate_test(plan, job)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    job[phase].update(status="failed", error=str(error))
                else:
                    job[phase].update(result, status="success")
                print(f"Finished {phase} {job['name']}: {job[phase]['status']}, "
                      f"elapsed={job[phase]['seconds']:.1f}s" +
                      (f"; {job[phase]['error']}" if job[phase].get("error") else ""), flush=True)
                save_plan(plan)
                write_report(plan)
            if active and time.monotonic() >= next_status:
                for gpu, (_, _, job, started) in active.items():
                    print(f"[{phase} GPU {gpu} {job['name']} elapsed={time.perf_counter()-started:.0f}s] "
                          f"{latest_progress(job[phase]['log'])}", flush=True)
                next_status = time.monotonic() + status_every
            if active:
                time.sleep(1)
    finally:
        if active:
            stop_workers(active, phase)
        save_plan(plan)
        write_report(plan)


def load_or_create_plan(candidate, *, resume=False):
    root = Path(candidate["output_dir"])
    path = root / "ablation.json"
    if path.is_file():
        if not resume:
            raise FileExistsError(f"Output already contains an experiment: {root}; use --resume or a new directory")
        previous = read_json(path)
        if any(previous.get(key) != candidate[key] for key in ("version", "settings", "data")):
            raise ValueError("--resume requires identical settings and source/split hashes; use the original arguments")
        expected = [(j["name"], j["batch_size"], j["learning_rate"], j["directory"]) for j in candidate["experiments"]]
        actual = [(j["name"], j["batch_size"], j["learning_rate"], j["directory"]) for j in previous["experiments"]]
        if actual != expected:
            raise ValueError("Saved experiment grid does not match the four planned runs")
        return previous
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output must be empty for a new experiment: {root}")
    return candidate


def launch(plan, *, status_every=30):
    root = Path(plan["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".launcher.lock"
    # Never silently run --resume while another launcher still owns the output.
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
    except FileExistsError as error:
        raise FileExistsError(f"Launcher lock exists: {lock}. Do not start two launchers on the same output. "
                              "If the previous launcher crashed, verify it has exited before removing only this lock.") from error
    started = time.perf_counter()
    try:
        save_plan(plan)
        print(f"Fixed VisA split: train={plan['data']['training_samples']}, validation={plan['data']['validation_samples']}; "
              f"FULL test={plan['data']['test_samples']}; epochs={plan['settings']['epochs']}", flush=True)
        print(f"Reference ACC={plan['settings']['baseline_accuracy']:.2%}; no new 0% baseline is run. "
              f"Training GPUs={plan['settings']['gpus']}; all testing GPU={plan['settings']['eval_gpu']}", flush=True)
        run_phase(plan, "train", plan["settings"]["gpus"], status_every=status_every)
        print("All owned training processes have exited; starting serial full-test evaluations.", flush=True)
        run_phase(plan, "test", [plan["settings"]["eval_gpu"]], status_every=status_every)
    finally:
        plan["launcher_last_wall_seconds"] = time.perf_counter() - started
        save_plan(plan)
        write_report(plan)
        lock.unlink()
    rows = write_report(plan)
    for row in rows:
        if row["status"] == "success":
            score = lambda key: "N/A" if row.get(key) is None else f"{100 * row[key]:.2f}%"
            print(f"{row['experiment']} prune={row['prune_rate']} ACC={score('accuracy')} "
                  f"F1={score('f1')} BACC={score('bacc')} RelAcc={row['relative_accuracy_pct']:.2f}%", flush=True)
    print(f"Summary: {root / 'summary.csv'}", flush=True)
    return 0 if all(row["status"] == "success" for row in rows) else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.report_only:
            plan = read_json(args.report_only.resolve() / "ablation.json")
            if (args.report_only.resolve() / ".launcher.lock").exists():
                raise ValueError("A launcher lock exists; do not rewrite reports while an experiment may be running")
            if Path(plan["output_dir"]) != args.report_only.resolve():
                raise ValueError("Saved output path changed; report from the original output directory")
            for job in plan["experiments"]:
                if job["test"]["status"] == "success":
                    validate_test(plan, job)
            rows = write_report(plan)
            print(f"Rebuilt {len(rows)} result rows: {Path(plan['output_dir']) / 'summary.csv'}")
            return 0 if all(row["status"] == "success" for row in rows) else 1
        plan = prepare(args)
        if args.dry_run:
            preview = dict(plan)
            preview["commands"] = [dict(experiment=job["name"], train=command_for(plan, job, "train"),
                                         test=command_for(plan, job, "test")) for job in plan["experiments"]]
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            return 0
        plan = load_or_create_plan(plan, resume=args.resume)
        return launch(plan, status_every=args.status_every)
    except KeyboardInterrupt:
        print("Interrupted. Complete stages are retained; partial training cannot optimizer-resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
