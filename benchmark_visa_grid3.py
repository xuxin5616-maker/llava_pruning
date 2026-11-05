"""Matched VisA letterbox/stretch-1152 / fixed-3x3 input and tail ablation.

Independent entry point: existing training, inference and benchmark defaults stay
unchanged. CPU image preparation happens once, outside all generate timings.
"""

import argparse
from collections import deque
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import benchmark_visa_methods as common


PROJECT = Path(__file__).resolve().parent
VERSION = "2.0-visa-letterbox-stretch-grid3"
CANVAS = 1152


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+", type=int, default=[3, 4])
    p.add_argument("--resize-modes", nargs="+", choices=("letterbox", "stretch"),
                   default=["letterbox", "stretch"],
                   help="letterbox preserves aspect ratio and pads; stretch directly resizes. Default: BOTH")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--previous-output", type=Path,
                        help="Prior VisA benchmark directory; reuse its checkpoint from run.json")
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--data-root", default="/home/yz/xxy/data/datasets/VisA")
    p.add_argument("--visa-split-csv")
    p.add_argument("--prune-rates", nargs="+", type=float, default=[70.])
    p.add_argument("--distill-share", type=float, default=.7)
    p.add_argument("--tails", nargs="+", type=int, default=[0, 9])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("output/visa_grid3_letterbox_stretch_70_tail0_9_01"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def letterbox(image):
    """RGB, aspect-preserving BICUBIC, centered black padding; no source edits."""
    from PIL import Image
    image = image.convert("RGB")
    width, height = image.size
    scale = min(CANVAS / width, CANVAS / height)
    resized = (min(CANVAS, max(1, round(width * scale))),
               min(CANVAS, max(1, round(height * scale))))
    offset = ((CANVAS - resized[0]) // 2, (CANVAS - resized[1]) // 2)
    result = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    result.paste(image.resize(resized, Image.Resampling.BICUBIC), offset)
    return result, dict(original_size=[width, height], resized_size=list(resized), paste_xy=list(offset))


def resize_image(image, mode):
    """Use identical interpolation; change only aspect preservation / padding."""
    from PIL import Image
    if mode == "letterbox":
        result, geometry = letterbox(image)
    elif mode == "stretch":
        image = image.convert("RGB")
        geometry = dict(original_size=list(image.size), resized_size=[CANVAS, CANVAS], paste_xy=[0, 0])
        result = image.resize((CANVAS, CANVAS), Image.Resampling.BICUBIC)
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    return result, dict(geometry, resize_mode=mode)


def schedule(rates, tails, share, gpus):
    jobs = []
    for name, roi in (("base_only", "base_only"), ("anyres_only", "anyres_only"),
                      ("base_anyres", "anyres_max_9")):
        jobs.append(dict(job_id=name, method="llava_unpruned", roi_mode=roi, prune_rate=0., tail=0,
                         config=f"input={name};canvas=1152;grid=3x3;prune=0%;tail=0"))
    for rate in rates:
        for tail in tails:
            token = f"{rate:g}".replace(".", "p")
            jobs.append(dict(job_id=f"hybrid_{token}_tail{tail}", method="hybrid_baseline",
                roi_mode="anyres_max_9", prune_rate=rate, tail=tail,
                config=f"input=base_anyres;canvas=1152;grid=3x3;prune={rate:g}%;lambda={share:g};"
                       f"base_full;input_branch=base_text;mapping=fast;base_policy=anyres_first;tail={tail}"))
    # Start the full unpruned reference immediately; presentation order stays above.
    for job in jobs:
        # Keep BOTH tail settings on the same card for a cleaner Lat comparison.
        job.update(gpu=gpus[-1] if job["method"] == "hybrid_baseline" else gpus[0],
                   status="pending", reason="")
    return jobs


def input_fingerprint(samples):
    rows = []
    for sample in samples:
        stat = sample.image.stat()
        rows.append((sample.sample_id, sample.gt, str(sample.image), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def cache_fingerprint(records, root):
    values = [(r["image"], common.digest(root / r["image"])) for r in records]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def verify(spec, *, checkpoint_hash=False):
    if spec["version"] != VERSION or spec["source_identity"] != common.source_identity():
        raise ValueError("Version/source changed; use a new output directory")
    if common.digest(Path(spec["model_path"]) / "config.json") != spec["model_config_sha256"]:
        raise ValueError("Model configuration changed")
    if common.digest(spec["input_json"]) != spec["input_sha256"]:
        raise ValueError("Prepared image manifest changed")
    if checkpoint_hash:
        if common.checkpoint_identity(spec["checkpoint"]["path"]) != spec["checkpoint"]:
            raise ValueError("Student checkpoint/provenance changed")
    else:
        stat = Path(spec["checkpoint"]["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (spec["checkpoint"]["size"], spec["checkpoint"]["mtime_ns"]):
            raise ValueError("Student checkpoint changed")
    records = [json.loads(line) for line in Path(spec["input_json"]).read_text(encoding="utf-8").splitlines()]
    if cache_fingerprint(records, Path(spec["prepared_root"])) != spec["prepared_images_sha256"]:
        raise ValueError("A prepared image changed; refusing mixed results")


def setup(args):
    from PIL import Image
    from llava_pruning.visa_data import load_visa_test
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        spec = json.loads((output / "run.json").read_text(encoding="utf-8"))
        verify(spec, checkpoint_hash=True)
        samples, info = load_visa_test(spec["raw_data_root"], spec["visa_split_csv"])
        if input_fingerprint(samples) != spec["raw_inputs_sha256"] or info != spec["dataset_info"]:
            raise ValueError("Original VisA input/split changed")
        return output, spec
    if len(args.resize_modes) != 1:
        raise ValueError("Single-mode setup requires exactly one resize mode")
    mode = args.resize_modes[0]
    if (not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0
            or args.warmup < 0 or not 0 <= args.seed < 2**32
            or not math.isfinite(args.distill_share) or not 0 < args.distill_share < 1):
        raise ValueError("Invalid GPUs/warmup/seed/lambda")
    if (not args.prune_rates or len(set(args.prune_rates)) != len(args.prune_rates)
            or any(not math.isfinite(r) or not 0 < r < 100 for r in args.prune_rates)
            or not args.tails or len(set(args.tails)) != len(args.tails)
            or any(t < 0 or t > 26 for t in args.tails)):
        raise ValueError("Invalid percentage pruning rates / tail counts")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory or --resume: {output}")
    checkpoint = args.checkpoint
    if args.previous_output:
        prior = json.loads((args.previous_output / "run.json").read_text(encoding="utf-8"))
        checkpoint = prior["checkpoint"]["path"]
    if checkpoint is None:
        raise ValueError("Supply --checkpoint or --previous-output (the previous VisA summary directory)")
    identity = common.checkpoint_identity(checkpoint)
    model = Path(args.model_path).expanduser().resolve()
    if json.loads((model / "config.json").read_text(encoding="utf-8")).get("num_hidden_layers") != 28:
        raise ValueError("Expected the existing 28-layer Qwen2 model")
    data = Path(args.data_root).expanduser().resolve()
    split = str(Path(args.visa_split_csv).expanduser().resolve()) if args.visa_split_csv else None
    samples, info = load_visa_test(data, split)
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "prepared"
    (prepared / "images").mkdir(parents=True)
    records = []
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        with Image.open(sample.image) as image:
            image, geometry = resize_image(image, mode)
        relative = f"images/{index:06d}.png"
        image.save(prepared / relative, compress_level=1)
        records.append(dict(sample.source, image=relative, image_size=[CANVAS, CANVAS], transform=geometry))
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(samples):
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(samples) - index - 1) / (index + 1)
            print(f"Prepare {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    input_path = output / "visa_test_grid3.jsonl"
    input_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    jobs = schedule(args.prune_rates, args.tails, args.distill_share, args.gpus)
    for job in jobs:
        job["config"] = f"resize={mode};" + job["config"]
    spec = dict(version=VERSION, resize_mode=mode, created_utc=datetime.now(timezone.utc).isoformat(),
        model_path=str(model), model_config_sha256=common.digest(model / "config.json"),
        checkpoint=identity, raw_data_root=str(data), visa_split_csv=split, dataset_info=info,
        raw_inputs_sha256=input_fingerprint(samples), prepared_root=str(prepared), input_json=str(input_path),
        input_sha256=common.digest(input_path), prepared_images_sha256=cache_fingerprint(records, prepared),
        sample_count=len(samples), source_identity=common.source_identity(), gpus=args.gpus,
        seed=args.seed, warmup=args.warmup, distill_share=args.distill_share,
        jobs=jobs,
        transform=dict(mode=mode, size=[CANVAS, CANVAS], resample="BICUBIC",
            padding_rgb=[0, 0, 0] if mode == "letterbox" else None,
            alignment="center" if mode == "letterbox" else None, grid_pinpoints=[[CANVAS, CANVAS]],
            padding_policy=("Padding is part of the shared square input" if mode == "letterbox" else
                            "Direct resize; no aspect preservation or added padding"),
            image_sizes_policy="1152x1152 in ALL trials"),
        definitions=dict(common.DEFINITIONS,
            relative_accuracy="100*ACC / measured Base+AnyRes unpruned ACC with the SAME resize mode",
            timing_policy="Shared resident-input TTFT and full-generation peak memory; CPU preparation, PNG saving, loading, H2D and warmups excluded; ALL online pruning included",
            tail_policy="Extra removal; original two-stage layer-average pruning budget unchanged",
            parallelism="One process per GPU; separate GPUs run concurrently; GPU ID saved in each config"),
        preparation_seconds=time.perf_counter() - started)
    common.save_json(output / "run.json", spec)
    return output, spec


def make_method(core, job, spec):
    if job["method"] == "llava_unpruned":
        from llava_pruning.methods import FastVMethod
        return FastVMethod(2, (0,), frozenset(), 1, True)
    from llava_pruning.pool_distill import PoolDistillMethod
    from runMyPruningPoolSpatial import MethodSettings
    return PoolDistillMethod(core, spec["checkpoint"]["path"], rate=job["prune_rate"],
        share=spec["distill_share"], mode="base_full", settings=MethodSettings(prune_rate=0),
        pruning_audit="lean", tail_drop_layers=job["tail"], predictor_input="base_text",
        base_mapping="fast", base_policy="anyres_first")


def visual_counts(core, job):
    side = core.get_vision_tower().num_patches_per_side
    if side != 27:
        raise ValueError("Expected the 27x27 SigLIP patch grid")
    from llava_pruning.visa_comparison_methods import native_counts
    grid, counts = native_counts(core, (CANVAS, CANVAS))
    if tuple(grid) != (3, 3):
        raise RuntimeError(f"Unexpected crop grid: {grid}")
    base, anyres, structure = counts
    return (base if job["roi_mode"] == "base_only" else
            anyres + structure if job["roi_mode"] == "anyres_only" else sum(counts))


@contextmanager
def observe_views(core):
    """Shape-only hook on the actual vision model; never request attentions."""
    counts = []
    def record(module, args, kwargs):
        pixels = args[0] if args else kwargs["pixel_values"]
        if pixels.ndim != 4 or tuple(pixels.shape[-2:]) != (384, 384):
            raise RuntimeError("Unexpected vision pixel batch")
        counts.append(int(pixels.shape[0]))
    handle = core.get_vision_tower().vision_tower.register_forward_pre_hook(record, with_kwargs=True)
    try:
        yield counts
    finally:
        handle.remove()


def evaluate(backend, method, samples, spec, job, output, total_memory):
    from llava_pruning.prompts import GENERIC_PROMPT_V0
    from llava_pruning.visa_comparison_methods import observe_prefill, observed_budget
    core = backend.model.get_model()
    original = visual_counts(core, job)
    records = common.restore_records(output / "predictions.jsonl", samples)
    resumed = len(records)

    def generate(index):
        sample = samples[index]
        with observe_prefill(core) as lengths, observe_views(core) as views:
            result = backend.generate(sample, GENERIC_PROMPT_V0, method, job["prune_rate"], job["roi_mode"],
                capture_visualization=False, capture_attention=False, random_seed=spec["seed"] + index,
                do_sample=False, include_pruning_time=True, benchmark_metrics=True)
        expected_views = {"base_only":1, "anyres_only":9, "anyres_max_9":10}[job["roi_mode"]]
        if sum(views) != expected_views:
            raise RuntimeError(f"Expected {expected_views} encoded views, got {views}")
        budget = observed_budget(job["method"], job["prune_rate"], result["stats"], lengths, original, len(core.layers))
        budget["encoded_views"] = sum(views)
        return dict(question_id=sample.sample_id, gt=sample.gt, answer=result["answer"],
                    **{key:result[key] for key in common.TIMING}, budget=budget)

    context = method.generation_context(backend.model) if hasattr(method, "generation_context") else nullcontext()
    with context:
        indices = common.warmup_indices(len(samples), spec["warmup"]) if resumed < len(samples) else []
        for position, index in enumerate(indices, 1):
            generate(index)
            print(f"{job['job_id']} warmup {position}/{len(indices)}", flush=True)
        started = time.perf_counter()
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            for index in range(resumed, len(samples)):
                row = generate(index)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
                records.append(row)
                if index == resumed or (index+1) % 25 == 0 or index+1 == len(samples):
                    elapsed = time.perf_counter() - started
                    eta = elapsed * (len(samples)-index-1) / (index+1-resumed)
                    print(f"{job['job_id']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s "
                          f"Lat={row['lat_seconds']*1000:.2f}ms", flush=True)
    return dict(common.summarize_records(records, len(samples), job, total_memory),
                resumed_samples=resumed, fresh_samples=len(records)-resumed, warmup_samples=len(indices))


def worker(manifest, job_id):
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    job = next(j for j in spec["jobs"] if j["job_id"] == job_id)
    output = manifest.parent / job_id
    output.mkdir(exist_ok=True)
    report = dict(job_id=job_id, status="failed", complete=False, gpu=job["gpu"], input_sha256=spec["input_sha256"])
    try:
        verify(spec)
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.data import load_samples
        from llava_pruning.prompts import GENERIC_PROMPT_V0
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each worker requires exactly one visible CUDA GPU")
        samples = load_samples(spec["input_json"], spec["prepared_root"])
        if len(samples) != spec["sample_count"] or any(s.text != GENERIC_PROMPT_V0 for s in samples):
            raise ValueError("Frozen test list/prompt changed")
        torch.manual_seed(spec["seed"])
        backend = LlavaBackend(spec["model_path"], job["roi_mode"], image_token_order="base_first")
        common.validate_single_gpu(backend.model, torch)
        core = backend.model.get_model()
        if len(core.layers) != 28 or backend.vision_tower.image_size != 384:
            raise ValueError("Expected 28 LLM layers and a 384-pixel vision input")
        for config in (backend.model.config, core.config):
            config.image_grid_pinpoints = [[CANVAS, CANVAS]]
        inference = dict(backend.inference_config, image_grid_pinpoints=[[CANVAS, CANVAS]])
        device = torch.cuda.get_device_properties(0)
        common.verify_runtime(output, inference, device.name, device.total_memory)
        report["inference"] = inference
        method = make_method(core, job, spec)
        report.update(evaluate(backend, method, samples, spec, job, output, device.total_memory))
        report["status"] = "success"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        common.save_json(output / "metrics.json", report)
    return 0 if report["status"] == "success" else 1


def write_reports(output, spec):
    reference = output / "base_anyres/metrics.json"
    base = json.loads(reference.read_text(encoding="utf-8")) if reference.is_file() else {}
    denominator = base.get("accuracy") if base.get("status") == "success" and base.get("complete") else None
    from llava_pruning.pool_ablation_report import compact_result_row
    rows, budgets = [], []
    for job in spec["jobs"]:
        path = output / job["job_id"] / "metrics.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if result and (result.get("job_id") != job["job_id"] or result.get("input_sha256") != spec["input_sha256"]):
            raise ValueError("Mismatched saved report")
        status = result.get("status", "pending")
        config = job["config"] + f";gpu={job['gpu']}"
        row = dict.fromkeys(common.FIELDS); row["config"] = config
        if status == "success" and result.get("complete"):
            row = compact_result_row(dict(result, baseline_accuracy=denominator or 1.), config)
            if not denominator:
                row["relative_accuracy_pct"] = None
        else:
            row["config"] += f";status={status}"
        rows.append(row)
        budgets.append(dict(result.get("budget", {}), config=row["config"], status=status,
                            reason=result.get("error", ""), target_average_prune_rate=job["prune_rate"]))
    common.save_csv(output / "summary.csv", rows, common.FIELDS)
    common.save_csv(output / "budget_summary.csv", budgets, common.BUDGET_FIELDS)


def merge_resize_reports(output, modes):
    """Keep per-mode ratios as computed from each mode's own reference."""
    for name, fields in (("summary.csv", common.FIELDS), ("budget_summary.csv", common.BUDGET_FIELDS)):
        import csv
        rows = []
        for mode in modes:
            path = output / mode / name
            if path.is_file():
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows.extend(csv.DictReader(stream))
        common.save_csv(output / name, rows, fields)


def run_both(args, *, process_factory=None):
    """One single-GPU controller per resize mode; all five jobs stay on that GPU."""
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        spec = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if (spec.get("kind") != "paired_resize" or spec["version"] != VERSION
                or spec["source_identity"] != common.source_identity()):
            raise ValueError("Paired benchmark version/source changed; use a new output directory")
        if common.checkpoint_identity(spec["checkpoint"]["path"]) != spec["checkpoint"]:
            raise ValueError("Student checkpoint/provenance changed")
        for key, value in spec["configuration"].items():
            setattr(args, key, Path(value) if key in {"checkpoint", "output_dir"} and value is not None else value)
    else:
        if (len(args.resize_modes) != 2 or len(set(args.resize_modes)) != 2
                or not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0):
            raise ValueError("Select distinct resize modes and one/two distinct nonnegative GPUs")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use a new/empty output directory or --resume: {output}")
        checkpoint = args.checkpoint
        if args.previous_output:
            prior = json.loads((args.previous_output / "run.json").read_text(encoding="utf-8"))
            checkpoint = prior["checkpoint"]["path"]
        if checkpoint is None:
            raise ValueError("Supply --checkpoint or --previous-output")
        identity = common.checkpoint_identity(checkpoint)
        configuration = {key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()
                         if key not in {"worker", "manifest", "resume"}}
        configuration.update(checkpoint=identity["path"], previous_output=None, output_dir=str(output))
        spec = dict(version=VERSION, kind="paired_resize", source_identity=common.source_identity(),
            checkpoint=identity, configuration=configuration, resize_modes=args.resize_modes,
            gpu_by_mode={mode:args.gpus[index % len(args.gpus)] for index,mode in enumerate(args.resize_modes)},
            interpretation="Each resize mode has its OWN full unpruned ACC denominator. All jobs within a mode use the same GPU.",
            complete=False)
        output.mkdir(parents=True)
        common.save_json(output / "run.json", spec)

    factory = process_factory or subprocess.Popen
    active, failures = {}, []
    expected_rows = 3 + len(args.prune_rates) * len(args.tails)
    # A single selected GPU runs both modes sequentially; two GPUs run in parallel.
    pending = deque(spec["resize_modes"])
    last_progress = time.perf_counter()
    try:
        while pending or active:
            busy = {spec["gpu_by_mode"][mode] for mode in active}
            for mode in list(pending):
                gpu = spec["gpu_by_mode"][mode]
                if gpu in busy:
                    continue
                folder = output / mode
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--resize-modes", mode,
                    "--gpus", str(gpu), "--checkpoint", spec["checkpoint"]["path"],
                    "--model-path", args.model_path, "--data-root", args.data_root,
                    "--prune-rates", *map(str,args.prune_rates), "--distill-share", str(args.distill_share),
                    "--tails", *map(str,args.tails), "--warmup", str(args.warmup), "--seed", str(args.seed),
                    "--output-dir", str(folder)]
                if args.visa_split_csv:
                    command.extend(["--visa-split-csv", args.visa_split_csv])
                if args.resume and (folder / "run.json").is_file():
                    command.append("--resume")
                log_path = output / (mode + ".log")
                log = log_path.open("a", encoding="utf-8")
                try:
                    process = factory(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=os.name != "nt",
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1"))
                except BaseException:
                    log.close(); raise
                active[mode] = (process, log, time.perf_counter())
                pending.remove(mode); busy.add(gpu)
                print(f"Start {mode}: GPU {gpu}, {expected_rows} trials; log={log_path}", flush=True)
            for mode, (process, log, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del active[mode]
                import csv
                table = output / mode / "summary.csv"
                rows = []
                if table.is_file():
                    with table.open(encoding="utf-8-sig", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                complete = len(rows) == expected_rows and all(row.get("acc") not in (None, "") for row in rows)
                if code or not complete:
                    failures.append(mode)
                    if not complete:
                        print(f"Incomplete {mode} summary: expected {expected_rows} completed rows", flush=True)
                print(f"Finished {mode}: exit={code}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
                merge_resize_reports(output, spec["resize_modes"])
            if time.perf_counter() - last_progress >= 30:
                merge_resize_reports(output, spec["resize_modes"])
                for mode, (_, _, started) in active.items():
                    print(f"{mode}: GPU {spec['gpu_by_mode'][mode]}, elapsed={time.perf_counter()-started:.0f}s; "
                          f"progress and ETA: {output/(mode+'.log')}", flush=True)
                last_progress = time.perf_counter()
            if active:
                time.sleep(.25)
    except BaseException:
        for process, log, _ in active.values():
            if process.poll() is None:
                if os.name == "posix": process.send_signal(signal.SIGINT)
                else: process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            log.close()
        raise
    merge_resize_reports(output, spec["resize_modes"])
    spec.update(complete=not failures, failed_modes=failures)
    common.save_json(output / "run.json", spec)
    print(f"Combined summary: {output/'summary.csv'}; separate tables: letterbox/ and stretch/", flush=True)
    return 1 if failures else 0


def run(args, *, process_factory=None):
    if getattr(args, "resume", False):
        saved = json.loads((args.output_dir.expanduser().resolve() / "run.json").read_text(encoding="utf-8"))
        if saved.get("kind") == "paired_resize":
            return run_both(args, process_factory=process_factory)
        args.resize_modes = [saved.get("resize_mode", "letterbox")]
    if len(getattr(args, "resize_modes", ["letterbox"])) > 1:
        return run_both(args, process_factory=process_factory)
    output, spec = setup(args)
    write_reports(output, spec)
    queues = {gpu:deque() for gpu in spec["gpus"]}
    order = [spec["jobs"][2], *spec["jobs"][:2], *spec["jobs"][3:]]
    for job in order:
        queues[job["gpu"]].append(job)
    active, failures = {}, []
    factory = process_factory or subprocess.Popen
    last_progress = time.perf_counter()
    print(f"{spec['sample_count']} official VisA test images; fixed 3x3; GPUs={spec['gpus']}; no visualizations", flush=True)
    try:
        while active or any(queues.values()):
            for gpu, queue in queues.items():
                while gpu not in active and queue:
                    job = queue.popleft()
                    report = output / job["job_id"] / "metrics.json"
                    old = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
                    if old.get("status") == "success" and old.get("complete"):
                        print(f"Reuse completed {job['job_id']}", flush=True); continue
                    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", job["job_id"],
                               "--manifest", str(output / "run.json")]
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
                report = output / job["job_id"] / "metrics.json"
                result = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
                if code or result.get("status") != "success" or not result.get("complete"):
                    failures.append(job["job_id"])
                    folder = report.parent; folder.mkdir(exist_ok=True)
                    common.save_json(report, dict(result, job_id=job["job_id"], status="failed", complete=False,
                        input_sha256=spec["input_sha256"], error=result.get("error", f"Worker exited {code} without a complete report")))
                print(f"Finished GPU {gpu}: {job['job_id']}; exit={code}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
                write_reports(output, spec)
            if time.perf_counter() - last_progress >= 30:
                for gpu, (_, _, job, started) in active.items():
                    count = len(common.read_prefix(output / job["job_id"] / "predictions.jsonl")[0])
                    print(f"GPU {gpu}: {job['job_id']} saved={count}/{spec['sample_count']} "
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
    print(f"Summary: {output/'summary.csv'}; budget audit: {output/'budget_summary.csv'}", flush=True)
    return 1 if failures else 0


def main():
    args = parser().parse_args()
    if args.worker:
        if not args.manifest:
            raise ValueError("Worker requires --manifest")
        return worker(args.manifest.resolve(), args.worker)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
