"""Export MMAD Base/AnyRes pruning images at 50% layer-average pruning."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from llava_pruning.mmad_data import load_binary, load_frozen
from llava_pruning.native_pruning_visualization import capture_inputs, view_masks
from llava_pruning.pool_compact_visualization import anchor_decisions
from llava_pruning.pool_distill_visualization import (
    anchor_masks, capture_baseline_views, input_rgb, white_positions,
)


ROOT = Path(__file__).resolve().parent
SERVER_DATA = Path("/home/yz/xxy/data/datasets")
DEFAULT_MODEL = Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov")
METHODS = ("mmadmode", "fastv", "clse")
LABELS = {"mmadmode": "AnomalyPruning (before tail 8)", "fastv": "FastV", "clse": "CLSE"}
RATE = 50
SEED = 42
CANVAS = 1152
GRID = (3, 3)
TILE_SIDE = 27
PATCH_SIZE = 14
METHOD_FILES = ("base.png", "anyres.png", "comparison.png", "visualization.json")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--data-root", type=Path, default=SERVER_DATA / "MMAD")
    cli.add_argument("--annotations", type=Path, help="Override the MMAD annotation JSON")
    cli.add_argument("--image-root", type=Path, help="Override the MMAD image root")
    cli.add_argument("--mvtec-root", type=Path, default=SERVER_DATA / "MVTec-AD")
    cli.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    cli.add_argument("--checkpoint", type=Path, default=ROOT / "output/attention_predictor_twig_train_01/best.pt")
    cli.add_argument("--gpus", type=int, nargs="+", metavar="ID",
                     help="GPU IDs, one worker per GPU (e.g. --gpus 0 1 2 3)")
    cli.add_argument("--resume", action="store_true", help="Reuse a frozen run and skip fully exported samples")
    cli.add_argument("--worker", type=int, nargs=2, help=argparse.SUPPRESS)
    return cli


def file_identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_png(image, path):
    temporary = path.with_suffix(".png.tmp")
    image.save(temporary, format="PNG", compress_level=3)
    temporary.replace(path)


def sample_folder(index):
    return Path("images") / f"{index:06d}"


def shard_samples(samples, rank, count):
    if not 0 <= rank < count:
        raise ValueError("The worker rank must be in [0, worker count)")
    return list(enumerate(samples, start=1))[rank::count]


def write_index(output, samples):
    fields = ["sample_number", "question_id", "dataset", "category", "ground_truth",
              "source_image", "original", *(f"{m}_comparison" for m in METHODS)]
    temporary = output / "index.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples, start=1):
            path = sample_folder(index)
            writer.writerow({
                "sample_number": index, "question_id": sample.sample_id, "dataset": sample.source["dataset"],
                "category": sample.category, "ground_truth": "abnormal" if sample.gt else "normal",
                "source_image": sample.source["image"], "original": (path / "original.png").as_posix(),
                **{f"{m}_comparison": (path / m / "comparison.png").as_posix() for m in METHODS},
            })
    temporary.replace(output / "index.csv")


def load_run(args, output):
    model = args.model_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = model / "config.json"
    if json.loads(config_path.read_text(encoding="utf-8")).get("num_hidden_layers") != 28:
        raise ValueError("Use the same 28-layer OneVision checkpoint as the MMAD benchmark")
    settings = {
        "model_path": str(model), "model_config_sha256": sha256(config_path),
        "checkpoint": file_identity(checkpoint), "data_root": str(args.data_root.expanduser().resolve()),
        "image_root": str(args.image_root.expanduser().resolve()) if args.image_root else None,
        "mvtec_root": str(args.mvtec_root.expanduser().resolve()) if args.mvtec_root else None,
        "annotation_override": str(args.annotations.expanduser().resolve()) if args.annotations else None,
        "methods": list(METHODS), "seed": SEED, "rate": RATE, "rate_semantics": "layer_average",
        "excluded_sources": [], "mask_blend": 0.5, "resize": [CANVAS, CANVAS],
        "mmadmode": {"distill_share": 0.7, "tail_drop_layers": 8, "snapshot": "before tail"},
    }
    manifest = output / "run.json"
    frozen = output / "samples.jsonl"
    if args.resume:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["settings"] != settings or sha256(frozen) != saved["samples_sha256"]:
            raise ValueError("The existing run's settings or sample list changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use an empty output directory or --resume: {output}")
        rows, info = load_binary(args.data_root, args.annotations, args.image_root, SEED,
                                 exclude_sources=(), mvtec_root=args.mvtec_root)
        for row in rows:
            row["image_identity"] = file_identity(Path(row["resolved_image"]))
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "samples.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(frozen)
        saved = {"settings": settings, "dataset": info, "sample_count": len(rows),
                 "samples_sha256": sha256(frozen),
                 "note": "VisA is included for visual review; the MMAD-mode student was trained on VisA."}
        atomic_json(manifest, saved)
    samples = load_frozen(frozen)
    if len(samples) != saved["sample_count"]:
        raise ValueError("The frozen MMAD sample count changed")
    for sample in samples:
        if file_identity(sample.image) != sample.source["image_identity"]:
            raise ValueError(f"Source image changed: {sample.image}")
    write_index(output, samples)
    return model, checkpoint, samples


def overlay(rgb, keep):
    """Blend the real RGB view with its white dropout rendering at 50% opacity."""
    with Image.fromarray(rgb) as original, Image.fromarray(white_positions(rgb, keep, PATCH_SIZE)) as masked:
        return Image.blend(original, masked, 0.5)


def save_panels(output, name, original_path, views, keep, valid, panel_size=384):
    if (len(views) != 1 + GRID[0] * GRID[1] or keep.shape != (len(views), TILE_SIDE**2)
            or valid.shape != keep.shape):
        raise ValueError("Expected one Base view and nine real 27×27 AnyRes token views")
    # Ignore unencoded positions in the displayed mask. Keep the native 6px crop edges.
    display_keep = keep | ~valid
    height, width = views.shape[1:3]
    with overlay(views[0], display_keep[0]) as base, \
            Image.new("RGB", (GRID[0] * width, GRID[1] * height)) as anyres:
        for idx, (crop, mask) in enumerate(zip(views[1:], display_keep[1:])):
            row, col = divmod(idx, GRID[0])
            with overlay(crop, mask) as tile:
                anyres.paste(tile, (col * width, row * height))
        save_png(base, output / "base.png")
        save_png(anyres, output / "anyres.png")
        with Image.open(original_path) as original, \
                Image.new("RGB", (panel_size * 3 + 40, panel_size + 74), "white") as comparison:
            draw = ImageDraw.Draw(comparison)
            font = ImageFont.load_default(size=17)
            draw.text((10, 8), f"{LABELS[name]} | Layer-average pruning target: 50%", fill="black", font=font)
            for i, (title, view) in enumerate(zip(("Original", "Base", "AnyRes"), (original, base, anyres))):
                x = 10 + i * (panel_size + 10)
                draw.text((x, 34), title, fill="black", font=font)
                resample = Image.Resampling.LANCZOS if i == 0 else Image.Resampling.NEAREST
                with ImageOps.contain(view, (panel_size, panel_size), method=resample) as panel:
                    comparison.paste(panel, (x + (panel_size-panel.width)//2,
                                             58 + (panel_size-panel.height)//2))
            save_png(comparison, output / "comparison.png")


def decisions(backend, method, name, sample, seed):
    core = backend.model.get_model()
    options = dict(capture_visualization=name == "clse", capture_attention=False,
                   random_seed=seed, do_sample=False)
    if name == "mmadmode":
        with capture_baseline_views(method) as trace:
            result = backend.generate(sample, sample.text, method, RATE, "anyres_max_9", **options)
        adapter = method.adapter
        if tuple(adapter.geometry.grid) != GRID:
            raise ValueError(f"Unexpected MMAD geometry: {adapter.geometry.grid}")
        selected = trace["selected"].cpu().tolist()
        _, keep, newlines = anchor_masks(
            anchor_decisions(adapter.device_decisions), GRID, selected,
            final_newline=core.config.mm_patch_merge_type.endswith("_add_newl"),
        )
        if int(keep.sum()) + newlines != adapter.plan["final_llm_tokens"]:
            raise ValueError("Rendered anchors differ from the observed MMAD token budget")
        valid = np.ones_like(keep)
        stats = result["stats"]
        meta = dict(stage="post-distillation before tail 8", packed_tokens=adapter.plan["baseline_packed_tokens"],
                    packed_kept=len(selected), actual_average_prune_rate=stats["actual_average_prune_rate"],
                    two_stage_average_prune_rate=stats["two_stage_average_prune_rate"])
    else:
        with capture_inputs() as trace:
            result = backend.generate(sample, sample.text, method, RATE, "anyres_max_9", **options)
        metadata = trace["metadata"]
        if tuple(metadata["grid_patches"]) != GRID:
            raise ValueError(f"Unexpected MMAD geometry: {metadata['grid_patches']}")
        if name == "fastv":
            packed = core.get_fastv_image_masks()[0][0]["keep"]
            stage = "FastV effective mask after block 2"
        else:
            packed = result["stages"][-1]["mask"]["keep"]
            stage = "CLSE after block 1"
        keep, valid, meta = view_masks(packed, metadata, side=TILE_SIDE)
        meta.update(stage=stage, actual_average_prune_rate=result["stats"]["actual_average_prune_rate"])
    meta.update(base_kept=int(keep[0].sum()), base_spatial=int(valid[0].sum()),
                anyres_kept=int(keep[1:].sum()), anyres_spatial=int(valid[1:].sum()),
                mark="white at 50% opacity at removed spatial token positions")
    return input_rgb(trace["pixels"], backend.processor), keep, valid, meta, result["answer"]


def method_for(core, name, checkpoint):
    from benchmark_mmad_binary import make_trial_method
    job = {"method": "hybrid_baseline" if name == "mmadmode" else name, "prune_rate": RATE}
    if name == "mmadmode":
        job.update(variant="joint_self", tail_drop_layers=8)
    return make_trial_method(job, core, checkpoint)


def complete(path):
    return all((path / file).is_file() and (path / file).stat().st_size > 0 for file in METHOD_FILES)


def save_original(folder, sample):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "original.png"
    if not path.is_file() or not path.stat().st_size:
        with Image.open(sample.image) as opened, opened.convert("RGB") as image:
            save_png(image, path)
    return path


def run_shard(output, model_path, checkpoint, samples, rank=0, count=1):
    import torch
    from benchmark_visa_grid3 import validate_grid
    from llava_pruning.backend import LlavaBackend
    from llava_pruning.mmad_data import canonical_answer

    assigned = shard_samples(samples, rank, count)
    label = f"shard {rank + 1}/{count}"
    print(f"[{label}] {len(assigned)} samples assigned", flush=True)
    pending = [name for name in METHODS if any(
        not complete(output / sample_folder(i) / name)
        for i, _ in assigned
    )]
    if not pending:
        for i, sample in assigned:
            save_original(output / sample_folder(i), sample)
        print(f"[{label}] Already complete", flush=True)
        return
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one server GPU")
    backend = LlavaBackend(model_path, "anyres_max_9", image_token_order="base_first",
                           image_resize=(CANVAS, CANVAS))
    validate_grid(backend.model.config)
    core = backend.model.get_model()
    for name in pending:
        method = method_for(core, name, checkpoint)
        ctx = method.generation_context(backend.model) if hasattr(method, "generation_context") else nullcontext()
        with torch.inference_mode(), ctx:
            for step, (i, sample) in enumerate(assigned, start=1):
                folder = output / sample_folder(i)
                target = folder / name
                original = save_original(folder, sample)
                if complete(target):
                    continue
                target.mkdir(parents=True, exist_ok=True)
                views, keep, valid, meta, answer = decisions(backend, method, name, sample, SEED + i - 1)
                save_panels(target, name, original, views, keep, valid)
                prediction, option = canonical_answer(sample.source, answer)
                prediction_label = {"A": "abnormal", "B": "normal"}.get(prediction)
                atomic_json(target / "visualization.json", {
                    "sample_id": sample.sample_id, "method": name, "target_layer_average": RATE,
                    "answer": answer, "prediction": prediction_label, "selected_option": option, **meta,
                })
                if step == 1 or step % 25 == 0 or step == len(assigned):
                    print(f"[{label} {name}] {step}/{len(assigned)} (sample {i:06d})", flush=True)
        del method
        torch.cuda.empty_cache()
    print(f"[{label}] Done", flush=True)


def run_worker(output, rank, count):
    saved = json.loads((output / "run.json").read_text(encoding="utf-8"))
    frozen = output / "samples.jsonl"
    if sha256(frozen) != saved["samples_sha256"]:
        raise ValueError("The frozen samples differ from the coordinator's manifest")
    samples = load_frozen(frozen)
    if len(samples) != saved["sample_count"]:
        raise ValueError("The frozen MMAD sample count changed")
    settings = saved["settings"]
    run_shard(output, Path(settings["model_path"]), Path(settings["checkpoint"]["path"]),
              samples, rank, count)


def launch_gpus(output, gpus):
    logs = output / "worker_logs"
    logs.mkdir(exist_ok=True)
    workers = []
    failures = []
    with ExitStack() as stack:
        try:
            for rank, gpu in enumerate(gpus):
                log_path = logs / f"gpu_{gpu}.log"
                log_file = stack.enter_context(log_path.open("a", encoding="utf-8"))
                log_file.write(f"\n=== GPU {gpu} / shard {rank + 1} of {len(gpus)} ===\n")
                log_file.flush()
                command = [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    "--output-dir", str(output), "--worker", str(rank), str(len(gpus)),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                worker = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
                workers.append((gpu, worker, log_path))
                print(f"GPU {gpu}: shard {rank + 1}/{len(gpus)} -> {log_path}", flush=True)
            for gpu, worker, log_path in workers:
                status = worker.wait()
                if status:
                    failures.append(f"GPU {gpu} (exit {status}, {log_path})")
        finally:
            still_running = [worker for _, worker, _ in workers if worker.poll() is None]
            for worker in still_running:
                worker.terminate()
            for worker in still_running:
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
    if failures:
        raise RuntimeError("Failed workers; check logs and use --resume: " + ", ".join(failures))


def run(args):
    output = args.output_dir.expanduser().resolve()
    if args.worker is not None:
        if args.gpus is not None or args.resume:
            raise ValueError("The internal worker only reads an existing manifest")
        return run_worker(output, *args.worker)
    if args.gpus is not None and (min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    model_path, checkpoint, samples = load_run(args, output)
    print(f"MMAD: {len(samples)} samples × {len(METHODS)} methods, 50% layer-average", flush=True)
    if args.gpus is None:
        run_shard(output, model_path, checkpoint, samples)
    else:
        launch_gpus(output, args.gpus)
    for index in range(1, len(samples) + 1):
        folder = output / sample_folder(index)
        original = folder / "original.png"
        if not original.is_file() or not original.stat().st_size or any(
            not complete(folder / name) for name in METHODS
        ):
            raise RuntimeError(f"Incomplete export for sample {index:06d}; use --resume")
    print(f"Done: {output / 'index.csv'}", flush=True)


if __name__ == "__main__":
    run(parser().parse_args())
