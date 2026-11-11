"""Compare FastV and distilled layer-average pruning on Base-only MVTec."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import benchmark_visa_methods as common
from compare_distilled_visa100 import student_identity
from llava_pruning.base_only_pruning import BOUNDARY, MODES, RATE, BaseOnlyPruning
from visualize_mvtec_baseline import freeze, restored, samples_for


PROJECT = Path(__file__).resolve().parent
VERSION = "base-only-layer-average-v3"
PREVIOUS_VERSION = "base-only-layer-average-v2"
LEGACY_VERSION = "base-only-layer-average-70-v1"
RATES = (50, RATE)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=int, nargs="+", default=[3, 4], help="One or two physical GPU IDs")
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--checkpoint", default="output/attention_predictor_twig_train_01/best.pt")
    p.add_argument("--input-json", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl")
    p.add_argument("--data-root", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/")
    p.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prune-rate", type=int, choices=RATES,
                   help="Target LLM layer-average pruning percentage (default: 70; resume uses saved rate)")
    p.add_argument("--source-run", type=Path, help="Existing Base-only run whose numbered samples will be rerendered")
    p.add_argument("--sample-indices", type=int, nargs="+",
                   help="Zero-based numbers from the source run's sample_NNNNNNNNN.png filenames")
    p.add_argument("--mask-opacity", type=float, help="White overlay opacity from 0 to 1 (default: 1)")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--resume", action="store_true", help="Continue with the saved inputs, checkpoint, and GPU assignments")
    p.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def verify(spec):
    valid_rates = {VERSION: RATES, PREVIOUS_VERSION: RATES, LEGACY_VERSION: (RATE,)}
    if spec["prune_rate"] not in valid_rates.get(spec["version"], ()):
        raise ValueError("Unsupported Base-only visualization version or saved prune rate")
    for path, digest in ((spec["input_json"], spec["input_sha256"]),
                         (Path(spec["model_path"]) / "config.json", spec["model_config_sha256"])):
        if common.digest(path) != digest:
            raise ValueError(f"Saved input or model configuration changed: {path}")


def setup(args):
    if (args.source_run is None) != (args.sample_indices is None):
        raise ValueError("--source-run and --sample-indices must be supplied together")
    opacity = 1.0 if args.mask_opacity is None else args.mask_opacity
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("--mask-opacity must be between 0 and 1")
    rate = RATE if args.prune_rate is None else args.prune_rate
    default_output = Path(f"output/mvtec_base_only_fastv_distilled_{rate}_gpu34_01")
    source_run = args.source_run.expanduser().resolve() if args.source_run else None
    if source_run is not None:
        default_output = source_run / f"mask_{100 * opacity:g}pct"
    output = (args.output_dir or default_output).expanduser().resolve()
    manifest = output / "run.json"
    if args.resume:
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        verify(spec)
        if args.prune_rate is not None and args.prune_rate != spec["prune_rate"]:
            raise ValueError("Requested prune rate differs from the saved experiment; use a new output directory")
        if args.mask_opacity is not None and opacity != spec.get("mask_opacity", 1.0):
            raise ValueError("Requested mask opacity differs from the saved experiment")
        if source_run is not None and (str(source_run) != spec.get("source_run")
                                       or args.sample_indices != spec.get("sample_indices")):
            raise ValueError("Requested sample indices or source run differ from the saved experiment")
        if student_identity(spec["checkpoint"]["path"], "visa") != spec["checkpoint"]:
            raise ValueError("The saved student checkpoint or training manifest changed")
        return output, spec

    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        raise ValueError("Provide one or two distinct nonnegative GPU IDs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty directory or --resume: {output}")
    if source_run is None:
        rows = freeze(args)
        model_path = Path(args.model_path).expanduser().resolve()
        data_root = Path(args.data_root).expanduser().resolve()
        checkpoint = student_identity(args.checkpoint, "visa")
        prompt_version, seed = args.prompt_version, args.seed
    else:
        source = json.loads((source_run / "run.json").read_text(encoding="utf-8"))
        verify(source)
        if "sample_indices" in source:
            raise ValueError("Select sample numbers from the original full run")
        if args.prune_rate is not None and args.prune_rate != source["prune_rate"]:
            raise ValueError("Requested prune rate differs from the source run")
        indices = args.sample_indices
        if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= source["sample_count"]:
            raise ValueError("Sample indices must be distinct and within the source run")
        frozen_rows = Path(source["input_json"]).read_text(encoding="utf-8").splitlines()
        if len(frozen_rows) != source["sample_count"]:
            raise ValueError("Source run's frozen sample count changed")
        rows = [json.loads(frozen_rows[index]) for index in indices]
        model_path, data_root = Path(source["model_path"]), Path(source["data_root"])
        checkpoint = student_identity(source["checkpoint"]["path"], "visa")
        if checkpoint != source["checkpoint"]:
            raise ValueError("The source run's student checkpoint or training manifest changed")
        rate, prompt_version, seed = source["prune_rate"], source["prompt_version"], source["seed"]
        selected = ", ".join(f"{index}={row['question_id']}" for index, row in zip(indices, rows))
        print(f"Selected samples from {source_run} (filename index=question_id): {selected}", flush=True)
    config = model_path / "config.json"
    if json.loads(config.read_text(encoding="utf-8")).get("num_hidden_layers") != 28:
        raise ValueError("Expected the 28-block LLaVA OneVision model")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "input_frozen.jsonl"
    frozen.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    spec = dict(
        version=VERSION, model_path=str(model_path), model_config_sha256=common.digest(config),
        checkpoint=checkpoint, data_root=str(data_root),
        input_json=str(frozen), input_sha256=common.digest(frozen), sample_count=len(rows),
        prompt_version=prompt_version, seed=seed, gpus=args.gpus, mask_opacity=opacity,
        jobs=[dict(mode=mode, gpu=str(args.gpus[i % len(args.gpus)])) for i, mode in enumerate(MODES)],
        rate_semantics="layer_average", prune_rate=rate, pruning_layer=BOUNDARY, roi_mode="base_only",
        definitions=dict(fastv="Native visibility mask", distilled="Student attention, physical drop of Base top-k complement",
                         visualization="White mask blended over removed Base positions; comparison starts with the disk original"),
    )
    if source_run is not None:
        spec.update(source_run=str(source_run), sample_indices=args.sample_indices)
    common.save_json(manifest, spec)
    return output, spec


def save_png(image, path):
    temporary = path.with_suffix(".png.tmp")
    image.save(temporary, format="PNG", compress_level=2)
    temporary.replace(path)


def render_samples(backend, method, samples, spec, folder):
    from PIL import Image
    from llava.mm_utils import process_base_image
    from llava_pruning.pool_distill_visualization import input_rgb, white_positions

    mode = method.mode
    pictures = folder / "visualizations"
    pictures.mkdir(parents=True, exist_ok=True)
    sample_indices = spec.get("sample_indices")
    rows = restored(folder, samples, mode, sample_indices=sample_indices)
    opacity = spec.get("mask_opacity", 1.0)
    patch_size = int(backend.vision_tower.config.patch_size)
    started = time.perf_counter()
    with (folder / "predictions.jsonl").open("a", encoding="utf-8") as stream:
        for index in range(len(rows), len(samples)):
            sample = samples[index]
            sample_index = index if sample_indices is None else sample_indices[index]
            result = backend.generate(
                sample, sample.source["prompt"], method, 0, "base_only",
                capture_visualization=False, capture_attention=False, random_seed=spec["seed"] + sample_index,
                do_sample=False, include_pruning_time=True, benchmark_metrics=False,
            )
            stats = result["stats"]
            with Image.open(sample.image) as original:
                pixels = process_base_image(original.convert("RGB"), backend.processor)
            rgb = input_rgb(pixels.unsqueeze(0), backend.processor)[0]
            output_name = f"visualizations/sample_{sample_index:09d}.png"
            with Image.fromarray(white_positions(rgb, stats["keep"], patch_size)) as masked:
                if opacity == 1.0:
                    save_png(masked, folder / output_name)
                else:
                    with Image.fromarray(rgb) as base, Image.blend(base, masked, opacity) as blended:
                        save_png(blended, folder / output_name)
            row = dict(index=index, question_id=sample.sample_id, gt=sample.gt, answer=result["answer"],
                       selection_mode=mode, output=output_name,
                       **{key: value for key, value in stats.items() if key != "keep"})
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            done = index + 1
            if done == len(rows) + 1 or done % 25 == 0 or done == len(samples):
                eta = (time.perf_counter() - started) * (len(samples) - done) / (done - len(rows))
                print(f"{mode} {done}/{len(samples)} ETA={eta:.1f}s", flush=True)


def worker(manifest, mode):
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    verify(spec)
    job = next(job for job in spec["jobs"] if job["mode"] == mode)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu"]:
        raise ValueError("Worker visible GPU does not match the saved assignment")
    folder = manifest.parent / mode
    samples = samples_for(spec)
    if len(restored(folder, samples, mode, sample_indices=spec.get("sample_indices"))) == len(samples):
        print(f"Reuse completed {mode} images", flush=True)
        return 0
    import torch
    from llava_pruning.backend import LlavaBackend

    torch.manual_seed(spec["seed"])
    backend = LlavaBackend(spec["model_path"], "base_only")
    common.validate_single_gpu(backend.model, torch)
    device = torch.cuda.get_device_properties(0)
    folder.mkdir(parents=True, exist_ok=True)
    inference = dict(backend.inference_config, attention_source=mode,
                     prune_rate=spec["prune_rate"], rate_semantics="layer_average", pruning_layer=BOUNDARY)
    common.verify_runtime(folder, inference, device.name, device.total_memory)
    method = BaseOnlyPruning(backend.model.get_model(), mode, spec["checkpoint"]["path"],
                             prune_rate=spec["prune_rate"])
    with torch.inference_mode():
        render_samples(backend, method, samples, spec, folder)
    return 0


def comparison_images(output, samples, *, prune_rate=RATE, sample_indices=None):
    from PIL import Image, ImageDraw
    from llava_pruning.pool_compact_visualization import font
    from llava_pruning.visualization import prediction_caption

    fastv, distilled = (restored(output / mode, samples, mode, sample_indices=sample_indices) for mode in MODES)
    if len(fastv) != len(samples) or len(distilled) != len(samples):
        raise RuntimeError("Both methods must have results for every sample")
    directory = output / "comparisons"
    directory.mkdir(exist_ok=True)
    for sample, left, right in zip(samples, fastv, distilled):
        path = directory / Path(left["output"]).name
        if path.is_file():
            continue
        with (Image.open(sample.image) as original, Image.open(output / "fastv" / left["output"]) as first,
              Image.open(output / "distilled" / right["output"]) as second):
            height = first.height
            width = max(1, round(original.width * height / original.height))
            with original.convert("RGB").resize((width, height), Image.Resampling.LANCZOS) as reference:
                gap, header = 16, 72
                with Image.new("RGB", (width + first.width + second.width + gap * 4, height + header + gap), "white") as canvas:
                    draw = ImageDraw.Draw(canvas)
                    draw.text((gap, 8), f"ID {sample.sample_id} | Base only | {prune_rate}% layer average | after block {BOUNDARY}",
                              fill="black", font=font(17))
                    panels = (("Original", reference, None), ("FastV", first, left), ("Distilled", second, right))
                    x = gap
                    for title, picture, record in panels:
                        draw.text((x, 31), title, fill="black", font=font(16))
                        if record is not None:
                            caption = prediction_caption(sample.gt, record["answer"])
                            draw.text((x, 51), caption, fill="black", font=font(13))
                        canvas.paste(picture, (x, header))
                        x += picture.width + gap
                    save_png(canvas, path)
    print(f"Saved three PNGs per sample for {len(samples)} MVTec samples under {output}", flush=True)


def run(args):
    output, spec = setup(args)
    jobs = spec["jobs"]
    concurrency = len(spec["gpus"])
    for offset in range(0, len(jobs), concurrency):
        processes = []
        try:
            for job in jobs[offset:offset + concurrency]:
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", job["mode"],
                           "--manifest", str(output / "run.json")]
                processes.append(subprocess.Popen(
                    command, cwd=PROJECT, start_new_session=os.name == "posix",
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpu"]),
                ))
            codes = [process.wait() for process in processes]
        except KeyboardInterrupt:
            print(f"Interrupted. Use --resume --output-dir {output}", flush=True)
            return 130
        finally:
            for process in processes:
                if process.poll() is None:
                    if os.name == "posix":
                        process.send_signal(signal.SIGINT)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if any(codes):
            return 1
    comparison_images(output, samples_for(spec), prune_rate=spec["prune_rate"],
                      sample_indices=spec.get("sample_indices"))
    return 0


def main():
    args = parser().parse_args()
    try:
        return worker(args.manifest, args.worker) if args.worker else run(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
