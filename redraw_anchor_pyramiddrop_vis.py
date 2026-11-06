"""Combine cached PNGs: original AnyRes, pruned Base, and pruned AnyRes.

CPU only, Pillow + standard library. No model, Torch, GPU or inference.
Reads existing visualizations; one PNG per sample/method/rate.
"""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import re
import time

from PIL import Image, ImageDraw, ImageFont


# Edit these defaults, or pass --results-dir / --output-dir.
RESULTS_DIR = Path("output/visa_anchor_pyramiddrop_full_vis_01")
OUTPUT_DIR = None  # Default: a sibling directory named RESULTS_DIR + '_compact'.
PANEL_HEIGHT = 768
VIEW_SIZE = 384
STAGES = {"anchorprune": ("pre_llm", "AnchorPrune", 0),
          "pyramiddrop": ("after_layer_21", "PyramidDrop", 21)}
INDEX_FIELDS = ("method", "prune_rate", "sample", "question_id", "gt", "prediction",
                "measured_answer", "visualization_answer", "grid_width", "grid_height",
                "output", "status", "error")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Experiment directory, or its visualizations directory")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--panel-height", type=int, default=PANEL_HEIGHT,
                   help="All panels' display height; nearest-neighbor resizing. 0 keeps native AnyRes height")
    return p


@lru_cache(maxsize=4)
def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def label(gt):
    return "Abnormal" if gt == 1 else "Normal" if gt == 0 else "Unknown"


def prediction(answer):
    match = re.match(r"^\s*([AB])\b", answer, re.IGNORECASE) if isinstance(answer, str) else None
    return label(1 if match.group(1).upper() == "A" else 0) if match else "Unparsed"


def read_body(path):
    """Legacy files have a variable text header ABOVE the real 384x384 view."""
    with Image.open(path) as image:
        if image.width != VIEW_SIZE or image.height < VIEW_SIZE:
            raise ValueError(f"Expected a cached 384-pixel-wide view: {path}")
        return image.convert("RGB").crop((0, image.height - VIEW_SIZE, VIEW_SIZE, image.height))


def render_sample(folder, method, rate, *, panel_height=PANEL_HEIGHT):
    if panel_height < 0:
        raise ValueError("panel-height must be nonnegative")
    saved = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    grid = saved.get("native_grid")
    if (not isinstance(grid, list) or len(grid) != 2
            or any(type(n) is not int or n < 1 for n in grid)):
        raise ValueError("Missing or invalid native_grid [width, height]")
    width, height = grid
    if saved.get("encoder_views", 1 + width * height) != 1 + width * height:
        raise ValueError("Saved encoder-view count disagrees with the native crop grid")
    prefix, title, layer = STAGES[method]
    stages = saved.get("stages", [])
    stage = next((s for s in stages if s.get("after_layer") == layer), None)
    if stage is None:
        raise ValueError(f"Final pruning stage is missing: {prefix}; partial stages are not substituted")
    base = read_body(folder / f"{prefix}_base.png")
    mosaic = Image.new("RGB", (width * VIEW_SIZE, height * VIEW_SIZE), "white")
    original = Image.new("RGB", mosaic.size, "white")
    for tile in range(width * height):
        view = read_body(folder / f"{prefix}_anyres_{tile+1:02d}.png")
        raw_view = read_body(folder / f"input_anyres_{tile+1:02d}.png")
        row, col = divmod(tile, width)
        mosaic.paste(view, (col * VIEW_SIZE, row * VIEW_SIZE))
        original.paste(raw_view, (col * VIEW_SIZE, row * VIEW_SIZE))
        view.close(); raw_view.close()
    target_height = panel_height or mosaic.height
    base = base.resize((target_height, target_height), Image.Resampling.NEAREST)
    mosaic = mosaic.resize((max(1, round(mosaic.width * target_height / mosaic.height)), target_height),
                           Image.Resampling.NEAREST)
    original = original.resize(mosaic.size, Image.Resampling.NEAREST)
    margin, gap, header, footer = 20, 24, 100, 36
    canvas = Image.new("RGB", (2 * margin + original.width + base.width + 2 * gap + mosaic.width,
                               header + target_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    text = f"{title} | Average pruning {rate:g}%"
    if layer:
        text += f" | After layer {layer}"
    draw.text((margin, 10), text, fill="black", font=font(22))
    answer = saved.get("visualization_answer", saved.get("measured_answer", ""))
    pred = prediction(answer)
    sample_id = saved.get("question_id", folder.name)
    caption = f"ID: {sample_id} | GT: {label(saved.get('gt'))} | Pred: {pred}"
    draw.text((margin, 43), caption, fill="black", font=font(16))
    middle = margin + original.width + gap
    right = middle + base.width + gap
    draw.text((margin, 73), f"Original AnyRes ({width} x {height})", fill="black", font=font(19))
    draw.text((middle, 73), "Base (pruned)", fill="black", font=font(19))
    draw.text((right, 73), "AnyRes (pruned)", fill="black", font=font(19))
    canvas.paste(original, (margin, header))
    canvas.paste(base, (middle, header))
    canvas.paste(mosaic, (right, header))
    note = "White = pruned | Gray = native unpadding"
    if stage.get("max9_resized"):
        note += " | AnyRes mask projected from packed tokens"
    draw.text((margin, header + target_height + 9), note, fill="black", font=font(14))
    base.close(); mosaic.close(); original.close()
    return canvas, dict(question_id=sample_id, gt=saved.get("gt"), prediction=pred,
                        measured_answer=saved.get("measured_answer", ""), visualization_answer=answer,
                        grid_width=width, grid_height=height)


def run(args):
    source = args.results_dir.expanduser().resolve()
    views = source / "visualizations" if (source / "visualizations").is_dir() else source
    output = (args.output_dir.expanduser().resolve() if args.output_dir
              else source.with_name(source.name + "_compact"))
    if output.is_relative_to(views) or views.is_relative_to(output):
        raise ValueError("Use a separate output directory so the original visualizations remain unchanged")
    if args.panel_height < 0:
        raise ValueError("panel-height must be nonnegative")
    jobs = []
    for method in STAGES:
        for group in sorted((views / method).glob("prune_*")):
            match = re.fullmatch(r"prune_(\d+(?:\.\d+)?)", group.name)
            if group.is_dir() and match:
                jobs.extend((method, float(match.group(1)), group.name, folder)
                            for folder in sorted(group.glob("sample_*")) if folder.is_dir())
    if not jobs:
        raise FileNotFoundError("No cached sample folders found. This script needs the existing per-view pruning PNGs; summary.csv alone is insufficient.")
    output.mkdir(parents=True, exist_ok=True)
    errors, started = 0, time.perf_counter()
    last_progress = started
    with (output / "redraw_index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for index, (method, rate, group, folder) in enumerate(jobs, 1):
            row = dict(method=method, prune_rate=rate, sample=folder.name, status="success", error="")
            try:
                picture, metadata = render_sample(folder, method, rate, panel_height=args.panel_height)
                target = output / method / group / (folder.name + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    picture.save(target, compress_level=2)
                finally:
                    picture.close()
                row.update(metadata, output=target.relative_to(output).as_posix())
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors += 1
                row.update(status="failed", error=str(error))
                print(f"FAILED {method}/{group}/{folder.name}: {error}", flush=True)
            writer.writerow(row); stream.flush()
            if index == 1 or index % 100 == 0 or index == len(jobs) or time.perf_counter() - last_progress >= 15:
                elapsed = time.perf_counter() - started
                print(f"Redraw {index}/{len(jobs)} | failed={errors} | elapsed={elapsed:.1f}s | ETA={elapsed*(len(jobs)-index)/index:.1f}s", flush=True)
                last_progress = time.perf_counter()
    print(f"Saved {len(jobs)-errors}/{len(jobs)} images: {output}; index: {output/'redraw_index.csv'}", flush=True)
    return int(errors > 0)


def main():
    try:
        return run(parser().parse_args())
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
