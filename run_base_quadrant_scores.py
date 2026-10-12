"""Base-quadrant references for token-level negative cosine and L2 residuals.

Base is the complete image, NOT a center-75% crop. Split its native token grid
by token centers into four spatial quadrants and average unit token vectors.
Both Base and AnyRes tokens reference the corresponding BASE quadrant mean.
Save four layer overviews, with raw region sums on BOTH views, and configs.
Only SigLIP executes; the existing loader still loads the LLM weights.
"""
from __future__ import annotations

RESULTS_DIR = r"outputs/feature_scores_02"
MODEL_PATH = r""
OUTPUT_DIR = r""
DATA_ROOT = r""
SCRIPT_VERSION = "1.0-base-quadrant-reference"

import argparse
import csv
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from base_quadrant_math import METRICS, QUADRANTS, region_totals, score_quadrants
from feature_residual_capture import LAYERS, STAGES, capture_stage_scores
from llava_pruning.score_visualization import COLORMAP, MISSING_COLOR, OVERLAY_ALPHA, prepare_views, stitch_score_map
from runVFlowOpt import base_score_map, check_output, prepare_inputs, script_path, write_json
from runVFlowOpt_unit import TOKEN_NORM_EPS
from visualize_patch_means import tile_rectangles


DEFINITION = {
    "base_source": "full original image; no center crop; no four-image re-encoding",
    "features": "1-based SigLIP block outputs before post-layernorm; no projector",
    "normalization": {"formula": "u_i = f_i / max(norm(f_i), epsilon)",
                      "axis": "channels", "epsilon": TOKEN_NORM_EPS},
    "reference": "mean of Base unit tokens in the SAME spatial quadrant, including the queried Base token",
    "reference_used_by": ["Base", "AnyRes"], "mean_renormalized": False,
    "quadrants": list(QUADRANTS),
    "quadrant_assignment": "token center in normalized ORIGINAL-image coordinates; x/y < 0.5 left/top; ties right/bottom",
    "odd_grid": "27x27 cannot split equally; pixel centers on the 384x384 Base determine membership",
    "anyres_mapping": "subtract canvas padding from each tile token center, divide by resized content width/height",
    "neg_cos": "-cos(u_i, mean_Base_quadrant)",
    "l2": "norm(u_i - mean_Base_quadrant)",
    "local_neighborhood_windows": None, "softmax": False,
    "token_unit": "native 14x14 encoder patch, 27x27 tokens per 384x384 view; no heatmap smoothing",
    "invalid": "AnyRes patches not fully covered by image content excluded; zero-norm cosine undefined (NaN)",
    "sum": "signed RAW finite token scores summed separately per layer, metric, view and AnyRes spatial region",
    "base_sum_regions": "map Base token centers through actual AnyRes resize/padding to crop IDs; each token counted once",
    "anyres_sum_regions": "each actual crop independently; no forced 3x3 grid when preprocessing selects another grid",
    "empty_sum": "NaN / NA if no finite scores; never treat missing scores as zero",
    "sum_caution": "Base and AnyRes sums have different token counts; sums are NOT directly resolution-invariant",
    "colormap": COLORMAP, "overlay_alpha": OVERLAY_ALPHA,
    "missing_color": MISSING_COLOR,
    "color_scales": {"raw": {"neg_cos": [-1., 1.], "l2": [0., 2.]},
                     "minmax": "joint Base+AnyRes per image/layer/metric; display only; sums stay RAW"},
    "interpretation": "custom feature differences, not attention, entropy or defect probabilities",
    "model_forward_modified": False, "llm_forward": False, "projector_forward": False,
}


def display_scores(base, tiles, metric, color_scale):
    if metric not in METRICS or color_scale not in ("raw", "minmax"):
        raise ValueError("Unknown metric/color scale")
    base, tiles = np.asarray(base), np.asarray(tiles)
    if np.isinf(base).any() or np.isinf(tiles).any():
        raise ValueError("Infinite quadrant scores")
    finite = np.concatenate((base[np.isfinite(base)], tiles[np.isfinite(tiles)]))
    raw_range = [float(finite.min()), float(finite.max())] if finite.size else None
    if color_scale == "raw":
        return base.copy(), tiles.copy(), DEFINITION["color_scales"]["raw"][metric], raw_range
    if raw_range is None:
        return base.copy(), tiles.copy(), [0., 1.], None
    low, high = raw_range
    if low == high:
        return (np.where(np.isfinite(base), .5, np.nan),
                np.where(np.isfinite(tiles), .5, np.nan), [0., 1.], raw_range)
    return (base - low) / (high - low), (tiles - low) / (high - low), [0., 1.], raw_range


def summary_rows(scores, geometry):
    for si, (stage, layer) in enumerate(zip(STAGES, LAYERS)):
        for metric in METRICS:
            for view in ("base", "tile"):
                sums, counts, valid_counts = region_totals(scores, si, view, metric, geometry)
                for index, (total, count, valid) in enumerate(zip(sums, counts, valid_counts)):
                    row, col = divmod(index, geometry.grid[0])
                    yield {"stage": stage, "layer": layer, "view": "anyres" if view == "tile" else view,
                           "metric": metric, "region_id": index + 1, "row": row + 1, "col": col + 1,
                           "raw_sum": float(total) if count else "",
                           "finite_token_count": int(count), "valid_token_count": int(valid),
                           "undefined_token_count": int(valid - count),
                           "raw_mean": float(total / count) if count else ""}


def draw_figures(image, base_image, scores, geometry, folder, caption, color_scale):
    from matplotlib import colormaps, rc_context, patheffects
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    folder = Path(folder)
    filenames = [f"{stage}_quadrant_scores.png" for stage in STAGES]
    if any((folder / name).exists() for name in filenames):
        raise FileExistsError("Quadrant figures already exist; use a new output directory")
    rectangles = tile_rectangles(asdict(geometry))
    cmap = colormaps[COLORMAP].with_extremes(bad=MISSING_COLOR)
    stroke = [patheffects.withStroke(linewidth=2, foreground="black")]

    def panel(ax, source, heatmap=None, norm=None, totals=None):
        ax.imshow(source, extent=(0, 1, 1, 0), interpolation="nearest")
        if heatmap is not None:
            rgba = cmap(norm(np.ma.masked_invalid(heatmap)))
            rgba[..., 3] = np.where(np.isfinite(heatmap), OVERLAY_ALPHA, 1.)
            ax.imshow(rgba, extent=(0, 1, 1, 0), interpolation="nearest")
        for index, rect in enumerate(rectangles):
            if rect is None:
                continue
            x, y, w, h = rect
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=.75))
            if totals is None:
                ax.text(x + .008, y + .014, str(index + 1), color="white", fontsize=8,
                        ha="left", va="top", path_effects=stroke)
            else:
                value = totals[index]
                ax.text(x + w / 2, y + h / 2, f"{value:.2f}" if np.isfinite(value) else "NA",
                        color="white", fontsize=9, ha="center", va="center", path_effects=stroke,
                        bbox={"facecolor": "black", "alpha": .4, "edgecolor": "none", "pad": 1.5})
        ax.axvline(.5, color="#ffff00", linestyle="--", linewidth=1.)
        ax.axhline(.5, color="#ffff00", linestyle="--", linewidth=1.)
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
        ax.set_aspect(source.height / source.width)

    displays = {}
    for si, (stage, layer) in enumerate(zip(STAGES, LAYERS)):
        with rc_context({"font.size": 10, "axes.titlesize": 12}):
            fig = Figure(figsize=(15, 10.5), dpi=150, layout="constrained", facecolor="white")
            FigureCanvasAgg(fig)
            grid = fig.add_gridspec(4, 3, height_ratios=[1., 1., .05, .18])
            axes = [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)]
            panel(axes[0][0], base_image)
            panel(axes[1][0], image)
            axes[0][0].set_title("Base reference: four quadrant means")
            axes[1][0].set_title(f"Original: AnyRes {geometry.grid[0]} x {geometry.grid[1]} regions")
            ranges = {}
            for col, metric in enumerate(METRICS, start=1):
                b, t, limits, raw_range = display_scores(scores[f"base_{metric}"][si],
                                                       scores[f"tile_{metric}"][si], metric, color_scale)
                norm = Normalize(*limits)
                base_sums = region_totals(scores, si, "base", metric, geometry)[0]
                tile_sums = region_totals(scores, si, "tile", metric, geometry)[0]
                panel(axes[0][col], base_image, base_score_map(b, geometry), norm, base_sums)
                panel(axes[1][col], image, stitch_score_map(t, geometry), norm, tile_sums)
                label = "Negative cosine" if metric == "neg_cos" else "L2 residual"
                axes[0][col].set_title(f"Base: {label} | raw sums")
                axes[1][col].set_title(f"AnyRes: {label} | raw sums")
                bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(grid[2, col]),
                                   orientation="horizontal")
                bar.set_label(label if color_scale == "raw" else f"{label}: joint min-max display")
                ranges[metric] = {"raw_range": raw_range, "color_limits": list(limits)}
            footer = fig.add_subplot(grid[3, :])
            footer.set_axis_off()
            footer.text(.5, .45,
                        "Both views use BASE quadrant means of unit tokens; means are not renormalized.\n"
                        "Yellow dashed = quadrants; white = AnyRes regions; numbers = signed RAW score sums.\n"
                        "Sums depend on token count (see region_sums.csv). Gray = padding / unencoded / undefined.\n"
                        + ("Fixed raw scales across all layers/images." if color_scale == "raw" else
                           "Min-max is shared by Base/AnyRes per layer/metric; colors across layers/images are not absolute comparisons.")
                        + " Scores are NOT defect probabilities.",
                        ha="center", va="center", fontsize=9, transform=footer.transAxes)
            fig.suptitle(f"{caption}\nSigLIP layer {layer} | Base quadrant reference | {color_scale}", fontsize=14)
            try:
                fig.savefig(folder / filenames[si], dpi=150, facecolor="white")
            finally:
                fig.clear()
            displays[stage] = ranges
    return {"figures": filenames, "display_ranges": displays}


def save_sample(image, base, scores, shapes, geometry, sample, folder, color_scale, config):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    sample_config = {**config, "script_version": SCRIPT_VERSION, "definition": DEFINITION,
                     "color_scale": color_scale, "id": sample["id"], "image": str(sample["image"]),
                     "gt": sample.get("gt"), "geometry": asdict(geometry), "feature_shapes": shapes,
                     "base_quadrant_token_counts": scores["base_quadrant_counts"][0].tolist(),
                     "source_metadata": sample.get("source_metadata")}
    write_json(folder / "config.json", sample_config)
    np.savez_compressed(folder / "quadrant_scores.npz", **scores, layers=np.asarray(LAYERS),
                        stages=np.asarray(STAGES), quadrants=np.asarray(QUADRANTS))
    rows = list(summary_rows(scores, geometry))
    with (folder / "region_sums.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gt = {0: "Normal", 1: "Abnormal"}.get(sample.get("gt"), "Unknown")
    display = draw_figures(image, base, scores, geometry, folder,
                           f"ID: {sample['id']} | GT: {gt} | No LLM prediction", color_scale)
    write_json(folder / "config.json", {**sample_config, **display})


def run(model_path, output_dir, results_dir=None, input_json=None, data_root=None,
        limit=None, color_scale="raw"):
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401

    if color_scale not in ("raw", "minmax"):
        raise ValueError("color-scale must be raw or minmax")
    samples, source_run, source_root = prepare_inputs(results_dir, input_json, data_root, limit)
    output = check_output(output_dir, source_root)
    model_path = model_path or source_run.get("model_path")
    if not model_path:
        raise ValueError("Pass --model-path or set MODEL_PATH; SigLIP features must be recomputed")
    checkpoint = script_path(str(model_path))
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Local checkpoint not found: {checkpoint}")

    import torch
    from llava_pruning.backend import LlavaBackend

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one visible CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=5")
    print(f"Loading checkpoint for {len(samples)} images; only SigLIP executes.", flush=True)
    backend = LlavaBackend(checkpoint, roi_mode="anyres_max_9")
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

    tower = backend.vision_tower
    if not isinstance(tower, SigLipVisionTower):
        raise ValueError("This probe requires the existing SigLIP tower")
    tower.eval()
    vision = tower.vision_tower.vision_model
    if len(vision.encoder.layers) != 26 or tuple(vision.embeddings.patch_embedding.kernel_size) != (14, 14):
        raise ValueError("Expected the existing 26-block, patch14 SigLIP tower")
    config = {"script_version": SCRIPT_VERSION, "model_path": str(checkpoint),
              "results_dir": str(source_root) if source_root else None,
              "input_json": str(input_json) if input_json else None,
              "data_root": str(data_root) if data_root else None, "output_dir": str(output),
              "sample_count": len(samples), "limit": limit,
              "layers": list(LAYERS), "metrics": list(METRICS), "color_scale": color_scale,
              "roi_mode": "anyres_max_9", "global_reference": "full_base_quadrants",
              "definition": DEFINITION, "vision_dtype": str(tower.dtype),
              "llm_weights_loaded": True, "llm_forward": False, "projector_forward": False,
              "training": False, "pruning": False,
              "loader_config": {k: v for k, v in backend.inference_config.items() if k != "attention_source"}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    manifest = {**config, "status": "running", "completed_samples": 0}
    write_json(output / "run.json", manifest)
    try:
        for index, sample in enumerate(samples, start=1):
            with Image.open(sample["image"]) as source:
                image = source.convert("RGB")
            pixels, base, _, geometry = prepare_views(image, backend.processor, backend.model.config,
                                                      14, global_reference="base")
            if (geometry.tile_size, geometry.patch_size, geometry.token_side) != (384, 14, 27):
                raise ValueError("Unexpected image/token geometry")
            if sample.get("saved_geometry") and asdict(geometry) != sample["saved_geometry"]:
                raise ValueError("AnyRes geometry differs from the saved run; verify checkpoint/preprocessing")
            pixels = pixels.to(device=tower.device, dtype=tower.dtype)
            scores, shapes = capture_stage_scores(tower, pixels, geometry, score_fn=score_quadrants)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sample["id"])[:80]
            folder = output / f"{index:06d}_{safe_id}"
            save_sample(image, base, scores, shapes, geometry, sample, folder, color_scale, config)
            manifest["completed_samples"] = index
            write_json(output / "run.json", manifest)
            print(f"[{index}/{len(samples)}] {sample['id']}: four layers, two metrics, "
                  f"{geometry.tile_count} regions; {folder}", flush=True)
        manifest["status"] = "complete"
        write_json(output / "run.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", manifest)
        raise
    print(f"Saved Base-quadrant visualizations and configs to {output}", flush=True)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results-dir", help="Reuse image paths/geometry, not cached scalar scores")
    source.add_argument("--input-json", help="Fresh records; requires --data-root")
    parser.add_argument("--model-path")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", help="New/empty directory")
    parser.add_argument("--limit", type=int, help="Optional sample count; default all samples")
    parser.add_argument("--color-scale", choices=("raw", "minmax"), default="raw",
                        help="raw: fixed metric ranges; minmax: joint Base/AnyRes display only; labels always RAW sums")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = None if args.input_json else script_path(args.results_dir or RESULTS_DIR)
    data = script_path(args.data_root or DATA_ROOT) if args.data_root or DATA_ROOT else None
    if args.output_dir or OUTPUT_DIR:
        output = script_path(args.output_dir or OUTPUT_DIR)
    elif results:
        output = results.with_name(results.name + "_base_quadrants")
    else:
        parser.error("--input-json requires --output-dir")
    try:
        run(args.model_path or MODEL_PATH or None, output, results_dir=results,
            input_json=script_path(args.input_json) if args.input_json else None,
            data_root=data, limit=args.limit, color_scale=args.color_scale)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
