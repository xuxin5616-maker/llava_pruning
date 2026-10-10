"""Visualize local SigLIP feature residuals at layers 7/14/21/26.

Base and every AnyRes crop retain the native 27x27 feature grid. For each
3x3/5x5/7x7 window, average valid neighboring vectors excluding the center,
then compute ||center - neighbor_mean||_2. The optional cosine diagnostic
uses the same raw-vector mean. This is a custom local-change probe, not the
original VFlowOpt score and not an anomaly probability.

Reuse the existing LLaVA checkpoint and AnyRes preprocessing. Only the inner
SigLIP encoder is executed. No entropy, Softmax, projector, LLM generation,
training or pruning is performed. The existing loader still loads LLM weights
and preserves its attention implementation.
"""
from __future__ import annotations

# Paths relative to this script, or absolute paths. Set MODEL_PATH to your
# existing LLaVA-OneVision checkpoint. With RESULTS_DIR, run.json.model_path
# is also accepted. Cached scalar scores are NOT features; SigLIP is rerun.
RESULTS_DIR = r"outputs/feature_scores_02"
MODEL_PATH = r""
OUTPUT_DIR = r""  # default: sibling <results>_feature_residual
DATA_ROOT = r""   # optional relocation of original images
SCRIPT_VERSION = "3.1-integrated-feature-residual"

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from feature_residual_math import WINDOWS
from feature_residual_capture import LAYERS, STAGES, capture_stage_scores
from visualize_patch_means import (
    read_json, resolve_image, sample_folders, tile_rectangles, validate_geometry,
)
from llava_pruning.score_visualization import (
    COLORMAP, MISSING_COLOR, OVERLAY_ALPHA, ScoreGeometry, load_score_samples,
    prepare_views, stitch_score_map,
)

RESIDUAL_DEFINITION = {
    "l2": "R_i(w) = ||f_i - mean(f_j for valid j in N_w(i), j != i)||_2",
    "cosine": "1 - cos(f_i, mean(f_j for valid j in N_w(i), j != i))",
    "windows": list(WINDOWS),
    "feature_location": "1-based SigLIP block outputs, before final post_layernorm",
    "base": "native 27x27 grid; no 6x6 regrouping",
    "anyres": "27x27 per crop, stitched logical patch grid across crop boundaries",
    "features": "raw vectors; no L2 normalization before neighbor averaging",
    "padding": "only patches fully inside resized image content are centers/neighbors",
    "edges": "available neighbors only; exclude center; no zero/reflect padding",
    "crop_gaps": "logical adjacency bridges each crop's six unencoded bottom/right pixels",
    "crop_boundary_context": "crops are encoded independently; cross-crop feature differences can include crop context and position resets",
    "undefined": "no neighbors -> NaN; zero-norm cosine -> NaN; zero-vector L2 is valid",
    "accumulation": "float64 on CPU; scalar scores saved in float32",
    "interpretation": "local feature departure, NOT attention or defect probability",
    "entropy": False, "softmax": False, "original_vflowopt_method": False,
}


def script_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(__file__).resolve().parent / path).resolve()


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def base_score_map(scores, geometry):
    """Actual 378x378 patch support; the remaining six pixels stay gray."""
    side, patch = geometry.token_side, geometry.patch_size
    scores = np.asarray(scores)
    if scores.shape != (side * side,):
        raise ValueError("Base scores do not match the native feature grid")
    result = np.full((geometry.tile_size, geometry.tile_size), np.nan, dtype=np.float32)
    result[:side * patch, :side * patch] = scores.reshape(side, side).repeat(patch, 0).repeat(patch, 1)
    return result


def display_scores(base, tiles, color_scale):
    """One common range for all six maps in a stage; never normalize per crop."""
    if color_scale not in ("raw", "minmax"):
        raise ValueError("color-scale must be raw or minmax")
    base, tiles = np.asarray(base), np.asarray(tiles)
    if np.isinf(base).any() or np.isinf(tiles).any():
        raise ValueError("Infinite residual scores")
    finite = np.concatenate((base[np.isfinite(base)], tiles[np.isfinite(tiles)]))
    if not finite.size:
        return base.copy(), tiles.copy(), (0.0, 1.0), None
    low, high = float(finite.min()), float(finite.max())
    if color_scale == "raw":
        # Always show zero as zero; avoid a degenerate colorbar for constant maps.
        return base.copy(), tiles.copy(), (0.0, max(high, 1e-6)), [low, high]
    if high == low:
        b = np.where(np.isfinite(base), 0.5, np.nan)
        t = np.where(np.isfinite(tiles), 0.5, np.nan)
    else:
        b, t = (base - low) / (high - low), (tiles - low) / (high - low)
    return b.astype(np.float32), t.astype(np.float32), (0.0, 1.0), [low, high]


def draw_figures(image, base_image, scores, geometry, folder, caption, metric="l2", color_scale="raw"):
    """Four layer figures; each contains Base and AnyRes at all three windows."""
    from matplotlib import colormaps, rc_context, patheffects
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    folder = Path(folder)
    if metric not in ("l2", "cosine"):
        raise ValueError("metric must be l2 or cosine")
    filenames = [f"{stage}_residual_{metric}_{color_scale}.png" for stage in STAGES]
    if any((folder / name).exists() for name in filenames):
        raise FileExistsError("Residual figures already exist; use a new output directory")
    cmap = colormaps[COLORMAP].with_extremes(bad=MISSING_COLOR)
    rectangles = tile_rectangles(asdict(geometry))
    display_metadata, files = {}, []

    def boundaries(ax):
        for index, rect in enumerate(rectangles, start=1):
            if rect is None:
                continue
            x, y, w, h = rect
            ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=.7,
                                   linestyle="--", edgecolor="white", alpha=.8))
            ax.text(x + .007, y + .013, str(index), fontsize=9, color="white",
                    ha="left", va="top",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

    def panel(ax, source, heatmap=None, norm=None, crops=False):
        ax.imshow(source, extent=(0, 1, 1, 0), interpolation="nearest")
        if heatmap is not None:
            rgba = cmap(norm(np.ma.masked_invalid(heatmap)))
            rgba[..., 3] = np.where(np.isfinite(heatmap), OVERLAY_ALPHA, 1.0)
            ax.imshow(rgba, extent=(0, 1, 1, 0), interpolation="nearest")
        if crops:
            boundaries(ax)
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
        ax.set_aspect(source.height / source.width)

    for index, (layer, stage) in enumerate(zip(LAYERS, STAGES)):
        b, t, limits, raw_range = display_scores(
            scores["base_" + metric][index], scores["tile_" + metric][index], color_scale)
        norm = Normalize(*limits)
        with rc_context({"font.size": 10, "axes.titlesize": 11}):
            fig = Figure(figsize=(16, 8.8), dpi=150)
            FigureCanvasAgg(fig)
            axes = fig.subplots(2, 4, squeeze=False)
            fig.subplots_adjust(left=.018, right=.916, top=.86, bottom=.13, wspace=.055, hspace=.18)
            panel(axes[0, 0], base_image)
            axes[0, 0].set_title("Base reference (384 x 384)")
            panel(axes[1, 0], image, crops=True)
            axes[1, 0].set_title("Original + crop IDs")
            for wi, window in enumerate(WINDOWS):
                panel(axes[0, wi + 1], base_image, base_score_map(b[wi], geometry), norm)
                axes[0, wi + 1].set_title(f"Base: {window} x {window}")
                panel(axes[1, wi + 1], image, stitch_score_map(t[wi], geometry), norm, crops=True)
                axes[1, wi + 1].set_title(f"AnyRes: {window} x {window}")
            color_ax = fig.add_axes((.935, .18, .014, .63))
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax)
            bar.set_label(("L2 feature residual" if metric == "l2" else "1 - cosine similarity")
                          if color_scale == "raw" else "Joint layer min-max score")
            fig.suptitle(f"{caption}\nSigLIP layer {layer}: local feature residual ({metric}, {color_scale})",
                         y=.98, fontsize=13)
            range_text = "no defined values" if raw_range is None else (
                f"raw range [{raw_range[0]:.6g}, {raw_range[1]:.6g}]")
            fig.text(.5, .066, f"One shared color scale for Base, all crops, and all three windows; {range_text}.",
                     ha="center", fontsize=10)
            fig.text(.5, .037, "Neighbor vector mean excludes the center. Gray = padding / partial-padding patch / unencoded pixels / undefined score.",
                     ha="center", fontsize=9)
            fig.text(.5, .013, "Layers use separate scales. Colors across different layers/images are not absolute comparisons. Scores are NOT defect probabilities.",
                     ha="center", fontsize=9)
            filename = f"{stage}_residual_{metric}_{color_scale}.png"
            try:
                fig.savefig(folder / filename, dpi=150)
            finally:
                fig.clear()
            files.append(filename)
            display_metadata[stage] = {"raw_range": raw_range, "colorbar_limits": list(limits)}
    return {"figures": files, "display_ranges": display_metadata,
            "display_scope": "per sample, per stage, joint Base/AnyRes/all windows",
            "metric": metric, "color_scale": color_scale,
            "colormap": COLORMAP, "overlay_alpha": OVERLAY_ALPHA}


def summary_rows(scores):
    for si, (layer, stage) in enumerate(zip(LAYERS, STAGES)):
        for wi, window in enumerate(WINDOWS):
            for view, l2, cosine, counts in [
                ("base", scores["base_l2"][si, wi], scores["base_cosine"][si, wi],
                 scores["base_neighbor_counts"][si, wi]),
                *[(f"crop_{ti + 1}", scores["tile_l2"][si, wi, ti],
                   scores["tile_cosine"][si, wi, ti], scores["neighbor_counts"][si, wi, ti])
                  for ti in range(scores["tile_l2"].shape[2])],
            ]:
                def mean(values):
                    finite = values[np.isfinite(values)]
                    return float(finite.mean(dtype=np.float64)) if finite.size else ""
                yield {"stage": stage, "layer": layer, "window": window, "view": view,
                       "valid_l2_count": int(np.isfinite(l2).sum()),
                       "mean_l2": mean(l2),
                       "valid_cosine_count": int(np.isfinite(cosine).sum()),
                       "mean_cosine": mean(cosine), "mean_neighbor_count": mean(counts[counts > 0])}


def save_sample(image, base, scores, shapes, geometry, sample, folder, metric, color_scale):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    # Full features are not retained, only residuals and audit masks/counts.
    np.savez_compressed(folder / "residuals.npz", **scores, stages=np.asarray(STAGES),
                        layers=np.asarray(LAYERS), windows=np.asarray(WINDOWS))
    gt = {0: "Normal", 1: "Abnormal"}.get(sample.get("gt"), "Unknown")
    display = draw_figures(image, base, scores, geometry, folder,
                           f"ID: {sample['id']} | GT: {gt} | No LLM prediction", metric, color_scale)
    rows = list(summary_rows(scores))
    with (folder / "crop_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "id": sample["id"], "image": str(sample["image"]), "gt": sample.get("gt"),
        "geometry": asdict(geometry), "feature_shapes": shapes,
        "script_version": SCRIPT_VERSION, "definition": RESIDUAL_DEFINITION,
        "score_axes": {"base": ["stage", "window", "token_row_major"],
                       "tile": ["stage", "window", "crop_row_major", "token_row_major"],
                       "base_valid": ["stage", "token_row_major"],
                       "tile_valid": ["stage", "crop_row_major", "token_row_major"]},
        "source_metadata": sample.get("source_metadata"),
        **display,
    }
    write_json(folder / "metadata.json", metadata)


def prepare_inputs(results_dir=None, input_json=None, data_root=None, limit=None):
    """Only image paths and geometry are reused; cached similarity scores ignored."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if bool(results_dir) == bool(input_json):
        raise ValueError("Provide exactly one of results-dir or input-json")
    if input_json:
        if not data_root:
            raise ValueError("--input-json requires --data-root")
        samples = load_score_samples(input_json, data_root, limit)
        source_run, source_root = {}, None
    else:
        source_root = Path(results_dir).expanduser().resolve()
        run_path = source_root / "run.json"
        if not run_path.is_file() and (source_root / "metadata.json").is_file():
            run_path = source_root.parent / "run.json"
        source_run = read_json(run_path) if run_path.is_file() else {}
        folders = sample_folders(source_root)
        if limit:
            folders = folders[:limit]
        samples = []
        for folder in folders:
            metadata = read_json(folder / "metadata.json")
            geometry = validate_geometry(metadata)
            if (geometry["tile_size"], geometry["patch_size"]) != (384, 14):
                raise ValueError("Saved geometry is not the current 384/14 SigLIP setup")
            samples.append({"id": str(metadata.get("id", folder.name)),
                            "image": resolve_image(metadata, source_run, data_root),
                            "gt": metadata.get("gt"), "saved_geometry": geometry,
                            "source_metadata": str(folder / "metadata.json")})
    if not samples:
        raise ValueError("No input images found")
    for sample in samples:
        if not Path(sample["image"]).is_file():
            raise FileNotFoundError(f"Image not found: {sample['image']}")
        # Catch invalid images/moved data before allocating GPU model weights.
        with Image.open(sample["image"]) as source:
            if sample.get("saved_geometry") and source.size != sample["saved_geometry"]["original_size"]:
                raise ValueError(f"Original image size changed: {sample['image']}")
            source.verify()
    return samples, source_run, source_root


def check_output(output, source_root):
    output = Path(output).resolve()
    source_root = Path(source_root).resolve() if source_root else None
    if source_root and (output == source_root or output in source_root.parents or source_root in output.parents):
        raise ValueError("Output must be separate from the source result directory (use a sibling)")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    return output


def run(model_path, output_dir, results_dir=None, input_json=None, data_root=None,
        limit=None, metric="l2", color_scale="raw"):
    # Plotting/input checks precede loading a large checkpoint.
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    samples, source_run, source_root = prepare_inputs(results_dir, input_json, data_root, limit)
    output = check_output(output_dir, source_root)
    model_path = model_path or source_run.get("model_path")
    if not model_path:
        raise ValueError("Set MODEL_PATH or pass --model-path; raw features must be recomputed")
    checkpoint = script_path(str(model_path))
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Local checkpoint not found: {checkpoint}. Pass --model-path for this machine.")
    if metric not in ("l2", "cosine") or color_scale not in ("raw", "minmax"):
        raise ValueError("Unsupported metric/color scale")

    import torch
    from llava_pruning.backend import LlavaBackend
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one visible CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0 python runVFlowOpt.py ...")
    print(f"Loading existing checkpoint for {len(samples)} images; only SigLIP is executed.", flush=True)
    backend = LlavaBackend(checkpoint, roi_mode="anyres_max_9")
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    tower = backend.vision_tower
    if not isinstance(tower, SigLipVisionTower):
        raise ValueError("This probe requires the existing SigLIP tower")
    tower.eval()
    vision = tower.vision_tower.vision_model
    if len(vision.encoder.layers) != 26:
        raise ValueError("Expected 26 active SigLIP blocks")
    patch_kernel = tuple(vision.embeddings.patch_embedding.kernel_size)
    if patch_kernel != (14, 14):
        raise ValueError(f"Expected 14x14 patch embedding, found {patch_kernel}")

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script_version": SCRIPT_VERSION, "model_path": str(checkpoint),
        "results_dir": str(source_root) if source_root else None,
        "input_json": str(input_json) if input_json else None,
        "data_root": str(data_root) if data_root else None,
        "sample_count": len(samples), "completed_samples": 0, "status": "running",
        "stages": list(STAGES), "definition": RESIDUAL_DEFINITION,
        "metric": metric, "color_scale": color_scale,
        "vision_dtype": str(tower.dtype),
        "llm_weights_loaded": True, "llm_forward": False, "projector_forward": False,
        "training": False, "pruning": False,
        "loader_config": {k: v for k, v in backend.inference_config.items() if k != "attention_source"},
    }
    write_json(output / "run.json", manifest)
    try:
        for index, sample in enumerate(samples, start=1):
            with Image.open(sample["image"]) as source:
                image = source.convert("RGB")
            pixels, base, tiles, geometry = prepare_views(image, backend.processor, backend.model.config, 14)
            if (geometry.tile_size, geometry.patch_size, geometry.token_side) != (384, 14, 27):
                raise ValueError("Unexpected image/token geometry")
            if sample.get("saved_geometry") and asdict(geometry) != sample["saved_geometry"]:
                raise ValueError("AnyRes geometry differs from the saved run; verify checkpoint and preprocessing configuration")
            pixels = pixels.to(device=tower.device, dtype=tower.dtype)
            scores, shapes = capture_stage_scores(tower, pixels, geometry)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sample["id"])[:80]
            folder = output / f"{index:06d}_{safe_id}"
            save_sample(image, base, scores, shapes, geometry, sample, folder, metric, color_scale)
            manifest["completed_samples"] = index
            write_json(output / "run.json", manifest)
            print(f"[{index}/{len(samples)}] {sample['id']}: four layers x three windows; {folder}", flush=True)
        manifest["status"] = "complete"
        write_json(output / "run.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "run.json", manifest)
        raise
    print(f"Saved feature residual visualizations to {output}", flush=True)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results-dir", help="Reuse saved image metadata, not cached scores")
    source.add_argument("--input-json", help="Fresh image records; requires --data-root")
    parser.add_argument("--model-path", help="Same local LLaVA-OneVision checkpoint as classification")
    parser.add_argument("--data-root", help="Dataset root (or relocation root)")
    parser.add_argument("--output-dir", help="New/empty output directory")
    parser.add_argument("--limit", type=int, help="Process first N samples")
    parser.add_argument("--metric", choices=("l2", "cosine"), default="l2")
    parser.add_argument("--color-scale", choices=("raw", "minmax"), default="raw",
                        help="Raw L2 by default; minmax is joint within each layer, never per crop")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = None if args.input_json else script_path(args.results_dir or RESULTS_DIR)
    data = script_path(args.data_root or DATA_ROOT) if args.data_root or DATA_ROOT else None
    model = args.model_path or MODEL_PATH or None
    if args.output_dir or OUTPUT_DIR:
        output = script_path(args.output_dir or OUTPUT_DIR)
    elif results:
        output = results.with_name(results.name + "_feature_residual")
    else:
        parser.error("--input-json requires --output-dir")
    try:
        run(model_path=model, output_dir=output, results_dir=results,
            input_json=script_path(args.input_json) if args.input_json else None,
            data_root=data, limit=args.limit, metric=args.metric, color_scale=args.color_scale)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
