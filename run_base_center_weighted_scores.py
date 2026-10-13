"""Center-weighted Base reference for SigLIP token visualization.

L2-unitize tokens. Average the central 0.50-by-0.50 rectangle and the
0.75-by-0.75 rectangle excluding the inner rectangle separately, then use
0.80*inner_mean + 0.20*ring_mean as one Base reference. Outside 0.75 is
excluded from reference construction. Positive cosine is the primary score;
negative cosine and L2 against the same reference are diagnostics.
"""
from __future__ import annotations

# Existing LLaVA pruning project. Leave empty for automatic discovery:
# placing this file on Desktop finds Desktop/llava_pruning; placing it inside
# that project uses the project itself. Relative data paths resolve there.
PROJECT_DIR = r""
RESULTS_DIR = r"outputs/feature_scores_02"
MODEL_PATH = r""
OUTPUT_DIR = r""  # default: sibling <results>_center_weighted
DATA_ROOT = r""

INNER_FRACTION = 0.50
OUTER_FRACTION = 0.75
INNER_WEIGHT = 0.80
RING_WEIGHT = 0.20
EPS = 1e-12
METRICS = ("cosine", "neg_cos", "l2")
SCRIPT_VERSION = "1.1-center-weighted-base-integrated"

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

def _field(geometry, key):
    return geometry[key] if isinstance(geometry, dict) else getattr(geometry, key)

def _geometry_values(geometry):
    grid = tuple(int(v) for v in _field(geometry, "grid"))
    tile_size = int(_field(geometry, "tile_size"))
    patch_size = int(_field(geometry, "patch_size"))
    resized = tuple(int(v) for v in _field(geometry, "resized_size"))
    paste = tuple(int(v) for v in _field(geometry, "paste_xy"))
    if len(grid) != 2 or min(grid) < 1:
        raise ValueError("Geometry grid must be positive (columns, rows)")
    if (tile_size, patch_size) != (384, 14):
        raise ValueError("This scorer requires the current 384/14 SigLIP geometry")
    if (len(resized) != 2 or min(resized) < 1 or len(paste) != 2
            or min(paste) < 0
            or paste[0] + resized[0] > grid[0] * tile_size
            or paste[1] + resized[1] > grid[1] * tile_size):
        raise ValueError("Image content does not fit the AnyRes canvas")
    return grid, tile_size, patch_size, resized, paste

def center_masks(geometry, inner_fraction=0.5, outer_fraction=0.75):
    """Return flattened center / ring / outer masks using pixel centers.

    A token is selected iff its actual center is in the centered rectangle.
    Patch center is (index + 0.5) * 14, divided by 384. Boundaries use
    [left, right), [top, bottom). Under defaults the 27x27 grid has
    196 inner tokens, 245 ring tokens, and 441 tokens inside the outer rectangle.
    """
    _, tile_size, patch, _, _ = _geometry_values(geometry)
    if (not np.isfinite(inner_fraction) or not np.isfinite(outer_fraction)
            or not 0 < inner_fraction < outer_fraction <= 1):
        raise ValueError("Need 0 < inner_fraction < outer_fraction <= 1")
    side = tile_size // patch
    y, x = np.mgrid[:side, :side]
    x = (x + 0.5) * patch / tile_size
    y = (y + 0.5) * patch / tile_size

    def rectangle(fraction):
        lower = (1.0 - fraction) / 2.0
        upper = (1.0 + fraction) / 2.0
        return ((x >= lower) & (x < upper)
                & (y >= lower) & (y < upper)).reshape(-1)

    inner = rectangle(inner_fraction)
    outer = rectangle(outer_fraction)
    ring = outer & ~inner
    if not np.any(inner) or not np.any(ring):
        raise ValueError("Center and ring must both contain tokens")
    return inner, ring, outer

def token_layout(geometry):
    """Return Base-to-crop region IDs and complete-content AnyRes masks.

    Base represents the complete original image resized to 384x384. Its
    normalized pixel centers are mapped into the padded AnyRes canvas using
    resized_size and paste_xy. All Base patches are spatially eligible.
    AnyRes patches touching padding are excluded, even if only partly padded.
    Tile and token order are both row-major. IDs are zero-based.
    """
    grid, tile_size, patch, resized, paste = _geometry_values(geometry)
    side = tile_size // patch
    yy, xx = np.mgrid[:side, :side]
    base_x = paste[0] + (xx + 0.5) * patch / tile_size * resized[0]
    base_y = paste[1] + (yy + 0.5) * patch / tile_size * resized[1]
    base_cols = np.floor(base_x / tile_size).astype(np.int32)
    base_rows = np.floor(base_y / tile_size).astype(np.int32)
    base_region = (base_rows * grid[0] + base_cols).reshape(-1)
    count = grid[0] * grid[1]
    masks = []
    for index in range(count):
        row, col = divmod(index, grid[0])
        x0 = xx * patch + col * tile_size
        y0 = yy * patch + row * tile_size
        valid = ((x0 >= paste[0]) & (y0 >= paste[1])
                 & (x0 + patch <= paste[0] + resized[0])
                 & (y0 + patch <= paste[1] + resized[1]))
        masks.append(valid.reshape(-1))
    return {
        "base_region": base_region,
        "base_valid": np.ones(side * side, dtype=bool),
        "tile_valid": np.stack(masks),
        "tile_region": np.broadcast_to(
            np.arange(count, dtype=np.int32)[:, None],
            (count, side * side)).copy(),
    }

def unit_token_features(features, eps=EPS):
    """L2-unitize channel-last floating features; tiny/zero vectors stay zero.

    Returns unit vectors, original norms, and a direction-validity mask.
    Cosine for direction-invalid tokens is undefined and later stored as NaN.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    features = np.asarray(features)
    if (features.ndim < 2 or features.shape[-1] < 1
            or not np.issubdtype(features.dtype, np.floating)
            or not np.isfinite(features).all()):
        raise ValueError("Features must be finite floating-point vectors")
    values = features.astype(np.float64, copy=False)
    norms = np.linalg.norm(values, axis=-1)
    if not np.isfinite(norms).all():
        raise ValueError("Feature norms are not finite")
    valid = norms > eps
    unit = np.zeros_like(values)
    np.divide(values, np.maximum(norms, eps)[..., None],
              out=unit, where=valid[..., None])
    return unit, norms, valid

def score_center_weighted(features, geometry, inner_fraction=0.5,
                          outer_fraction=0.75, inner_weight=0.8,
                          ring_weight=0.2, eps=EPS):
    """One layer [Base+tiles,729,C] -> reference and score arrays.

    Means are computed separately before applying the 0.8 / 0.2 weights;
    group size must not change those total weights. Cosine normalizes its
    reference internally. L2 uses the unnormalized weighted reference.
    Tiny/zero token cosine is NaN; its L2 is still mathematically defined.
    A zero/tiny weighted reference raises ValueError rather than making up a
    cosine direction. Invalid AnyRes padding scores are all NaN.

    All return values are NumPy arrays, including audit scalars, so callers
    may stack them across encoder layers and save them without object arrays.
    Input features are never modified.
    """
    grid, tile_size, patch, _, _ = _geometry_values(geometry)
    side = tile_size // patch
    features = np.asarray(features)
    if (features.ndim != 3
            or features.shape[:2] != (1 + grid[0] * grid[1], side * side)
            or features.shape[-1] < 1):
        raise ValueError("Expected features [Base + tiles, 729, channels]")
    if (not np.isfinite(inner_weight) or not np.isfinite(ring_weight)
            or min(inner_weight, ring_weight) < 0
            or not np.isclose(inner_weight + ring_weight, 1.0,
                              rtol=0, atol=1e-12)):
        raise ValueError("Weights must be nonnegative and sum to one")
    inner, ring, outer = center_masks(
        geometry, inner_fraction, outer_fraction)
    layout = token_layout(geometry)
    unit, original_norms, direction_valid = unit_token_features(features, eps)
    inner_mean = unit[0, inner].mean(axis=0)
    ring_mean = unit[0, ring].mean(axis=0)
    reference = inner_weight * inner_mean + ring_weight * ring_mean
    reference_norm = np.linalg.norm(reference)
    if not np.isfinite(reference_norm) or reference_norm <= eps:
        raise ValueError(
            "Center-weighted reference has zero/tiny norm; cosine is undefined")
    reference_unit = reference / reference_norm
    cosine = np.einsum("vnc,c->vn", unit, reference_unit)
    cosine = np.clip(cosine, -1.0, 1.0)
    cosine[~direction_valid] = np.nan
    l2 = np.linalg.norm(unit - reference, axis=-1)
    cosine[1:][~layout["tile_valid"]] = np.nan
    l2[1:][~layout["tile_valid"]] = np.nan
    result = {
        "base_cosine": cosine[0].astype(np.float32),
        "base_neg_cos": (-cosine[0]).astype(np.float32),
        "base_l2": l2[0].astype(np.float32),
        "tile_cosine": cosine[1:].astype(np.float32),
        "tile_neg_cos": (-cosine[1:]).astype(np.float32),
        "tile_l2": l2[1:].astype(np.float32),
        "base_foreground_score": ((cosine[0] + 1.0) / 2.0).astype(np.float32),
        "tile_foreground_score": ((cosine[1:] + 1.0) / 2.0).astype(np.float32),
        "base_center_mask": inner,
        "base_ring_mask": ring,
        "base_outer_mask": outer,
        "tile_valid": layout["tile_valid"],
        "base_direction_valid": direction_valid[0],
        "tile_direction_valid": direction_valid[1:],
        "base_feature_norm": original_norms[0].astype(np.float32),
        "tile_feature_norm": original_norms[1:].astype(np.float32),
        "reference": reference.astype(np.float32),
        "reference_unit": reference_unit.astype(np.float32),
        "reference_norm": np.asarray(reference_norm, dtype=np.float32),
        "inner_mean": inner_mean.astype(np.float32),
        "ring_mean": ring_mean.astype(np.float32),
        "inner_count": np.asarray(inner.sum(), dtype=np.int32),
        "ring_count": np.asarray(ring.sum(), dtype=np.int32),
        "outer_count": np.asarray(outer.sum(), dtype=np.int32),
        "inner_fraction": np.asarray(inner_fraction, dtype=np.float32),
        "outer_fraction": np.asarray(outer_fraction, dtype=np.float32),
        "inner_weight": np.asarray(inner_weight, dtype=np.float32),
        "ring_weight": np.asarray(ring_weight, dtype=np.float32),
        "zero_token_counts": np.sum(~direction_valid, axis=1).astype(np.int32),
    }
    return result


def score_with_config(features, geometry):
    return score_center_weighted(
        features, geometry, inner_fraction=INNER_FRACTION,
        outer_fraction=OUTER_FRACTION, inner_weight=INNER_WEIGHT,
        ring_weight=RING_WEIGHT, eps=EPS)


def _bootstrap_project():
    here = Path(__file__).resolve().parent
    requested = PROJECT_DIR or os.environ.get("LLAVA_PRUNING_PROJECT_DIR")
    candidates = ([Path(requested).expanduser()] if requested else
                  [here, here / "llava_pruning", Path.cwd()])
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = here / candidate
        root = candidate.resolve()
        if ((root / "llava_pruning" / "score_visualization.py").is_file()
                and (root / "visualize_patch_means.py").is_file()):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise FileNotFoundError(
        "Cannot find the existing llava_pruning project. Put this script in its "
        "root or set PROJECT_DIR / LLAVA_PRUNING_PROJECT_DIR.")


PROJECT_ROOT = _bootstrap_project()
# Reuse the existing read-only hooks; no changes to encoder dtype/attention.
from feature_residual_capture import LAYERS, STAGES, capture_stage_scores
from visualize_patch_means import (
    read_json, resolve_image, sample_folders, tile_rectangles, validate_geometry,
)
from llava_pruning.score_visualization import (
    COLORMAP, MISSING_COLOR, OVERLAY_ALPHA, ScoreGeometry, load_score_samples,
    prepare_views, stitch_score_map,
)

REFERENCE_DEFINITION = {
    "formula": (f"r = {INNER_WEIGHT:g} * mean(unit Base tokens in center {INNER_FRACTION:g}) + "
                f"{RING_WEIGHT:g} * mean(unit Base tokens in center {OUTER_FRACTION:g} minus center {INNER_FRACTION:g})"),
    "inner_fraction": INNER_FRACTION, "outer_fraction": OUTER_FRACTION,
    "inner_weight": INNER_WEIGHT, "ring_weight": RING_WEIGHT,
    "fractions": "width AND height ratios, not area fractions; actual patch centers decide membership",
    "group_weights": "mean each group separately, then weight; independent of group token count",
    "reference_source": "Base only; all four encoder stages independently",
    "outside_outer_region": "excluded from reference only; still scored and displayed",
    "primary_score": "cos(unit token, r), high means agreement with center reference",
    "diagnostics": ["-cos(unit token, r)", "||unit token - r||_2"],
    "reference_renormalized_for_l2": False,
    "feature_location": "1-based SigLIP block output before final post_layernorm",
    "token_normalization": "channel-wise L2; zero/tiny tokens have no cosine direction",
    "epsilon": EPS,
    "padding": "AnyRes patch must be fully inside actual image content",
    "zero_reference": "fail explicitly; do not invent a cosine direction",
    "interpretation": "reference agreement, NOT a foreground or anomaly probability",
    "llm_forward": False, "projector_forward": False, "training": False, "pruning": False,
    "model_forward_modified": False,
    "foreground_score_fields": "(cosine + 1) / 2 compatibility fields; NOT foreground probabilities",
}


def script_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


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


def metric_limits(base, tiles, metric, color_scale):
    if color_scale not in ("sample", "fixed", "raw"):
        raise ValueError("color-scale must be sample, fixed or raw")
    if metric not in METRICS:
        raise ValueError("Unknown metric")
    lower, upper = ((0., 2.) if metric == "l2" else (-1., 1.))
    if color_scale in ("fixed", "raw"):
        return lower, upper
    values = np.concatenate((base[np.isfinite(base)], tiles[np.isfinite(tiles)]))
    if not values.size:
        return lower, upper
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        low, high = max(lower, low - .01), min(upper, high + .01)
    return low, high


def region_statistics(scores, stage_index, view, metric, geometry):
    values = np.asarray(scores[f"{view}_{metric}"][stage_index])
    layout = token_layout(geometry)
    groups = layout["base_region"] if view == "base" else layout["tile_region"]
    eligible = layout["base_valid"] if view == "base" else layout["tile_valid"]
    means = np.full(geometry.tile_count, np.nan)
    sums = means.copy()
    counts = np.zeros(geometry.tile_count, dtype=np.int32)
    geometric_counts = counts.copy()
    for index in range(geometry.tile_count):
        selected = eligible & (groups == index)
        geometric_counts[index] = int(selected.sum())
        finite = selected & np.isfinite(values)
        counts[index] = int(finite.sum())
        if counts[index]:
            sums[index] = values[finite].sum(dtype=np.float64)
            means[index] = sums[index] / counts[index]
    return means, sums, counts, geometric_counts


def draw_center_figures(image, base, scores, geometry, folder, caption, color_scale="sample"):
    from matplotlib import colormaps, rc_context, patheffects
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    folder = Path(folder)
    filenames = [f"{stage}_center_weighted.png" for stage in STAGES]
    if any((folder / name).exists() for name in filenames):
        raise FileExistsError("Center-weighted figures already exist")
    rectangles = tile_rectangles(asdict(geometry))
    cmap = colormaps[COLORMAP].with_extremes(bad=MISSING_COLOR)
    names = {"cosine": "Positive cosine (primary)", "neg_cos": "Negative cosine", "l2": "L2 to weighted reference"}
    ranges = {}

    def regions(ax, means=None):
        for index, rect in enumerate(rectangles):
            if rect is None:
                continue
            x, y, width, height = rect
            ax.add_patch(Rectangle((x, y), width, height, fill=False, linewidth=.6,
                                  edgecolor="white", alpha=.65))
            if means is None:
                ax.text(x + .009, y + .02, str(index + 1), color="white", fontsize=8,
                        va="top", path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
            else:
                value = means[index]
                ax.text(x + width / 2, y + height / 2,
                        f"{value:.2f}" if np.isfinite(value) else "--",
                        ha="center", va="center", color="white", fontsize=8,
                        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

    def roi(ax):
        for fraction, color, style in ((OUTER_FRACTION, "#00d9ff", "--"),
                                       (INNER_FRACTION, "#ffff00", "-")):
            start = (1 - fraction) / 2
            ax.add_patch(Rectangle((start, start), fraction, fraction, fill=False,
                                  linewidth=1.5, edgecolor=color, linestyle=style))

    def panel(ax, source, heatmap=None, norm=None, means=None, reference=False):
        ax.imshow(source, extent=(0, 1, 1, 0), interpolation="nearest")
        if heatmap is not None:
            rgba = cmap(norm(np.ma.masked_invalid(heatmap)))
            rgba[..., 3] = np.where(np.isfinite(heatmap), OVERLAY_ALPHA, 1.0)
            ax.imshow(rgba, extent=(0, 1, 1, 0), interpolation="nearest")
        regions(ax, means)
        if reference:
            roi(ax)
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
        ax.set_aspect(source.height / source.width)

    for si, (layer, stage) in enumerate(zip(LAYERS, STAGES)):
        with rc_context({"font.size": 9, "axes.titlesize": 10}):
            fig = Figure(figsize=(18, 9), dpi=140)
            FigureCanvasAgg(fig)
            grid = fig.add_gridspec(2, 7, width_ratios=(1, 1, .04, 1, .04, 1, .04),
                                   left=.018, right=.94, top=.86, bottom=.14,
                                   wspace=.22, hspace=.18)
            base_ax, original_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
            panel(base_ax, base, reference=True)
            base_ax.set_title(f"Base reference: center {INNER_WEIGHT:.1f}, ring {RING_WEIGHT:.1f}")
            panel(original_ax, image, reference=True)
            original_ax.set_title("Original + actual AnyRes regions")
            stage_ranges = {}
            for mi, metric in enumerate(METRICS):
                b, t = scores[f"base_{metric}"][si], scores[f"tile_{metric}"][si]
                limits = metric_limits(b, t, metric, color_scale)
                norm = Normalize(*limits)
                col = 1 + 2 * mi
                ax_b, ax_t = fig.add_subplot(grid[0, col]), fig.add_subplot(grid[1, col])
                means_b = region_statistics(scores, si, "base", metric, geometry)[0]
                means_t = region_statistics(scores, si, "tile", metric, geometry)[0]
                panel(ax_b, base, base_score_map(b, geometry), norm, means_b)
                panel(ax_t, image, stitch_score_map(t, geometry), norm, means_t)
                ax_b.set_title(f"Base: {names[metric]}")
                ax_t.set_title(f"AnyRes: {names[metric]}")
                color_ax = fig.add_subplot(grid[:, col + 1])
                bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax)
                bar.ax.tick_params(labelsize=8)
                stage_ranges[metric] = list(limits)
            reference_norm = float(scores["reference_norm"][si])
            inner_count = int(scores["inner_count"][si])
            ring_count = int(scores["ring_count"][si])
            fig.suptitle(f"{caption}\nSigLIP layer {layer} | Center-weighted Base reference | "
                         f"inner {inner_count}, ring {ring_count}, reference norm {reference_norm:.4f}",
                         y=.982, fontsize=12)
            fig.text(.5, .083, f"Yellow: center {INNER_FRACTION:.2f} x {INNER_FRACTION:.2f}; "
                     f"cyan: outer {OUTER_FRACTION:.2f} x {OUTER_FRACTION:.2f}. "
                     f"Reference = {INNER_WEIGHT:.2f} * inner mean + {RING_WEIGHT:.2f} * ring mean.",
                     ha="center", fontsize=9)
            fig.text(.5, .053, "All tokens L2-normalized first. Numbers = finite token-score MEANS per actual AnyRes region; "
                     "each metric shares one colorbar between Base and AnyRes.",
                     ha="center", fontsize=9)
            fig.text(.5, .024, "Gray = padding / unencoded pixels / undefined cosine. "
                     "Cosine is reference agreement, not defect probability. L2 uses the weighted mean without renormalizing it.",
                     ha="center", fontsize=9)
            try:
                fig.savefig(folder / filenames[si], dpi=140)
            finally:
                fig.clear()
            ranges[stage] = stage_ranges
    return {"figures": filenames, "colorbar_limits": ranges, "color_scale": color_scale,
            "region_annotation": "mean of finite token scores, not raw sum",
            "colormap": COLORMAP, "overlay_alpha": OVERLAY_ALPHA}


def save_center_sample(image, base, scores, shapes, geometry, sample, folder, color_scale):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "center_scores.npz", **scores,
                        stages=np.asarray(STAGES), layers=np.asarray(LAYERS))
    gt = {0: "Normal", 1: "Abnormal"}.get(sample.get("gt"), "Unknown")
    display = draw_center_figures(image, base, scores, geometry, folder,
                                 f"ID: {sample['id']} | GT: {gt} | No LLM prediction", color_scale)
    rows = []
    for si, (layer, stage) in enumerate(zip(LAYERS, STAGES)):
        for view in ("base", "tile"):
            for metric in METRICS:
                means, sums, counts, geometric = region_statistics(scores, si, view, metric, geometry)
                for ti in range(geometry.tile_count):
                    def finite_or_blank(value):
                        return float(value) if np.isfinite(value) else ""
                    rows.append({"stage": stage, "layer": layer, "view": view, "metric": metric,
                                 "crop_id": ti + 1, "mean": finite_or_blank(means[ti]),
                                 "sum": finite_or_blank(sums[ti]), "finite_count": int(counts[ti]),
                                 "geometric_count": int(geometric[ti])})
    with (folder / "region_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"id": sample["id"], "image": str(sample["image"]), "gt": sample.get("gt"),
                "geometry": asdict(geometry), "feature_shapes": shapes,
                "script_version": SCRIPT_VERSION, "definition": REFERENCE_DEFINITION,
                "score_axes": {"base": ["stage", "token_row_major"],
                               "tile": ["stage", "crop_row_major", "token_row_major"],
                               "reference": ["stage", "channel"]},
                "source_metadata": sample.get("source_metadata"), **display}
    write_json(folder / "metadata.json", metadata)


def run_center(model_path, output_dir, results_dir=None, input_json=None, data_root=None,
               limit=None, color_scale="sample"):
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    if color_scale not in ("sample", "fixed", "raw"):
        raise ValueError("Unsupported color scale")
    samples, source_run, source_root = prepare_inputs(results_dir, input_json, data_root, limit)
    output = check_output(output_dir, source_root)
    checkpoint_raw = model_path or source_run.get("model_path")
    if not checkpoint_raw:
        raise ValueError("Set MODEL_PATH or --model-path: original features must be recomputed")
    checkpoint = script_path(str(checkpoint_raw))
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Local checkpoint not found: {checkpoint}")
    # Validate requested rectangles before allocating any model weights.
    center_masks(ScoreGeometry.create((384, 384), (1, 1), 384, 14),
                 INNER_FRACTION, OUTER_FRACTION)
    if (not np.isfinite([INNER_WEIGHT, RING_WEIGHT]).all()
            or min(INNER_WEIGHT, RING_WEIGHT) < 0
            or not np.isclose(INNER_WEIGHT + RING_WEIGHT, 1, rtol=0, atol=1e-12)):
        raise ValueError("Center/ring weights must be nonnegative and sum to one")

    import torch
    from llava_pruning.backend import LlavaBackend
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one visible CUDA GPU")
    print(f"Loading the existing checkpoint for {len(samples)} images; only SigLIP is executed.", flush=True)
    backend = LlavaBackend(checkpoint, roi_mode="anyres_max_9")
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    tower = backend.vision_tower
    if not isinstance(tower, SigLipVisionTower):
        raise ValueError("This probe requires the existing SigLIP tower")
    tower.eval()
    vision = tower.vision_tower.vision_model
    if len(vision.encoder.layers) != 26 or tuple(vision.embeddings.patch_embedding.kernel_size) != (14, 14):
        raise ValueError("Expected 26 active SigLIP blocks and 14x14 patch embedding")

    output.mkdir(parents=True, exist_ok=True)
    manifest = {"script_version": SCRIPT_VERSION, "project_root": str(PROJECT_ROOT),
                "model_path": str(checkpoint), "sample_count": len(samples), "completed_samples": 0,
                "results_dir": str(source_root) if source_root else None,
                "input_json": str(input_json) if input_json else None,
                "data_root": str(data_root) if data_root else None,
                "stages": list(STAGES), "definition": REFERENCE_DEFINITION,
                "color_scale": color_scale, "status": "running",
                "limit": limit, "layers": list(LAYERS), "metrics": list(METRICS),
                "output_dir": str(output), "global_reference": "full_base_center_weighted",
                "llm_weights_loaded": True, "llm_forward": False, "projector_forward": False,
                "vision_dtype": str(tower.dtype),
                "loader_config": {k: v for k, v in backend.inference_config.items() if k != "attention_source"}}
    write_json(output / "run.json", manifest)
    try:
        for index, sample in enumerate(samples, start=1):
            with Image.open(sample["image"]) as source:
                image = source.convert("RGB")
            pixels, base, tiles, geometry = prepare_views(
                image, backend.processor, backend.model.config, 14, global_reference="base")
            if (geometry.tile_size, geometry.token_side) != (384, 27):
                raise ValueError("Unexpected processed geometry")
            if sample.get("saved_geometry") and asdict(geometry) != sample["saved_geometry"]:
                raise ValueError("AnyRes geometry changed; verify checkpoint and preprocessing")
            pixels = pixels.to(device=tower.device, dtype=tower.dtype)
            scores, shapes = capture_stage_scores(tower, pixels, geometry, score_fn=score_with_config)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sample["id"])[:80]
            folder = output / f"{index:06d}_{safe_id}"
            save_center_sample(image, base, scores, shapes, geometry, sample, folder, color_scale)
            manifest["completed_samples"] = index
            write_json(output / "run.json", manifest)
            print(f"[{index}/{len(samples)}] {sample['id']}: saved four center-weighted layer figures", flush=True)
        manifest["status"] = "complete"
        write_json(output / "run.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "run.json", manifest)
        raise
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--results-dir", help="Reuse image paths and geometry from a previous score run")
    group.add_argument("--input-json", help="Fresh image records, requires --data-root")
    parser.add_argument("--model-path")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", help="New or empty folder")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--color-scale", choices=("sample", "fixed", "raw"), default="sample",
                        help="Same raw-score limits for Base/AnyRes per metric and layer; fixed/raw uses cosine [-1,1], L2 [0,2]; neither mode min-max rescales stored scores")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = None if args.input_json else script_path(args.results_dir or RESULTS_DIR)
    data = script_path(args.data_root or DATA_ROOT) if args.data_root or DATA_ROOT else None
    if args.output_dir or OUTPUT_DIR:
        output = script_path(args.output_dir or OUTPUT_DIR)
    elif results:
        output = results.with_name(results.name + "_center_weighted")
    else:
        parser.error("--input-json requires --output-dir")
    try:
        run_center(args.model_path or MODEL_PATH or None, output, results_dir=results,
                   input_json=script_path(args.input_json) if args.input_json else None,
                   data_root=data, limit=args.limit, color_scale=args.color_scale)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
