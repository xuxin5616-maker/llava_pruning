"""Base 6x6 / AnyRes 2x2 entropy blocks: plain or local-deviation mode, CPU only.

Based on the user-supplied Softmax-corrected script. Base is resized to 384x384
and split into 36 analysis blocks of 64x64 pixels. This does not change the
SigLIP model's patch embedding or number of model tokens. Each AnyRes crop
has four 192x192 analysis blocks covering all 384x384 pixels. A 3x3 crop layout
therefore gives a 6x6 stitched analysis grid; other crop layouts are preserved.

--entropy-mode plain (default) displays ordinary histogram entropy. Softmax
is over all 36 Base blocks and independently over each crop's four blocks,
including padding. Only entropy_overview.png is rendered in this mode.
--entropy-mode local retains the neighborhood-deviation figures described below.

AnyRes deviation is abs(H_i - median(neighbor entropies)) in raw nats.
Windows are 3x3, 5x5 and 7x7 in the stitched logical patch grid, exclude the
center and cross crop boundaries. Only full-content patches participate:
fully/partly padded patches are retained in raw entropy but excluded from
deviation, neighbor medians and deviation Softmax. No-neighbor scores are NaN.

The supplied script's log2 convention and independent per-view Softmax are
preserved: Base Softmax covers 36 blocks; deviation Softmax covers only defined
scores within each crop. Raw comparisons are also always plotted. Base and
AnyRes use different colorbars; all three AnyRes windows share one scale.
No model, Torch, training, LLM inference or pruning is performed. This custom
local-deviation extension is not VFlowOpt's original importance score.
"""

from __future__ import annotations

RESULTS_DIR = r"outputs/feature_scores_02"
OUTPUT_DIR = r""  # default: sibling <results>_vflowopt_<plain|local>_<color-scale>
DATA_ROOT = r""
SCRIPT_VERSION = "2.1-plain-local-coarse-grids"
SOURCE_SCRIPT_SHA256 = "9ef3d3ce499acf844b4cdcff1c5c0c1d40b771279005188c55f8dfba2907d114"
BASE_GRID_SIDE = 6
ANYRES_GRID_SIDE = 2
WINDOWS = (3, 5, 7)
PAPER_URL = "https://arxiv.org/html/2508.05211v2#S3.SS1"
AUTHOR_CODE_URL = "https://github.com/sihany077/VFlowOpt/blob/main/src/LLaVA-OneVision/llava/model/multimodal_encoder/siglip_encoder.py"

import argparse
import csv
import json
import math
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from visualize_patch_means import read_json, resolve_image, sample_folders, tile_rectangles, validate_geometry
from llava_pruning.score_visualization import (
    COLORMAP, MISSING_COLOR, OVERLAY_ALPHA, ScoreGeometry, stitch_score_map,
)

ENTROPY_DEFINITION = {
    "formula": "H_i = -sum(p_k * ln(p_k)) over nonzero bins",
    "grayscale": "floor((R + G + B) / 3), arithmetic RGB mean",
    "bins": 256, "unit": "nat",
    "pixel_input": "uint8 RGB after Base/AnyRes resize and padding, before normalization",
    "base": {"grid": [6, 6], "patch_pixels": [64, 64], "count": 36,
             "analysis_grid_only": True, "model_patch_embedding_changed": False},
    "anyres": {"grid_per_crop": [2, 2], "patch_pixels": [192, 192], "count_per_crop": 4,
               "analysis_grid_only": True, "model_patch_embedding_changed": False},
    "aggregation": "recompute histograms over each large pixel block; NOT mean of old token entropies",
    "raw_padding_policy": "raw entropy retains full patch histograms, including padding pixels",
    "layer_dependent": False,
}
DEVIATION_DEFINITION = {
    "formula": "D_i(w) = abs(H_i - median(H_j for j in valid neighbors excluding i))",
    "unit": "nat", "windows": [3, 5, 7], "max_neighbor_counts": [8, 24, 48],
    "reference": "local spatial context in the current image, NOT a normal reference image",
    "grid": "stitched logical patch lattice; crop boundaries are crossed",
    "physical_gap": "none: 2x192 covers all 384 pixels of each crop",
    "validity": "only patches fully inside actual resized image content, excluding partial padding",
    "edge_policy": "use available valid neighbors, no zero/reflect/repeat padding",
    "no_neighbor_policy": "NaN, excluded from display and Softmax",
    "mad_standardization": False, "paper_method": False,
}
SOFTMAX_DEFINITION = {
    "formula": "w_i = exp(s_i / ln(2)) / sum_j exp(s_j / ln(2))",
    "input": "Base entropy or AnyRes absolute entropy deviation in nats",
    "input_after_conversion": "bit", "output": "dimensionless weight, NOT defect probability",
    "temperature": 1.0,
    "raw_entropy_cache_only": "tile_entropy_softmax covers all 4 crop blocks including padding; plain figures use it, local figures use deviation Softmax",
    "scope": "Base: all 36 blocks; AnyRes: defined deviations independently per crop and per window",
    "no_defined_scores": "all NaN; no fabricated uniform distribution for a padded/unsupported crop",
    "post_softmax_minmax": False,
    "source": "log2 conversion and per-view convention retained from supplied script",
}
PLAIN_SOFTMAX_DEFINITION = {
    **SOFTMAX_DEFINITION,
    "input": "Base and AnyRes histogram entropy in nats",
    "scope": "Base: all 36 blocks; AnyRes: all 4 blocks independently per crop, including padding",
    "no_defined_scores": "raw block entropies are always finite; padding is hidden only in the display",
}
PLAIN_MINMAX_DEFINITION = {
    "formula": "(H - min) / (max - min)", "constant_value": 0.5,
    "scope": "shared across Base and AnyRes blocks with any visible content in one image",
    "softmax_applied": False, "paper_method": False,
}


MINMAX_DEFINITION = {
    "formula": "(s - min) / (max - min)", "constant_value": 0.5,
    "base_scope": "36 Base entropies only",
    "anyres_scope": "all finite AnyRes deviations jointly over all crops and all three windows",
    "softmax_applied": False, "paper_method": False,
}


def entropy_geometry(source_geometry):
    """Separate the analysis grid from the untouched source model geometry."""
    if source_geometry.tile_size != 384 or source_geometry.patch_size != 14:
        raise ValueError("Expected the current SigLIP384/14 source AnyRes geometry")
    return replace(source_geometry, patch_size=source_geometry.tile_size // ANYRES_GRID_SIDE)


def visible_blocks(geometry):
    """Blocks with any content overlap (plain mode); local uses full-content only."""
    starts = np.arange(geometry.token_side) * geometry.patch_size
    masks = []
    for index in range(geometry.tile_count):
        x0, y0, x1, y1 = geometry.tile_content_box(index)
        xs = (starts < x1) & (starts + geometry.patch_size > x0)
        ys = (starts < y1) & (starts + geometry.patch_size > y0)
        masks.append((ys[:, None] & xs[None, :]).ravel())
    return np.stack(masks)


def patch_entropy(rgb, patch_size):
    """Return FP32 raw entropies, token-row-major, on exact patch support.

    Remainder rows/columns are unused, matching a stride=kernel patch embedding.
    Black/constant patches have H=0 (valid data, not a missing score).
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Entropy requires an H x W x 3 uint8 RGB array before normalization")
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    rows, cols = rgb.shape[0] // patch_size, rgb.shape[1] // patch_size
    if not rows or not cols:
        raise ValueError("Image is smaller than one patch")
    # Promote before addition so uint8 channels cannot overflow.
    gray = rgb[:rows * patch_size, :cols * patch_size].astype(np.uint16).sum(axis=-1) // 3
    patches = gray.reshape(rows, patch_size, cols, patch_size).transpose(0, 2, 1, 3)
    patches = patches.reshape(rows * cols, patch_size * patch_size).astype(np.int64)
    offsets = np.arange(rows * cols, dtype=np.int64)[:, None] * 256
    histograms = np.bincount((patches + offsets).ravel(), minlength=rows * cols * 256)
    probabilities = histograms.reshape(rows * cols, 256).astype(np.float64) / (patch_size ** 2)
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    entropy = -(probabilities * logs).sum(axis=-1)
    return np.maximum(entropy, 0).astype(np.float32)


def entropy_softmax(entropy_nats):
    """Softmax of bit-valued entropy along tokens, NOT pixels or joined views.

    Input is the unchanged raw-nat cache. Author SigLIP code uses log2 entropy
    and softmax(entropies[i].flatten(), dim=0) for each vision batch item.
    Subtracting the maximum preserves the formula while avoiding overflow.
    """
    entropy = np.asarray(entropy_nats, dtype=np.float64)
    if entropy.ndim not in (1, 2) or not entropy.size or not np.isfinite(entropy).all():
        raise ValueError("Softmax requires finite, nonempty token or view-by-token entropies")
    if np.any(entropy < 0):
        raise ValueError("Entropy must be nonnegative")
    shifted_bits = (entropy - entropy.max(axis=-1, keepdims=True)) / math.log(2)
    weights = np.exp(shifted_bits)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights.astype(np.float32)


def prepare_entropy_views(image, geometry):
    """Reconstruct current SigLIP Base + AnyRes RGB pixels from saved geometry.

    Current local SigLipImageProcessor.size == crop_size == (384, 384). Pillow
    RGB's default resize is BICUBIC, exactly as the local mm_utils.py code.
    The processor's subsequent same-size resize does not alter these pixels.
    """
    if image.mode != "RGB" or image.size != geometry.original_size:
        raise ValueError("Source RGB image size must match saved geometry")
    size = geometry.tile_size
    base = image.resize((size, size), Image.Resampling.BICUBIC)
    padded = Image.new("RGB", (geometry.grid[0] * size, geometry.grid[1] * size), (0, 0, 0))
    resized = image.resize(geometry.resized_size, Image.Resampling.BICUBIC)
    padded.paste(resized, geometry.paste_xy)
    tiles = [padded.crop((col * size, row * size, (col + 1) * size, (row + 1) * size))
             for row in range(geometry.grid[1]) for col in range(geometry.grid[0])]
    return base, tiles


def configured_path(raw):
    if not raw.strip():
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number

def base_score_map(scores, geometry):
    """Map 36 analysis blocks across the full 384x384 Base view."""
    values = np.asarray(scores)
    if values.shape != (BASE_GRID_SIDE ** 2,):
        raise ValueError("Base must have exactly 36 analysis block scores")
    if geometry.tile_size % BASE_GRID_SIDE:
        raise ValueError("Base view size must be divisible by 6")
    block = geometry.tile_size // BASE_GRID_SIDE
    return values.reshape(BASE_GRID_SIDE, BASE_GRID_SIDE).repeat(block, 0).repeat(block, 1)


def full_content_tokens(geometry):
    """Full-patch image support; partial black padding is not normal context."""
    starts = np.arange(geometry.token_side) * geometry.patch_size
    masks = []
    for index in range(geometry.tile_count):
        x0, y0, x1, y1 = geometry.tile_content_box(index)
        xs = (starts >= x0) & (starts + geometry.patch_size <= x1)
        ys = (starts >= y0) & (starts + geometry.patch_size <= y1)
        masks.append((ys[:, None] & xs[None, :]).ravel())
    return np.stack(masks)


def assemble_patch_grid(values, geometry):
    values = np.asarray(values)
    side = geometry.token_side
    if values.shape != (geometry.tile_count, side ** 2):
        raise ValueError("AnyRes scores must have [crop, row-major patch] shape")
    return values.reshape(geometry.grid[1], geometry.grid[0], side, side).transpose(
        0, 2, 1, 3).reshape(geometry.grid[1] * side, geometry.grid[0] * side)


def split_patch_grid(values, geometry):
    values = np.asarray(values)
    side = geometry.token_side
    expected = (geometry.grid[1] * side, geometry.grid[0] * side)
    if values.shape != expected:
        raise ValueError("Stitched logical patch grid shape does not match geometry")
    return values.reshape(geometry.grid[1], side, geometry.grid[0], side).transpose(
        0, 2, 1, 3).reshape(geometry.tile_count, side ** 2)


def local_entropy_deviation(tile_entropy, geometry, windows=WINDOWS):
    """Return D, local medians and valid-neighbor counts in [window,crop,token]."""
    values = np.asarray(tile_entropy, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Raw entropy must be finite and nonnegative")
    if not windows or any(type(w) is not int or w < 3 or w % 2 == 0 for w in windows):
        raise ValueError("Neighborhood windows must be odd integers >= 3")
    valid = full_content_tokens(geometry)
    grid = assemble_patch_grid(np.where(valid, values, np.nan), geometry)
    deviations, medians, counts = [], [], []
    for window in windows:
        radius = window // 2
        padded = np.pad(grid, radius, constant_values=np.nan)
        neighborhoods = np.lib.stride_tricks.sliding_window_view(
            padded, (window, window)).copy().reshape(*grid.shape, window ** 2)
        neighborhoods[..., window ** 2 // 2] = np.nan  # exclude center
        count = np.isfinite(neighborhoods).sum(axis=-1).astype(np.int16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # isolated or padded centers
            median = np.nanmedian(neighborhoods, axis=-1)
        usable = np.isfinite(grid) & (count > 0)
        deviation = np.full(grid.shape, np.nan, dtype=np.float32)
        deviation[usable] = np.abs(grid[usable] - median[usable])
        median = np.where(usable, median, np.nan).astype(np.float32)
        count = np.where(np.isfinite(grid), count, 0).astype(np.int16)
        deviations.append(split_patch_grid(deviation, geometry))
        medians.append(split_patch_grid(median, geometry))
        counts.append(split_patch_grid(count, geometry))
    return (np.stack(deviations), np.stack(medians), np.stack(counts), valid)


def deviation_softmax(deviations):
    """Use supplied bit-entropy Softmax on defined scores, independently per crop."""
    values = np.asarray(deviations, dtype=np.float64)
    if values.ndim != 3 or np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Expected nonnegative/NaN deviations [window,crop,token]")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for window_index in range(values.shape[0]):
        for crop_index in range(values.shape[1]):
            finite = np.isfinite(values[window_index, crop_index])
            if finite.any():
                result[window_index, crop_index, finite] = entropy_softmax(
                    values[window_index, crop_index, finite])
    return result


def value_range(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else None


def minmax_scores(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    limits = value_range(values)
    if limits is not None:
        low, high = limits
        result[finite] = .5 if low == high else (values[finite] - low) / (high - low)
    return result, limits


def weight_color_limits(value_range):
    if value_range is None:
        return 0., 1.
    low, high = value_range
    if low == high:
        margin = max(high * .05, 1e-8)
        return max(0., low - margin), min(1., high + margin)
    return low, high


def plain_score_display(base_entropy, tile_entropy, geometry, color_scale):
    """Normalize before expansion; never remove padding from a Softmax group."""
    base = np.asarray(base_entropy)
    tiles = np.asarray(tile_entropy)
    if base.shape != (36,) or tiles.shape != (geometry.tile_count, geometry.token_side ** 2):
        raise ValueError("Plain entropy arrays do not match the analysis grids")
    if not np.isfinite(base).all() or not np.isfinite(tiles).all() or min(base.min(), tiles.min()) < 0:
        raise ValueError("Raw entropy must be finite and nonnegative")
    visible = visible_blocks(geometry)
    displayed_raw = np.where(visible, tiles, np.nan)
    if color_scale == "minmax":
        joint = np.concatenate((base, tiles[visible]))
        low, high = float(joint.min()), float(joint.max())
        if high == low:
            b = np.full(base.shape, .5, dtype=np.float32)
            t = np.where(visible, .5, np.nan).astype(np.float32)
        else:
            b = ((base - low) / (high - low)).astype(np.float32)
            t = ((displayed_raw - low) / (high - low)).astype(np.float32)
        return b, t, {"base_limits": (0., 1.), "anyres_limits": (0., 1.),
                      "base_label": "Entropy (shared min-max)", "anyres_label": "Entropy (shared min-max)",
                      "shared_reference_nats": (low, high), "score_mode": "minmax"}
    if color_scale == "raw":
        return base, displayed_raw, {
            "base_limits": (0., max(float(base.max()), 1e-8)),
            "anyres_limits": (0., max(float(tiles[visible].max()), 1e-8)) if visible.any() else (0., 1.),
            "base_label": "Base entropy (nat)", "anyres_label": "AnyRes entropy (nat)",
            "score_mode": "raw"}
    if color_scale not in {"softmax", "sample", "fixed", "layer"}:
        raise ValueError("Unknown color scale")
    b = entropy_softmax(base)
    t = np.where(visible, entropy_softmax(tiles), np.nan)
    return b, t, {
        "base_limits": (0., 1.) if color_scale == "fixed" else weight_color_limits(value_range(b)),
        "anyres_limits": (0., 1.) if color_scale == "fixed" else weight_color_limits(value_range(t)),
        "base_label": "Base Softmax weight", "anyres_label": "AnyRes Softmax weight",
        "score_mode": "softmax"}


def score_display(base_entropy, deviations, color_scale):
    """Different score meanings require separate Base/AnyRes normalization."""
    if color_scale == "minmax":
        base, base_reference = minmax_scores(base_entropy)
        local, local_reference = minmax_scores(deviations)
        return base, local, {
            "base_limits": (0., 1.), "anyres_limits": (0., 1.),
            "base_label": "Base entropy (min-max)",
            "anyres_label": "AnyRes deviation (min-max)",
            "base_reference_nats": base_reference, "anyres_reference_nats": local_reference,
            "score_mode": "minmax"}
    if color_scale == "raw":
        base, local = np.asarray(base_entropy), np.asarray(deviations)
        local_range = value_range(local)
        return base, local, {
            "base_limits": (0., max(float(base.max()), 1e-8)),
            "anyres_limits": (0., max(local_range[1] if local_range else 0., 1e-8)),
            "base_label": "Base entropy (nat)",
            "anyres_label": "AnyRes deviation (nat)",
            "score_mode": "raw"}
    if color_scale not in {"softmax", "sample", "fixed", "layer"}:
        raise ValueError("Unknown color scale")
    base, local = entropy_softmax(base_entropy), deviation_softmax(deviations)
    fixed = color_scale == "fixed"
    return base, local, {
        "base_limits": (0., 1.) if fixed else weight_color_limits(value_range(base)),
        "anyres_limits": (0., 1.) if fixed else weight_color_limits(value_range(local)),
        "base_label": "Base Softmax weight",
        "anyres_label": "AnyRes Softmax weight",
        "score_mode": "softmax"}


def draw_figures(image, base, base_entropy, deviations, geometry, folder, caption, color_scale,
                 entropy_mode="local"):
    """Plain: one overview. Local: retain raw/selected overviews and comparisons."""
    from matplotlib import colormaps, patheffects, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    folder = Path(folder)
    cmap = colormaps[COLORMAP].with_extremes(bad=MISSING_COLOR)
    if entropy_mode == "plain":
        selected_base, selected_local, setup = plain_score_display(base_entropy, deviations, geometry, color_scale)
    elif entropy_mode == "local":
        raw_base, raw_local, raw_setup = score_display(base_entropy, deviations, "raw")
        soft_base, soft_local, soft_setup = score_display(base_entropy, deviations, "softmax")
        selected_base, selected_local, setup = score_display(base_entropy, deviations, color_scale)
    else:
        raise ValueError("Unknown entropy mode")
    files = []

    def panel(ax, source, heatmap=None, norm=None, tile_edges=False):
        ax.imshow(source, extent=(0, 1, 1, 0), interpolation="nearest")
        if heatmap is not None:
            values = np.asarray(heatmap)
            rgba = cmap(norm(np.ma.masked_invalid(values)))
            rgba[..., 3] = np.where(np.isfinite(values), OVERLAY_ALPHA, 1.)
            ax.imshow(rgba, extent=(0, 1, 1, 0), interpolation="nearest")
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
        ax.set_aspect(source.height / source.width)
        if tile_edges:
            effect = [patheffects.Stroke(linewidth=2, foreground="black"), patheffects.Normal()]
            for index, rect in enumerate(tile_rectangles(geometry.__dict__)):
                if rect is None:
                    continue
                x, y, width, height = rect
                ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="white",
                                      linewidth=.6, linestyle="--", path_effects=effect))
                ax.text(x + .01, y + .01, str(index + 1), va="top", color="white", fontsize=8,
                        bbox=dict(facecolor="black", alpha=.65, pad=1, edgecolor="none"))

    def colorbar(fig, axes, limits, label):
        fig.colorbar(ScalarMappable(norm=Normalize(*limits), cmap=cmap), ax=axes,
                     fraction=.025, pad=.02, label=label)

    def save(fig, name, footer):
        fig.suptitle(caption, fontsize=11, y=.995)
        fig.supxlabel(footer, fontsize=8)
        target = folder / name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        try:
            fig.savefig(target, dpi=150, facecolor="white")
        finally:
            fig.clear()
        files.append(name)

    def overview(name, base_scores, local_scores, spec):
        fig = Figure(figsize=(12, 9), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3)
        base_norm, local_norm = Normalize(*spec["base_limits"]), Normalize(*spec["anyres_limits"])
        panel(axes[0, 0], base)
        panel(axes[0, 1], base, base_score_map(base_scores, geometry), base_norm)
        panel(axes[0, 2], image, tile_edges=True)
        axes[0, 0].set_title("Base reference")
        axes[0, 1].set_title(f"Base entropy: 6x6 blocks | {spec['score_mode']}")
        axes[0, 2].set_title("Original + AnyRes crop IDs")
        colorbar(fig, [axes[0, 1]], spec["base_limits"], spec["base_label"])
        for wi, window in enumerate(WINDOWS):
            panel(axes[1, wi], image, stitch_score_map(local_scores[wi], geometry),
                  local_norm, tile_edges=True)
            axes[1, wi].set_title(f"AnyRes: {window}x{window} entropy deviation | {spec['score_mode']}")
        colorbar(fig, list(axes[1]), spec["anyres_limits"], spec["anyres_label"])
        footer = ("Base: 36 blocks of 64x64 pixels | AnyRes: 4 blocks of 192x192 pixels per crop\n"
                  "D = |H - median(neighbors)|; center excluded; logical neighborhoods cross crop boundaries.\n"
                  "All three AnyRes windows share one scale; Base has a separate scale (colors are not comparable). Gray = padding / undefined.")
        if spec["score_mode"] == "softmax":
            footer += "\nSoftmax uses bits, independently per view and window; AnyRes includes defined scores only. Weights are NOT defect probabilities."
        elif spec["score_mode"] == "minmax":
            footer += "\nMin-max: Base separately; all AnyRes windows jointly. Constant group = 0.5. No Softmax."
        save(fig, name, footer)

    with rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        if entropy_mode == "plain":
            fig = Figure(figsize=(16, 5), layout="constrained")
            FigureCanvasAgg(fig)
            axes = fig.subplots(1, 4)
            panel(axes[0], base)
            panel(axes[1], base, base_score_map(selected_base, geometry), Normalize(*setup["base_limits"]))
            panel(axes[2], image, tile_edges=True)
            panel(axes[3], image, stitch_score_map(selected_local, geometry),
                  Normalize(*setup["anyres_limits"]), tile_edges=True)
            for ax, title in zip(axes, ("Base reference", f"Base entropy: 6x6 | {setup['score_mode']}",
                                       "Original + AnyRes crop IDs", f"AnyRes entropy: 2x2 per crop | {setup['score_mode']}")):
                ax.set_title(title)
            shared = color_scale in {"minmax", "fixed"}
            if shared:
                colorbar(fig, list(axes), setup["base_limits"],
                         "Entropy (shared min-max)" if color_scale == "minmax" else "Entropy Softmax weight")
            else:
                colorbar(fig, list(axes[:2]), setup["base_limits"], setup["base_label"])
                colorbar(fig, list(axes[2:]), setup["anyres_limits"], setup["anyres_label"])
            footer = ("Ordinary histogram entropy | Base: 36 blocks of 64x64 px | AnyRes: 4 blocks of 192x192 px per crop\n"
                      "No neighborhood subtraction. Gray = geometric padding. Analysis blocks are NOT model tokens.\n")
            if setup["score_mode"] == "softmax":
                footer += "Softmax of bit entropy: Base 36 blocks sum to 1; each AnyRes crop's 4 blocks sum to 1, including padding. Not defect probabilities.\n"
            elif setup["score_mode"] == "minmax":
                footer += "One raw-entropy min/max for Base + visible AnyRes blocks; constant range = 0.5. No Softmax.\n"
            footer += ("Shared color scale." if shared else
                       "Separate Base/AnyRes colorbars: the same color does NOT imply the same value. All AnyRes crops share one scale.")
            save(fig, "entropy_overview.png", footer)
            return {"figures": files, "display": setup, "colorbar_mode": "shared" if shared else "separate"}

        overview("entropy_raw_overview.png", raw_base, raw_local, raw_setup)
        overview("entropy_overview.png", selected_base, selected_local, setup)

        fig = Figure(figsize=(12, 4.8), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3)
        panel(axes[0], base)
        panel(axes[1], base, base_score_map(raw_base, geometry), Normalize(*raw_setup["base_limits"]))
        panel(axes[2], base, base_score_map(soft_base, geometry), Normalize(*soft_setup["base_limits"]))
        for ax, title in zip(axes, ("Base reference", "Base: 6x6 raw entropy", "Base: 36-block Softmax")):
            ax.set_title(title)
        colorbar(fig, [axes[1]], raw_setup["base_limits"], raw_setup["base_label"])
        colorbar(fig, [axes[2]], soft_setup["base_limits"], soft_setup["base_label"])
        save(fig, "base_entropy_6x6.png",
             "Base scoring blocks are 64x64 pixels, covering all 384x384 pixels.\n"
             "Softmax is over 36 blocks using bit entropy. This analysis does not change model tokens.")

        for wi, window in enumerate(WINDOWS):
            fig = Figure(figsize=(12, 4.8), layout="constrained")
            FigureCanvasAgg(fig)
            axes = fig.subplots(1, 3)
            panel(axes[0], image, tile_edges=True)
            panel(axes[1], image, stitch_score_map(raw_local[wi], geometry),
                  Normalize(*raw_setup["anyres_limits"]), tile_edges=True)
            panel(axes[2], image, stitch_score_map(soft_local[wi], geometry),
                  Normalize(*soft_setup["anyres_limits"]), tile_edges=True)
            for ax, title in zip(axes, ("Original + crop IDs", f"{window}x{window}: raw entropy deviation",
                                       f"{window}x{window}: deviation Softmax")):
                ax.set_title(title)
            colorbar(fig, [axes[1]], raw_setup["anyres_limits"], raw_setup["anyres_label"])
            colorbar(fig, [axes[2]], soft_setup["anyres_limits"], soft_setup["anyres_label"])
            save(fig, f"anyres_deviation_{window}x{window}.png",
                 f"Window {window}x{window}: up to {window * window - 1} neighbors, center excluded; full-content patches only.\n"
                 "Raw and Softmax use separate colorbars. Each scale is shared across the 3x3, 5x5 and 7x7 figures.\n"
                 "Softmax is independent per crop over defined scores; Gray = undefined / padding / no patch coverage.")

    return {"figures": files, "display": setup, "raw_display": raw_setup,
            "softmax_display": soft_setup, "colorbar_mode": "separate_base_anyres_shared_anyres_windows"}


def write_means(path, base_entropy, tile_entropy, deviations, base_weights, local_weights, valid, geometry):
    def mean_or_blank(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        return float(finite.mean(dtype=np.float64)) if finite.size else ""

    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["view", "crop_id", "window", "total_blocks", "full_content_blocks",
                         "defined_score_blocks", "mean_full_content_entropy_nats", "mean_deviation_nats",
                         "mean_softmax_weight", "softmax_weight_sum"])
        writer.writerow(["base", "", "", 36, 36, 36, mean_or_blank(base_entropy), "",
                         mean_or_blank(base_weights), float(base_weights.sum(dtype=np.float64))])
        for wi, window in enumerate(WINDOWS):
            for ci in range(geometry.tile_count):
                finite = np.isfinite(deviations[wi, ci])
                weight_sum = float(local_weights[wi, ci, finite].sum(dtype=np.float64)) if finite.any() else ""
                writer.writerow(["anyres", ci + 1, window, geometry.token_side ** 2, int(valid[ci].sum()),
                                 int(finite.sum()), mean_or_blank(tile_entropy[ci, valid[ci]]),
                                 mean_or_blank(deviations[wi, ci]), mean_or_blank(local_weights[wi, ci]),
                                 weight_sum])


def write_plain_means(path, base_entropy, tile_entropy, geometry):
    """Keep raw/Softmax means over ALL blocks, and separately count visible blocks."""
    b, t = entropy_softmax(base_entropy), entropy_softmax(tile_entropy)
    visible = visible_blocks(geometry)
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["view", "crop_id", "total_blocks", "visible_blocks", "mean_entropy_nats",
                         "mean_visible_entropy_nats", "mean_softmax_weight", "softmax_weight_sum"])
        writer.writerow(["base", "", len(b), len(b), float(base_entropy.mean()), float(base_entropy.mean()),
                         float(b.mean(dtype=np.float64)), float(b.sum(dtype=np.float64))])
        for i, values in enumerate(tile_entropy):
            writer.writerow(["anyres", i + 1, len(values), int(visible[i].sum()), float(values.mean()),
                             float(values[visible[i]].mean()) if visible[i].any() else "",
                             float(t[i].mean(dtype=np.float64)), float(t[i].sum(dtype=np.float64))])


def run(results_dir, output_dir=None, data_root=None, color_scale="softmax", limit=None, entropy_mode="plain"):
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # check before creating outputs
    if color_scale not in {"softmax", "raw", "minmax", "sample", "fixed", "layer"}:
        raise ValueError("Unknown color scale")
    if entropy_mode not in {"plain", "local"}:
        raise ValueError("entropy_mode must be plain or local")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    results = Path(results_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else results.with_name(
        f"{results.name}_vflowopt_{entropy_mode}_{color_scale}")
    if output == results or output.is_relative_to(results) or results.is_relative_to(output):
        raise ValueError("Output must be separate from the source result tree")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    folders = sample_folders(results)
    if limit is not None:
        folders = folders[:limit]
    run_path = results / "run.json"
    if len(folders) == 1 and folders[0] == results and not run_path.is_file():
        run_path = results.parent / "run.json"
    source_run = read_json(run_path) if run_path.is_file() else {}
    samples = []
    for folder in folders:
        metadata = read_json(folder / "metadata.json")
        geometry = ScoreGeometry(**validate_geometry(metadata))
        if geometry.tile_size != 384 or geometry.patch_size != 14:
            raise ValueError("Expected the current SigLIP384/14 AnyRes geometry")
        image_path = resolve_image(metadata, source_run, data_root)
        with Image.open(image_path) as image:
            if image.size != geometry.original_size:
                raise ValueError(f"Original image size differs from saved geometry: {image_path}")
        samples.append((folder, metadata, geometry, image_path))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"script_version": SCRIPT_VERSION, "source_script_sha256": SOURCE_SCRIPT_SHA256,
                "source_results": str(results), "paper": PAPER_URL, "entropy": ENTROPY_DEFINITION,
                "entropy_mode": entropy_mode,
                "deviation": DEVIATION_DEFINITION if entropy_mode == "local" else None,
                "softmax": SOFTMAX_DEFINITION if entropy_mode == "local" else PLAIN_SOFTMAX_DEFINITION,
                "minmax": MINMAX_DEFINITION if entropy_mode == "local" else PLAIN_MINMAX_DEFINITION,
                "color_scale": color_scale, "colormap": COLORMAP, "overlay_alpha": OVERLAY_ALPHA,
                "sample_count": len(samples), "completed_samples": 0, "complete": False,
                "model_loaded": False, "llm_generation": False, "pruning": False,
                "source_scores_read": False, "score_location": "offline before AnyRes unpadding/downsampling/newlines",
                "data_root_override": str(Path(data_root).resolve()) if data_root else None,
                "numpy_version": np.__version__, "pillow_version": Image.__version__}
    manifest_path = output / "run.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_manifest()
    print(f"runVFlowOpt {SCRIPT_VERSION} | {len(samples)} samples | mode={entropy_mode} | "
          f"Base=6x6 | AnyRes=2x2 per crop | color-scale={color_scale} | CPU", flush=True)
    if color_scale in {"sample", "layer"}:
        print(f"'{color_scale}' is a compatibility alias of Softmax. Use --color-scale raw for raw values.", flush=True)
    try:
        for index, (folder, metadata, source_geometry, image_path) in enumerate(samples, 1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            base, tiles = prepare_entropy_views(image, source_geometry)
            geometry = entropy_geometry(source_geometry)
            base_entropy = patch_entropy(base, geometry.tile_size // BASE_GRID_SIDE)
            tile_entropy = np.stack([patch_entropy(tile, geometry.patch_size) for tile in tiles])
            base_weights = entropy_softmax(base_entropy)
            tile_weights = entropy_softmax(tile_entropy)
            extra_arrays, extra_metadata = {}, {}
            if entropy_mode == "local":
                deviations, medians, counts, valid = local_entropy_deviation(tile_entropy, geometry)
                local_weights = deviation_softmax(deviations)
                base_display, local_display, _ = score_display(base_entropy, deviations, color_scale)
                plot_values = deviations
                extra_arrays = {"tile_deviation": deviations, "tile_deviation_softmax": local_weights,
                                "tile_neighbor_median": medians, "tile_neighbor_count": counts,
                                "tile_full_content": valid, "windows": np.asarray(WINDOWS),
                                "tile_deviation_display": local_display,
                                "tile_deviation_softmax_scope": np.asarray("per_window_per_crop_defined_scores_only")}
                extra_metadata = {"anyres_deviation_shape": list(deviations.shape),
                                  "full_content_patch_counts": valid.sum(axis=-1).tolist(),
                                  "defined_deviation_counts": np.isfinite(deviations).sum(axis=-1).tolist()}
            else:
                base_display, local_display, _ = plain_score_display(base_entropy, tile_entropy, geometry, color_scale)
                plot_values = tile_entropy
                extra_arrays = {"tile_entropy_display": local_display,
                                "tile_visible": visible_blocks(geometry)}
            target = output / folder.name
            target.mkdir(exist_ok=False)
            np.savez_compressed(target / "entropy.npz",
                                base_entropy=base_entropy, tile_entropy=tile_entropy,
                                base_softmax=base_weights,
                                tile_entropy_softmax=tile_weights,
                                base_display=base_display,
                                base_grid_side=np.asarray(6), base_patch_size=np.asarray(64),
                                anyres_grid_side=np.asarray(2), patch_size=np.asarray(192),
                                source_model_patch_size=np.asarray(14), entropy_mode=np.asarray(entropy_mode),
                                unit=np.asarray("nat"),
                                softmax_input_unit=np.asarray("bit"),
                                tile_entropy_softmax_scope=np.asarray("per_crop_all_4_including_padding"),
                                **extra_arrays)
            if entropy_mode == "local":
                write_means(target / "entropy_patch_means.csv", base_entropy, tile_entropy, deviations,
                            base_weights, local_weights, valid, geometry)
            else:
                write_plain_means(target / "entropy_patch_means.csv", base_entropy, tile_entropy, geometry)
            gt = {0: "Normal", 1: "Abnormal"}.get(metadata.get("gt"), "Unknown")
            display = draw_figures(image, base, base_entropy, plot_values, geometry, target,
                                   f"ID: {metadata.get('id', folder.name)} | GT: {gt} | Mode: {entropy_mode} | No LLM prediction",
                                   color_scale, entropy_mode)
            saved = {"id": metadata.get("id", folder.name), "image": str(image_path), "gt": metadata.get("gt"),
                     "source_metadata": str(folder / "metadata.json"), "geometry": source_geometry.__dict__,
                     "entropy_geometry": geometry.__dict__, "entropy_mode": entropy_mode,
                     "entropy": ENTROPY_DEFINITION, "deviation": manifest["deviation"], "softmax": manifest["softmax"],
                     "base_score_shape": list(base_entropy.shape), "anyres_entropy_shape": list(tile_entropy.shape),
                     "score_axes": (["window_3_5_7"] if entropy_mode == "local" else []) + ["crop_row_major", "analysis_block_row_major"],
                     "stitched_anyres_grid_shape": [geometry.grid[1] * 2, geometry.grid[0] * 2],
                     "entropy_upper_bound_nats": {"base": math.log(256), "anyres": math.log(256)},
                     "color_scale": color_scale, "script_version": SCRIPT_VERSION, **extra_metadata, **display}
            (target / "metadata.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["completed_samples"] = index
            save_manifest()
            print(f"[{index}/{len(samples)}] {saved['id']}: {geometry.tile_count} crops -> {target}", flush=True)
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_manifest()
        raise
    manifest["complete"] = True
    save_manifest()
    print(f"Done: {output}", flush=True)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--results-dir", type=Path, help="Previous feature-score results, or one sample folder")
    parser.add_argument("--output-dir", type=Path, help="New/empty output directory; default sibling <results>_vflowopt_<mode>_<color-scale>")
    parser.add_argument("--data-root", type=Path, help="Optional relocated dataset root, as in visualize_patch_means.py")
    parser.add_argument("--entropy-mode", choices=("plain", "local"), default="plain",
                        help="plain (default): ordinary entropy; local: absolute deviation from 3/5/7 neighbor medians. "
                             "Both use Base 6x6 and AnyRes 2x2 blocks per crop; model geometry is unchanged.")
    parser.add_argument("--color-scale", choices=("softmax", "raw", "minmax", "sample", "fixed", "layer"),
                        default="softmax",
                        help="Overview mode; separate Base/AnyRes scales, one shared AnyRes scale across windows. "
                             "Local mode also generates raw + Softmax comparisons. sample/layer alias Softmax.")
    parser.add_argument("--limit", type=positive_int, default=None, help="First N samples; default all")
    return parser


def main():
    args = build_parser().parse_args()
    source = args.results_dir or configured_path(RESULTS_DIR)
    if source is None:
        raise ValueError("Set RESULTS_DIR at the top or pass --results-dir")
    run(source, args.output_dir or configured_path(OUTPUT_DIR), args.data_root or configured_path(DATA_ROOT),
        args.color_scale, args.limit, args.entropy_mode)


if __name__ == "__main__":
    main()
