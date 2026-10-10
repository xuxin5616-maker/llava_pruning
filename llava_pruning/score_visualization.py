"""Read-only feature probes and spatially aligned AnyRes score figures.

Each token is L2-normalized along channels BEFORE computing view means.
Global = -cos(unit tile token, mean(unit Base tokens)); local = -cos(unit tile
token, mean(unit tokens of that tile)). Neither score is a defect probability.
All view means include every encoded token, including padding-context tokens.
Only the display excludes geometric image padding. Scores are computed before
AnyRes unpadding, max_9 feature downsampling, and structural newline insertion.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


LAYERS = (7, 14, 21, 26)  # 1-based encoder block outputs, before post_layernorm.
STAGES = tuple(f"siglip_{layer:02d}" for layer in LAYERS) + ("projector",)
LABELS = tuple(f"SigLIP layer {layer}" for layer in LAYERS) + ("After projector",)
MISSING_COLOR = "#b8b8b8"
OVERLAY_ALPHA = 0.70
COLORMAP = "jet"
TILES_PER_PAGE = 4
FEATURE_NORMALIZATION = "l2_per_token_before_mean"


def _unit_token_features(values):
    """Unit vectors along channels; zero vectors stay zero, inputs untouched.

    The caller validates finite FP32 features. Scaling by the largest component
    first avoids overflow/underflow in the norm without changing direction.
    """
    scale = values.abs().amax(dim=-1, keepdim=True)
    scaled = values / scale.masked_fill(scale == 0, 1.)
    length = scaled.norm(dim=-1, keepdim=True)
    return scaled / length.masked_fill(length == 0, 1.)


def score_tokens(features):
    """Score [Base + tiles, spatial tokens, channels] in detached FP32.

    Normalize each token along channels, THEN average over spatial tokens.
    Original forward features are never modified. Zero vectors remain zero in
    means; zero-norm tokens/references give NaN (undefined cosine), not invented
    low/high scores. Nonfinite features fail.
    """
    import torch
    if features.ndim != 3 or features.shape[0] < 2 or min(features.shape[1:]) < 1:
        raise ValueError("Expected features shaped [Base + at least one tile, tokens, channels]")
    values = features.detach().float()
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite visual features; cannot produce trustworthy scores")
    values = _unit_token_features(values)
    means = values.mean(dim=1)
    tokens = values[1:]
    token_norms = tokens.norm(dim=-1)

    def cosine_to(references):
        reference_norms = references.norm(dim=-1)
        denominator = token_norms * reference_norms[:, None]
        dot = (tokens * references[:, None, :]).sum(dim=-1)
        scores = (-dot / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0)
        scores = scores.masked_fill(denominator <= 1e-12, float("nan"))
        return scores.cpu().numpy()

    return {"global": cosine_to(means[:1].expand(tokens.shape[0], -1)),
            "local": cosine_to(means[1:])}


def capture_scores(model, pixels):
    """Run only the existing encode_images path; always remove read-only hooks."""
    import torch
    tower = model.get_vision_tower()
    blocks = tower.vision_tower.vision_model.encoder.layers
    if len(blocks) != 26:
        raise ValueError(f"Expected the current 26-block SigLIP path, found {len(blocks)}; "
                         "do not silently relabel another model's layers")
    if model.training:
        raise ValueError("Feature visualization requires model.eval()")
    captured, shapes, handles = {}, {}, []

    def hook_for(stage):
        def observe(module, args, output):
            features = output[0] if isinstance(output, (tuple, list)) else output
            if stage in captured:
                raise ValueError(f"Unexpected repeated forward at {stage}")
            captured[stage] = score_tokens(features)
            shapes[stage] = list(features.shape)
            # Returning None preserves the forward result, including its dtype.
        return observe

    try:
        for layer, stage in zip(LAYERS, STAGES):
            handles.append(blocks[layer - 1].register_forward_hook(hook_for(stage)))
        handles.append(model.get_model().mm_projector.register_forward_hook(hook_for("projector")))
        with torch.inference_mode():
            model.encode_images(pixels)
    finally:
        for handle in handles:
            handle.remove()
    if tuple(captured) != STAGES:
        raise RuntimeError(f"Missing or out-of-order feature stages: {list(captured)}")
    expected = pixels.shape[0]
    if any(shape[0] != expected for shape in shapes.values()):
        raise ValueError("Feature batch does not preserve Base + tile order")
    scores = {kind: np.stack([captured[name][kind] for name in STAGES])
              for kind in ("global", "local")}
    return scores, shapes


@dataclass(frozen=True)
class ScoreGeometry:
    original_size: tuple[int, int]
    grid: tuple[int, int]
    tile_size: int
    patch_size: int
    resized_size: tuple[int, int]
    paste_xy: tuple[int, int]

    @classmethod
    def create(cls, original_size, grid, tile_size, patch_size):
        if min(*original_size, *grid, tile_size, patch_size) < 1 or patch_size > tile_size:
            raise ValueError("Invalid image/token geometry")
        width, height = original_size
        target_w, target_h = grid[0] * tile_size, grid[1] * tile_size
        scale_w, scale_h = target_w / width, target_h / height
        # Match resize_and_pad_image exactly, including ceil and odd padding.
        if scale_w < scale_h:
            new_w, new_h = target_w, min(math.ceil(height * scale_w), target_h)
        else:
            new_h, new_w = target_h, min(math.ceil(width * scale_h), target_w)
        return cls(tuple(original_size), tuple(grid), tile_size, patch_size,
                   (new_w, new_h), ((target_w - new_w) // 2, (target_h - new_h) // 2))

    @property
    def token_side(self):
        return self.tile_size // self.patch_size

    @property
    def tile_count(self):
        return self.grid[0] * self.grid[1]

    def tile_content_box(self, index):
        if not 0 <= index < self.tile_count:
            raise IndexError(index)
        row, col = divmod(index, self.grid[0])
        left = self.paste_xy[0] - col * self.tile_size
        top = self.paste_xy[1] - row * self.tile_size
        width, height = self.resized_size
        clip = lambda value: max(0, min(self.tile_size, value))
        return clip(left), clip(top), clip(left + width), clip(top + height)


def prepare_views(image, processor, config, patch_size):
    """Reuse actual AnyRes pixels, reconstruct only their display geometry."""
    from llava.mm_utils import (get_anyres_image_grid_shape, process_anyres_image,
                                resize_and_pad_image, divide_to_patches)
    tile_size = processor.crop_size["height"]
    if processor.crop_size["width"] != tile_size:
        raise ValueError("This SigLIP probe requires square processed views")
    grid = get_anyres_image_grid_shape(image.size, config.image_grid_pinpoints, tile_size)
    geometry = ScoreGeometry.create(image.size, grid, tile_size, patch_size)
    pixels = process_anyres_image(image, processor, config.image_grid_pinpoints)
    if tuple(pixels.shape) != (1 + geometry.tile_count, 3, tile_size, tile_size):
        raise ValueError("Processor resize differs from the tile geometry")
    padded = resize_and_pad_image(image, (grid[0] * tile_size, grid[1] * tile_size))
    tiles = divide_to_patches(padded, tile_size)
    # The global input is the original, square-resized image, not the padded mosaic.
    base = image.resize((tile_size, tile_size))
    return pixels, base, tiles, geometry


def tile_score_map(scores, geometry, index):
    """Exact patch support (e.g. 27*14=378 of 384), not stretched to 384."""
    side, patch = geometry.token_side, geometry.patch_size
    if np.asarray(scores).shape != (side * side,):
        raise ValueError("Score count does not match the patch embedding grid")
    result = np.full((geometry.tile_size, geometry.tile_size), np.nan, dtype=np.float32)
    support = np.asarray(scores).reshape(side, side).repeat(patch, 0).repeat(patch, 1)
    result[:side * patch, :side * patch] = support
    x0, y0, x1, y1 = geometry.tile_content_box(index)
    valid = np.zeros(result.shape, dtype=bool)
    valid[y0:y1, x0:x1] = True
    result[~valid] = np.nan
    return result


def stitch_score_map(scores, geometry):
    if np.asarray(scores).shape != (geometry.tile_count, geometry.token_side ** 2):
        raise ValueError("Score views do not match the AnyRes tile grid")
    rows = []
    for row in range(geometry.grid[1]):
        rows.append(np.concatenate([
            tile_score_map(scores[row * geometry.grid[0] + col], geometry,
                           row * geometry.grid[0] + col)
            for col in range(geometry.grid[0])], axis=1))
    padded = np.concatenate(rows, axis=0)
    left, top = geometry.paste_xy
    width, height = geometry.resized_size
    return padded[top:top + height, left:left + width]


def color_limits(scores, geometry, scale):
    if scale == "fixed":
        return -1.0, 1.0
    if scale != "sample":
        raise ValueError("Unknown color scale")
    low, high = math.inf, -math.inf
    for kind in ("global", "local"):
        for stage in scores[kind]:
            # Only tokens overlapping actual displayed image content affect limits.
            for index, token_scores in enumerate(stage):
                displayed = tile_score_map(token_scores, geometry, index)
                finite = displayed[np.isfinite(displayed)]
                if finite.size:
                    low, high = min(low, float(finite.min())), max(high, float(finite.max()))
    if not math.isfinite(low):
        raise ValueError("No finite scores on the visible image")
    if high - low < 1e-6:
        low, high = max(-1.0, low - 0.01), min(1.0, high + 0.01)
    return low, high


def draw_figures(image, base, tiles, scores, geometry, folder, caption, scale):
    """Five-layer overview + bounded-width pages of individual tile overlays."""
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.cm import ScalarMappable
    from matplotlib import patheffects

    folder = Path(folder)
    limits = color_limits(scores, geometry, scale)
    norm = Normalize(*limits)
    cmap = colormaps[COLORMAP].with_extremes(bad=MISSING_COLOR)
    files = []

    def panel(ax, source, heatmap=None):
        ax.imshow(source, extent=(0, 1, 1, 0), interpolation="nearest")
        if heatmap is not None:
            # Opaque gray marks missing pixels; valid scores use a fixed opacity.
            rgba = cmap(norm(np.ma.masked_invalid(heatmap)))
            rgba[..., 3] = np.where(np.isfinite(heatmap), OVERLAY_ALPHA, 1.0)
            ax.imshow(rgba, extent=(0, 1, 1, 0), interpolation="nearest")
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
        ax.set_aspect(source.height / source.width)

    def boundaries(ax):
        width, height = geometry.resized_size
        left, top = geometry.paste_xy
        effect = [patheffects.Stroke(linewidth=2.0, foreground="black"), patheffects.Normal()]
        for col in range(1, geometry.grid[0]):
            x = (col * geometry.tile_size - left) / width
            if 0 < x < 1:
                ax.axvline(x, color="white", lw=0.8, ls="--", path_effects=effect)
        for row in range(1, geometry.grid[1]):
            y = (row * geometry.tile_size - top) / height
            if 0 < y < 1:
                ax.axhline(y, color="white", lw=0.8, ls="--", path_effects=effect)
        for index in range(geometry.tile_count):
            x0, y0, x1, y1 = geometry.tile_content_box(index)
            if x1 > x0 and y1 > y0:
                row, col = divmod(index, geometry.grid[0])
                x = (col * geometry.tile_size + x0 - left) / width
                y = (row * geometry.tile_size + y0 - top) / height
                ax.text(x + 0.01, y + 0.01, str(index + 1), va="top", fontsize=8,
                        color="white", bbox=dict(facecolor="black", alpha=0.7, pad=1, edgecolor="none"))

    def finish(fig, axes, name):
        fig.get_layout_engine().set(w_pad=0.12, h_pad=0.08)
        fig.suptitle(caption + "\nFeature score = -cosine similarity; higher = more dissimilar (not a defect probability)",
                     fontsize=10)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                     fraction=0.025, pad=0.015, label="Raw -cos score (all panels share this scale)")
        fig.supxlabel(f"{scale} scale [{limits[0]:.4f}, {limits[1]:.4f}] | "
                      "Gray: padding / no patch coverage / undefined cosine | "
                      f"Nearest token display; overlay opacity {OVERLAY_ALPHA}", fontsize=8)
        target = folder / name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        fig.savefig(target, dpi=150, facecolor="white")
        fig.clear()
        files.append(name)

    with rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig = Figure(figsize=(12, 14), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(STAGES), 4, squeeze=False)
        for row, label in enumerate(LABELS):
            panel(axes[row, 0], base)
            panel(axes[row, 1], image)
            boundaries(axes[row, 1])
            for col, kind in enumerate(("global", "local"), start=2):
                panel(axes[row, col], image, stitch_score_map(scores[kind][row], geometry))
                boundaries(axes[row, col])
            axes[row, 0].set_ylabel(label, fontsize=11)
        for ax, title in zip(axes[0], ("Base reference", "Original + tile IDs",
                                       "Global: Base mean", "Local: each tile mean")):
            ax.set_title(title)
        finish(fig, axes, "scores_overview.png")

        for start in range(0, geometry.tile_count, TILES_PER_PAGE):
            count = min(TILES_PER_PAGE, geometry.tile_count - start)
            fig = Figure(figsize=(max(9, count * 5), 13), layout="constrained")
            FigureCanvasAgg(fig)
            axes = fig.subplots(len(STAGES), count * 2, squeeze=False)
            for row, label in enumerate(LABELS):
                for offset in range(count):
                    index = start + offset
                    for j, kind in enumerate(("global", "local")):
                        ax = axes[row, offset * 2 + j]
                        panel(ax, tiles[index], tile_score_map(scores[kind][row, index], geometry, index))
                        if row == 0:
                            ax.set_title(f"Tile {index + 1} | {kind.title()}")
                axes[row, 0].set_ylabel(label, fontsize=11)
            finish(fig, axes, f"scores_tiles_{start // TILES_PER_PAGE + 1:02d}.png")
    return {"color_limits": list(limits), "figures": files}


def load_score_samples(input_json, data_root, limit=None):
    """Same image/ID conventions as run.py; masks, bbox and text are not needed."""
    from .data import _records, _resolve
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    records = _records(Path(input_json).expanduser())
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        records = records[:limit]
    result, seen = [], set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object")
        sample_id = str(record.get("question_id", record.get("id", "")))
        if not sample_id or sample_id in seen:
            raise ValueError(f"Empty or duplicate question_id: {sample_id!r}")
        seen.add(sample_id)
        if not isinstance(record.get("image"), str):
            raise ValueError(f"Record {index} needs an image path")
        gt = record.get("gt")
        if gt is not None and gt not in (0, 1):
            raise ValueError(f"Record {index} gt must be 0 or 1")
        result.append({"id": sample_id, "image": _resolve(root, record["image"], image=True), "gt": gt})
    if not result:
        raise ValueError("No images in input")
    return result


def _display_options(display_mode, color_scale):
    """Validate before model loading/writing; retain original token defaults."""
    if display_mode not in {"tokens", "patch-means"}:
        raise ValueError("display_mode must be tokens or patch-means")
    if color_scale is None:
        color_scale = "fixed" if display_mode == "patch-means" else "sample"
    if color_scale not in {"sample", "fixed"}:
        raise ValueError("Unknown color scale")
    metadata = {"display_mode": display_mode, "color_scale": color_scale}
    if display_mode == "patch-means":
        # Import before loading the checkpoint so a missing companion fails early.
        import visualize_patch_means as patch_vis
        metadata.update(displayed_stages=patch_vis.STAGES,
                        figure_layout=patch_vis.FIGURE_LAYOUT,
                        tile_mean_policy=patch_vis.MEAN_POLICY,
                        score_kind="global", score_normalization=patch_vis.SCORE_SCALING["method"],
                        score_scaling=patch_vis.SCORE_SCALING,
                        cross_layer_average=patch_vis.CROSS_LAYER_AVERAGE,
                        display_values="global_score_01")
    return color_scale, metadata


def save_sample(image, base, tiles, scores, shapes, geometry, sample, folder, scale=None,
                display_mode="tokens"):
    scale, display_options = _display_options(display_mode, scale)
    folder.mkdir(parents=True, exist_ok=False)
    gt = {0: "Normal", 1: "Abnormal"}.get(sample["gt"], "Unknown")
    # The numeric arrays are small token scores, never full hidden states or attention matrices.
    np.savez_compressed(folder / "scores.npz", global_scores=scores["global"],
                        local_scores=scores["local"], stages=np.asarray(STAGES),
                        feature_normalization=np.asarray(FEATURE_NORMALIZATION))
    if display_mode == "patch-means":
        import visualize_patch_means as patch_vis
        # Reuse the EXACT cache selection, averaging, scaling, and renderer used
        # offline. No extra encode_images() call, and no token-map PNGs generated.
        patch_geometry = patch_vis.validate_geometry({"geometry": asdict(geometry)})
        selected = patch_vis.load_scores(folder / "scores.npz", patch_geometry,
                                         {"score_shape": list(scores["global"].shape)})
        means, counts = patch_vis.aggregate_scores(selected)
        means, counts = patch_vis.build_display_means(means, counts)
        limits = patch_vis.render_figure(image, means, patch_geometry, sample["id"], sample["gt"],
                                         folder / "patch_means.png", scale)
        patch_vis.write_values(folder / "patch_means.csv", means, counts, patch_geometry,
                               selected["global"].shape[-1])
        display = {"figures": ["patch_means.png"], "color_limits": limits,
                   "values_file": "patch_means.csv"}
    else:
        display = draw_figures(image, base, tiles, scores, geometry, folder,
                              f"ID: {sample['id']} | GT: {gt} | No LLM prediction", scale)
    metadata = {"id": sample["id"], "image": str(sample["image"]), "gt": sample["gt"],
                "geometry": asdict(geometry), "feature_shapes": shapes,
                "score_shape": list(scores["global"].shape),
                "score_axes": ["stage", "tile_row_major", "token_row_major"],
                "feature_normalization": FEATURE_NORMALIZATION,
                **display_options,
                **display}
    with (folder / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


def run_score_visualization(model_path, input_json, data_root, output_dir,
                            limit=None, color_scale=None, display_mode="tokens"):
    color_scale, display_options = _display_options(display_mode, color_scale)
    import torch
    from .backend import LlavaBackend
    # Fail on missing plotting libraries before loading the large checkpoint.
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    samples = load_score_samples(input_json, data_root, limit)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one visible CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=5 python visualize_scores.py ...")
    print(f"Loading checkpoint once for {len(samples)} images; no pruning or LLM generation.", flush=True)
    # Reuse the original loader so dtype/weights are unchanged. Decoder weights
    # are loaded, but only model.encode_images() is ever executed by this script.
    backend = LlavaBackend(model_path, roi_mode="anyres_max_9")
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    if not isinstance(backend.vision_tower, SigLipVisionTower):
        raise ValueError("This probe requires the current SigLIP vision tower")
    vision = backend.vision_tower.vision_tower.vision_model
    if len(vision.encoder.layers) != 26:
        raise ValueError(f"Expected 26 active SigLIP blocks, found {len(vision.encoder.layers)}")
    patch_size = vision.embeddings.patch_embedding.kernel_size[0]
    projection_dtype = str(next(backend.model.get_model().mm_projector.parameters()).dtype)
    output.mkdir(parents=True, exist_ok=True)
    config = {"model_path": str(Path(model_path).expanduser().resolve()),
              "input_json": str(Path(input_json).expanduser().resolve()),
              "data_root": str(Path(data_root).expanduser().resolve()),
              "sample_count": len(samples), "limit": limit, "stages": list(STAGES),
              "global_score": "-cos(unit_tile_token, mean(all unit Base tokens))",
              "local_score": "-cos(unit_tile_token, mean(all unit tokens of the same tile))",
              "feature_normalization": FEATURE_NORMALIZATION,
              "mean_policy": "L2-normalize each token along channels, then average all encoded tokens, including padding-context tokens",
              "zero_norm_policy": "zero vectors remain zero in means; their cosine scores are NaN",
              "feature_location": "block outputs before post_layernorm; actual projector output",
              "spatial_policy": "before unpad/max_9 downsample/newlines; nearest patch support; padding gray",
              "undefined_cosine": "NaN (gray); denominator <= 1e-12",
              "color_scale": color_scale, "colormap": COLORMAP, "overlay_alpha": OVERLAY_ALPHA,
              "score_dtype": "float32", "projector_dtype": projection_dtype,
              "llm_generation": False, "pruning": False,
              **display_options,
              "loader_config": {k: v for k, v in backend.inference_config.items() if k != "attention_source"}}
    with (output / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    for index, sample in enumerate(samples, start=1):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sample["id"])[:80]
        folder = output / f"{index:06d}_{safe_id}"
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB")
        pixels, base, tiles, geometry = prepare_views(
            image, backend.processor, backend.model.config, patch_size)
        pixels = pixels.to(device=backend.model.device, dtype=torch.float16)
        scores, shapes = capture_scores(backend.model, pixels)
        save_sample(image, base, tiles, scores, shapes, geometry, sample, folder, color_scale,
                    display_mode=display_mode)
        print(f"[{index}/{len(samples)}] {sample['id']}: {geometry.tile_count} tiles, saved {folder}", flush=True)
    print(f"Saved all feature-score visualizations to {output}", flush=True)
    return output
