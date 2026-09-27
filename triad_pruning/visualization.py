"""Project FastV decisions onto the source image for SigLip image packing.

This module needs only Pillow and NumPy. Its randomroi and pure-anyres layouts
mirror the corresponding llava_arch.py branches. Patch coordinates describe
spatial anchors, not the full receptive field of contextual tokens.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def build_randomroi_layout(pro_data, base_grid, patch_size):
    """Return row-major view layouts, with half-open source-image boxes."""
    required = {"original_size", "roi_boxes", "processed_view_sizes"}
    if not required.issubset(pro_data):
        raise ValueError("Missing randomroi crop metadata; use the updated mm_utils.py")
    width, height = map(int, pro_data["original_size"])
    if min(width, height, base_grid, patch_size) <= 0:
        raise ValueError("Image and patch dimensions must be positive")
    boxes = [[0, 0, width, height]] + pro_data["roi_boxes"]
    sizes = pro_data["processed_view_sizes"]
    if len(sizes) != len(boxes):
        raise ValueError("Crop count does not match processed view count")
    roi_count = len(boxes) - 1
    roi_pool = 1 if roi_count <= 1 else (2 if roi_count <= 4 else 3)
    layouts = []
    offset = 0
    for view_index, (box, size) in enumerate(zip(boxes, sizes)):
        box = list(map(int, box))
        model_width, model_height = map(int, size)
        if (box[2] <= box[0] or box[3] <= box[1]
                or model_width // patch_size != base_grid
                or model_height // patch_size != base_grid):
            raise ValueError("Invalid crop or unexpected vision token grid")
        pool = 1 if view_index == 0 else roi_pool
        grid = base_grid // pool
        if grid < 1:
            raise ValueError("Pooling leaves an empty ROI token grid")
        count = grid * grid
        layouts.append({
            "name": "global" if view_index == 0 else f"roi_{view_index - 1}",
            "source_box_xyxy": box,
            "processed_size": [model_width, model_height],
            "grid_shape": [grid, grid],
            "pool_factor": pool,
            "cell_size_processed_pixels": patch_size * pool,
            "token_offset": offset,
            "token_count": count,
        })
        offset += count
    return layouts


def build_anyres_layout(pro_data, base_grid, patch_size):
    """Mirror the anyres_max_9 unpad/downsample/token-order path in llava_arch."""
    width, height = map(int, pro_data["original_size"])
    grid_width, grid_height = map(int, pro_data["grid_patches"])
    if min(width, height, grid_width, grid_height, base_grid, patch_size) <= 0:
        raise ValueError("Invalid anyres image or patch grid")
    rows, cols = grid_height * base_grid, grid_width * base_grid
    if width / height > cols / rows:
        new_height = int(height * cols / width)
        padding = (rows - new_height) // 2
        rows -= 2 * padding
    else:
        new_width = int(width * rows / height)
        padding = (cols - new_width) // 2
        cols -= 2 * padding
    if min(rows, cols) <= 0:
        raise ValueError("Anyres unpadding produced an empty token grid")
    scale = math.sqrt(rows * cols / (9 * base_grid * base_grid))
    if scale > 1.1:
        rows, cols = int(rows // scale), int(cols // scale)
    if min(rows, cols) <= 0:
        raise ValueError("Anyres downsampling produced an empty token grid")
    global_count = base_grid * base_grid
    return [
        {
            "name": "global", "source_box_xyxy": [0, 0, width, height],
            "processed_size": [base_grid * patch_size, base_grid * patch_size],
            "grid_shape": [base_grid, base_grid], "pool_factor": 1,
            "cell_size_processed_pixels": patch_size,
            "token_offset": 0, "token_count": global_count,
            "token_row_stride": base_grid, "projection": "global",
        },
        {
            "name": "anyres", "source_box_xyxy": [0, 0, width, height],
            "processed_size": [cols * patch_size, rows * patch_size],
            "grid_shape": [rows, cols], "pool_factor": None,
            "cell_size_processed_pixels": patch_size,
            "token_offset": global_count, "token_count": rows * cols,
            "token_row_stride": cols + 1, "projection": "anyres",
            "grid_patches": [grid_width, grid_height],
            "row_newline_count": rows,
        },
    ]


def _spatial_indices(layout):
    rows, cols = layout["grid_shape"]
    stride = layout.get("token_row_stride", cols)
    return (layout["token_offset"] +
            np.arange(rows)[:, None] * stride + np.arange(cols)[None, :]).reshape(-1)


def project_view_mask(original_size, layout, keep):
    """Rasterize patch anchors using source-pixel centers.

    Returns (pruned, covered), both boolean arrays in original-image size.
    Pixels outside the crop or outside valid convolution/pooling support are
    not counted as pruned. In particular, 27x27 pooled by 2 becomes 13x13;
    it must not be stretched to fill the discarded last row and column.
    """
    width, height = map(int, original_size)
    rows, cols = layout["grid_shape"]
    keep = np.asarray(keep, dtype=bool)
    if keep.size != rows * cols:
        raise ValueError("Keep mask length does not match this view's grid")
    keep = keep.reshape(rows, cols)
    if layout.get("projection") == "anyres":
        xs = np.minimum(((np.arange(width) + 0.5) * cols / width).astype(int), cols - 1)
        ys = np.minimum(((np.arange(height) + 0.5) * rows / height).astype(int), rows - 1)
        return ~keep[ys[:, None], xs[None, :]], np.ones((height, width), dtype=bool)
    covered = np.zeros((height, width), dtype=bool)
    pruned = np.zeros_like(covered)
    x0, y0, x1, y1 = layout["source_box_xyxy"]
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(width, x1), min(height, y1)
    if left >= right or top >= bottom:
        return pruned, covered
    model_width, model_height = layout["processed_size"]
    cell = layout["cell_size_processed_pixels"]
    xs = np.floor((np.arange(left, right) + 0.5 - x0)
                  * model_width / (x1 - x0) / cell).astype(int)
    ys = np.floor((np.arange(top, bottom) + 0.5 - y0)
                  * model_height / (y1 - y0) / cell).astype(int)
    valid = (ys[:, None] < rows) & (xs[None, :] < cols)
    decisions = keep[np.minimum(ys, rows - 1)[:, None],
                     np.minimum(xs, cols - 1)[None, :]]
    covered[top:bottom, left:right] = valid
    pruned[top:bottom, left:right] = valid & ~decisions
    return pruned, covered


def prepare_image_masks(original_size, pro_data, image_mask, base_grid, patch_size):
    """Split a packed image mask and combine overlapping views explicitly."""
    if list(original_size) != list(pro_data.get("original_size", [])):
        raise ValueError("Original image size disagrees with preprocessing metadata")
    anyres = pro_data.get("mode") == "anyres_max_9"
    layouts = (build_anyres_layout(pro_data, base_grid, patch_size) if anyres
               else build_randomroi_layout(pro_data, base_grid, patch_size))
    keep = np.asarray(image_mask["keep"], dtype=bool)
    start, end = image_mask["span"]
    patch_count = sum(view["token_count"] for view in layouts)
    # Anyres adds one row-newline token per unpadded row; add_newl is optional.
    row_newlines = layouts[1]["grid_shape"][0] if anyres else 0
    final_newline = not anyres or bool(pro_data.get("final_newline", False))
    sequence_tokens = patch_count + row_newlines + int(final_newline)
    if keep.ndim != 1 or len(keep) != sequence_tokens or end - start != len(keep):
        raise ValueError(
            f"Expected {patch_count} patch tokens + {row_newlines + int(final_newline)} newlines, got {keep.size}. "
            "Image-token truncation or a different merge mode cannot be visualized safely."
        )
    width, height = original_size
    covered_any = np.zeros((height, width), dtype=bool)
    kept_any = np.zeros_like(covered_any)
    views = []
    pruned_patches = 0
    for layout in layouts:
        indices = _spatial_indices(layout)
        view_keep = keep[indices]
        pruned_patches += int((~view_keep).sum())
        pruned, covered = project_view_mask(original_size, layout, view_keep)
        covered_any |= covered
        kept_any |= covered & ~pruned
        view_info = dict(layout)
        view_info.update({
            "sequence_span": [start + layout["token_offset"],
                              start + int(indices[-1]) + 1],
            "kept_local_indices": np.flatnonzero(view_keep).tolist(),
            "pruned_local_indices": np.flatnonzero(~view_keep).tolist(),
        })
        if layout.get("projection") == "anyres":
            rows, cols = layout["grid_shape"]
            newline_indices = layout["token_offset"] + np.arange(rows) * (cols + 1) + cols
            view_info["row_newline_kept"] = keep[newline_indices].tolist()
        views.append({"metadata": view_info, "pruned": pruned, "covered": covered})
    return {
        "views": views,
        "combined_pruned": covered_any & ~kept_any,
        "covered": covered_any,
        "patch_tokens": patch_count,
        "sequence_tokens": sequence_tokens,
        "pruned_patch_tokens": pruned_patches,
        "image_newline_kept": bool(keep[-1]) if final_newline else None,
    }


def build_attention_overlay(original, image_attention, image_mask, prepared):
    """Show source views above their ranking-attention overlays, as in Triad."""
    scores = np.asarray(image_attention["scores"], dtype=np.float32)
    if (image_attention["span"] != image_mask["span"]
            or scores.ndim != 1
            or scores.size != prepared["sequence_tokens"]
            or not np.isfinite(scores).all()):
        raise ValueError("Ranking attention does not match the image token layout")
    # Structural newline tokens have no spatial cells.
    spatial_indices = np.concatenate([_spatial_indices(view["metadata"])
                                      for view in prepared["views"]])
    patch_scores = np.maximum(scores[spatial_indices], 0)
    maximum = float(patch_scores.max()) if patch_scores.size else 0.0
    scale = maximum if maximum > 0 else 1.0
    columns = []
    for view in prepared["views"]:
        layout = view["metadata"]
        width, height = layout["processed_size"]
        source = original.convert("RGB").crop(layout["source_box_xyxy"])
        source = source.resize((width, height), Image.Resampling.BILINEAR)
        indices = _spatial_indices(layout)
        rows, cols = layout["grid_shape"]
        grid = (np.maximum(scores[indices], 0) / scale).reshape(rows, cols)
        heat = Image.fromarray(grid.astype(np.float32), mode="F")
        heat = np.asarray(heat.resize((width, height), Image.Resampling.BILINEAR))
        heat = np.clip(heat, 0.0, 1.0)
        # JET-style map and the same 60/40 image/heat blend used by Triad-main.
        color = np.stack([
            np.clip(1.5 - np.abs(4 * heat - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * heat - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * heat - 1), 0, 1),
        ], axis=-1) * 255
        overlay = np.clip(0.6 * np.asarray(source) + 0.4 * color, 0, 255).astype(np.uint8)
        columns.append((layout["name"], source, Image.fromarray(overlay)))

    margin = 8
    label_height = 26
    canvas_width = sum(source.width for _, source, _ in columns) + margin * (len(columns) - 1)
    canvas_height = label_height + max(source.height for _, source, _ in columns) * 2 + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    overlay_y = label_height + max(source.height for _, source, _ in columns) + margin
    for name, source, overlay in columns:
        draw.text((x + 4, 5), name, fill="black")
        canvas.paste(source, (x, label_height))
        canvas.paste(overlay, (x, overlay_y))
        x += source.width + margin
    return canvas


def _blackout(image, pruned):
    pixels = np.array(image.convert("RGB"), copy=True)
    pixels[pruned] = 0
    return Image.fromarray(pixels)


def save_fastv_visualizations(original_images, pro_datas, image_masks, *,
                              base_grid, patch_size, output_dir, sample_id,
                              fastv_layer, keep_ratio, image_attentions=None,
                              save_prune=True, save_attention=False):
    """Save only comparison/attention PNGs and auditable decision metadata.

    Repeated samples/QA rounds receive numbered directories, so earlier results
    are preserved. Return paths for the existing stats JSONL.
    """
    if not original_images or not (len(original_images) == len(pro_datas) == len(image_masks)):
        raise ValueError("Images, crop metadata and FastV image masks must match one-to-one")
    if not (save_prune or save_attention):
        raise ValueError("At least one visualization type must be enabled")
    if save_attention and image_attentions is None:
        raise ValueError("Attention visualization requires ranking attention")
    if image_attentions is not None and len(image_attentions) != len(image_masks):
        raise ValueError("Ranking attention and image masks must match one-to-one")
    # Validate all image layouts before writing partial output.
    prepared = [
        prepare_image_masks(image.size, metadata, mask, base_grid, patch_size)
        for image, metadata, mask in zip(original_images, pro_datas, image_masks)
    ]
    overlays = (
        [build_attention_overlay(image, attention, mask, masks)
         for image, attention, mask, masks in zip(original_images, image_attentions, image_masks, prepared)]
        if save_attention else None
    )
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(sample_id or "sample"))[:64] or "sample"
    base_dir = Path(output_dir) / f"sample_{slug}"
    call_dir = base_dir
    suffix = 2
    while True:
        try:
            call_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            call_dir = base_dir.with_name(f"{base_dir.name}_{suffix}")
            suffix += 1
    paths = []
    for image_index, (image, image_mask, masks) in enumerate(zip(original_images, image_masks, prepared)):
        directory = call_dir / f"image_{image_index}"
        directory.mkdir(parents=True, exist_ok=False)
        original = image.convert("RGB")
        if save_prune:
            combined = _blackout(original, masks["combined_pruned"])
            global_blackout = _blackout(original, masks["views"][0]["pruned"])
            width, height = original.size
            panel_width = min(640, width)
            panel_height = max(1, round(height * panel_width / width))
            header_height = 64
            comparison = Image.new("RGB", (panel_width * 3, panel_height + header_height), "white")
            draw = ImageDraw.Draw(comparison)
            captions = ["Original", "Global view: pruned = black", "Combined: any kept view stays visible"]
            for index, (panel, caption) in enumerate(zip([original, global_blackout, combined], captions)):
                panel = panel.resize((panel_width, panel_height))
                comparison.paste(panel, (index * panel_width, header_height))
                draw.text((index * panel_width + 6, 5), caption, fill="black")
            draw.text((6, 25), f"FastV layer index={fastv_layer}, keep_ratio={keep_ratio}; "
                      f"pruned patches={masks['pruned_patch_tokens']}/{masks['patch_tokens']}", fill="black")
            draw.text((6, 43), "Uncovered margins unchanged. Black marks pruned spatial token anchors.", fill="black")
            comparison.save(directory / "comparison.png")
        if overlays is not None:
            overlays[image_index].save(directory / "attention_overlay.png")

        metadata = {
            "sample_id": None if sample_id is None else str(sample_id),
            "image_index": image_index,
            "original_size": list(original.size),
            "fastv_layer": fastv_layer,
            "keep_ratio": keep_ratio,
            "image_span": image_mask["span"],
            "patch_tokens": masks["patch_tokens"],
            "pruned_patch_tokens": masks["pruned_patch_tokens"],
            "image_newline_kept": masks["image_newline_kept"],
            "combined_rule": "black iff covered by a patch anchor and no covering view kept it",
            "uncovered_pixels": "unchanged; not classified as FastV pruning",
            "coordinate_convention": "half-open xyxy in original pixels; row-major token grids; pixel-center rasterization",
            "interpretation": "spatial token anchors, not exact pixel information removal or receptive fields",
            "attention_overlay": "ranking layer; last valid prompt token to image keys; shared scale across views; source above 60/40 JET overlay" if overlays is not None else None,
            "views": [view["metadata"] for view in masks["views"]],
        }
        with (directory / "decisions.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        paths.append({
            "image_index": image_index,
            "directory": str(directory.resolve()),
            "comparison": str((directory / "comparison.png").resolve()) if save_prune else None,
            "attention_overlay": str((directory / "attention_overlay.png").resolve()) if overlays is not None else None,
        })
    return paths
