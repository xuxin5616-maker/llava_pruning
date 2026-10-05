"""Project pruning decisions onto the source image for SigLip image packing.

This module needs only Pillow and NumPy. Its ROI, anyres and Base-copy layouts
mirror the corresponding llava_arch.py branches. Patch coordinates describe
spatial anchors, not the full receptive field of contextual tokens.
"""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .metrics import OPTION


def prediction_caption(ground_truth, answer):
    """Use the evaluation's A/B rule, never guess a class from free-form text."""
    truth = {0: "Normal", 1: "Abnormal"}.get(ground_truth, "Unknown")
    match = OPTION.match(answer) if isinstance(answer, str) else None
    predicted = ("Abnormal" if match.group(1).upper() == "A" else "Normal") if match else "Unparsed"
    return f"GT: {truth} | Pred: {predicted}"


@lru_cache(maxsize=1)
def _prediction_font():
    # Pillow 10.4 ships this font; no OS-specific font path is required.
    return ImageFont.load_default(size=24)


def _add_prediction_header(panel, caption):
    """Prepend a final-result label without covering or rescaling the figure."""
    if caption is None:
        return panel
    font = _prediction_font()
    left, top, right, bottom = font.getbbox(caption)
    padding = 12
    header_height = bottom - top + 2 * padding
    width = max(panel.width, right - left + 2 * padding)
    canvas = Image.new("RGB", (width, panel.height + header_height), "white")
    canvas.paste(panel, (0, header_height))
    ImageDraw.Draw(canvas).text((padding - left, padding - top), caption, font=font, fill="black")
    return canvas


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
    order = pro_data.get("image_token_order", "base_first")
    if order not in {"base_first", "anyres_first"}:
        raise ValueError(f"Unknown image token order: {order}")
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
    anyres_first = order == "anyres_first"
    # Without an extra final newline, move the final row newline behind base.
    moved_row_newline = anyres_first and not pro_data.get("final_newline", False)
    global_offset = rows * (cols + 1) - int(moved_row_newline) if anyres_first else 0
    return [
        {
            "name": "global", "source_box_xyxy": [0, 0, width, height],
            "processed_size": [base_grid * patch_size, base_grid * patch_size],
            "grid_shape": [base_grid, base_grid], "pool_factor": 1,
            "cell_size_processed_pixels": patch_size,
            "token_offset": global_offset, "token_count": global_count,
            "token_row_stride": base_grid, "projection": "global",
        },
        {
            "name": "anyres", "source_box_xyxy": [0, 0, width, height],
            "processed_size": [cols * patch_size, rows * patch_size],
            "grid_shape": [rows, cols], "pool_factor": None,
            "cell_size_processed_pixels": patch_size,
            "token_offset": 0 if anyres_first else global_count, "token_count": rows * cols,
            "token_row_stride": cols + 1, "projection": "anyres",
            "grid_patches": [grid_width, grid_height],
            "row_newline_count": rows,
        },
    ]


def build_ex_base_copy_layout(pro_data, base_grid, patch_size):
    """Three complete Base grids, followed by one structural newline."""
    width, height = map(int, pro_data["original_size"])
    sizes = pro_data.get("processed_view_sizes", [])
    if (min(width, height, base_grid, patch_size) <= 0
            or pro_data.get("base_view_count") != 3 or len(sizes) != 3
            or pro_data.get("image_token_order", "base_first") != "base_first"
            or pro_data.get("final_newline") is not True):
        raise ValueError("Invalid ex_base_copy metadata; expected three Base views and one final newline")
    if any(list(size) != list(sizes[0]) for size in sizes):
        raise ValueError("ex_base_copy views must have identical processed sizes")
    count = base_grid * base_grid
    layouts = []
    for index, (model_width, model_height) in enumerate(sizes):
        if model_width // patch_size != base_grid or model_height // patch_size != base_grid:
            raise ValueError("Unexpected ex_base_copy vision token grid")
        layouts.append({
            "name": f"Base {index + 1}", "projection": "base_copy",
            "source_box_xyxy": [0, 0, width, height],
            "processed_size": [model_width, model_height],
            "grid_shape": [base_grid, base_grid], "pool_factor": 1,
            "cell_size_processed_pixels": patch_size,
            "token_offset": index * count, "token_count": count,
        })
    return layouts


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
    if pro_data.get("mode") == "ex_base_copy":
        layouts = build_ex_base_copy_layout(pro_data, base_grid, patch_size)
    else:
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
            "kept_patch_tokens": int(view_keep.sum()),
            "pruned_patch_tokens": int((~view_keep).sum()),
            "prune_rate_percent": 100.0 * int((~view_keep).sum()) / view_keep.size,
        })
        if layout.get("projection") == "anyres":
            rows, cols = layout["grid_shape"]
            newline_indices = layout["token_offset"] + np.arange(rows) * (cols + 1) + cols
            if pro_data.get("image_token_order", "base_first") == "anyres_first" and not final_newline:
                newline_indices[-1] = sequence_tokens - 1
            view_info["row_newline_kept"] = keep[newline_indices].tolist()
        views.append({"metadata": view_info, "pruned": pruned, "covered": covered})
    return {
        "views": views,
        "combined_pruned": covered_any & ~kept_any,
        "covered": covered_any,
        "patch_tokens": patch_count,
        "sequence_tokens": sequence_tokens,
        "pruned_patch_tokens": pruned_patches,
        "pruned_sequence_tokens": int((~keep).sum()),
        "newline_tokens": sequence_tokens - patch_count,
        "pruned_newline_tokens": int((~keep).sum()) - pruned_patches,
        "image_newline_kept": bool(keep[-1]) if final_newline else None,
    }


def build_attention_overlay(original, image_attention, image_mask, prepared):
    """Show source views above their ranking-attention overlays, as in reference."""
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
    valid = np.asarray(image_attention.get("valid", np.ones(scores.size, dtype=bool)), dtype=bool)
    if valid.shape != scores.shape:
        raise ValueError("Attention validity mask does not match original token coordinates")
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
        heat_width, heat_height = width, height
        if layout.get("projection") == "base_copy":
            # 384 / patch14 gives 27*14=378 pixels of convolution support.
            # Do not stretch these anchors into the uncovered 6px margins.
            cell = layout["cell_size_processed_pixels"]
            heat_width, heat_height = cols * cell, rows * cell
        heat = np.asarray(heat.resize((heat_width, heat_height), Image.Resampling.BILINEAR))
        heat = np.clip(heat, 0.0, 1.0)
        # JET-style map and the same 60/40 image/heat blend used by reference visualization.
        color = np.stack([
            np.clip(1.5 - np.abs(4 * heat - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * heat - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * heat - 1), 0, 1),
        ], axis=-1) * 255
        overlay = np.array(source, copy=True)
        covered_overlay = np.clip(0.6 * overlay[:heat_height, :heat_width] + 0.4 * color,
                                  0, 255).astype(np.uint8)
        # A token deleted at an earlier stage has NO current attention score.
        # Gray it out rather than misrepresenting its missing value as low heat.
        if not valid[indices].all():
            active = Image.fromarray((valid[indices].reshape(rows, cols) * 255).astype(np.uint8))
            active = np.asarray(active.resize((heat_width, heat_height), Image.Resampling.NEAREST)) > 0
            covered_overlay[~active] = 96
        overlay[:heat_height, :heat_width] = covered_overlay
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


def _view_count_summaries(prepared):
    """Persist scalar results only; keep spatial masks/indices in memory for drawing."""
    fields = ("name", "token_count", "kept_patch_tokens", "pruned_patch_tokens", "prune_rate_percent")
    return [{key: view["metadata"][key] for key in fields} for view in prepared["views"]]


def build_anyres_comparison(original, prepared):
    """Show both independent views before their overlapping-coverage union.

    Counts use the original spatial token masks, not the rasterized black area.
    Structural newline tokens have no spatial anchors and are reported separately.
    """
    global_view, anyres_view = prepared["views"]
    if (global_view["metadata"].get("projection") != "global"
            or anyres_view["metadata"].get("projection") != "anyres"):
        raise ValueError("Separate anyres comparison requires global and anyres views")
    return _build_separate_view_comparison(original, prepared,
                                          ("Global view", "High-resolution (anyres)"))


def build_ex_base_copy_comparison(original, prepared):
    """Show each copy's decisions separately, not only overlapping coverage."""
    names = tuple(view["metadata"]["name"] for view in prepared["views"])
    if names != ("Base 1", "Base 2", "Base 3"):
        raise ValueError("Base-copy comparison requires exactly three Base views")
    return _build_separate_view_comparison(original, prepared, names, max_panel_width=384)


def _build_separate_view_comparison(original, prepared, names, *, max_panel_width=480):

    def view_caption(name, view):
        info = view["metadata"]
        return (f"{name}\n"
                f"Pruned: {info['pruned_patch_tokens']}/{info['token_count']} "
                f"({info['prune_rate_percent']:.2f}%)\n"
                "Spatial tokens only; black = pruned")

    captions = ["Original\nUnmodified source"]
    captions.extend(view_caption(name, view) for name, view in zip(names, prepared["views"]))
    captions.append("Combined (overlap)\nBlack only if ALL views prune\nNot the token pruning rate")
    masks = [None] + [view["pruned"] for view in prepared["views"]] + [prepared["combined_pruned"]]
    width, height = original.size
    # Keep labels legible even for small test/input images; never upscale source pixels.
    panel_width = max(280, min(max_panel_width, width))
    image_width = min(width, panel_width)
    image_height = max(1, round(height * image_width / width))
    header_height, footer_height = 62, 46
    canvas = Image.new("RGB", (panel_width * len(captions), header_height + image_height + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    source = original.convert("RGB").resize((image_width, image_height), Image.Resampling.BILINEAR)
    for index, (caption, mask) in enumerate(zip(captions, masks)):
        draw.multiline_text((index * panel_width + 6, 6), caption, fill="black", spacing=4)
        panel = source
        if mask is not None:
            # Resize the binary mask separately so interpolation cannot blur removed cells.
            display_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                source.size, Image.Resampling.NEAREST)
            panel = _blackout(source, np.asarray(display_mask) != 0)
        canvas.paste(panel, (index * panel_width + (panel_width - image_width) // 2, header_height))
    removed, total = prepared["pruned_sequence_tokens"], prepared["sequence_tokens"]
    footer = (f"Token totals: pruned {removed}/{total} ({100.0 * removed / total:.2f}%); "
              f"spatial {prepared['pruned_patch_tokens']}/{prepared['patch_tokens']}; "
              f"newlines {prepared['pruned_newline_tokens']}/{prepared['newline_tokens']} (not drawn).\n"
              "Per-view rates use each view's spatial tokens. Combined black area is NOT the token pruning rate.")
    draw.multiline_text((6, header_height + image_height + 6), footer, fill="black", spacing=4)
    return canvas


def save_fastv_visualizations(original_images, pro_datas, image_masks, *,
                              base_grid, patch_size, output_dir, sample_id,
                              fastv_layer, keep_ratio, image_attentions=None,
                              save_prune=True, save_attention=False, prediction_label=None):
    """Save only comparison/attention PNGs and compact pruning-count summaries.

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
        mode = pro_datas[image_index].get("mode")
        if save_prune and mode in {"anyres_max_9", "ex_base_copy"}:
            order_label = "; order=anyres_first" if pro_datas[image_index].get("image_token_order") == "anyres_first" else ""
            build_comparison = build_ex_base_copy_comparison if mode == "ex_base_copy" else build_anyres_comparison
            comparison = _stack_stage_panels([
                (f"FastV layer index={fastv_layer}, keep_ratio={keep_ratio}{order_label}\n"
                 "Spatial token anchors, not exact pixel information removal.",
                 build_comparison(original, masks))
            ])
        elif save_prune:
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
        if save_prune:
            _add_prediction_header(comparison, prediction_label).save(directory / "comparison.png")
        if overlays is not None:
            _add_prediction_header(overlays[image_index], prediction_label).save(directory / "attention_overlay.png")

        metadata = {
            "sample_id": None if sample_id is None else str(sample_id),
            "image_index": image_index,
            "method": "fastv",
            "image_token_order": pro_datas[image_index].get("image_token_order", "base_first"),
            "fastv_layer": fastv_layer,
            "keep_ratio": keep_ratio,
            "patch_tokens": masks["patch_tokens"],
            "pruned_patch_tokens": masks["pruned_patch_tokens"],
            "sequence_tokens": masks["sequence_tokens"],
            "pruned_sequence_tokens": masks["pruned_sequence_tokens"],
            "newline_tokens": masks["newline_tokens"],
            "pruned_newline_tokens": masks["pruned_newline_tokens"],
            "views": _view_count_summaries(masks),
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


def _stack_stage_panels(panels, *, max_width=1920):
    """Compose labelled stage rows without writing any intermediate images."""
    resized = []
    for title, panel in panels:
        if panel.width > max_width:
            panel = panel.resize((max_width, max(1, round(panel.height * max_width / panel.width))),
                                 Image.Resampling.LANCZOS)
        resized.append((title, panel))
    header, gap = 52, 12
    width = max(1000, max(panel.width for _, panel in resized))
    height = sum(header + panel.height + gap for _, panel in resized)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for title, panel in resized:
        draw.multiline_text((8, y + 6), title, fill="black", spacing=4)
        canvas.paste(panel, (0, y + header))
        y += header + panel.height + gap
    return canvas


def save_vico_visualizations(original, pro_data, stages, *, layer_stats,
                             base_grid, patch_size, output_dir, sample_id,
                             save_prune=True, save_attention=False, prediction_label=None):
    """Two multi-stage PNGs in original token coordinates, plus compact counts.

    Each row corresponds to an actual pruning boundary. Scores are PRE-drop
    at that boundary, decisions are POST-drop. Previously removed anchors have
    no scores and are gray in the attention panel, not filled with stale heat.
    """
    if not stages or not (save_prune or save_attention):
        raise ValueError("ViCo visualization requires captured stages and an enabled image type")
    comparisons, attentions, decisions = [], [], []
    previous_keep = None
    for stage in stages:
        mask = stage["mask"]
        prepared = prepare_image_masks(original.size, pro_data, mask, base_grid, patch_size)
        keep = np.asarray(mask["keep"], dtype=bool)
        if previous_keep is not None and np.any(keep & ~previous_keep):
            raise ValueError("ViCo visualization cannot restore previously pruned tokens")
        if int(keep.sum()) != stage["image_tokens_after"]:
            raise ValueError("ViCo stage token count disagrees with its cumulative mask")
        previous_keep = keep
        title = (f"ViCo - after layer {stage['after_layer']} | visual tokens: "
                 f"{stage['image_tokens_before']} -> {stage['image_tokens_after']} | "
                 f"removed now: {stage['removed_this_stage']} | "
                 f"cumulative removed: {stage['cumulative_prune_rate']:.2f}%")
        if pro_data.get("image_token_order") == "anyres_first":
            title += " | order=anyres_first"
        mode = pro_data.get("mode")
        if save_prune and mode in {"anyres_max_9", "ex_base_copy"}:
            build_comparison = build_ex_base_copy_comparison if mode == "ex_base_copy" else build_anyres_comparison
            row = build_comparison(original, prepared)
            comparisons.append((title + "\nCumulative decisions; spatial anchors, not exact pixel information removal.", row))
        elif save_prune:
            width, height = original.size
            panel_width = min(640, width)
            panel_height = max(1, round(height * panel_width / width))
            row = Image.new("RGB", (panel_width * 3, panel_height + 26), "white")
            draw = ImageDraw.Draw(row)
            panels = (original, _blackout(original, prepared["views"][0]["pruned"]),
                      _blackout(original, prepared["combined_pruned"]))
            captions = ("Original", "Global: pruned = black", "Combined: any kept view visible")
            for index, (panel, caption) in enumerate(zip(panels, captions)):
                draw.text((index * panel_width + 4, 4), caption, fill="black")
                row.paste(panel.resize((panel_width, panel_height)), (index * panel_width, 26))
            comparisons.append((title + "\nCumulative decisions; spatial anchors, not exact pixel information removal.", row))
        if save_attention:
            if "attention" not in stage:
                raise ValueError("ViCo attention images require captured independent ranking scores")
            row = build_attention_overlay(original, stage["attention"], mask, prepared)
            attentions.append((title + "\nPRE-drop ranking attention; gray = removed BEFORE this stage; scale normalized per stage.", row))
        # Whitelist scalar settings/results; never serialize stage masks or scores.
        stage_fields = ("after_layer", "scoring_layer", "target_keep_ratio", "image_tokens_before",
                        "image_tokens_after", "removed_this_stage", "cumulative_prune_rate")
        summary = {key: stage[key] for key in stage_fields if key in stage}
        summary.update({key: prepared[key] for key in (
            "sequence_tokens", "pruned_sequence_tokens", "patch_tokens", "pruned_patch_tokens",
            "newline_tokens", "pruned_newline_tokens")})
        summary["views"] = _view_count_summaries(prepared)
        decisions.append(summary)
    # Validate and build all panels before creating any output files.
    comparison = _stack_stage_panels(comparisons) if save_prune else None
    attention = _stack_stage_panels(attentions) if save_attention else None
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(sample_id or "sample"))[:64] or "sample"
    base_dir = Path(output_dir) / f"sample_{slug}"
    call_dir, suffix = base_dir, 2
    while True:
        try:
            call_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            call_dir = base_dir.with_name(f"{base_dir.name}_{suffix}")
            suffix += 1
    directory = call_dir / "image_0"
    directory.mkdir()
    if comparison is not None:
        _add_prediction_header(comparison, prediction_label).save(directory / "comparison.png")
    if attention is not None:
        _add_prediction_header(attention, prediction_label).save(directory / "attention_overlay.png")
    metadata = {"sample_id": str(sample_id), "method": "vico", "stages": decisions,
                "image_token_order": pro_data.get("image_token_order", "base_first")}
    (directory / "decisions.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return [{"image_index": 0, "directory": str(directory.resolve()),
             "comparison": str((directory / "comparison.png").resolve()) if save_prune else None,
             "attention_overlay": str((directory / "attention_overlay.png").resolve()) if save_attention else None}]
