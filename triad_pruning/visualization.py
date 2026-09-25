"""Project actual FastV decisions onto the source image (SigLip randomroi).

This module needs only Pillow and NumPy. The token layout mirrors
llava_arch.py's spatial_avgpool_auto_unpad_add_newl branch. Patch coordinates
describe spatial anchors, not the full receptive field of contextual tokens.
"""

import json
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
    layouts = build_randomroi_layout(pro_data, base_grid, patch_size)
    keep = np.asarray(image_mask["keep"], dtype=bool)
    start, end = image_mask["span"]
    patch_count = sum(view["token_count"] for view in layouts)
    # The supported merge mode appends exactly one structural newline token.
    if keep.ndim != 1 or len(keep) != patch_count + 1 or end - start != len(keep):
        raise ValueError(
            f"Expected {patch_count} patch tokens + 1 newline, got {keep.size}. "
            "Image-token truncation or a different merge mode cannot be visualized safely."
        )
    width, height = original_size
    covered_any = np.zeros((height, width), dtype=bool)
    kept_any = np.zeros_like(covered_any)
    views = []
    for layout in layouts:
        offset, count = layout["token_offset"], layout["token_count"]
        view_keep = keep[offset:offset + count]
        pruned, covered = project_view_mask(original_size, layout, view_keep)
        covered_any |= covered
        kept_any |= covered & ~pruned
        view_info = dict(layout)
        view_info.update({
            "sequence_span": [start + offset, start + offset + count],
            "kept_local_indices": np.flatnonzero(view_keep).tolist(),
            "pruned_local_indices": np.flatnonzero(~view_keep).tolist(),
        })
        views.append({"metadata": view_info, "pruned": pruned, "covered": covered})
    return {
        "views": views,
        "combined_pruned": covered_any & ~kept_any,
        "covered": covered_any,
        "patch_tokens": patch_count,
        "pruned_patch_tokens": int((~keep[:-1]).sum()),
        "image_newline_kept": bool(keep[-1]),
    }


def build_attention_overlay(original, image_attention, image_mask, prepared):
    """Show source views above their ranking-attention overlays, as in Triad."""
    scores = np.asarray(image_attention["scores"], dtype=np.float32)
    if (image_attention["span"] != image_mask["span"]
            or scores.ndim != 1
            or scores.size != prepared["patch_tokens"] + 1
            or not np.isfinite(scores).all()):
        raise ValueError("Ranking attention does not match the image token layout")
    # The final image-newline token has no spatial cell.
    patch_scores = np.maximum(scores[:-1], 0)
    maximum = float(patch_scores.max()) if patch_scores.size else 0.0
    scale = maximum if maximum > 0 else 1.0
    columns = []
    for view in prepared["views"]:
        layout = view["metadata"]
        width, height = layout["processed_size"]
        source = original.convert("RGB").crop(layout["source_box_xyxy"])
        source = source.resize((width, height), Image.Resampling.BILINEAR)
        offset, count = layout["token_offset"], layout["token_count"]
        rows, cols = layout["grid_shape"]
        grid = (patch_scores[offset:offset + count] / scale).reshape(rows, cols)
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


def _save_binary_mask(mask, path):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_fastv_visualizations(original_images, pro_datas, image_masks, *,
                              base_grid, patch_size, output_dir, sample_id,
                              fastv_layer, keep_ratio, image_attentions=None,
                              save_prune=True, save_attention=False):
    """Save originals, blackouts, comparison panels and auditable decisions.

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
        original.save(directory / "original.png")
        if save_prune:
            combined = _blackout(original, masks["combined_pruned"])
            combined.save(directory / "original_blackout.png")
            _save_binary_mask(masks["combined_pruned"], directory / "pruned_mask.png")
            _save_binary_mask(masks["covered"], directory / "coverage_mask.png")

            for view in masks["views"]:
                info = view["metadata"]
                projected = _blackout(original, view["pruned"])
                projected.save(directory / f"{info['name']}_on_original.png")
                projected.crop(info["source_box_xyxy"]).save(directory / f"{info['name']}_crop.png")

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
            draw.text((6, 43), "Uncovered margins unchanged. ROI decisions are also saved separately.", fill="black")
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
            "blackout": str((directory / "original_blackout.png").resolve()) if save_prune else None,
            "attention_overlay": str((directory / "attention_overlay.png").resolve()) if overlays is not None else None,
        })
    return paths
