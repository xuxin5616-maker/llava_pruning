"""Opt-in snapshots and white token-position masks for the hybrid baseline.

This module does not enable the existing model visualization flags or change
selection. It observes the actual CPU input views and the returned block2
selection; exporting provenance and drawing happen after generation.
"""

from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def normalized_id(value):
    value = str(value).removeprefix("sample_")
    return str(int(value)) if value.isascii() and value.isdigit() else value


def find_record(records, requested):
    matches = [(index, row) for index, row in enumerate(records)
               if isinstance(row, dict) and normalized_id(row.get("question_id", row.get("id", ""))) == normalized_id(requested)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one ID matching {requested!r} ignoring leading zeros; found {len(matches)}")
    return matches[0]


@contextmanager
def capture_baseline_views(method):
    """Use the same fast pipeline; retain only already-computed decisions."""
    from llava import mm_utils
    trace = {}
    adapter = method.adapter
    original_process = mm_utils.process_images
    original_select = adapter.select
    had_select = "select" in adapter.__dict__
    previous_select = adapter.__dict__.get("select")

    def process(*args, **kwargs):
        result = original_process(*args, **kwargs)
        if "pixels" in trace or not kwargs.get("return_pro_data"):
            raise RuntimeError("Visualization requires one real preprocessing call with crop metadata")
        pixels, metadata = result
        if isinstance(pixels, list):
            if len(pixels) != 1:
                raise ValueError("Single-sample visualization requires exactly one image")
            pixels = pixels[0]
        elif pixels.ndim == 5 and pixels.shape[0] == 1:
            pixels = pixels[0]
        if pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.device.type != "cpu":
            raise ValueError("Expected CPU-normalized Base + AnyRes input views")
        trace.update(pixels=pixels, metadata=metadata)
        return result

    def select(*args, **kwargs):
        result = original_select(*args, **kwargs)
        if "selected" in trace:
            raise RuntimeError("Expected one distilled block2 pruning decision")
        # No clone, CPU transfer or rendering in the scoring path. This tensor
        # is the same selection returned to the original physical-drop engine.
        trace["selected"] = result[0].detach()
        return result

    try:
        mm_utils.process_images = process
        adapter.select = select
        yield trace
    finally:
        mm_utils.process_images = original_process
        if had_select:
            adapter.select = previous_select
        else:
            delattr(adapter, "select")


def input_rgb(pixels, processor):
    """Undo this SigLIP processor's rescale/normalization, before FP16 input casting."""
    values = pixels.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    mean = np.asarray(processor.image_mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(processor.image_std, dtype=np.float32).reshape(1, 1, 1, 3)
    factor = float(processor.rescale_factor)
    if not math.isfinite(factor) or factor <= 0 or not np.isfinite(values).all():
        raise ValueError("Invalid processor normalization or input views")
    return np.rint((values * std + mean) / factor).clip(0, 255).astype(np.uint8)


def anchor_masks(decisions, grid, selected, *, final_newline=False):
    """Exact actual packed anchor positions; merged OTHER sources stay white."""
    from .pool_distill import source_records
    records = source_records(decisions, grid, final_newline=final_newline, fast=True)
    selected = [int(index) for index in selected]
    if selected != sorted(set(selected)) or any(not 0 <= i < len(records) for i in selected):
        raise ValueError("Distilled selection is not a unique sorted subset of the pre-LLM image span")
    chosen = set(selected)
    if any(i not in chosen for i, record in enumerate(records) if record is None):
        raise ValueError("The distilled decision unexpectedly removed a structural newline")
    before = np.zeros((len(decisions), 729), dtype=bool)
    after = np.zeros_like(before)
    for position, record in enumerate(records):
        if record is None:
            continue
        view, anchor = record["view"], record["anchor"]
        if before[view, anchor]:
            raise ValueError("Duplicate anchor in packed source records")
        before[view, anchor] = True
        if position in chosen:
            after[view, anchor] = True
    newlines = sum(record is None for record in records)
    if before.sum()+newlines != len(records) or after.sum()+newlines != len(selected):
        raise RuntimeError("Visible anchor counts do not reconcile with actual packed token budgets")
    return before, after, newlines


def white_positions(rgb, keep, patch_size=14):
    """Paint removed convolution-patch supports RGB255; retain the unencoded edge."""
    keep = np.asarray(keep, dtype=bool)
    side = math.isqrt(keep.size)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or side*side != keep.size:
        raise ValueError("Expected an RGB8 view and a square token-position mask")
    support = side*patch_size
    if support > min(rgb.shape[:2]):
        raise ValueError("Token grid exceeds the actual input crop")
    removed = (~keep.reshape(side, side)).repeat(patch_size, 0).repeat(patch_size, 1)
    result = rgb.copy()
    result[:support, :support][removed] = 255
    return result


def overview(views, masks, grid, *, title, caption, packed_count, patch_size=14,
             count_label="Packed image tokens (including newlines)"):
    """Base at left, actual row-major AnyRes grid at right; no view resizing."""
    width, height = grid
    if len(views) != 1+width*height or masks.shape != (len(views), 729):
        raise ValueError("Input views/masks disagree with the actual AnyRes grid")
    tile_h, tile_w = views.shape[1:3]
    label, gap = 30, 12
    cell_w, cell_h = tile_w+gap, tile_h+label+gap
    canvas_width = (width+1)*cell_w+gap
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    paragraphs = ((title,font),(caption,font),
        (f"{count_label}: {packed_count} | Actual AnyRes grid: {width} x {height}",small),
        ("White = no independent token at this position (merged sources included). Unencoded crop edge unchanged.",small))
    lines = []
    for paragraph, line_font in paragraphs:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and line_font.getlength(candidate) > canvas_width-2*gap:
                lines.append((line,line_font));line = word
            else:
                line = candidate
        lines.append((line,line_font))
    header = gap+sum(line_font.getbbox("Ag")[3]+6 for _,line_font in lines)
    canvas = Image.new("RGB", (canvas_width, header+height*cell_h+gap), "white")
    draw = ImageDraw.Draw(canvas)
    y = 8
    for line,line_font in lines:
        draw.text((gap,y),line,fill="black",font=line_font)
        y += line_font.getbbox("Ag")[3]+6
    for view, (rgb, mask) in enumerate(zip(views, masks)):
        if view == 0:
            x, y = gap, header+(height*cell_h-cell_h)//2
            name = "Base"
        else:
            row, col = divmod(view-1, width)
            x, y = gap+(col+1)*cell_w, header+row*cell_h
            name = f"AnyRes {view:02d} (row {row+1}, col {col+1})"
        draw.text((x, y), f"{name} | {int(mask.sum())}/729 tokens", fill="black", font=small)
        canvas.paste(Image.fromarray(white_positions(rgb, mask, patch_size)), (x, y+label))
    return canvas


STAGES = (("input", "Original input", "original_tokens"),
          ("pre_llm", "After MyPool", "pre_llm_kept"),
          ("post_distill", "After distillation (before tail)", "post_distill_kept"))


def save_individual_views(views, grid, output, *, stage, caption, counts, title=None, count_label="tokens"):
    """One actual-size Base/crop per PNG, with short GT/prediction labels."""
    width, height = grid
    if len(views) != 1+width*height or len(counts) != len(views):
        raise ValueError("Individual views/counts disagree with the native AnyRes grid")
    if title is None:
        title = dict((key, title) for key,title,_ in STAGES)[stage]
    output = Path(output)
    font = ImageFont.load_default(size=14)
    files = []
    for view, rgb in enumerate(views):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Individual images must be RGB8")
        name = "base" if view == 0 else f"anyres_{view:02d}"
        display = "Base" if view == 0 else f"AnyRes {view:02d} (r{(view-1)//width+1},c{(view-1)%width+1})"
        lines = []
        for paragraph in (title, f"{display} | {int(counts[view])}/729 {count_label}", caption):
            line = ""
            for word in paragraph.split():
                candidate = f"{line} {word}".strip()
                if line and font.getlength(candidate) > rgb.shape[1]-12:
                    lines.append(line);line = word
                else:
                    line = candidate
            lines.append(line)
        line_height = font.getbbox("Ag")[3]+5
        header = len(lines)*line_height+10
        panel = Image.new("RGB",(rgb.shape[1],header+rgb.shape[0]),"white")
        draw = ImageDraw.Draw(panel)
        for index,line in enumerate(lines):
            draw.text((6,5+index*line_height),line,fill="black",font=font)
        panel.paste(Image.fromarray(rgb),(0,header))
        filename = f"{stage}_{name}.png"
        panel.save(output/filename)
        files.append(filename)
    return files


def split_existing_overviews(output):
    """Lossless CPU export beside the old overviews; no model/data paths needed."""
    from .visualization import prediction_caption
    output = Path(output).expanduser().resolve()
    metadata = json.loads((output/"run.json").read_text(encoding="utf-8"))
    grid = tuple(metadata["native_grid"])
    if len(grid) != 2 or any(type(n) is not int or n < 1 for n in grid):
        raise ValueError("Saved native_grid must contain positive integer width/height")
    width,height = grid
    with (output/"view_counts.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_names = ["base"]+[f"anyres_{i:02d}" for i in range(1,1+width*height)]
    if [row["view"] for row in rows] != expected_names:
        raise ValueError("Saved view counts do not match Base + the native crop grid")
    sources = ("input_views.png","pre_llm_pruning.png","post_distill_pruning.png")
    extracted = []
    # Validate all three source images before writing any new files. Both
    # original overview versions have gap12, label30 and unscaled square views;
    # infer header height from the canvas rather than guessing its text/font.
    for filename in sources:
        with Image.open(output/filename) as source:
            if (source.width-12) % (width+1):
                raise ValueError(f"Unrecognized saved overview geometry: {filename}")
            tile = (source.width-12)//(width+1)-12
            cell_w,cell_h = tile+12,tile+30+12
            header = source.height-height*cell_h-12
            if tile < 1 or header < 1:
                raise ValueError(f"Invalid saved overview dimensions: {filename}")
            views = []
            source_rgb = source.convert("RGB")
            for view in range(1+width*height):
                if view == 0:
                    x,y = 12,header+(height*cell_h-cell_h)//2
                else:
                    row,col = divmod(view-1,width)
                    x,y = 12+(col+1)*cell_w,header+row*cell_h
                views.append(np.asarray(source_rgb.crop((x,y+30,x+tile,y+30+tile))).copy())
            extracted.append(views)
    caption = f"ID: {metadata['question_id']} | {prediction_caption(metadata.get('gt'),metadata.get('answer'))}"
    files = []
    for (stage,_,field),views in zip(STAGES,extracted):
        files += save_individual_views(views,grid,output,stage=stage,caption=caption,
                                       counts=[int(row[field]) for row in rows])
    (output/"individual_views.json").write_text(json.dumps(dict(question_id=metadata["question_id"],
        native_grid=list(grid),export_mode="lossless_from_existing_overviews_no_inference",
        sources=list(sources),outputs=files),ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Exported {len(files)} individual PNGs beside the existing overviews; CPU only: {output}",flush=True)
    return output


def save_baseline_visualizations(method, trace, processor, result, sample, output):
    from .pool_lean import export_source_decisions
    from .visualization import prediction_caption
    adapter = method.adapter
    if "pixels" not in trace or "selected" not in trace:
        raise RuntimeError("Missing real input views or the pre-tail distilled decision")
    views = input_rgb(trace["pixels"], processor)
    grid = tuple(adapter.geometry.grid)
    decisions = export_source_decisions(adapter.device_decisions)
    selected = trace["selected"].cpu().tolist()
    before, after, newlines = anchor_masks(decisions, grid, selected,
        final_newline=adapter.core.config.mm_patch_merge_type.endswith("_add_newl"))
    plan = adapter.plan
    if int(before.sum())+newlines != plan["pre_llm_tokens"] or int(after.sum())+newlines != plan["final_llm_tokens"]:
        raise RuntimeError("Visualization decisions differ from the actual two-stage budget")
    caption = f"ID: {sample.sample_id} | {prediction_caption(sample.gt, result['answer'])}"
    output = Path(output)
    files = []
    for (stage,_,_),masks in zip(STAGES,(np.ones_like(before),before,after)):
        masked = [white_positions(rgb,mask,int(adapter.geometry.patch_size)) for rgb,mask in zip(views,masks)]
        files += save_individual_views(masked,grid,output,stage=stage,caption=caption,
                                       counts=[int(mask.sum()) for mask in masks])
    rows = []
    for view in range(len(views)):
        rows.append(dict(view="base" if view == 0 else f"anyres_{view:02d}", original_tokens=729,
            pre_llm_kept=int(before[view].sum()), post_distill_kept=int(after[view].sum()),
            removed_before_llm=729-int(before[view].sum()),
            removed_by_distillation=int(before[view].sum()-after[view].sum()),
            cumulative_removed=729-int(after[view].sum())))
    with (output/"view_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return dict(native_grid=list(grid), encoder_views=len(views), raw_encoder_spatial_tokens=len(views)*729,
        baseline_packed_tokens=plan["baseline_packed_tokens"],
        pre_llm_tokens=plan["pre_llm_tokens"], post_distill_tokens=len(selected), protected_newlines=newlines,
        mask_semantics="independent retained token anchors only; merged sources also white; RGB255; post mask cumulative",
        post_snapshot="after block2 selection, before tail=9 removal", outputs=files)
