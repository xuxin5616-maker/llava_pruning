"""Save separate Base and AnyRes pruning overlays for six MVTec samples."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

import benchmark_visa_methods as common
import visualize_mvtec_baseline as baseline
from llava_pruning.pool_distill_visualization import input_rgb, white_positions
from llava_pruning.visa_comparison_methods import AverageFastV, VisionZipMethod, make_method


RATE = 50
INDICES = (55, 1200, 286, 403, 488, 1705)


def packed_view_keeps(image_mask, metadata, side):
    """Map packed decoder pruning decisions to the original encoder tile grid."""
    import torch
    from llava.model.llava_arch import unpad_image

    width, height = metadata["grid_patches"]
    raw_rows, raw_cols = height * side, width * side
    raw_positions = torch.arange(raw_rows * raw_cols).reshape(1, raw_rows, raw_cols)
    unpadded = unpad_image(raw_positions, metadata["original_size"])[0].numpy()
    rows, cols = unpadded.shape
    scale = math.sqrt(rows * cols / (9 * side**2))
    if scale > 1.1:
        rows, cols = int(rows // scale), int(cols // scale)
    packed = np.asarray(image_mask["keep"], dtype=bool)
    base_count = side**2
    anyres_end = base_count + rows * (cols + 1)
    start, end = image_mask["span"]
    if len(packed) != anyres_end + int(metadata["final_newline"]) or end - start != len(packed):
        raise ValueError("Decoder mask does not match the packed Base/AnyRes geometry")
    anyres = packed[base_count:anyres_end].reshape(rows, cols + 1)[:, :cols]
    # Max9 can pool the unpadded features. Project each coarse decision onto
    # its nearest cell; unpadded positions were outside the packed grid.
    with Image.fromarray(anyres) as mask:
        with mask.resize((unpadded.shape[1], unpadded.shape[0]), Image.Resampling.NEAREST) as projected:
            keep = np.ones((raw_rows, raw_cols), dtype=bool)
            keep.flat[unpadded.ravel()] = np.asarray(projected).ravel()
    tiles = keep.reshape(height, side, width, side).transpose(0, 2, 1, 3).reshape(width * height, base_count)
    return np.concatenate((packed[:base_count][None, :], tiles))


def visionzip_view_anchors(adapter, side):
    if len(adapter.view_anchors) != len(adapter.keeps):
        raise RuntimeError("Missing VisionZip view decisions")
    keeps = np.zeros((len(adapter.keeps), side**2), dtype=bool)
    for mask, indices, count in zip(keeps, adapter.view_anchors, adapter.keeps):
        anchors = indices.detach().cpu().numpy()
        if len(anchors) != count or np.any((anchors < 0) | (anchors >= mask.size)):
            raise ValueError("VisionZip anchors do not match the view budget")
        mask[anchors] = True
        if mask.sum() != count:
            raise ValueError("Duplicate VisionZip anchors")
    return keeps


def save_views(pixels, keeps, grid, patch_size, directory):
    width, height = grid
    if len(pixels) != 1 + width * height or len(pixels) != len(keeps):
        raise ValueError("The mask and image tiles have different view counts")
    tile_height, tile_width = pixels.shape[1:3]
    directory.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (width * tile_width, height * tile_height)) as mosaic:
        for view, (rgb, keep) in enumerate(zip(pixels, keeps)):
            with Image.fromarray(rgb) as source, Image.fromarray(white_positions(rgb, keep, patch_size)) as white:
                with Image.blend(source, white, .5) as blended:
                    if view == 0:
                        baseline.save_png(blended, directory / "base.png")
                    else:
                        row, col = divmod(view - 1, width)
                        mosaic.paste(blended, (col * tile_width, row * tile_height))
        baseline.save_png(mosaic, directory / "anyres.png")


def selected_samples(root):
    spec = json.loads((root / "run.json").read_text(encoding="utf-8"))
    baseline.verify(spec)
    if "sample_indices" in spec:
        raise ValueError("Use the original full run so the filename indices match")
    records, _ = common.read_prefix(Path(spec["input_json"]))
    if len(records) != spec["sample_count"] or max(INDICES) >= len(records):
        raise ValueError("Sample indices exceed the source run")
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "selected.jsonl"
        path.write_text("".join(json.dumps(records[index], ensure_ascii=False) + "\n" for index in INDICES),
                        encoding="utf-8")
        samples = baseline.samples_for(dict(spec, input_json=str(path), sample_count=len(INDICES)))
    return spec, samples


def run(args):
    if args.gpu < 0:
        raise ValueError("GPU ID must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from llava_pruning.backend import LlavaBackend

    root = args.source_run.expanduser().resolve()
    spec, samples = selected_samples(root)
    output = root / "fastv_visionzip_prune50_mask50" / args.method

    backend = LlavaBackend(spec["model_path"], "anyres_max_9", image_token_order="base_first")
    from llava.mm_utils import process_images
    common.validate_single_gpu(backend.model, torch)
    core = backend.model.get_model()
    tower = backend.vision_tower
    side, patch_size = int(tower.num_patches_per_side), int(tower.config.patch_size)
    if args.method == "fastv":
        method = AverageFastV(RATE, len(core.layers))
    elif args.method == "visionzip":
        method = VisionZipMethod(core, RATE, capture_anchors=True)
    else:
        method = make_method("clse", core, RATE, None)
    with torch.inference_mode():
        for index, sample in zip(INDICES, samples):
            with Image.open(sample.image) as image:
                pixels, crop_metadata = process_images([image.convert("RGB")], backend.processor,
                                                       core.config, return_pro_data=True)
            metadata = crop_metadata[0]
            result = backend.generate(sample, sample.source["prompt"], method, RATE, "anyres_max_9",
                                      capture_visualization=args.method == "clse", capture_attention=False,
                                      random_seed=spec["seed"] + index, do_sample=False,
                                      include_pruning_time=True, benchmark_metrics=False)
            if args.method == "visionzip":
                keeps = visionzip_view_anchors(method.adapter, side)
            elif args.method == "fastv":
                (image_mask,), = core.get_fastv_image_masks()
                keeps = packed_view_keeps(image_mask, metadata, side)
            else:
                stage, = result["stages"]
                keeps = packed_view_keeps(stage["mask"], metadata, side)
            directory = output / f"sample_{index:09d}"
            save_views(input_rgb(pixels[0], backend.processor), keeps, metadata["grid_patches"], patch_size, directory)
            print(f"{args.method} | {index}={sample.sample_id} | {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--method", choices=("fastv", "visionzip", "clse"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
