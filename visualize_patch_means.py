"""Redraw cached token scores as AnyRes tile means. No model, Torch or GPU.

Display SigLIP 7/14/21/26 and raw-score means of the first 2/3/4 selected layers.
Global only: original image plus scores scaled by (raw_score + 1) / 2.
Local and projector are ignored. Default shared color range: [0, 1].
This is fixed linear scaling, not per-layer min-max normalization.
Compact 2 x 4 layout: four single layers above; original + three means below.
The original image has tile boundaries but no tile-number labels.
Edit RESULTS_DIR below and run this file, or pass --results-dir. A trailing .log
on a validated score archive is removed; file contents, metadata and images stay
unchanged. Only NumPy, Pillow and Matplotlib are needed.
"""

from __future__ import annotations

# ===== 可以在这里填写上轮 visualize_scores.py 的结果目录 =====
# 相对路径以本脚本目录为起点。留空 OUTPUT_DIR 默认生成同级 <结果目录名>_patch_means。
RESULTS_DIR = r"outputs/feature_scores_02"
OUTPUT_DIR = r""
# 原图路径未变化时留空；换电脑/移动数据后可填写新的数据根目录。
DATA_ROOT = r""

SCRIPT_VERSION = "4.2-global-horizontal"

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image


SIGLIP_STAGES = ("siglip_07", "siglip_14", "siglip_21", "siglip_26")
# Older caches contain a fifth projector entry; never include it in display/averages.
CACHE_STAGES = SIGLIP_STAGES + ("projector",)
LAYER_GROUPS = {stage: (stage,) for stage in SIGLIP_STAGES}
LAYER_GROUPS.update({f"mean_first_{count}": SIGLIP_STAGES[:count] for count in (2, 3, 4)})
STAGES = tuple(LAYER_GROUPS)
LABELS = ("SigLIP layer 7", "SigLIP layer 14", "SigLIP layer 21", "SigLIP layer 26",
          "Mean: 7 + 14", "Mean: 7 + 14 + 21", "Mean: 7 + 14 + 21 + 26")
PANEL_STAGES = (SIGLIP_STAGES, ("original", *STAGES[4:]))
FIGURE_LAYOUT = {
    "rows": 2, "columns": 4, "panels": PANEL_STAGES,
    "original_tile_labels": False, "original_tile_boundaries": True,
    "heatmap_labels": "tile_id and scaled score", "colorbar": "shared_horizontal",
}
KINDS = ("global",)
COLORMAP = "jet"
OVERLAY_ALPHA = 0.70
MISSING_COLOR = "#b8b8b8"
MEAN_POLICY = "Arithmetic mean of all finite saved token scores in each tile, including padding-context tokens"
CROSS_LAYER_AVERAGE = {
    "method": "Equal-weight arithmetic mean of raw per-layer Global tile means, then fixed linear scaling to [0, 1]",
    "groups": LAYER_GROUPS,
    "missing_policy": "If any selected layer has no valid tile mean, the group mean is NaN; do not omit that layer",
    "count_policy": "CSV valid/total token counts sum token observations across the selected layers",
    "excluded_scores": "Local and projector are not read into the scoring pipeline or displayed",
}
SCORE_SCALING = {
    "method": "fixed_linear_0_1",
    "formula": "(raw_score + 1) / 2",
    "input_range": [-1.0, 1.0],
    "output_range": [0.0, 1.0],
    "scope": "Same mapping for every image, layer and tile; no per-layer min-max",
    "roundoff_policy": "Only raw overshoot up to 1e-6 is tolerated and clipped to the endpoints; larger errors fail",
}


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def find_score_file(folder):
    # Extension is only a discovery hint. np.load reads the actual binary format.
    suffixes = (".npz", ".npz.log", ".npy", ".npy.log")
    candidates = sorted(path for path in Path(folder).iterdir()
                        if path.is_file() and path.name.lower().endswith(suffixes))
    if not candidates:
        raise FileNotFoundError(f"No cached score file in {folder}; expected scores.npz (optional .log suffix)")
    if len(candidates) != 1:
        raise ValueError(f"Multiple score files in {folder}; keep exactly one source to avoid ambiguity: "
                         + ", ".join(path.name for path in candidates))
    return candidates[0]


def restore_score_filename(path):
    """Remove only the appended .log after load_scores has validated the archive."""
    path = Path(path)
    if not path.name.lower().endswith((".npz.log", ".npy.log")):
        return path
    target = path.with_suffix("")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Cannot rename {path}: destination already exists: {target}; nothing overwritten")
    path.rename(target)
    print(f"Renamed score file: {path} -> {target.name} (contents unchanged)", flush=True)
    return target


def sample_folders(results):
    results = Path(results)
    if not results.is_dir():
        raise NotADirectoryError(f"Result directory not found: {results}")
    if (results / "metadata.json").is_file():
        return [results]
    folders = sorted(path for path in results.iterdir() if path.is_dir() and (
        (path / "metadata.json").exists() or any(
            file.is_file() and file.name.lower().endswith((".npz", ".npz.log", ".npy", ".npy.log"))
            for file in path.iterdir())))
    if not folders:
        raise FileNotFoundError(f"No saved sample folders found under {results}")
    return folders


def validate_geometry(metadata):
    geometry = metadata.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("metadata.json needs saved geometry; PNG alone is not sufficient")
    geometry = dict(geometry)
    for key in ("original_size", "grid", "resized_size", "paste_xy"):
        value = geometry.get(key)
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(x) is not int or x < (0 if key == "paste_xy" else 1) for x in value)):
            raise ValueError(f"Invalid geometry.{key}")
        geometry[key] = tuple(value)
    for key in ("tile_size", "patch_size"):
        if type(geometry.get(key)) is not int or geometry[key] < 1:
            raise ValueError(f"Invalid geometry.{key}")
    if geometry["patch_size"] > geometry["tile_size"]:
        raise ValueError("Patch size exceeds tile size")
    width, height = geometry["original_size"]
    target_w, target_h = (value * geometry["tile_size"] for value in geometry["grid"])
    scale_w, scale_h = target_w / width, target_h / height
    if scale_w < scale_h:
        new_w, new_h = target_w, min(math.ceil(height * scale_w), target_h)
    else:
        new_h, new_w = target_h, min(math.ceil(width * scale_h), target_w)
    if (geometry["resized_size"] != (new_w, new_h)
            or geometry["paste_xy"] != ((target_w - new_w) // 2, (target_h - new_h) // 2)):
        raise ValueError("Saved resize/padding geometry is inconsistent; cannot align the overlay")
    return geometry


def load_scores(path, geometry, metadata):
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Cannot read binary score archive {path}. An extra .log suffix is OK, "
                         "but plain text logs or damaged files are not scores.") from error
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"{path} is a single NPY array, not the saved score archive. "
                         "Need global_scores and stages from visualize_scores.py; "
                         "the script will not guess an unknown array layout.")
    with archive:
        required = {"global_scores", "stages"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path} is missing score fields: {sorted(required - set(archive.files))}")
        stage_array = archive["stages"]
        if stage_array.ndim != 1 or stage_array.dtype.kind not in "US":
            raise ValueError(f"Invalid stages in {path}")
        stages = tuple(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in stage_array)
        if len(stages) != len(set(stages)) or set(stages) not in (set(SIGLIP_STAGES), set(CACHE_STAGES)):
            raise ValueError(f"Expected SigLIP layers 7/14/21/26 with optional projector; found {stages}")
        order = [stages.index(stage) for stage in SIGLIP_STAGES]
        expected = (len(stages), math.prod(geometry["grid"]), (geometry["tile_size"] // geometry["patch_size"]) ** 2)
        if metadata.get("score_shape") is not None and tuple(metadata["score_shape"]) != expected:
            raise ValueError("metadata.score_shape disagrees with geometry")
        scores = {}
        for kind in KINDS:
            values = archive[kind + "_scores"]
            if values.shape != expected or values.dtype.kind not in "fiu":
                raise ValueError(f"{kind}_scores must be a numeric array shaped {expected}; found {values.shape}")
            selected = values[order].astype(np.float64)
            if np.isinf(selected).any() or np.any(np.abs(selected[np.isfinite(selected)]) > 1.000001):
                raise ValueError(f"{kind}_scores contains infinite or out-of-range cosine scores")
            scores[kind] = selected
    return scores


def aggregate_scores(scores):
    """Average scores, not features. NaNs omitted with valid counts recorded."""
    means, counts = {}, {}
    for kind in KINDS:
        values = scores[kind]
        valid = np.isfinite(values)
        counts[kind] = valid.sum(axis=-1)
        totals = np.where(valid, values, 0).sum(axis=-1, dtype=np.float64)
        means[kind] = np.divide(totals, counts[kind], out=np.full(totals.shape, np.nan),
                                where=counts[kind] > 0)
    return means, counts


def build_display_means(means, counts):
    """Four single layers plus three equal-weight raw Global means; no normalization."""
    displayed, displayed_counts = {}, {}
    indices = [[SIGLIP_STAGES.index(stage) for stage in group] for group in LAYER_GROUPS.values()]
    for kind in KINDS:
        values = np.asarray(means[kind], dtype=np.float64)
        valid_counts = np.asarray(counts[kind])
        if values.ndim != 2 or values.shape[0] != len(SIGLIP_STAGES) or valid_counts.shape != values.shape:
            raise ValueError("Expected raw tile means/counts from exactly SigLIP layers 7/14/21/26")
        # Do not pool token counts or average already-normalized scores. Each layer
        # has equal weight. NaN propagates if any selected layer has no valid mean.
        displayed[kind] = np.stack([values[group].mean(axis=0) for group in indices])
        displayed_counts[kind] = np.stack([valid_counts[group].sum(axis=0) for group in indices])
    return displayed, displayed_counts


def tile_rectangles(geometry):
    """Visible tile rectangles in normalized original-image coordinates, row-major."""
    grid_w, grid_h = geometry["grid"]
    size = geometry["tile_size"]
    width, height = geometry["resized_size"]
    left, top = geometry["paste_xy"]
    rectangles = []
    for row in range(grid_h):
        for col in range(grid_w):
            x0, x1 = max(0, col * size - left), min(width, (col + 1) * size - left)
            y0, y1 = max(0, row * size - top), min(height, (row + 1) * size - top)
            rectangles.append(None if x1 <= x0 or y1 <= y0 else
                              (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height))
    return rectangles


def scale_scores_01(values):
    """Fixed cosine-range mapping, preserving missing values and raw input arrays."""
    raw = np.asarray(values, dtype=np.float64)
    if np.isinf(raw).any() or np.any(np.abs(raw[np.isfinite(raw)]) > 1.000001):
        raise ValueError("Cannot scale infinite or out-of-range cosine scores")
    # Match load_scores' existing roundoff tolerance, without hiding invalid data.
    return np.clip((raw + 1.0) / 2.0, 0.0, 1.0)


def color_limits(means, rectangles, scale):
    """Color limits in scaled units; inputs remain raw tile means."""
    if scale == "fixed":
        return 0.0, 1.0
    if scale != "sample":
        raise ValueError("Unknown color scale")
    visible = np.asarray([rect is not None for rect in rectangles])
    values = scale_scores_01(np.concatenate([means[kind][:, visible].ravel() for kind in KINDS]))
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 1.0  # All missing is rendered gray, never silently as zero.
    low, high = float(values.min()), float(values.max())
    if high - low < 5e-7:
        low, high = max(0.0, low - 0.005), min(1.0, high + 0.005)
    return low, high


def resolve_image(metadata, source_run, data_root=None):
    raw = metadata.get("image")
    if not isinstance(raw, str) or not raw:
        raise ValueError("metadata.json is missing the source image path")
    # Explicit relocation takes priority, even if the old path still exists.
    if data_root is None:
        path = Path(raw).expanduser()
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"Original image not found: {raw}. Supply --data-root with its new dataset root.")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    path_class = PureWindowsPath if PureWindowsPath(raw).drive or "\\" in raw else PurePosixPath
    original = path_class(raw)
    saved_root = source_run.get("data_root")
    if saved_root:
        try:
            relative = original.relative_to(path_class(saved_root))
        except ValueError as error:
            raise ValueError("Saved image is outside the saved data_root; refusing an ambiguous relocation") from error
        candidate = (root / Path(*relative.parts)).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Relocated image path escapes data_root")
        if candidate.is_file():
            return candidate
    else:
        # Without run.json, allow only an unambiguous root/imgs filename match.
        candidates = {path.resolve() for path in (root / original.name, root / "imgs" / original.name)
                      if path.is_file() and path.resolve().is_relative_to(root)}
        if len(candidates) == 1:
            return candidates.pop()
        if len(candidates) > 1:
            raise ValueError("Multiple matching images; retain the source run.json for exact relocation")
    raise FileNotFoundError(f"Cannot relocate {raw} under {root}; retain the original directory structure")


def render_figure(image, means, geometry, sample_id, gt, destination, scale="fixed"):
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    rectangles = tile_rectangles(geometry)
    displayed = {kind: scale_scores_01(means[kind]) for kind in KINDS}
    limits = color_limits(means, rectangles, scale)
    # All seven score panels use the same fixed transform. Normalize only selects
    # palette colors; optional sample limits do not alter the displayed values.
    norm, cmap = Normalize(*limits), colormaps[COLORMAP]
    if Path(destination).exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    with rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig = Figure(figsize=(16, 9), layout="constrained")
        FigureCanvasAgg(fig)
        fig.get_layout_engine().set(w_pad=0.12, h_pad=0.08)
        axes = fig.subplots(FIGURE_LAYOUT["rows"], FIGURE_LAYOUT["columns"], squeeze=False)
        for ax, stage in zip(axes.flat, (stage for row in PANEL_STAGES for stage in row)):
            score_row = None if stage == "original" else STAGES.index(stage)
            ax.imshow(image, extent=(0, 1, 1, 0), interpolation="nearest")
            for index, rect in enumerate(rectangles):
                if rect is None:
                    continue
                x, y, width, height = rect
                if score_row is not None:
                    value = displayed["global"][score_row, index]
                    finite = math.isfinite(value)
                    ax.add_patch(Rectangle((x, y), width, height, linewidth=0,
                                           facecolor=cmap(norm(value)) if finite else MISSING_COLOR,
                                           alpha=OVERLAY_ALPHA if finite else 1.0))
                ax.add_patch(Rectangle((x, y), width, height, fill=False, linewidth=1,
                                       edgecolor="white"))
                # Original is shown once, with boundaries only; score panels keep
                # tile IDs so the displayed values can still be located in CSV.
                if score_row is not None:
                    text = f"{index + 1}\n{value:.4f}" if finite else f"{index + 1}\nN/A"
                    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                            fontsize=8, color="white",
                            bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=2))
            ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
            ax.set_aspect(image.height / image.width)
            ax.set_title("Original" if score_row is None else LABELS[score_row], fontsize=11)
        ground_truth = {0: "Normal", 1: "Abnormal"}.get(gt, "Unknown")
        fig.suptitle(f"ID: {sample_id} | GT: {ground_truth}\n"
                     "Global tile means: (raw score + 1) / 2 | Cached scores only", fontsize=11)
        colorbar_label = "Global score [0, 1]: (raw mean -cos score + 1) / 2"
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal",
                     fraction=0.04, pad=0.025, shrink=0.75, aspect=50, label=colorbar_label)
        fig.supxlabel(f"Shared {scale} color range [{limits[0]:.4f}, {limits[1]:.4f}] | JET; opacity {OVERLAY_ALPHA}\n"
                      "Scaled scores, not probabilities. No per-layer min-max; Local and projector excluded. "
                      "Mean panels: average raw layer scores, then (raw score + 1) / 2.\n"
                      "Padding-context tokens included; NaNs omitted; missing selected layer -> gray group score. "
                      "CSV retains raw means and scaled scores.", fontsize=8)
        try:
            fig.savefig(destination, dpi=150, facecolor="white")
        finally:
            fig.clear()
    return list(limits)


def write_values(destination, means, counts, geometry, token_count):
    rectangles = tile_rectangles(geometry)
    scaled = scale_scores_01(means["global"])
    with Path(destination).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "tile_id", "tile_row", "tile_col", "visible",
                         "global_mean", "global_score_01", "global_valid_tokens", "total_tokens", "source_layers", "layer_count"])
        for row, stage in enumerate(STAGES):
            source_layers = LAYER_GROUPS[stage]
            for index, rect in enumerate(rectangles):
                tile_row, tile_col = divmod(index, geometry["grid"][0])
                values = [float(means[kind][row, index]) if np.isfinite(means[kind][row, index]) else "" for kind in KINDS]
                scaled_value = float(scaled[row, index]) if np.isfinite(scaled[row, index]) else ""
                writer.writerow([stage, index + 1, tile_row + 1, tile_col + 1, rect is not None,
                                 *values, scaled_value, int(counts["global"][row, index]),
                                 token_count * len(source_layers), "+".join(source_layers), len(source_layers)])


def run(results_dir, output_dir=None, data_root=None, color_scale="fixed"):
    results = Path(results_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else results.with_name(results.name + "_patch_means")
    if output == results or output.is_relative_to(results) or results.is_relative_to(output):
        raise ValueError("Output must be separate from the input result tree (use a sibling directory)")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    if color_scale not in {"sample", "fixed"}:
        raise ValueError("Use fixed or sample for scaled scores; per-layer normalization was removed")
    folders = sample_folders(results)
    run_path = results / "run.json"
    if len(folders) == 1 and folders[0] == results and not run_path.is_file():
        run_path = results.parent / "run.json"
    source_run = read_json(run_path) if run_path.is_file() else {}
    # Preflight filenames/geometry/images before writing or restoring .log names.
    samples = []
    for folder in folders:
        metadata = read_json(folder / "metadata.json")
        geometry = validate_geometry(metadata)
        source = find_score_file(folder)
        image_path = resolve_image(metadata, source_run, data_root)
        with Image.open(image_path) as image:
            if image.size != geometry["original_size"]:
                raise ValueError(f"Source image size no longer matches saved geometry: {image_path}")
        samples.append((folder, metadata, geometry, source, image_path))
    output.mkdir(parents=True, exist_ok=True)
    print(f"visualize_patch_means {SCRIPT_VERSION} | color-scale={color_scale}", flush=True)
    print(f"Redrawing {len(samples)} cached samples on CPU. No model/LLM loading or inference.", flush=True)
    for index, (folder, metadata, geometry, source, image_path) in enumerate(samples, start=1):
        scores = load_scores(source, geometry, metadata)
        source = restore_score_filename(source)
        means, counts = aggregate_scores(scores)
        means, counts = build_display_means(means, counts)
        target = output / folder.name
        target.mkdir(exist_ok=False)
        with Image.open(image_path) as original:
            image = original.convert("RGB")
        limits = render_figure(image, means, geometry, metadata.get("id", folder.name), metadata.get("gt"),
                               target / "patch_means.png", color_scale)
        write_values(target / "patch_means.csv", means, counts, geometry, scores["global"].shape[-1])
        saved = {"id": metadata.get("id", folder.name), "source_scores": str(source),
                 "source_metadata": str(folder / "metadata.json"), "image": str(image_path),
                 "gt": metadata.get("gt"), "geometry": geometry, "color_limits": limits,
                 "script_version": SCRIPT_VERSION, "color_scale": color_scale,
                 "figure_layout": FIGURE_LAYOUT,
                 "score_kind": "global", "score_normalization": SCORE_SCALING["method"],
                 "score_scaling": SCORE_SCALING,
                 "cross_layer_average": CROSS_LAYER_AVERAGE,
                 "display_values": "global_score_01"}
        with (target / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(saved, stream, ensure_ascii=False, indent=2)
        print(f"[{index}/{len(samples)}] {source.name} -> {target / 'patch_means.png'}", flush=True)
    manifest = {"source_results": str(results), "sample_count": len(samples), "complete": True,
                "script_version": SCRIPT_VERSION,
                "figure_layout": FIGURE_LAYOUT,
                "stages": STAGES, "mean_policy": MEAN_POLICY,
                "cross_layer_average": CROSS_LAYER_AVERAGE,
                "missing_policy": "NaN omitted with counts; all missing -> gray, CSV blank; infinity rejected",
                "color_scale": color_scale, "colormap": COLORMAP, "overlay_alpha": OVERLAY_ALPHA,
                "scale_scope": {"sample": "Scaled Global visible tile means, all seven score panels in each image",
                                "fixed": "Global [0, 1] across all images/panels"}[color_scale],
                "score_kind": "global", "score_normalization": SCORE_SCALING["method"],
                "score_scaling": SCORE_SCALING, "display_values": "global_score_01",
                "llm_generation": False, "model_loaded": False,
                "data_root_override": str(Path(data_root).resolve()) if data_root else None}
    with (output / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"Done: {output}", flush=True)
    return output


def configured_path(raw):
    if not raw.strip():
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--results-dir", type=Path, help="Previous visualize_scores.py output directory")
    parser.add_argument("--output-dir", type=Path, help="New/empty directory; default: sibling <results>_patch_means")
    parser.add_argument("--data-root", type=Path, help="Optional new dataset root if original images moved")
    parser.add_argument("--color-scale", choices=("fixed", "sample"), default="fixed",
                        help="fixed (default): [0, 1] across images/panels; sample: shared scaled-score range "
                             "across the seven Global panels of each image. Both use (raw_score + 1) / 2; "
                             "no per-layer min-max.")
    return parser


def main():
    args = build_parser().parse_args()
    source = args.results_dir or configured_path(RESULTS_DIR)
    if source is None:
        raise ValueError("Set RESULTS_DIR at the top or pass --results-dir")
    run(source, args.output_dir or configured_path(OUTPUT_DIR),
        args.data_root or configured_path(DATA_ROOT), args.color_scale)


if __name__ == "__main__":
    main()
