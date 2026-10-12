"""CPU-only pixel-entropy deviation: BOTH Base and whole AnyRes at 12x12 AND 18x18.

Both views use 3x3/5x5/7x7 neighborhoods, excluding the center:
    H = -sum(p * ln(p)), D = abs(H - median(valid neighboring H)).
Softmax(D / ln(2)) is computed separately per view, grid size AND window over
all defined cells, never per AnyRes crop. No feature extraction or LLM execution.

Read original images and geometry from previous feature-score metadata.
Save two PNG overviews per sample: Base and AnyRes, each with a 12x12 row and
an 18x18 row (reference + three windows per row). Raw arrays and provenance are
embedded in PNG text metadata; only a small run.json is written at the root.
No cached score NPZ is loaded, renamed, or modified. This is a custom pixel
analysis, NOT cross-entropy, model attention, or the original VFlowOpt method.

Example (no CUDA_VISIBLE_DEVICES/model-path required):
    python run_entropy_18_12.py --results-dir outputs/feature_scores_02 \
        --output-dir outputs/entropy_both_12_18_01
"""
from __future__ import annotations

# Defaults can also be edited here. Relative DEFAULT paths use this script's
# directory; command-line paths use the current working directory as usual.
RESULTS_DIR = r"outputs/feature_scores_02"
OUTPUT_DIR = r""  # default sibling: <results>_entropy_both_12_18
DATA_ROOT = r""   # optional dataset relocation; preserve root/imgs structure
SCRIPT_VERSION = "2.0-both-views-grids12-18-neighborhood-softmax"

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image

GRID_SIDES = (12, 18)  # Apply BOTH independently to BOTH views.
WINDOWS = (3, 5, 7)
COLORMAP = "jet"  # Preserve the user's existing visualization convention.
OVERLAY_ALPHA = 0.70
MISSING_COLOR = "#b8b8b8"
DEFINITION = {
    "entropy": "H = -sum(p * ln(p)); 256-bin histogram of floor((R+G+B)/3)",
    "entropy_unit": "nat",
    "deviation": "D_i(w) = abs(H_i - median(H_j in valid neighbors, j != i))",
    "windows": list(WINDOWS),
    "grid_sides": list(GRID_SIDES),
    "base": "12x12 AND 18x18 across the entire 384x384 Base image",
    "anyres": "12x12 AND 18x18 TOTAL across the saved padded AnyRes mosaic, NOT per crop",
    "four_groups": ["Base12", "Base18", "AnyRes12", "AnyRes18"],
    "grid_edges": "floor(i * axis_length / side); all pixels covered",
    "padding": "only cells fully inside image content are valid centers/neighbors",
    "edges": "available valid neighbors only; no zero/reflection padding",
    "undefined": "no valid neighbors -> NaN; invalid cells excluded from Softmax",
    "softmax": "softmax(D / ln(2)); temperature 1, bit-valued input",
    "softmax_scope": "entire view, independently for Base/AnyRes, each grid size and each window",
    "display": "actual Softmax weights; no post-Softmax score transformation",
    "color_limits": "separate per view/grid size/image; shared by the three windows within each row",
    "colormap": COLORMAP, "overlay_alpha": OVERLAY_ALPHA,
    "interpretation": "pixel-entropy contrast; NOT a defect probability",
    "feature_extraction": False, "model_loaded": False,
    "llm_generation": False, "pruning": False, "original_vflowopt_method": False,
}


@dataclass(frozen=True)
class Geometry:
    original_size: tuple[int, int]
    grid: tuple[int, int]  # columns, rows of the OLD encoder crops
    tile_size: int
    patch_size: int  # unchanged model patch size, NOT the entropy block size
    resized_size: tuple[int, int]
    paste_xy: tuple[int, int]

    @property
    def canvas_size(self):
        return self.grid[0] * self.tile_size, self.grid[1] * self.tile_size

    @property
    def content_box(self):
        x, y = self.paste_xy
        w, h = self.resized_size
        return x, y, x + w, y + h

    @classmethod
    def from_metadata(cls, metadata):
        raw = metadata.get("geometry")
        if not isinstance(raw, dict):
            raise ValueError("metadata.json needs saved geometry; PNGs alone are insufficient")
        values = {}
        for key in ("original_size", "grid", "resized_size", "paste_xy"):
            value = raw.get(key)
            if (not isinstance(value, (list, tuple)) or len(value) != 2 or
                    any(type(x) is not int or x < (0 if key == "paste_xy" else 1) for x in value)):
                raise ValueError(f"Invalid geometry.{key}")
            values[key] = tuple(value)
        for key, expected in (("tile_size", 384), ("patch_size", 14)):
            if type(raw.get(key)) is not int or raw[key] != expected:
                raise ValueError("Expected saved SigLIP 384/14 source geometry")
            values[key] = raw[key]
        geometry = cls(**values)
        w, h = geometry.original_size
        cw, ch = geometry.canvas_size
        scale_w, scale_h = cw / w, ch / h
        if scale_w < scale_h:
            resized = cw, min(math.ceil(h * scale_w), ch)
        else:
            resized = min(math.ceil(w * scale_h), cw), ch
        paste = (cw - resized[0]) // 2, (ch - resized[1]) // 2
        if geometry.resized_size != resized or geometry.paste_xy != paste:
            raise ValueError("Saved resize/padding geometry is inconsistent")
        return geometry


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_image(metadata, source_run, data_root=None):
    raw = metadata.get("image")
    if not isinstance(raw, str) or not raw:
        raise ValueError("metadata.json is missing the original image path")
    if data_root is None:
        image = Path(raw).expanduser()
        if image.is_file():
            return image.resolve()
        raise FileNotFoundError(f"Original image not found: {raw}. Use --data-root to relocate it.")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    path_class = PureWindowsPath if PureWindowsPath(raw).drive or "\\" in raw else PurePosixPath
    old_image = path_class(raw)
    if source_run.get("data_root"):
        try:
            relative = old_image.relative_to(path_class(source_run["data_root"]))
        except ValueError as error:
            raise ValueError("Image is outside the saved data_root; relocation is ambiguous") from error
        candidate = (root / Path(*relative.parts)).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Relocated image escapes data_root")
        if candidate.is_file():
            return candidate
    else:
        # Without a saved root, do not guess arbitrary nested directories.
        candidates = {p.resolve() for p in (root / old_image.name, root / "imgs" / old_image.name)
                      if p.is_file() and p.resolve().is_relative_to(root)}
        if len(candidates) == 1:
            return candidates.pop()
        if len(candidates) > 1:
            raise ValueError("Multiple matching images; retain run.json for exact relocation")
    raise FileNotFoundError(f"Cannot relocate {raw} under {root}")


def prepare_inputs(results_dir, data_root=None, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    results = Path(results_dir).expanduser().resolve()
    if not results.is_dir():
        raise NotADirectoryError(f"Result directory not found: {results}")
    single = (results / "metadata.json").is_file()
    if single:
        folders = [results]
    else:
        folders = sorted(p for p in results.iterdir() if p.is_dir() and (
            (p / "metadata.json").is_file() or any(
                f.is_file() and f.name.lower().endswith((".npz", ".npz.log", ".npy", ".npy.log"))
                for f in p.iterdir())))
    if not folders:
        raise FileNotFoundError("No sample metadata found. Supply the previous feature-score results directory.")
    run_path = results / "run.json"
    if single and not run_path.is_file():
        run_path = results.parent / "run.json"
    source_run = read_json(run_path) if run_path.is_file() else {}
    samples = []
    for folder in folders[:limit]:
        metadata = read_json(folder / "metadata.json")
        geometry = Geometry.from_metadata(metadata)
        image = resolve_image(metadata, source_run, data_root)
        with Image.open(image) as source:
            if source.size != geometry.original_size:
                raise ValueError(f"Image size differs from saved geometry: {image}")
            source.verify()
        samples.append({"folder": folder, "image": image, "geometry": geometry,
                        "id": str(metadata.get("id", folder.name)), "gt": metadata.get("gt")})
    return results, samples


def prepare_views(image, geometry):
    """Reproduce the existing Base resize and whole AnyRes padded RGB canvas."""
    if image.mode != "RGB" or image.size != geometry.original_size:
        raise ValueError("RGB source image must match saved geometry")
    base = image.resize((384, 384), Image.Resampling.BICUBIC)
    mosaic = Image.new("RGB", geometry.canvas_size, (0, 0, 0))
    mosaic.paste(image.resize(geometry.resized_size, Image.Resampling.BICUBIC), geometry.paste_xy)
    return base, mosaic


def grid_edges(length, side):
    if type(length) is not int or type(side) is not int or side < 1 or length < side:
        raise ValueError("Grid requires at least one pixel per cell")
    return (np.arange(side + 1, dtype=np.int64) * length) // side


def grid_entropy(image, side, content_box=None):
    """Recompute histograms over exact cells; do not average old token scores.

    Raw H includes each complete cell's pixels, including padding. Cells with
    any padding are marked invalid and excluded from subsequent processing.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB pixels")
    h, w = rgb.shape[:2]
    x_edges, y_edges = grid_edges(w, side), grid_edges(h, side)
    gray = (rgb.astype(np.uint16).sum(axis=-1) // 3).astype(np.uint8)
    entropy = np.empty((side, side), dtype=np.float64)
    for y in range(side):
        for x in range(side):
            block = gray[y_edges[y]:y_edges[y + 1], x_edges[x]:x_edges[x + 1]]
            hist = np.bincount(block.ravel(), minlength=256)
            p = hist[hist > 0].astype(np.float64) / block.size
            entropy[y, x] = max(0., float(-np.dot(p, np.log(p))))
    left, top, right, bottom = content_box if content_box is not None else (0, 0, w, h)
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError("Invalid content rectangle")
    valid = ((y_edges[:-1, None] >= top) & (y_edges[1:, None] <= bottom) &
             (x_edges[None, :-1] >= left) & (x_edges[None, 1:] <= right))
    return {"entropy_nats": entropy, "valid": valid, "x_edges": x_edges, "y_edges": y_edges}


def neighborhood_deviation(entropy, valid):
    """Both Base and AnyRes use the identical center-excluding median rule."""
    entropy = np.asarray(entropy, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if entropy.ndim != 2 or not entropy.size or valid.shape != entropy.shape:
        raise ValueError("Expected matching 2D entropy and validity grids")
    if not np.isfinite(entropy).all() or (entropy < 0).any():
        raise ValueError("Entropy must be finite and nonnegative")
    h, w = entropy.shape
    deviations = np.full((len(WINDOWS), h, w), np.nan)
    medians = np.full_like(deviations, np.nan)
    counts = np.zeros(deviations.shape, dtype=np.int16)
    for wi, window in enumerate(WINDOWS):
        radius = window // 2
        for y, x in np.argwhere(valid):
            top, left = max(0, y - radius), max(0, x - radius)
            bottom, right = min(h, y + radius + 1), min(w, x + radius + 1)
            neighbors = valid[top:bottom, left:right].copy()
            neighbors[y - top, x - left] = False
            values = entropy[top:bottom, left:right][neighbors]
            counts[wi, y, x] = len(values)
            if values.size:
                medians[wi, y, x] = np.median(values)
                deviations[wi, y, x] = abs(entropy[y, x] - medians[wi, y, x])
    return deviations, medians, counts


def view_softmax(deviations):
    """Softmax of bit-valued deviations over a WHOLE view, once per window."""
    values = np.asarray(deviations, dtype=np.float64)
    if (values.ndim != 3 or not values.size or np.isinf(values).any()
            or (values[np.isfinite(values)] < 0).any()):
        raise ValueError("Expected nonnegative/NaN deviations [window,row,col]")
    result = np.full(values.shape, np.nan)
    for wi, plane in enumerate(values):
        finite = np.isfinite(plane)
        if finite.any():
            logits = (plane[finite] - plane[finite].max()) / math.log(2.)
            weights = np.exp(logits)
            result[wi, finite] = weights / weights.sum()
    return result


def analyze_view(image, side, content_box=None):
    result = grid_entropy(image, side, content_box)
    deviation, median, counts = neighborhood_deviation(result["entropy_nats"], result["valid"])
    result.update(deviation_nats=deviation, neighbor_median_nats=median,
                  neighbor_count=counts, softmax=view_softmax(deviation))
    return result


def analyze_grids(image, content_box=None):
    """Recompute both grids from pixels, never resize/average one score grid."""
    return {side: analyze_view(image, side, content_box) for side in GRID_SIDES}


def expand_grid(values, x_edges, y_edges):
    """Piecewise-constant exact cell support, including 21/22-pixel Base cells."""
    values = np.asarray(values)
    if values.ndim < 2 or values.shape[:2] != (len(y_edges) - 1, len(x_edges) - 1):
        raise ValueError("Score grid and pixel edges disagree")
    return values.repeat(np.diff(y_edges), axis=0).repeat(np.diff(x_edges), axis=1)


def color_limits(weights):
    finite = np.asarray(weights)[np.isfinite(weights)]
    if not finite.size:
        return 0., 1.
    low, high = float(finite.min()), float(finite.max())
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-15):
        return 0., max(2 * high, 1e-6)  # constant map is a single middle color
    return low, high


def array_for_json(value):
    value = np.asarray(value)
    return np.where(np.isfinite(value), value, None).tolist() if value.dtype.kind == "f" else value.tolist()


def draw_overview(reference, analyses, destination, view, caption, provenance, content_box=None):
    """Two rows (12x12, 18x18), each with reference + three window heatmaps."""
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import FormatStrFormatter

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if set(analyses) != set(GRID_SIDES):
        raise ValueError("Overview requires both 12x12 and 18x18 analyses")
    for side in GRID_SIDES:
        if np.asarray(analyses[side]["softmax"]).shape != (len(WINDOWS), side, side):
            raise ValueError(f"Unexpected Softmax shape for {side}x{side} grid")
    cmap = colormaps[COLORMAP].with_extremes(bad=MISSING_COLOR)
    figure_data = {
        **provenance, "schema_version": 2, "view": view,
        "grid_order": list(GRID_SIDES), "grids": {},
    }
    with rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11}):
        fig = Figure(figsize=(16, 9.4), dpi=150)
        FigureCanvasAgg(fig)
        try:
            axes = fig.subplots(2, 4, squeeze=False)
            fig.subplots_adjust(left=.015, right=.885, top=.88, bottom=.14, wspace=.06, hspace=.25)
            for row, side in enumerate(GRID_SIDES):
                analysis = analyses[side]
                weights = analysis["softmax"]
                limits = color_limits(weights)
                norm = Normalize(*limits)
                counts = np.isfinite(weights).sum(axis=(1, 2)).tolist()
                figure_data["grids"][str(side)] = {
                    "analysis_grid": [side, side], "color_limits": list(limits),
                    "defined_counts_per_window": counts,
                    "arrays": {key: array_for_json(value) for key, value in analysis.items()},
                }
                for ax in axes[row]:
                    ax.imshow(reference, extent=(0, 1, 1, 0), interpolation="nearest")
                    ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
                    ax.set_aspect(reference.height / reference.width)
                axes[row, 0].set_title(f"{side} x {side} grid | Reference")
                for wi, window in enumerate(WINDOWS):
                    # Expand RGBA bytes instead of full-mosaic float64 colors.
                    cells = cmap(norm(np.ma.masked_invalid(weights[wi])), bytes=True)
                    cells[..., 3] = np.where(np.isfinite(weights[wi]), round(OVERLAY_ALPHA * 255), 255)
                    rgba = expand_grid(cells, analysis["x_edges"], analysis["y_edges"])
                    if content_box is not None:
                        left, top, right, bottom = content_box
                        rgba = rgba[top:bottom, left:right]
                    axes[row, wi + 1].imshow(rgba, extent=(0, 1, 1, 0), interpolation="nearest",
                                            aspect=reference.height / reference.width)
                    axes[row, wi + 1].set_title(f"{window} x {window} neighbors | n={counts[wi]}")
                slot = axes[row, 0].get_subplotspec().get_position(fig)
                color_ax = fig.add_axes((.905, slot.y0 + .02, .014, slot.height - .04))
                bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                                  format=FormatStrFormatter("%.3g"))
                bar.set_label(f"Softmax weight ({side} x {side})", fontsize=9)
            fig.suptitle(f"{caption}\n{view}: BOTH 12 x 12 and 18 x 18 total grids | Neighborhood entropy deviation", y=.98, fontsize=12)
            fig.text(.5, .09, "D = |H - median(neighbor H)|; center excluded. Softmax(D / ln(2)) separately for each view, grid size and window.",
                     ha="center", fontsize=9)
            fig.text(.5, .055, "Gray = padding / partial-padding cell / no valid neighbors. Each row's three windows share actual-weight color limits.",
                     ha="center", fontsize=9)
            fig.text(.5, .02, "Rows and views use separate scales; equal colors across them need not mean equal weights. Weights are NOT defect probabilities.",
                     ha="center", fontsize=9)
            with destination.open("xb") as stream:
                fig.savefig(stream, format="png", dpi=150, metadata={
                    "Description": "Two rows: grids 12x12 and 18x18. Each row has a reference and three neighborhood entropy Softmax maps; not model predictions.",
                    "EntropyAnalysis": json.dumps(figure_data, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                })
        finally:
            fig.clear()


def check_output(output_dir, source_root):
    output, source = Path(output_dir).expanduser().resolve(), Path(source_root).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source result tree; use a sibling directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    return output


def run(results_dir, output_dir=None, data_root=None, limit=None):
    # Check dependencies and all inputs before creating outputs. No model imports.
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    source, samples = prepare_inputs(results_dir, data_root, limit)
    output = check_output(output_dir or source.with_name(source.name + "_entropy_both_12_18"), source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script_version": SCRIPT_VERSION, "source_results": str(source),
        "data_root_override": str(Path(data_root).resolve()) if data_root else None,
        "sample_count": len(samples), "completed_samples": 0, "status": "running",
        "definition": DEFINITION, "numeric_data": "PNG text metadata key EntropyAnalysis",
        "numpy_version": np.__version__, "pillow_version": Image.__version__,
    }

    def save_manifest():
        with (output / "run.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)

    save_manifest()
    print(f"run_entropy_18_12 {SCRIPT_VERSION} | {len(samples)} samples | CPU only", flush=True)
    try:
        for index, sample in enumerate(samples, start=1):
            with Image.open(sample["image"]) as source_image:
                image = source_image.convert("RGB")
            geometry = sample["geometry"]
            base, mosaic = prepare_views(image, geometry)
            base_analyses = analyze_grids(base)
            anyres_analyses = analyze_grids(mosaic, geometry.content_box)
            target = output / sample["folder"].name
            target.mkdir(exist_ok=False)
            gt = {0: "Normal", 1: "Abnormal"}.get(sample["gt"], "Unknown")
            caption = f"ID: {sample['id']} | GT: {gt} | No LLM prediction"
            provenance = {"script_version": SCRIPT_VERSION, "definition": DEFINITION,
                          "source_metadata": str(sample["folder"] / "metadata.json"),
                          "image": str(sample["image"]), "id": sample["id"], "gt": sample["gt"],
                          "geometry": asdict(geometry)}
            draw_overview(base, base_analyses, target / "base_entropy_overview.png", "Base", caption, provenance)
            draw_overview(image, anyres_analyses, target / "anyres_entropy_overview.png", "AnyRes", caption,
                          provenance, geometry.content_box)
            manifest["completed_samples"] = index
            save_manifest()
            print(f"[{index}/{len(samples)}] {sample['id']}: saved two overviews -> {target}", flush=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_manifest()
        raise
    save_manifest()
    print(f"Done: {output}", flush=True)
    return output


def configured_path(value):
    if not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--results-dir", type=Path, help="Previous feature-score result directory, or one sample folder")
    parser.add_argument("--output-dir", type=Path, help="New/empty output directory, separate from input results")
    parser.add_argument("--data-root", type=Path, help="Optional new dataset root to relocate original images")
    parser.add_argument("--limit", type=positive_int, help="First N samples; default all samples")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = args.results_dir or configured_path(RESULTS_DIR)
    if results is None:
        parser.error("Set RESULTS_DIR or supply --results-dir")
    try:
        run(results, args.output_dir or configured_path(OUTPUT_DIR),
            args.data_root or configured_path(DATA_ROOT), args.limit)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
