"""Redraw cached Global scores at token resolution, labeling tile Top-K means.

CPU only: no feature extraction, model, Torch or GPU. Accepts the original
visualize_scores.py results or a visualize_patch_means.py results directory
whose source token archives still exist. Source files are never modified.
"""
from __future__ import annotations

# Relative paths are resolved against this script's project directory.
RESULTS_DIR = r"outputs/feature_scores_02"
OUTPUT_DIR = r""
DATA_ROOT = r""
SOURCE_RESULTS_DIR = r""  # Optional relocated ORIGINAL token-cache directory.
TOP_K = 10  # Ten tokens, not ten percent.
SCRIPT_VERSION = "1.0-token-heatmap-topk"

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from visualize_patch_means import (
    COLORMAP, LABELS, LAYER_GROUPS, MISSING_COLOR, OVERLAY_ALPHA, PANEL_STAGES,
    SCORE_SCALING, SIGLIP_STAGES, STAGES, configured_path, find_score_file,
    load_scores, read_json, resolve_image, sample_folders, scale_scores_01,
    tile_rectangles, validate_geometry,
)
from llava_pruning.score_visualization import ScoreGeometry, stitch_score_map


def build_token_panels(values):
    """Average selected layers at each token position, NOT across spatial tokens."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != len(SIGLIP_STAGES) or min(values.shape) < 1:
        raise ValueError("Expected Global scores [4 layers, tiles, tokens]")
    scale_scores_01(values)  # Validate range/Inf without altering raw values.
    # A missing score in ANY selected layer makes that combined token missing.
    return np.stack([values[[SIGLIP_STAGES.index(s) for s in group]].mean(axis=0)
                     for group in LAYER_GROUPS.values()])


def topk_means(panels, top_k=TOP_K):
    """Largest SIGNED scores, per panel/tile; all finite cached tokens eligible.

    Preserve the old mean script's inclusion of padding-context tokens. Do not
    rank absolute values, pool crops, or replace the token-level heatmap.
    With fewer than K finite tokens, use those available; all missing -> NaN.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top-k must be a positive integer (number of tokens)")
    values = np.asarray(panels, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("Expected scores [panels, tiles, tokens]")
    scale_scores_01(values)
    means = np.full(values.shape[:2], np.nan, dtype=np.float64)
    finite_counts = np.isfinite(values).sum(axis=-1)
    selected_counts = np.minimum(finite_counts, top_k)
    for panel, tile in np.ndindex(values.shape[:2]):
        finite = values[panel, tile][np.isfinite(values[panel, tile])]
        k = int(selected_counts[panel, tile])
        if k:
            means[panel, tile] = np.partition(finite, finite.size - k)[-k:].mean()
    return means, finite_counts, selected_counts


def _saved_path(raw, relative_to):
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


def _run_metadata(root):
    path = root / "run.json"
    if not path.is_file() and (root / "metadata.json").is_file():
        path = root.parent / "run.json"
    return read_json(path) if path.is_file() else {}


def prepare_samples(results_dir, data_root=None, source_results_dir=None):
    """Resolve raw archives even when the input folder only has block-mean plots."""
    root = Path(results_dir).expanduser().resolve()
    folders = sample_folders(root)
    run_info = _run_metadata(root)
    override = Path(source_results_dir).expanduser().resolve() if source_results_dir else None
    protected = {root}
    samples = []
    for folder in folders:
        display_meta = read_json(folder / "metadata.json")
        derived = 'source_scores' in display_meta or 'source_results' in run_info
        if derived:
            if override:
                source_folder = override if (override / 'metadata.json').is_file() else override / folder.name
                archive = find_score_file(source_folder)
                protected.add(override)
            else:
                raw = display_meta.get('source_scores')
                archive = _saved_path(raw, folder) if raw else None
                if archive is None or not archive.is_file():
                    cache_root = run_info.get('source_results')
                    source_folder = (_saved_path(cache_root, root) / folder.name if cache_root
                                     else archive.parent if archive else folder)
                    try:
                        archive = find_score_file(source_folder)
                    except (OSError, ValueError) as error:
                        raise FileNotFoundError(
                            f"Original token archive missing for {folder.name}. Tile means/PNGs "
                            "cannot recover Top-K. Use --source-results-dir with the original "
                            "visualize_scores.py cache directory.") from error
            metadata_path = archive.parent / 'metadata.json'
            metadata = read_json(metadata_path)
            if validate_geometry(metadata) != validate_geometry(display_meta):
                raise ValueError(f"Source and derived geometry disagree for {folder.name}")
            if str(metadata.get('id', folder.name)) != str(display_meta.get('id', folder.name)):
                raise ValueError(f"Source and derived sample IDs disagree for {folder.name}")
        else:
            archive = find_score_file(folder)
            metadata_path = folder / 'metadata.json'
            metadata = display_meta
        geometry = validate_geometry(metadata)
        source_root = archive.parent.parent
        if not (source_root / 'run.json').is_file():
            source_root = archive.parent
        protected.add(source_root.resolve())
        source_run = _run_metadata(source_root)
        # The derived metadata may contain an already-relocated image path.
        image_meta = dict(metadata)
        if derived and not data_root and Path(display_meta.get('image', '')).is_file():
            image_meta['image'] = display_meta['image']
        image_path = resolve_image(image_meta, source_run, data_root)
        with Image.open(image_path) as image:
            if image.size != geometry['original_size']:
                raise ValueError(f"Image size differs from cached geometry: {image_path}")
            image.verify()
        samples.append({'folder': folder.name, 'metadata': metadata, 'geometry': geometry,
                        'archive': archive.resolve(), 'source_metadata': metadata_path.resolve(),
                        'image': image_path})
    return root, samples, protected


def check_output(output_dir, protected):
    output = Path(output_dir).expanduser().resolve()
    for source in protected:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate from all input/cache directories; use a sibling")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    return output


def render(image, panels, means, geometry, metadata, destination, top_k):
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    if Path(destination).exists():
        raise FileExistsError(destination)
    geom = ScoreGeometry(**geometry)
    rectangles = tile_rectangles(geometry)
    norm = Normalize(0, 1)
    cmap = colormaps[COLORMAP].with_extremes(bad=MISSING_COLOR)
    scaled_means = scale_scores_01(means)
    with rc_context({'font.family': 'DejaVu Sans', 'font.size': 9}):
        fig = Figure(figsize=(16, 9), layout='constrained')
        FigureCanvasAgg(fig)
        fig.get_layout_engine().set(w_pad=.12, h_pad=.08)
        axes = fig.subplots(2, 4, squeeze=False)
        try:
            for ax, stage in zip(axes.flat, (s for row in PANEL_STAGES for s in row)):
                index = None if stage == 'original' else STAGES.index(stage)
                ax.imshow(image, extent=(0, 1, 1, 0), interpolation='nearest')
                if index is not None:
                    # Repeat each token only over its true patch support, stitch,
                    # then unpad. No block-mean fill and no heatmap smoothing.
                    heat = stitch_score_map(scale_scores_01(panels[index]), geom)
                    rgba = cmap(norm(np.ma.masked_invalid(heat)))
                    rgba[..., 3] = np.where(np.isfinite(heat), OVERLAY_ALPHA, 1.)
                    ax.imshow(rgba, extent=(0, 1, 1, 0), interpolation='nearest')
                for tile, rect in enumerate(rectangles):
                    if rect is None:
                        continue
                    x, y, w, h = rect
                    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='white', linewidth=.7))
                    if index is not None:
                        value = scaled_means[index, tile]
                        label = f'{tile + 1}\n{value:.4f}' if np.isfinite(value) else f'{tile + 1}\nN/A'
                        ax.text(x + w / 2, y + h / 2, label, color='white', fontsize=8,
                                ha='center', va='center',
                                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': .65, 'pad': 2})
                ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(1, 0))
                ax.set_aspect(image.height / image.width)
                ax.set_title('Original' if index is None else LABELS[index], fontsize=11)
            gt = {0: 'Normal', 1: 'Abnormal'}.get(metadata.get('gt'), 'Unknown')
            reference = metadata.get('global_reference', 'saved reference')
            fig.suptitle(f"ID: {metadata.get('id', '')} | GT: {gt} | No LLM prediction\n"
                         f'Token-level Global heatmaps | Tile labels: Top-{top_k} mean | Reference: {reference}', fontsize=11)
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal',
                         fraction=.04, pad=.025, shrink=.75, aspect=50,
                         label='Token score [0, 1]: (raw negative-cosine score + 1) / 2')
            fig.supxlabel(
                f'Color = EACH token score; number = mean of the {top_k} largest finite scores in that tile (or all if fewer).\n'
                'Mean panels: average selected layers at each token FIRST, then select Top-K. Global reference unchanged.\n'
                'As in source config, ranking includes padding-context tokens; only image content is drawn. '
                'Gray = missing / unencoded. Scores are not probabilities.', fontsize=8)
            fig.savefig(destination, dpi=150, facecolor='white')
        finally:
            fig.clear()


def write_table(path, means, counts, selected_counts, geometry, top_k):
    rectangles = tile_rectangles(geometry)
    scaled = scale_scores_01(means)
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stage', 'tile_id', 'visible', 'top_k', 'selected_tokens',
                         'finite_tokens', 'topk_raw_mean', 'topk_score_01', 'source_layers'])
        for row, stage in enumerate(STAGES):
            for tile, rect in enumerate(rectangles):
                raw, value = means[row, tile], scaled[row, tile]
                writer.writerow([stage, tile + 1, rect is not None, top_k,
                                 int(selected_counts[row, tile]), int(counts[row, tile]),
                                 float(raw) if np.isfinite(raw) else '',
                                 float(value) if np.isfinite(value) else '', '+'.join(LAYER_GROUPS[stage])])


def write_json(path, data):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run(results_dir, output_dir=None, data_root=None, source_results_dir=None, top_k=TOP_K):
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top-k must be a positive integer')
    root, samples, protected = prepare_samples(results_dir, data_root, source_results_dir)
    output = check_output(output_dir or root.with_name(root.name + f'_token_top{top_k}'), protected)
    # Validate the plotting dependency before creating output.
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    output.mkdir(parents=True, exist_ok=True)
    definition = {
        'score_kind': 'global', 'top_k': top_k, 'top_k_unit': 'tokens, not percent',
        'ranking': 'largest signed finite raw scores within each panel and AnyRes tile',
        'token_scope': 'all finite cached tokens, including padding-context; same scope as source means',
        'cross_layer_order': 'mean raw scores at each token position across selected layers, then per-tile Top-K mean',
        'missing_policy': 'missing selected layer -> missing group token; fewer than K finite tokens -> use available; none -> NaN',
        'score_scaling': SCORE_SCALING, 'global_reference_recomputed': False,
        'heatmap': 'token-level, nearest-neighbor; no spatial averaging; fixed colorbar [0,1]',
        'annotation': 'tile ID and scaled Top-K mean; not the value of every colored token',
        'colormap': COLORMAP, 'overlay_alpha': OVERLAY_ALPHA, 'stages': STAGES,
        'panels': PANEL_STAGES, 'model_loaded': False, 'llm_generation': False,
    }
    manifest = {**definition, 'script_version': SCRIPT_VERSION, 'input_results': str(root),
                'source_results_override': str(source_results_dir) if source_results_dir else None,
                'data_root_override': str(data_root) if data_root else None,
                'sample_count': len(samples), 'completed_samples': 0, 'complete': False, 'status': 'running'}
    write_json(output / 'run.json', manifest)
    print(f'{SCRIPT_VERSION}: {len(samples)} samples, Top-{top_k} tokens per tile; CPU only.', flush=True)
    try:
        for index, sample in enumerate(samples, start=1):
            metadata, geometry = sample['metadata'], sample['geometry']
            values = load_scores(sample['archive'], geometry, metadata)['global']
            panels = build_token_panels(values)
            means, counts, selected = topk_means(panels, top_k)
            target = output / sample['folder']
            target.mkdir(exist_ok=False)
            with Image.open(sample['image']) as original:
                render(original.convert('RGB'), panels, means, geometry, metadata,
                       target / 'token_topk.png', top_k)
            write_table(target / 'token_topk.csv', means, counts, selected, geometry, top_k)
            write_json(target / 'metadata.json', {
                **definition, 'script_version': SCRIPT_VERSION,
                'id': metadata.get('id', sample['folder']), 'gt': metadata.get('gt'),
                'image': str(sample['image']), 'geometry': geometry,
                'source_scores': str(sample['archive']), 'source_metadata': str(sample['source_metadata']),
                'global_reference': metadata.get('global_reference', 'unknown'),
                'global_reference_transform': metadata.get('global_reference_transform'),
            })
            manifest['completed_samples'] = index
            write_json(output / 'run.json', manifest)
            print(f"[{index}/{len(samples)}] {sample['folder']} -> token_topk.png", flush=True)
        manifest.update(complete=True, status='complete')
        write_json(output / 'run.json', manifest)
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', manifest)
        raise
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=SCRIPT_VERSION)
    parser.add_argument('--results-dir', help='Original token results OR the 4.2 patch-mean results')
    parser.add_argument('--source-results-dir', help='Relocated original token results if recorded paths are unavailable')
    parser.add_argument('--data-root', help='Relocated original image root, if necessary')
    parser.add_argument('--output-dir', help='New/empty directory separate from all source results')
    parser.add_argument('--top-k', type=int, default=TOP_K, help='Number of highest-score tokens per tile (default 10, not 10%%)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(configured_path(args.results_dir or RESULTS_DIR),
            configured_path(args.output_dir or OUTPUT_DIR), configured_path(args.data_root or DATA_ROOT),
            configured_path(args.source_results_dir or SOURCE_RESULTS_DIR), args.top_k)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
