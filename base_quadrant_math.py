"""Token-level scores against spatially corresponding Base quadrant means."""
from __future__ import annotations

import numpy as np

from feature_residual_math import tile_validity
from runVFlowOpt_unit import unit_token_features


METRICS = ("neg_cos", "l2")
QUADRANTS = ("top_left", "top_right", "bottom_left", "bottom_right")


def token_layout(geometry):
    """Map native token centers through the actual AnyRes resize/padding.

    Base directly resizes the whole image to a square. AnyRes coordinates are
    mapped back through the padded canvas. Midline ties go right/bottom.
    Region IDs are actual AnyRes crop IDs, not an assumed equal 3x3 partition.
    """
    side, patch, size = geometry.token_side, geometry.patch_size, geometry.tile_size
    if side < 2:
        raise ValueError("Four Base quadrants require at least a 2x2 token grid")
    yy, xx = np.indices((side, side), dtype=np.float64)
    x, y = (xx.ravel() + .5) * patch, (yy.ravel() + .5) * patch
    base_x, base_y = x / size, y / size
    quadrant = lambda x, y: ((y >= .5).astype(np.int32) * 2 + (x >= .5).astype(np.int32))
    base_q = quadrant(base_x, base_y)
    left, top = geometry.paste_xy
    width, height = geometry.resized_size
    cols, rows = geometry.grid
    base_col = np.floor((base_x * width + left) / size).astype(np.int32)
    base_row = np.floor((base_y * height + top) / size).astype(np.int32)
    if np.any((base_col < 0) | (base_col >= cols) | (base_row < 0) | (base_row >= rows)):
        raise ValueError("Base centers mapped outside the AnyRes canvas")
    tile_q = []
    for index in range(geometry.tile_count):
        row, col = divmod(index, cols)
        tile_q.append(quadrant((col * size + x - left) / width,
                               (row * size + y - top) / height))
    valid = tile_validity(geometry)
    return {"base_quadrant": base_q,
            "tile_quadrant": np.where(valid, np.asarray(tile_q), -1).astype(np.int32),
            "base_region": base_row * cols + base_col,
            "base_valid": np.ones(side * side, dtype=bool), "tile_valid": valid}


def _compare(tokens, references, valid):
    delta = tokens - references
    l2 = np.sqrt(np.einsum("...c,...c->...", delta, delta))
    products = np.linalg.norm(tokens, axis=-1) * np.linalg.norm(references, axis=-1)
    neg_cos = np.full(products.shape, np.nan, dtype=np.float64)
    np.divide(-np.einsum("...c,...c->...", tokens, references), products,
              out=neg_cos, where=valid & (products > 0))
    return {"neg_cos": np.clip(neg_cos, -1., 1.).astype(np.float32),
            "l2": np.where(valid, l2, np.nan).astype(np.float32)}


def score_quadrants(features, geometry):
    """One encoder layer: both views reference Base means, never tile means.

    Mean includes all Base tokens in that quadrant, including the queried Base
    token. It is neither leave-one-out nor a local-neighborhood residual.
    """
    values = np.asarray(features)
    expected = (1 + geometry.tile_count, geometry.token_side ** 2)
    if values.ndim != 3 or values.shape[:2] != expected:
        raise ValueError("Expected Base + AnyRes features matching native token geometry")
    unit = unit_token_features(values).astype(np.float64)
    layout = token_layout(geometry)
    counts = np.bincount(layout["base_quadrant"], minlength=4)
    if np.any(counts == 0):
        raise ValueError("Each Base quadrant must contain at least one token")
    means = np.stack([unit[0, layout["base_quadrant"] == q].mean(axis=0) for q in range(4)])
    base = _compare(unit[0], means[layout["base_quadrant"]], layout["base_valid"])
    tiles = _compare(unit[1:], means[np.maximum(layout["tile_quadrant"], 0)], layout["tile_valid"])
    return {**layout, **{"base_" + key: value for key, value in base.items()},
            **{"tile_" + key: value for key, value in tiles.items()},
            "base_quadrant_counts": counts.astype(np.int32),
            "base_quadrant_mean_norms": np.linalg.norm(means, axis=-1).astype(np.float32)}


def region_totals(scores, stage_index, view, metric, geometry):
    """Return signed raw sums, finite counts and geometric counts per crop.

    A token contributes once, by its center; never area-weighted or rescaled.
    Empty/undefined regions have NaN sums rather than misleading zeros.
    """
    if view not in ("base", "tile") or metric not in METRICS:
        raise ValueError("Unknown view/metric")
    values = np.asarray(scores[f"{view}_{metric}"][stage_index])
    valid = np.asarray(scores[f"{view}_valid"][stage_index], dtype=bool)
    groups = (scores["base_region"][stage_index] if view == "base" else
              np.broadcast_to(np.arange(geometry.tile_count)[:, None], values.shape))
    sums = np.full(geometry.tile_count, np.nan, dtype=np.float64)
    counts = np.zeros(geometry.tile_count, dtype=np.int32)
    valid_counts = counts.copy()
    for index in range(geometry.tile_count):
        mask = valid & (groups == index)
        selected = mask & np.isfinite(values)
        valid_counts[index] = int(mask.sum())
        counts[index] = int(selected.sum())
        if counts[index]:
            sums[index] = values[selected].sum(dtype=np.float64)
    return sums, counts, valid_counts
