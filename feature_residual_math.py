"""Raw-vector local residuals. NumPy only; no feature normalization.

The AnyRes grid is logical: adjacent crops touch in token space even though
the 384/14 encoder leaves six unencoded pixels at each crop edge. Padding and
partial-padding patches are excluded, not interpreted as zero features.
"""
from __future__ import annotations

import numpy as np

WINDOWS = (3, 5, 7)


def _box_sum(values, radius):
    """Truncated box sums along the first two axes, including the center."""
    height, width = values.shape[:2]
    integral = np.zeros((height + 1, width + 1) + values.shape[2:], dtype=values.dtype)
    integral[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    y, x = np.arange(height), np.arange(width)
    top, bottom = np.maximum(y - radius, 0), np.minimum(y + radius + 1, height)
    left, right = np.maximum(x - radius, 0), np.minimum(x + radius + 1, width)
    return (integral[bottom[:, None], right] - integral[top[:, None], right]
            - integral[bottom[:, None], left] + integral[top[:, None], left])


def local_residuals(features, valid=None, windows=WINDOWS, channel_chunk=128):
    """Return [window, row, col] L2, 1-cosine and valid-neighbor counts.

    Accumulate in float64 in bounded channel chunks instead of allocating
    [H,W,window,window,C]. Only the saved scalar scores are cast to float32.
    A valid zero vector has defined L2 but undefined cosine. No-neighbor
    centers, masked centers and undefined cosine are NaN, never zero.
    """
    features = np.asarray(features)
    if features.ndim != 3 or min(features.shape) < 1 or not np.issubdtype(features.dtype, np.number):
        raise ValueError("Expected nonempty real features [height, width, channels]")
    if np.iscomplexobj(features) or not np.isfinite(features).all():
        raise ValueError("Nonfinite or complex features are not supported")
    if type(channel_chunk) is not int or channel_chunk < 1:
        raise ValueError("channel_chunk must be a positive integer")
    windows = tuple(windows)
    if not windows or any(type(w) is not int or w < 3 or w % 2 != 1 for w in windows):
        raise ValueError("windows must be odd integers >= 3")
    valid = np.ones(features.shape[:2], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != features.shape[:2]:
        raise ValueError("Validity mask does not match the feature grid")
    counts = np.stack([_box_sum(valid.astype(np.int64), w // 2) - valid for w in windows])
    defined = valid[None] & (counts > 0)
    denominator = np.maximum(counts, 1)[..., None]
    square_error = np.zeros(counts.shape, dtype=np.float64)
    dot = np.zeros_like(square_error)
    mean_square = np.zeros_like(square_error)
    center_square = np.zeros(features.shape[:2], dtype=np.float64)
    for start in range(0, features.shape[2], channel_chunk):
        center = np.asarray(features[..., start:start + channel_chunk], dtype=np.float64)
        masked = np.where(valid[..., None], center, 0.)
        center_square += np.einsum("...c,...c->...", center, center)
        for wi, window in enumerate(windows):
            mean = (_box_sum(masked, window // 2) - masked) / denominator[wi]
            difference = center - mean
            square_error[wi] += np.einsum("...c,...c->...", difference, difference)
            dot[wi] += np.einsum("...c,...c->...", center, mean)
            mean_square[wi] += np.einsum("...c,...c->...", mean, mean)
    l2 = np.where(defined, np.sqrt(square_error), np.nan)
    norm_product = np.sqrt(center_square)[None] * np.sqrt(mean_square)
    cosine_defined = defined & (norm_product > 0)
    similarity = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(dot, norm_product, out=similarity, where=cosine_defined)
    cosine = 1. - np.clip(similarity, -1., 1.)
    if np.isinf(l2).any() or np.isinf(norm_product).any():
        raise ValueError("Residual computation overflowed float64")
    if np.any(l2[np.isfinite(l2)] > np.finfo(np.float32).max):
        raise ValueError("Residual score cannot be represented in float32")
    return {"l2": l2.astype(np.float32), "cosine": cosine.astype(np.float32),
            "neighbor_counts": np.where(valid[None], counts, 0).astype(np.int32)}


def tile_validity(geometry):
    """One boolean per token: its entire pixel patch must be image content."""
    side, patch = geometry.token_side, geometry.patch_size
    y, x = np.indices((side, side)) * patch
    masks = []
    for index in range(geometry.tile_count):
        left, top, right, bottom = geometry.tile_content_box(index)
        masks.append((x >= left) & (y >= top) & (x + patch <= right) & (y + patch <= bottom))
    return np.asarray(masks).reshape(geometry.tile_count, side * side)


def stitch_tokens(tokens, geometry):
    """[crop, token, ...] -> [logical row, logical col, ...], row-major."""
    tokens = np.asarray(tokens)
    cols, rows = geometry.grid
    side = geometry.token_side
    if tokens.shape[:2] != (geometry.tile_count, side * side):
        raise ValueError("Tile tokens do not match geometry")
    tail = tokens.shape[2:]
    blocks = tokens.reshape((rows, cols, side, side) + tail)
    return blocks.transpose((0, 2, 1, 3) + tuple(range(4, blocks.ndim))).reshape(
        (rows * side, cols * side) + tail)


def _split_window_scores(values, geometry):
    cols, rows = geometry.grid
    side = geometry.token_side
    return values.reshape(len(WINDOWS), rows, side, cols, side).transpose(0, 1, 3, 2, 4).reshape(
        len(WINDOWS), geometry.tile_count, side * side)


def stage_residuals(features, geometry):
    """Compute one captured layer, preserving Base then row-major crop order."""
    features = np.asarray(features)
    side = geometry.token_side
    if features.ndim != 3 or features.shape[:2] != (1 + geometry.tile_count, side * side):
        raise ValueError("Feature batch must match Base + native AnyRes token geometry")
    valid = tile_validity(geometry)
    base = local_residuals(features[0].reshape(side, side, -1))
    tiles = local_residuals(stitch_tokens(features[1:], geometry), stitch_tokens(valid, geometry))
    return {
        "base_l2": base["l2"].reshape(len(WINDOWS), -1),
        "base_cosine": base["cosine"].reshape(len(WINDOWS), -1),
        "base_neighbor_counts": base["neighbor_counts"].reshape(len(WINDOWS), -1),
        "tile_l2": _split_window_scores(tiles["l2"], geometry),
        "tile_cosine": _split_window_scores(tiles["cosine"], geometry),
        "neighbor_counts": _split_window_scores(tiles["neighbor_counts"], geometry),
        "base_valid": np.ones(side * side, dtype=bool), "tile_valid": valid,
    }
