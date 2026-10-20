#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Haotian Liu
# Copyright 2024 Senqiao Yang
# Modified for non-CLS SigLIP / LLaVA-OneVision and single-file inference.
"""VisionZip 单文件主实验：放入原 llava_pruning-ex 目录即可运行。

CUDA_VISIBLE_DEVICES=5 python runVisionZip.py --output-dir outputs/visionzip_70_01

复用本工程模型加载、MVTec/VisA 数据读取、提示词、分类指标和 Lat/Mem 计量。
主图和每个 AnyRes 裁块独立使用 --prune-rate 设定的剪枝率（默认70）；0仍走本方法。
生成所有图片的透明度可视化、最终分类 CSV/JSONL、指标与 token 审计。

官方源码提交: 8f86b55c6f000eb033e6912538af2dd7dcb30502
https://github.com/JIA-Lab-research/VisionZip/tree/8f86b55c6f000eb033e6912538af2dd7dcb30502
无 CLS 排名: https://arxiv.org/html/2412.04467v1#A2 Algorithm 3
保持官方 anchor + mean(assigned OTHER tokens)、K 头平均和稀疏 AnyRes 顺序。
本文件末尾附完整 Apache-2.0 许可证。真实结果需在原服务器权重上运行。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import types
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except ImportError:
    torch = None  # --help / --self-test do not require the model/CUDA stack.

PROJECT = Path(__file__).resolve().parent
DEFAULT_MODEL = "/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/"
DEFAULT_INPUT = "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"
PRUNE_RATE = 70
SOURCE_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"
SOURCE_URL = f"https://github.com/JIA-Lab-research/VisionZip/tree/{SOURCE_COMMIT}"
SCRIPT_VERSION = "3.1-zero-identity"


# Reuse the project's loader, dataset protocol, metrics and passive TTFT timer.
from llava_pruning.backend import LlavaBackend
from llava_pruning.data import load_samples
from llava_pruning.metrics import Accuracy, METRIC_RULE, OPTION
from llava_pruning.visualization import prediction_caption
from benchmark_fastv import (DEFAULT_DATA_ROOTS, DEFINITIONS, benchmark_prompt,
                             measurement_summary, validate_single_gpu, warmup_indices)

OFFICIAL_SOURCE_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"


DOMINANT_RATIO = 54 / 64


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


def token_budget(token_count, prune_rate=0.70, dominant_count=None, contextual_count=None):
    """Return (dominant, contextual), retaining ceil(N * (1 - rate)).

    With no reduction all tokens are dominant (no contextual merging).
    Otherwise the split follows the paper's 54:10 budget. An explicit pair overrides
    the rate. A single explicit count allocates the rest of the rate-derived
    budget to the other category. Decimal arithmetic avoids float artefacts
    such as ceil(10 * (1 - 0.70)) == 4.
    """
    token_count = _count(token_count, "token_count")
    if isinstance(prune_rate, bool) or not isinstance(prune_rate, Real):
        raise ValueError("prune_rate must be a finite number between 0 and 1")
    if not math.isfinite(float(prune_rate)) or not 0 <= prune_rate <= 1:
        raise ValueError("prune_rate must be a finite number between 0 and 1")
    retained = int((Decimal(token_count) * (Decimal(1) - Decimal(str(prune_rate)))).to_integral_value(rounding=ROUND_CEILING))
    if dominant_count is None and contextual_count is None:
        dominant_count = token_count if retained == token_count else round(retained * DOMINANT_RATIO)
        contextual_count = retained - dominant_count
    elif dominant_count is None:
        contextual_count = _count(contextual_count, "contextual_count")
        dominant_count = retained - contextual_count
    elif contextual_count is None:
        dominant_count = _count(dominant_count, "dominant_count")
        contextual_count = retained - dominant_count
    dominant_count = _count(dominant_count, "dominant_count")
    contextual_count = _count(contextual_count, "contextual_count")
    if dominant_count + contextual_count > token_count:
        raise ValueError("dominant_count + contextual_count cannot exceed token_count")
    return dominant_count, contextual_count


def _validate_inputs(features, scores, metric, dominant_count, contextual_count):
    for name, value, ndim in (("features", features, 3), ("scores", scores, 2), ("metric", metric, 3)):
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}-dimensional torch tensor")
        if not value.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    if features.shape[:2] != scores.shape or features.shape[:2] != metric.shape[:2]:
        raise ValueError("features, scores and metric must have matching batch/token dimensions")
    if features.shape[0] < 1 or features.shape[2] < 1 or metric.shape[2] < 1:
        raise ValueError("batch and feature dimensions must be nonempty")
    if not features.device == scores.device == metric.device:
        raise ValueError("features, scores and metric must be on the same device")
    dominant_count = _count(dominant_count, "dominant_count")
    contextual_count = _count(contextual_count, "contextual_count")
    if dominant_count + contextual_count > features.shape[1]:
        raise ValueError("dominant_count + contextual_count cannot exceed the token count")
    return dominant_count, contextual_count


@dataclass
class TokenDecisions:
    """GPU-only packing indices; optional audit tensors are exported after timing."""
    token_count: int
    retained_count: int
    dominant_indices: Any
    audit: dict | None = None

    def export(self):
        if self.audit is None:
            raise ValueError("Detailed decisions were not captured; enable --save-prune-vis")
        arrays = {key: value.detach().cpu().tolist() for key, value in self.audit.items()}
        arrays["dominant_indices"] = self.dominant_indices.detach().cpu().tolist()
        return [
            {"original_token_count": self.token_count,
             "retained_token_count": self.retained_count,
             **{key: value[index] for key, value in arrays.items()}}
            for index in range(self.dominant_indices.shape[0])
        ]


def compress_tokens(features, scores, metric, dominant_count, contextual_count, *, capture=False):
    """Return compressed [B,D+C,F] and GPU-only decisions for sparse packing.

    features [B,N,F], scores [B,N], metric [B,N,K]. Dominant top-k features
    retain spatial order. Context follows evenly spaced anchors of ascending
    non-dominant tokens; other tokens join their maximum-cosine anchor.

    Optional audit decisions use original patch indices. cluster_assignments has N entries:
    -1 for dominant/dropped tokens, otherwise the contextual output slot
    (0..C-1). Anchors and their merged tokens share that assignment.
    merged_indices excludes anchors. discarded_indices is nonempty only at
    contextual_count=0. importance contains raw received attention scores.

    Feature computation never casts or moves tensors, preserving source FP16
    arithmetic. Source has no key-norm epsilon: invalid norms fail clearly.
    """
    dominant_count, contextual_count = _validate_inputs(
        features, scores, metric, dominant_count, contextual_count
    )
    batch, token_count, feature_dim = features.shape
    ranked = scores.topk(dominant_count, dim=1).indices
    mask = torch.ones((batch, token_count), dtype=torch.bool, device=features.device)
    mask.scatter_(1, ranked, False)
    spatial = torch.arange(token_count, device=features.device).expand(batch, -1)
    dominant_indices = spatial[~mask].view(batch, dominant_count)
    dominant_features = features.masked_select(~mask.unsqueeze(-1)).view(batch, dominant_count, feature_dim)
    remaining_count = token_count - dominant_count
    remaining_indices = spatial[mask].view(batch, remaining_count)
    filtered_features = features.masked_select(mask.unsqueeze(-1)).view(batch, remaining_count, feature_dim)
    assignments = (torch.full((batch, token_count), -1, device=features.device, dtype=torch.long)
                   if capture else None)
    contextual_indices = spatial[:, :0]
    merged_indices = spatial[:, :0]
    merge_counts = features.new_empty((batch, 0))
    contextual_features = features[:, :0, :]

    if contextual_count:
        filtered_metric = metric[mask].view(batch, remaining_count, metric.shape[2])
        norms = filtered_metric.norm(dim=-1, keepdim=True)
        if not bool((norms > 0).all()) or not bool(torch.isfinite(norms).all()):
            raise ValueError("VisionZip encountered a zero/non-finite key norm; source normalization would be invalid")
        normalized = filtered_metric / norms
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("VisionZip key normalization produced non-finite values")
        step = max(1, remaining_count // contextual_count)
        anchors = torch.arange(0, remaining_count, step, device=features.device)[:contextual_count]
        non_anchor_mask = torch.ones(remaining_count, dtype=torch.bool, device=features.device)
        non_anchor_mask[anchors] = False
        contextual_indices = remaining_indices[:, anchors]
        merged_indices = remaining_indices[:, non_anchor_mask]
        similarity = torch.bmm(normalized[:, non_anchor_mask, :], normalized[:, anchors, :].transpose(1, 2))
        if not bool(torch.isfinite(similarity).all()):
            raise ValueError("VisionZip cosine similarity produced non-finite values")
        cluster = similarity.argmax(dim=2)
        one_hot = torch.zeros(batch, remaining_count - contextual_count, contextual_count, dtype=features.dtype, device=features.device)
        one_hot.scatter_(2, cluster.unsqueeze(-1), 1)
        merge_counts = one_hot.sum(dim=1)
        counts = merge_counts.clamp(min=1).unsqueeze(-1)
        aggregated = torch.bmm(one_hot.transpose(1, 2), filtered_features[:, non_anchor_mask, :]) / counts
        contextual_features = filtered_features[:, anchors, :] + aggregated
        if capture:
            assignments.scatter_(1, contextual_indices, torch.arange(contextual_count, device=features.device).expand(batch, -1))
            assignments.scatter_(1, merged_indices, cluster)

    compressed = torch.cat((dominant_features, contextual_features), dim=1)
    if not bool(torch.isfinite(compressed).all()):
        raise ValueError("VisionZip feature aggregation produced non-finite values")
    # No .cpu()/.tolist() here: packing consumes GPU indices directly.
    # Audit capture is optional and serialization happens AFTER model.generate.
    audit = None
    if capture:
        audit = {
            "dominant_ranked_indices": ranked,
            "contextual_indices": contextual_indices,
            "merged_indices": merged_indices,
            "non_dominant_indices": remaining_indices,
            "discarded_indices": remaining_indices if contextual_count == 0 else spatial[:, :0],
            "cluster_assignments": assignments,
            "merge_counts": merge_counts,
            "importance": scores,
        }
    decisions = TokenDecisions(token_count, dominant_count + contextual_count, dominant_indices, audit)
    return compressed, decisions


# ---------- Encoder capture / source sparse AnyRes packing ----------
def baseline_packed_count(original_size, grid, base_grid, merge_type):
    """Count the current spatial_unpad/max_9 sequence without changing features."""
    width, height = original_size
    grid_width, grid_height = grid
    rows, cols = grid_height * base_grid, grid_width * base_grid
    if width / height > cols / rows:
        scale_factor = cols / width
        new_height = int(height * scale_factor)
        rows -= 2 * ((rows - new_height) // 2)
    else:
        scale_factor = rows / height
        new_width = int(width * scale_factor)
        cols -= 2 * ((cols - new_width) // 2)
    times = math.sqrt(rows * cols / (9 * base_grid ** 2))
    if times > 1.1:
        rows, cols = int(rows // times), int(cols // times)
    if min(rows, cols) < 1:
        raise ValueError("Baseline AnyRes grid is empty")
    return base_grid ** 2 + rows * (cols + 1) + int(merge_type == "spatial_unpad_add_newl")


def pack_sparse_views(features, decisions, grid, dominant_count):
    """Algebraic equivalent of official restore_image_features_sorted, no CLS.

    Global Base remains first. High-resolution dominants follow original global
    row-major coordinates; contextual tokens are appended in tile/anchor order.
    """
    import torch
    grid_width, grid_height = grid
    if features.ndim != 3 or features.shape[0] != 1 + grid_width * grid_height:
        raise ValueError("AnyRes encoder view count does not match preprocessing grid")
    if decisions.dominant_indices.shape != (features.shape[0], dominant_count):
        raise ValueError("VisionZip decisions do not match views")
    side = math.isqrt(decisions.token_count)
    if side * side != decisions.token_count or min(grid_width, grid_height) < 1:
        raise ValueError("Expected square encoder token grids and nonempty AnyRes grid")
    if decisions.dominant_indices.device != features.device:
        raise ValueError("Packing indices and features must be on the same device")
    local = decisions.dominant_indices[1:]
    tiles = torch.arange(grid_width * grid_height, device=features.device).unsqueeze(1)
    row, col = tiles // grid_width, tiles % grid_width
    positions = ((row * side + local // side) * (grid_width * side) + col * side + local % side).flatten()
    dominants = features[1:, :dominant_count].reshape(-1, features.shape[-1])
    dominants = dominants[torch.argsort(positions)]
    contexts = features[1:, dominant_count:].reshape(-1, features.shape[-1])
    return torch.cat((features[0], dominants, contexts), dim=0)


def pack_full_views(features, decisions, grid, original_size, image_newline, merge_type):
    """Pack a complete selected grid without calling the reference model path.

    The zero-reduction endpoint of VisionZip has an ordinary dense spatial grid.
    Reproduce its geometry: unpad -> max9 resize -> row newlines -> Base first.
    Sparse packing is only valid after compression and is not an identity on a
    complete grid. Keep operation order/dtypes equal to the checkpoint contract.
    """
    from llava.model.llava_arch import unpad_image
    if merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}:
        raise ValueError("Full-grid packing requires spatial_unpad or spatial_unpad_add_newl")
    width, height = grid
    count = decisions.token_count
    side = math.isqrt(count)
    if (side * side != count or decisions.retained_count != count
            or tuple(features.shape[:2]) != (1 + width * height, count)
            or decisions.dominant_indices.shape != features.shape[:2]):
        raise ValueError("Full-grid packing requires every original token, with no contextual merge")
    expected = torch.arange(count, device=features.device).expand(features.shape[0], -1)
    if not torch.equal(decisions.dominant_indices, expected):
        raise ValueError("Full-grid packing requires original row-major token order")
    base, tiles = features[0], features[1:]
    tiles = tiles.view(height, width, side, side, -1)
    tiles = tiles.permute(4, 0, 2, 1, 3).contiguous()
    tiles = tiles.flatten(1, 2).flatten(2, 3)
    tiles = unpad_image(tiles, original_size)
    _, h, w = tiles.shape
    times = math.sqrt(h * w / (9 * side ** 2))
    if times > 1.1:
        tiles = torch.nn.functional.interpolate(
            tiles[None], [int(h // times), int(w // times)], mode="bilinear")[0]
    newline_count = tiles.shape[1]
    tiles = torch.cat((tiles, image_newline[:, None, None].expand(
        *tiles.shape[:-1], 1).to(tiles.device)), dim=-1)
    tiles = tiles.flatten(1, 2).transpose(0, 1)
    packed = torch.cat((base, tiles), dim=0)
    if merge_type == "spatial_unpad_add_newl":
        packed = torch.cat((packed, image_newline[None].to(device=packed.device, dtype=packed.dtype)), dim=0)
        newline_count += 1
    return packed, newline_count


class VisionZipAdapter:
    def __init__(self, core):
        self.core = core
        self.tower = core.get_vision_tower()
        self.capture_decisions = False
        self.decisions = None
        self.crop_metadata = None
        self.stats = {}

    def configure(self, rate):
        rate = validate_prune_rate(rate)
        self.rate = rate
        self.decisions = None
        self.crop_metadata = None
        self.stats = {"mode": "visionzip_siglip_adaptation", "requested_prune_rate_percent": rate}
        blocks = self.tower.vision_tower.vision_model.encoder.layers
        if len(blocks) != 26 or self.tower.num_patches != 729:
            raise ValueError("This adapter requires the current 26-active-block SigLIP patch14/384 tower")
        if self.core.config.image_aspect_ratio != "anyres_max_9":
            raise ValueError("VisionZip currently requires --roi-mode anyres_max_9")
        if self.core.config.mm_patch_merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}:
            raise ValueError("VisionZip requires the current spatial_unpad checkpoint for baseline accounting")
        if getattr(self.core.config, "image_token_order", "base_first") != "base_first":
            raise ValueError("Source-faithful VisionZip requires base_first order")

    def encode_images(self, images):
        import torch
        if self.core.training:
            raise ValueError("VisionZip is an inference-only method; call model.eval()")
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("Expected one flattened encoder crop batch")
        attention = self.tower.vision_tower.vision_model.encoder.layers[-1].self_attn
        captured = {}
        handles = []

        def capture_key(module, args, output):
            batch, tokens, _ = output.shape
            captured["metric"] = output.view(batch, tokens, attention.num_heads,
                                              attention.head_dim).permute(0, 2, 1, 3).mean(1)

        def capture_attention(module, args, output):
            # This custom SigLIP attention already computes and returns probabilities.
            # Mean over heads, then over query positions: Appendix A.2, Algorithm 3.
            weights = output[1]
            if weights is None:
                raise RuntimeError("Selected SigLIP attention probabilities are missing")
            captured["scores"] = weights.mean(dim=1).mean(dim=1)

        try:
            handles.append(attention.k_proj.register_forward_hook(capture_key))
            handles.append(attention.register_forward_hook(capture_attention))
            hidden = self.tower(images)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {"metric", "scores"} or hidden.shape[1] != 729:
            raise RuntimeError("Incomplete final-active-block SigLIP capture")
        dominant, contextual = token_budget(hidden.shape[1], self.rate / 100)
        zipped, self.decisions = compress_tokens(hidden, captured["scores"], captured["metric"],
                                                 dominant, contextual, capture=self.capture_decisions)
        self.dominant_count = dominant
        self.contextual_count = contextual
        self.stats = {
            "mode": "visionzip_siglip_adaptation", "source_commit": SOURCE_COMMIT,
            "requested_prune_rate_percent": self.rate,
            "selection_layer": 26, "ranking": "mean_received_attention_over_heads_and_queries",
            "encoder_views": int(hidden.shape[0]), "tokens_per_view_before": int(hidden.shape[1]),
            "tokens_per_view_after": dominant + contextual,
            "dominant_per_view": dominant, "contextual_per_view": contextual,
            "encoder_tokens_before": int(hidden.shape[0] * hidden.shape[1]),
            "encoder_tokens_after": int(hidden.shape[0] * (dominant + contextual)),
            "actual_encoder_prune_rate_percent": 100 * (1 - (dominant + contextual) / hidden.shape[1]),
            "packing": "base_compressed + global_row_major_tile_dominants + tile_contextual; no row newlines",
            "budget_scope": "per_raw_encoder_view; ceil(remaining); 54:10 when compressed, all dominant when full",
        }
        return self.core.mm_projector(zipped)

    def pack(self, image_features, image_sizes, modalities):
        from llava.mm_utils import get_anyres_image_grid_shape
        if len(image_features) != 1 or len(image_sizes) != 1 or list(modalities) != ["image"]:
            raise ValueError("VisionZip adapter supports batch size 1 with one image only")
        grid = get_anyres_image_grid_shape(image_sizes[0], self.core.config.image_grid_pinpoints,
                                           self.tower.image_size)
        full_grid = self.decisions.retained_count == self.decisions.token_count
        if full_grid:
            packed, newlines = pack_full_views(
                image_features[0], self.decisions, grid, image_sizes[0], self.core.image_newline,
                self.core.config.mm_patch_merge_type)
            self.stats["packing"] = "complete selected grid: unpad + max9 resize + row newlines + Base first"
        else:
            packed = pack_sparse_views(image_features[0], self.decisions, grid, self.dominant_count)
            newlines = 0
        # Geometry from the same config used by process_images; no extra encoder pass.
        size = int(self.tower.image_size)
        self.crop_metadata = {
            "mode": "anyres_max_9", "original_size": list(image_sizes[0]),
            "grid_patches": list(grid),
            "processed_view_sizes": [[size, size], [grid[0] * size, grid[1] * size]],
        }
        if not self.capture_decisions:
            self.decisions = None
        before = baseline_packed_count(image_sizes[0], grid, self.tower.num_patches_per_side,
                                        self.core.config.mm_patch_merge_type)
        if full_grid and packed.shape[0] != before:
            raise RuntimeError("Full-grid VisionZip packing does not match baseline token count")
        self.stats.update(baseline_packed_tokens=before, final_llm_visual_tokens=int(packed.shape[0]),
                          actual_packed_reduction_percent=100 * (1 - packed.shape[0] / before),
                          newline_tokens_after=newlines, anyres_grid=list(grid),
                          full_grid_identity=full_grid)
        return [packed]


# ---------- Per-view geometry / alpha visualization ----------
@dataclass(frozen=True)
class ScoreGeometry:
    original_size: tuple[int, int]
    grid: tuple[int, int]
    tile_size: int
    patch_size: int
    resized_size: tuple[int, int]
    paste_xy: tuple[int, int]

    @classmethod
    def create(cls, original_size, grid, tile_size, patch_size):
        if min(*original_size, *grid, tile_size, patch_size) < 1 or patch_size > tile_size:
            raise ValueError("Invalid image/token geometry")
        width, height = original_size
        target_w, target_h = grid[0] * tile_size, grid[1] * tile_size
        scale_w, scale_h = target_w / width, target_h / height
        # Match resize_and_pad_image exactly, including ceil and odd padding.
        if scale_w < scale_h:
            new_w, new_h = target_w, min(math.ceil(height * scale_w), target_h)
        else:
            new_h, new_w = target_h, min(math.ceil(width * scale_h), target_w)
        return cls(tuple(original_size), tuple(grid), tile_size, patch_size,
                   (new_w, new_h), ((target_w - new_w) // 2, (target_h - new_h) // 2))

    @property
    def token_side(self):
        return self.tile_size // self.patch_size

    @property
    def tile_count(self):
        return self.grid[0] * self.grid[1]

    def tile_content_box(self, index):
        if not 0 <= index < self.tile_count:
            raise IndexError(index)
        row, col = divmod(index, self.grid[0])
        left = self.paste_xy[0] - col * self.tile_size
        top = self.paste_xy[1] - row * self.tile_size
        width, height = self.resized_size
        clip = lambda value: max(0, min(self.tile_size, value))
        return clip(left), clip(top), clip(left + width), clip(top + height)


def tile_score_map(scores, geometry, index):
    """Exact patch support (e.g. 27*14=378 of 384), not stretched to 384."""
    side, patch = geometry.token_side, geometry.patch_size
    if np.asarray(scores).shape != (side * side,):
        raise ValueError("Score count does not match the patch embedding grid")
    result = np.full((geometry.tile_size, geometry.tile_size), np.nan, dtype=np.float32)
    support = np.asarray(scores).reshape(side, side).repeat(patch, 0).repeat(patch, 1)
    result[:side * patch, :side * patch] = support
    x0, y0, x1, y1 = geometry.tile_content_box(index)
    valid = np.zeros(result.shape, dtype=bool)
    valid[y0:y1, x0:x1] = True
    result[~valid] = np.nan
    return result


def stitch_score_map(scores, geometry):
    if np.asarray(scores).shape != (geometry.tile_count, geometry.token_side ** 2):
        raise ValueError("Score views do not match the AnyRes tile grid")
    rows = []
    for row in range(geometry.grid[1]):
        rows.append(np.concatenate([
            tile_score_map(scores[row * geometry.grid[0] + col], geometry,
                           row * geometry.grid[0] + col)
            for col in range(geometry.grid[0])], axis=1))
    padded = np.concatenate(rows, axis=0)
    left, top = geometry.paste_xy
    width, height = geometry.resized_size
    return padded[top:top + height, left:left + width]


VISIBILITY = {"dominant": 1.0, "contextual": 0.60, "merged": 0.18}


MISSING_RGB = (184, 184, 184)


LEGEND = (
    "Visibility over white: dominant 1.00 | contextual anchor 0.60 | merged source 0.18",
    "Merged source information contributes to contextual tokens; faded anchors are not deleted image content.",
    "Gray: padding / no patch coverage. Visibility is not attention or a defect probability.",
)


@lru_cache(maxsize=1)
def _font():
    # Support the existing server's older Pillow as well as Pillow 10.4+.
    try:
        return ImageFont.load_default(size=16)
    except TypeError:
        for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, 16)
            except OSError:
                pass
        return ImageFont.load_default()


def _integer_indices(value, name, count):
    indices = np.asarray(value)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer index list")
    indices = indices.astype(np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= count):
        raise ValueError(f"{name} index is outside the encoder patch grid")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate token indices")
    return indices


def token_visibility(decision, token_count):
    """Validate an exact three-state partition; do not infer missing decisions."""
    if not isinstance(decision, Mapping):
        raise ValueError("Each encoder view decision must be an object")
    required = [name + "_indices" for name in VISIBILITY]
    if not all(name in decision for name in required):
        raise ValueError("Each decision needs dominant_indices, contextual_indices, merged_indices")
    groups = [_integer_indices(decision[name], name, token_count) for name in required]
    partition = np.concatenate(groups)
    if partition.size != token_count or np.unique(partition).size != token_count:
        raise ValueError("Dominant, contextual and merged indices must be a disjoint complete partition")
    if groups[2].size and not groups[1].size:
        raise ValueError("Merged source tokens require at least one contextual target")
    visibility = np.empty(token_count, dtype=np.float32)
    for state, indices in zip(VISIBILITY, groups):
        visibility[indices] = VISIBILITY[state]
    for field in ("scores", "importance"):
        if field in decision:
            scores = np.asarray(decision[field])
            if scores.shape != (token_count,) or not np.isfinite(scores).all():
                raise ValueError("Ranking scores must be finite values for the original encoder patch grid")
    for field, expected in (("original_token_count", token_count),
                            ("retained_token_count", groups[0].size + groups[1].size)):
        if field in decision and decision[field] != expected:
            raise ValueError(f"{field} disagrees with the actual decision counts")
    for field, expected in (("dominant_ranked_indices", groups[0]),
                            ("non_dominant_indices", np.concatenate(groups[1:]))):
        if field in decision:
            actual = _integer_indices(decision[field], field, token_count)
            if not np.array_equal(np.sort(actual), np.sort(expected)):
                raise ValueError(f"{field} disagrees with the dominant/contextual/merged partition")
    if "discarded_indices" in decision and len(decision["discarded_indices"]):
        raise ValueError("This merged-state visualization requires empty discarded_indices")
    if "cluster_assignments" in decision:
        assignments = np.asarray(decision["cluster_assignments"])
        if assignments.shape != (token_count,) or assignments.dtype.kind not in "iu":
            raise ValueError("Cluster assignments must be an integer for every original encoder token")
        if (np.any(assignments[groups[0]] != -1)
                or not np.array_equal(assignments[groups[1]], np.arange(groups[1].size))
                or np.any(assignments[groups[2]] < 0)
                or np.any(assignments[groups[2]] >= groups[1].size)):
            raise ValueError("Cluster assignments disagree with contextual anchor slots")
        if "merge_counts" in decision:
            expected = np.bincount(assignments[groups[2]], minlength=groups[1].size)
            actual = np.asarray(decision["merge_counts"])
            if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.array_equal(actual, expected):
                raise ValueError("Merge counts disagree with source-to-contextual assignments")
    elif "merge_counts" in decision:
        raise ValueError("Merge counts need cluster_assignments for a reproducible audit")
    return visibility


def blend_visibility(image, visibility):
    """Nearest patch support is rasterized by the caller, never interpolated here."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    visibility = np.asarray(visibility, dtype=np.float32)
    if visibility.shape != pixels.shape[:2]:
        raise ValueError("Visibility map must match displayed source pixels")
    valid = np.isfinite(visibility)
    if np.any((visibility[valid] < 0) | (visibility[valid] > 1)):
        raise ValueError("Visibility must be between zero and one")
    alpha = np.nan_to_num(visibility, nan=0)[..., None]
    result = np.rint(pixels * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    result[~valid] = MISSING_RGB
    return Image.fromarray(result)


def _base_visibility(values, side, patch_size, processed_size):
    width, height = processed_size
    if width // patch_size != side or height // patch_size != side:
        raise ValueError("Processed view dimensions do not match the encoder patch grid")
    result = np.full((height, width), np.nan, dtype=np.float32)
    support = np.asarray(values).reshape(side, side).repeat(patch_size, 0).repeat(patch_size, 1)
    result[:support.shape[0], :support.shape[1]] = support
    return result


def _compact_stats(stats):
    if stats is None:
        return {}
    if not isinstance(stats, Mapping):
        raise ValueError("VisionZip statistics must be an object of scalar values")
    result = {}
    for key, value in stats.items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("VisionZip statistics contain a nonfinite scalar")
            result[str(key)] = value
    return result


def prepare_visionzip_views(image, pro_data, decisions, base_grid, patch_size):
    """Reconstruct actual encoder views, including ceil resize and odd padding.

    AnyRes tile ordering follows encoder input (Base then row-major tiles),
    independent of eventual packed image-token ordering.
    """
    if list(image.size) != list(pro_data.get("original_size", [])):
        raise ValueError("Source image size disagrees with preprocessing metadata")
    if (isinstance(base_grid, bool) or isinstance(patch_size, bool)
            or not isinstance(base_grid, int) or not isinstance(patch_size, int)
            or min(base_grid, patch_size) < 1):
        raise ValueError("Encoder grid and patch dimensions must be positive integers")
    if not decisions:
        raise ValueError("VisionZip visualization requires encoder view decisions")
    values = [token_visibility(decision, base_grid ** 2) for decision in decisions]
    image = image.convert("RGB")
    mode = pro_data.get("mode", "randomroi")
    sizes = pro_data.get("processed_view_sizes", [])
    views, geometry, combined = [], None, None

    def add_view(name, source, visibility, details):
        index = len(views)
        decision = decisions[index]
        counts = {state + "_tokens": len(decision[state + "_indices"]) for state in VISIBILITY}
        count = base_grid ** 2
        summary = {"view_index": index, "name": name, "encoder_grid_shape": [base_grid, base_grid],
                   "token_count": count, **counts,
                   "output_tokens": counts["dominant_tokens"] + counts["contextual_tokens"],
                   "compression_rate_percent": 100.0 * counts["merged_tokens"] / count, **details}
        views.append({"source": source, "visualization": blend_visibility(source, visibility),
                      "visibility": visibility, "summary": summary})

    if mode in {"anyres", "anyres_max_9", "anyres_only"}:
        no_base = mode == "anyres_only"
        grid = tuple(map(int, pro_data["grid_patches"]))
        if len(grid) != 2 or min(grid) < 1:
            raise ValueError("Invalid AnyRes tile grid")
        if not sizes:
            raise ValueError("AnyRes visualization needs processed_view_sizes from mm_utils")
        if no_base:
            width, height = map(int, sizes[0])
            if width % grid[0] or height % grid[1]:
                raise ValueError("AnyRes dimensions do not divide into encoder tiles")
            tile_width, tile_height = width // grid[0], height // grid[1]
        else:
            tile_width, tile_height = map(int, sizes[0])
        if tile_width != tile_height or tile_width // patch_size != base_grid:
            raise ValueError("AnyRes visualization requires square processed encoder views")
        expected_sizes = [[grid[0] * tile_width, grid[1] * tile_height]]
        if not no_base:
            expected_sizes.insert(0, [tile_width, tile_height])
        if [list(map(int, size)) for size in sizes] != expected_sizes:
            raise ValueError("AnyRes processed sizes disagree with tile-grid metadata")
        geometry = ScoreGeometry.create(image.size, grid, tile_width, patch_size)
        if len(decisions) != geometry.tile_count + int(not no_base):
            raise ValueError("Decision view count differs from Base + AnyRes encoder inputs")
        offset = int(not no_base)
        if not no_base:
            source = image.resize((tile_width, tile_height))
            add_view("global", source, _base_visibility(values[0], base_grid, patch_size, source.size),
                     {"processed_size": list(source.size), "source_box_xyxy": [0, 0, *image.size]})
        # Match resize_and_pad_image's Pillow default resampling and exact geometry.
        padded = Image.new("RGB", (grid[0] * tile_width, grid[1] * tile_height), "black")
        padded.paste(image.resize(geometry.resized_size), geometry.paste_xy)
        for tile_index in range(geometry.tile_count):
            row, col = divmod(tile_index, grid[0])
            box = (col * tile_width, row * tile_height,
                   (col + 1) * tile_width, (row + 1) * tile_height)
            source = padded.crop(box)
            add_view(f"tile_{tile_index + 1:02d}", source,
                     tile_score_map(values[offset + tile_index], geometry, tile_index),
                     {"tile_index": tile_index, "tile_row_col": [row, col],
                      "processed_size": list(source.size), "padded_canvas_box_xyxy": list(box),
                      "content_box_xyxy": list(geometry.tile_content_box(tile_index))})
        combined = blend_visibility(image.resize(geometry.resized_size),
                                    stitch_score_map(np.stack(values[offset:]), geometry))
    elif mode in {"randomroi", "ex_base_copy"}:
        boxes = ([[0, 0, *image.size]] * 3 if mode == "ex_base_copy" else
                 [[0, 0, *image.size]] + pro_data.get("roi_boxes", []))
        if len(sizes) != len(boxes) or len(decisions) != len(boxes):
            raise ValueError("Decision view count differs from Base + crop encoder inputs")
        for index, (box, size) in enumerate(zip(boxes, sizes)):
            width, height = map(int, size)
            box = list(map(int, box))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid source crop box")
            source = image.crop(box).resize((width, height))
            name = f"base_{index + 1}" if mode == "ex_base_copy" else ("global" if index == 0 else f"roi_{index:02d}")
            add_view(name, source, _base_visibility(values[index], base_grid, patch_size, source.size),
                     {"processed_size": [width, height], "source_box_xyxy": box})
    else:
        raise ValueError(f"Unsupported VisionZip visualization mode: {mode}")
    return {"views": views, "anyres_combined": combined, "geometry": geometry}


def _panel_row(panels, max_panel_width=480):
    font = _font()
    prepared = []
    for title, source in panels:
        width = min(max_panel_width, source.width)
        height = max(1, round(source.height * width / source.width))
        prepared.append((title, source.resize((width, height), Image.Resampling.NEAREST)))
    gap, label_height = 12, 30
    width = max(1200, sum(source.width for _, source in prepared) + gap * (len(prepared) - 1))
    height = max(source.height for _, source in prepared) + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw, x = ImageDraw.Draw(canvas), 0
    for title, source in prepared:
        draw.text((x + 4, 4), title, font=font, fill="black")
        canvas.paste(source, (x, label_height))
        x += source.width + gap
    return canvas


def _captioned(panel, lines):
    font = _font()
    line_height, padding = 24, 8
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(panel.width, int(max(measure.textlength(line, font=font) for line in lines)) + 2 * padding)
    header = len(lines) * line_height + 2 * padding
    canvas = Image.new("RGB", (width, panel.height + header), "white")
    canvas.paste(panel, (0, header))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((padding, padding + index * line_height), line, font=font, fill="black")
    return canvas


def save_visionzip_visualizations(image, pro_data, decisions, stats, base_grid, patch_size,
                                 output_dir, sample_id, answer=None, ground_truth=None):
    """Save every encoder view plus an overview with the actual final A/B class.

    Decisions require dominant_indices, contextual_indices and merged_indices
    in local encoder row-major coordinates; scores is optional. Returned paths
    follow other pruning visualizers' one-image list convention. No inference
    happens in this display-only module.
    """
    prepared = prepare_visionzip_views(image, pro_data, decisions, base_grid, patch_size)
    source_stats = _compact_stats(stats)
    summaries = [view["summary"] for view in prepared["views"]]
    counts = {state + "_tokens": sum(view[state + "_tokens"] for view in summaries) for state in VISIBILITY}
    before = len(summaries) * base_grid ** 2
    after = counts["dominant_tokens"] + counts["contextual_tokens"]
    caption = prediction_caption(ground_truth, answer)
    run_title = (f"VisionZip SigLIP adaptation | encoder tokens {before} -> {after} | "
                 f"token reduction {100.0 * (before - after) / before:.4f}%")
    lines = [f"ID: {sample_id} | {caption}", run_title, *LEGEND]
    panels = [("Original", image.convert("RGB"))]
    if pro_data.get("mode") != "anyres_only":
        panels.append(("Global encoder view", prepared["views"][0]["visualization"]))
    if prepared["anyres_combined"] is not None:
        panels.append(("AnyRes tiles stitched (crop padding)", prepared["anyres_combined"]))
    else:
        panels.extend((view["summary"]["name"], view["visualization"]) for view in prepared["views"][1:])
    comparison = _captioned(_panel_row(panels), lines)
    raw_decisions = []
    for decision in decisions:
        raw = {state + "_indices": np.asarray(decision[state + "_indices"], dtype=np.int64).tolist()
               for state in VISIBILITY}
        for field in ("scores", "importance", "cluster_assignments", "merge_counts",
                      "dominant_ranked_indices", "non_dominant_indices", "discarded_indices",
                      "original_token_count", "retained_token_count"):
            if field in decision:
                raw[field] = np.asarray(decision[field]).tolist()
        raw_decisions.append(raw)
    match = OPTION.match(answer) if isinstance(answer, str) else None
    metadata = {"sample_id": str(sample_id), "method": "visionzip_siglip_adapted",
                "answer": answer, "ground_truth": ground_truth,
                "predicted_class": (1 if match.group(1).upper() == "A" else 0) if match else None,
                "prediction_caption": caption, "encoder_tokens_before": before,
                "encoder_tokens_after": after, **counts,
                "encoder_compression_rate_percent": 100.0 * (before - after) / before,
                "statistics": source_stats, "views": summaries, "decisions": raw_decisions,
                "image_token_order": pro_data.get("image_token_order", "base_first"),
                "encoder_view_order": "Base (if present), then AnyRes tiles row-major / source crop order",
                "geometry": asdict(prepared["geometry"]) if prepared["geometry"] else None,
                "visibility": VISIBILITY, "visibility_background": "white", "legend": list(LEGEND),
                "spatial_policy": "original encoder patch anchors before AnyRes unpadding/downsampling/newlines"}
    # Ensure bad metadata fails before any output is created.
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
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
    comparison.save(directory / "comparison.png")
    view_paths = []
    for view in prepared["views"]:
        summary = view["summary"]
        title = (f"{summary['name']} | dominant {summary['dominant_tokens']} | "
                 f"contextual {summary['contextual_tokens']} | merged {summary['merged_tokens']}")
        panel = _panel_row([("Encoder source", view["source"]), (title, view["visualization"])])
        target = directory / f"view_{summary['view_index']:02d}_{summary['name']}.png"
        _captioned(panel, lines).save(target)
        view_paths.append(str(target.resolve()))
    tile_views = [view for view in prepared["views"] if "tile_index" in view["summary"]]
    pages = []
    for start in range(0, len(tile_views), 4):
        page = _panel_row([(f"Tile {view['summary']['tile_index'] + 1}", view["visualization"])
                           for view in tile_views[start:start + 4]], max_panel_width=384)
        target = directory / f"tiles_{start // 4 + 1:02d}.png"
        _captioned(page, lines).save(target)
        pages.append(str(target.resolve()))
    (directory / "decisions.json").write_text(encoded, encoding="utf-8")
    return [{"image_index": 0, "directory": str(directory.resolve()),
             "comparison": str((directory / "comparison.png").resolve()),
             "views": view_paths, "tile_pages": pages,
             "decisions": str((directory / "decisions.json").resolve())}]

# ---------- Runtime-only model-instance integration; never writes model code ----------
@contextmanager
def visionzip_generation(model, adapter):
    """Use VisionZip prefill at ALL rates, including zero. Never substitute baseline."""
    name = "prepare_inputs_labels_for_multimodal"
    had_local = name in model.__dict__
    saved_local = model.__dict__.get(name)

    def prepare(self, input_ids, position_ids, attention_mask, past_key_values,
                labels, images, modalities=("image",), image_sizes=None):
        # Hugging Face cached decoding does not re-encode images.
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if self.training or labels is not None or past_key_values is not None:
            raise ValueError("VisionZip single-file image prefill requires eval mode, no labels or existing cache")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("VisionZip requires batch size 1 and one image per sample")
        modes = [modalities] if isinstance(modalities, str) else list(modalities)
        if modes != ["image"] or image_sizes is None or len(image_sizes) != 1:
            raise ValueError("VisionZip requires one image with its original image_size")
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError("Expected a list containing one Base+AnyRes crop batch")
            crops = images[0]
        elif torch.is_tensor(images) and images.ndim == 5 and images.shape[0] == 1:
            crops = images[0]
        else:
            raise ValueError("Expected [1, views, 3, H, W] or [tensor(views, 3, H, W)]")
        if not torch.is_tensor(crops) or crops.ndim != 4 or crops.shape[0] < 2:
            raise ValueError("Expected Base plus at least one AnyRes encoder view")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("Prompt attention mask must match token IDs")
        valid = (attention_mask[0].bool() if attention_mask is not None
                 else torch.ones_like(input_ids[0], dtype=torch.bool))
        prompt_ids = input_ids[0][valid]
        from llava.constants import IMAGE_TOKEN_INDEX
        placeholders = torch.where(prompt_ids == IMAGE_TOKEN_INDEX)[0]
        if placeholders.numel() != 1:
            raise ValueError("Prompt must contain exactly one unmasked <image> placeholder")
        split = int(placeholders.item())
        # At rate=0 this still calls VisionZip scoring/selection/projector/packing.
        # No fallback to the original multimodal implementation on image prefill.
        projected = adapter.encode_images(crops)
        visual = adapter.pack([projected], image_sizes, modes)[0]
        text_ids = torch.cat((prompt_ids[:split], prompt_ids[split + 1:]))
        text = self.get_model().embed_tokens(text_ids)
        pieces = (text[:split], visual, text[split:])
        embeddings = torch.cat([piece.to(self.device) for piece in pieces], dim=0)
        limit = getattr(self.config, "tokenizer_model_max_length", None)
        if limit is not None and embeddings.shape[0] > limit:
            if adapter.stats.get("full_grid_identity", False):
                # A complete grid also preserves the reference's final truncation.
                adapter.stats["packed_prompt_tokens_before_truncation"] = embeddings.shape[0]
                embeddings = embeddings[:limit]
                adapter.stats["packed_prompt_tokens_after_truncation"] = embeddings.shape[0]
            else:
                raise ValueError("VisionZip packed prompt exceeds tokenizer_model_max_length; refusing unreported visual token truncation")
        # Exactly preserve the original generation's optional-mask/position convention.
        new_mask = (torch.ones((1, embeddings.shape[0]), dtype=attention_mask.dtype,
                               device=attention_mask.device) if attention_mask is not None else None)
        new_positions = (torch.arange(embeddings.shape[0], dtype=position_ids.dtype,
                                      device=position_ids.device).unsqueeze(0)
                         if position_ids is not None else None)
        return None, new_positions, new_mask, past_key_values, embeddings.unsqueeze(0), None

    model.prepare_inputs_labels_for_multimodal = types.MethodType(prepare, model)
    try:
        yield adapter
    finally:
        if had_local:
            setattr(model, name, saved_local)
        else:
            delattr(model, name)


class SingleFileVisionZip:
    """No registry, external method module, JSON config, or vendor edits needed."""
    name = "visionzip"

    def __init__(self, core):
        self.adapter = VisionZipAdapter(core)

    def configure(self, core, prune_rate, *, capture_attention=False, capture_visualization=True):
        prune_rate = validate_prune_rate(prune_rate)
        core.configure_fastv(enabled=False)
        self.adapter.configure(prune_rate)

    def stats(self, core, prune_rate):
        return dict(self.adapter.stats)

    def visualization_data(self, core, *, capture_attention=False):
        return {"decisions": self.adapter.decisions.export() if self.adapter.decisions else []}


def validate_prune_rate(value):
    if type(value) is not int or not 0 <= value <= 99:
        raise ValueError("prune_rate must be an integer in 0..99")
    return value


def source_metadata(prune_rate=PRUNE_RATE):
    prune_rate = validate_prune_rate(prune_rate)
    dominant, contextual = token_budget(729, prune_rate / 100)
    return {
        "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
        "paper_non_cls_algorithm": "https://arxiv.org/html/2412.04467v1#A2",
        "adaptations": ["non-CLS received attention averaged over heads and query positions",
                        "27x27 SigLIP grid, no CLIP CLS feature",
                        "final active SigLIP block 26 is original penultimate block"],
        "selection": "per-view attention top-k, spatially ordered dominant output",
        "merge": "raw K mean over heads then L2; anchor hidden + mean assigned OTHER hidden; clamp count min=1",
        "budget": (f"each Base and AnyRes view independently retains ceil(729*(1-{prune_rate}/100))="
                   f"{dominant + contextual}: dominant{dominant}/contextual{contextual}"),
        "requested_prune_rate_percent": prune_rate,
        "mode": "visionzip_siglip_adaptation",
        "zero_policy": "Run the method: score/select all tokens in original order, no merge; "
                       "complete-grid AnyRes packing preserves unpad/max9/newlines. No model-path bypass.",
        "packing": ("complete selected grid with unpad/max9/newlines" if prune_rate == 0 else
                    "official sparse Base + spatial tile dominants + tile contextual, no unpad/max9 resize/newlines"),
        "visualization": {"visibility_over_white": VISIBILITY,
                          "merged_sources": "information contributes to contextual features"},
    }


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def self_test(output_dir, prune_rate=PRUNE_RATE):
    """Geometry/opacity preview only, explicitly synthetic; no model prediction."""
    prune_rate = validate_prune_rate(prune_rate)
    dominant_count, contextual_count = token_budget(729, prune_rate / 100)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Self-test output must be new/empty: {output}")
    y, x = np.mgrid[:384, :384]
    radius = np.sqrt((x-192)**2+(y-192)**2)
    ring = ((radius > 75) & (radius < 145))
    base = np.where(ring, 80, 170).astype(np.uint8)
    image = Image.fromarray(np.stack([base, base, base], axis=-1))
    decisions = []
    for view in range(5):
        ranking = np.roll(np.arange(729), view * 37)
        dominant = sorted(ranking[:dominant_count].tolist())
        remaining = [i for i in range(729) if i not in set(dominant)]
        anchors = (remaining[::max(1, len(remaining)//contextual_count)][:contextual_count]
                   if contextual_count else [])
        anchor_set = set(anchors)
        merged = [i for i in remaining if i not in anchor_set]
        decisions.append({"dominant_indices": dominant, "contextual_indices": anchors,
                          "merged_indices": merged})
    result = save_visionzip_visualizations(
        image, {"mode": "anyres_max_9", "original_size": [384, 384],
                "grid_patches": [2, 2], "processed_view_sizes": [[384,384], [768,768]]},
        decisions, {"mode": "SYNTHETIC_GEOMETRY_TEST_NO_MODEL_INFERENCE", "prune_rate": prune_rate},
        27, 14, output, "SYNTHETIC_self_test_no_inference", answer=None, ground_truth=None)
    print("Synthetic geometry test only; no model inference or real classification.")
    print(result[0]["comparison"])
    return result


def parse_bool(value):
    """Parse only True/False, case-insensitively; never use bool(string)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected True or False for --save-prune-vis")


def build_parser():
    parser = argparse.ArgumentParser(description="Independent VisionZip/SigLIP runner; all input samples, FP16 + FlashAttention2")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", choices=("mvtec", "visa"), default="mvtec")
    parser.add_argument("--input-json", help="MVTec JSON/JSONL; defaults to the existing question_musc.jsonl")
    parser.add_argument("--data-root", help="Default follows --dataset; same paths as benchmark_fastv.py")
    parser.add_argument("--visa-split-csv", help="VisA split CSV or split_csv directory; auto-detects split_csv/1cls.csv")
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3", "v0_generic"),
                        help="Default: MVTec v0; VisA category-free v0 with A/B choices")
    parser.add_argument("--output-dir", type=Path,
                        help="New/empty directory; default outputs/visionzip_<rate>_<timestamp>")
    parser.add_argument("--prune-rate", type=int, choices=range(100), default=PRUNE_RATE,
                        metavar="0..99", help="Per raw encoder view (default 70); 0 runs identity selection/full-grid packing")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup images excluded from metrics; all images then evaluated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-prune-vis", type=parse_bool, nargs="?", const=True, default=True,
                        metavar="{True,False}", help="Preserve source visualizations (default True); use False for Lat/Mem comparison")
    parser.add_argument("--self-test", action="store_true", help="Synthetic geometry preview only; no weights or CUDA needed")
    parser.add_argument("--verify-zero", action="store_true",
                        help="With --prune-rate 0: verification-only paired runs against original packing; no performance report")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser


def prepare_inputs(args):
    """Reuse project readers and freeze the prompt selection before loading weights."""
    validate_prune_rate(args.prune_rate)
    if args.verify_zero and (args.prune_rate != 0 or args.self_test):
        raise ValueError("--verify-zero requires --prune-rate 0 and cannot be combined with --self-test")
    if args.warmup < 1:
        raise ValueError("--warmup must be >= 1")
    data_root = Path(args.data_root or DEFAULT_DATA_ROOTS[args.dataset]).expanduser().resolve()
    if args.dataset == "visa":
        if args.input_json is not None:
            raise ValueError("--dataset visa reads the test split directly; omit --input-json")
        if args.prompt_version not in (None, "v0", "v0_generic"):
            raise ValueError("VisA uses category-free v0 with A/B choices; omit --prompt-version")
        from llava_pruning.visa_data import load_visa_test
        samples, dataset_info = load_visa_test(data_root, args.visa_split_csv)
        if not dataset_info["normal"] or not dataset_info["abnormal"]:
            raise ValueError("VisA requires normal AND abnormal test images to measure TNR/Recall. "
                             "Check split_csv/1cls.csv; training images will not be added.")
        prompt_version = "v0_generic"
        source = None
    else:
        if args.visa_split_csv is not None:
            raise ValueError("--visa-split-csv requires --dataset visa")
        source = Path(args.input_json or DEFAULT_INPUT).expanduser().resolve()
        samples = load_samples(source, data_root)
        dataset_info = {"layout": "question_json", "source": str(source)}
        prompt_version = args.prompt_version or "v0"
    if not samples:
        raise ValueError("Input dataset is empty")
    if not 0 <= args.seed <= 2**32 - len(samples):
        raise ValueError("seed + sample index must remain within NumPy's 32-bit range")
    prompts = [benchmark_prompt(sample, prompt_version) for sample in samples]
    return samples, prompts, {
        "dataset": args.dataset, "dataset_info": dataset_info,
        "input_json": str(source) if source else None, "data_root": str(data_root),
        "prompt_version": prompt_version,
    }


TIMING_FIELDS = ("lat_seconds", "generation_seconds", "generated_tokens",
                 "peak_allocated_bytes", "peak_reserved_bytes")
TOKEN_FIELDS = ("encoder_views", "tokens_per_view_before", "tokens_per_view_after",
                "dominant_per_view", "contextual_per_view", "actual_encoder_prune_rate_percent",
                "baseline_packed_tokens", "final_llm_visual_tokens", "actual_packed_reduction_percent")
SUMMARY_FIELDS = (
    "dataset", "method", "prune_rate", "status", "complete", "expected_samples", "evaluated_samples",
    "labeled_samples", "correct", "incorrect", "unparsed", "parsed_samples",
    "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "tnr",
    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_std_ms",
    "mem_peak_allocated_gib", "mem_peak_reserved_gib", "mem_peak_allocated_pct", "device_total_gib",
    "generation_mean_ms", "mean_generated_tokens", "warmup_samples",
    "capture_visualization_decisions", "model_load_seconds", "run_wall_seconds", "error",
)


def generate_sample(backend, method, sample, prompt, rate, seed, *, measure, capture, trace=None):
    """Benchmark and normal generation share one backend; serialize nothing here.

    The public backend's visualization switches stay off (they request FastV/ViCo
    capture). VisionZip owns its separate, explicitly recorded audit switch.
    Capturing extra GPU decision tensors can affect measured memory/latency;
    performance experiments should set --save-prune-vis False.
    """
    method.adapter.capture_decisions = bool(capture)
    method.configure(backend.model.get_model(), rate)
    with visionzip_generation(backend.model, method.adapter):
        with capture_generation_trace(backend.model, trace) if trace is not None else nullcontext():
            return backend.generate(
                sample, prompt, method, rate, "anyres_max_9",
                capture_visualization=False, capture_attention=False,
                random_seed=seed, do_sample=False, include_pruning_time=True,
                benchmark_metrics=measure,
            )


@contextmanager
def capture_generation_trace(model, trace):
    """Read-only tensor snapshots for verification, NEVER for performance metrics.

    Wrap the already-installed prefill path, so the candidate cannot accidentally
    skip its own VisionZip adapter. Restore every hook/instance method on failure.
    """
    def snapshot(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [snapshot(item) for item in value]
        if not torch.is_tensor(value):
            raise TypeError("Verification expected a tensor/None or list of tensors")
        return value.detach().cpu().clone()

    original_prepare = model.prepare_inputs_labels_for_multimodal
    original_generate = model.generate
    names = ("prepare_inputs_labels_for_multimodal", "generate")
    saved = {name: (name in model.__dict__, model.__dict__.get(name)) for name in names}
    handles = []

    def observe_projector(module, args, output):
        if "selected_features" in trace:
            raise RuntimeError("Verification expects exactly one projector call per image")
        trace["selected_features"] = snapshot(args[0])
        trace["projected_features"] = snapshot(output)

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        if result[4] is not None:
            if "llm_inputs_embeds" in trace:
                raise RuntimeError("Verification expects exactly one image prefill")
            trace["input_ids"] = snapshot(args[0])
            trace["pixels"] = snapshot(args[5] if len(args) > 5 else kwargs["images"])
            trace["position_ids"] = snapshot(result[1])
            trace["attention_mask"] = snapshot(result[2])
            trace["llm_inputs_embeds"] = snapshot(result[4])
        return result

    def generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        trace["generated_token_ids"] = snapshot(result)
        return result

    try:
        handles.append(model.get_model().mm_projector.register_forward_hook(observe_projector))
        model.prepare_inputs_labels_for_multimodal = prepare
        model.generate = generate
        yield trace
    finally:
        for handle in handles:
            handle.remove()
        for name, (had_local, value) in saved.items():
            if had_local:
                setattr(model, name, value)
            else:
                delattr(model, name)


TRACE_FIELDS = ("pixels", "input_ids", "selected_features", "projected_features",
                "llm_inputs_embeds", "position_ids", "attention_mask", "generated_token_ids")


def compare_traces(reference, candidate):
    """Exact tensor/dtype/shape equality, not just equal A/B labels or accuracy."""
    def compare(left, right):
        if left is None or right is None:
            return {"equal": left is None and right is None}
        if isinstance(left, list) or isinstance(right, list):
            if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
                return {"equal": False, "reason": "different input container/length"}
            items = [compare(a, b) for a, b in zip(left, right)]
            return {"equal": all(item["equal"] for item in items), "items": items}
        equal = left.dtype == right.dtype and torch.equal(left, right)
        result = {"equal": equal, "reference_shape": list(left.shape), "candidate_shape": list(right.shape),
                  "reference_dtype": str(left.dtype), "candidate_dtype": str(right.dtype)}
        if not equal and left.shape == right.shape and left.numel():
            diff = (left.double() - right.double()).abs()
            maximum = diff.max().item()
            result["max_abs_diff"] = maximum if math.isfinite(maximum) else None
        return result

    return {name: (compare(reference[name], candidate[name])
                   if name in reference and name in candidate else {"equal": False, "reason": "missing capture"})
            for name in TRACE_FIELDS}


class ReferenceOnlyMethod:
    """Only the --verify-zero comparator uses this; never a replacement candidate."""
    def configure(self, core, rate, **kwargs):
        core.configure_fastv(enabled=False)

    def stats(self, core, rate):
        return {"mode": "original_reference_for_verification_only"}


def verify_zero_equivalence(backend, method, samples, prompts, seed, output):
    """Run both paths for every selected sample; stop/report on first mismatch."""
    summary = {
        "status": "running", "complete": False, "expected_samples": len(samples),
        "checked_samples": 0, "matched_samples": 0, "error": None,
        "comparison": "Exact dtype/shape/value equality for all trace stages AND full generated token IDs/answer",
        "performance_metrics": "Not measured: paired inference and CPU snapshots are diagnostic only",
    }
    report_path = Path(output) / "zero_equivalence.json"
    _write_json(report_path, summary)
    try:
        with (Path(output) / "zero_equivalence.jsonl").open("x", encoding="utf-8") as stream:
            for index, (sample, prompt) in enumerate(zip(samples, prompts)):
                candidate_trace, reference_trace = {}, {}
                # Candidate first, through the method. Reference is a separate comparison run.
                candidate = generate_sample(backend, method, sample, prompt, 0, seed + index,
                                            measure=False, capture=False, trace=candidate_trace)
                if not candidate["stats"].get("full_grid_identity", False):
                    raise RuntimeError("Candidate did not execute VisionZip full-grid packing")
                with capture_generation_trace(backend.model, reference_trace):
                    reference = backend.generate(
                        sample, prompt, ReferenceOnlyMethod(), 0, "anyres_max_9",
                        capture_visualization=False, capture_attention=False,
                        random_seed=seed + index, do_sample=False, include_pruning_time=True,
                        benchmark_metrics=False)
                stages = compare_traces(reference_trace, candidate_trace)
                match = all(item["equal"] for item in stages.values()) and candidate["answer"] == reference["answer"]
                row = {"question_id": sample.sample_id, "equal": match, "stages": stages,
                       "candidate_answer": candidate["answer"], "reference_answer": reference["answer"]}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                summary["checked_samples"] += 1
                summary["matched_samples"] += int(match)
                _write_json(report_path, summary)
                # Release large CPU snapshots before the next sample.
                del candidate_trace, reference_trace
                if not match:
                    raise RuntimeError(f"Zero-rate mismatch at {sample.sample_id}; see {output}/zero_equivalence.jsonl")
                print(f"[{index+1}/{len(samples)}] id={sample.sample_id}: EXACT MATCH (features, inputs, token IDs, answer)", flush=True)
        summary.update(status="passed", complete=summary["checked_samples"] == len(samples))
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write_json(report_path, summary)
    return summary


def save_sample_visualization(backend, method, sample, result, output_dir):
    """CPU serialization/rendering only, called after the measured generate."""
    adapter = method.adapter
    if adapter.decisions is None or adapter.crop_metadata is None:
        raise RuntimeError("VisionZip did not capture decisions/geometry for visualization")
    try:
        decisions = adapter.decisions.export()
        # Drop GPU audit buffers before rendering, and before the next input.
        adapter.decisions = None
        with Image.open(sample.image) as opened:
            image = opened.convert("RGB")
        return save_visionzip_visualizations(
            image=image, pro_data=adapter.crop_metadata, decisions=decisions, stats=result["stats"],
            base_grid=int(backend.vision_tower.num_patches_per_side),
            patch_size=int(backend.vision_tower.config.patch_size),
            output_dir=output_dir, sample_id=sample.sample_id,
            answer=result["answer"], ground_truth=sample.gt,
        )
    finally:
        adapter.decisions = None


def load_backend_for_run(args, factory, spec, output):
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    backend = factory(args.model_path, roi_mode="anyres_max_9", image_token_order="base_first")
    validate_single_gpu(backend.model, torch)
    torch.cuda.synchronize()
    inference = dict(backend.inference_config)
    inference.update(attention_source="SigLIP block 26 native received attention",
                     score_dtype=inference.get("vision_dtype"))
    spec["inference"] = inference
    _write_json(output / "run.json", spec)
    return backend, time.perf_counter() - started


def run_experiment(args, *, backend_factory=LlavaBackend):
    """One rate, all records; common model loading, metrics and Lat/Mem protocol."""
    save_prune_vis = parse_bool(args.save_prune_vis)
    samples, prompts, dataset_spec = prepare_inputs(args)
    rate = args.prune_rate
    capture = save_prune_vis and not args.verify_zero
    output = Path(args.output_dir or PROJECT / "outputs" /
                  f"visionzip_{rate:02d}_{datetime.now():%Y%m%d_%H%M%S}").expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new/empty: {output}")
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("Inference needs the existing Torch/CUDA/FlashAttention2 environment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one visible GPU, e.g. CUDA_VISIBLE_DEVICES=5 python runVisionZip.py ...")
    output.mkdir(parents=True, exist_ok=True)
    rate_dir = output / f"prune_{rate:02d}"
    rate_dir.mkdir()
    spec = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_version": SCRIPT_VERSION,
        "method": "visionzip_siglip_adaptation",
        "verification_only": args.verify_zero,
        "method_details": source_metadata(rate),
        "model_path": str(Path(args.model_path).expanduser().resolve()), **dataset_spec,
        "samples": len(samples), "prune_rate": rate, "roi_mode": "anyres_max_9",
        "image_token_order": "base_first", "seed": args.seed, "do_sample": False,
        "decoding": "greedy, max_new_tokens<=512, original LLaVA context budget",
        "warmup_requested": args.warmup, "save_prune_vis": save_prune_vis,
        "capture_visualization_decisions": capture, "include_pruning_time": True,
        "definitions": {key: DEFINITIONS[key] for key in
                        ("lat", "memory", "memory_percent", "generation", "accuracy", "units")},
        "rate_definition": "Nonzero rate is per RAW encoder view, NOT final packed LLM tokens. "
                           "Sparse packing removes row newlines/unpadding/max9 resizing; "
                           "actual packed reduction is reported per sample and may be negative.",
        "visualization_timing": "PNG/JSON rendering/export is outside generate timing. "
                                "Extra GPU decision capture is included when enabled; use False for comparison.",
    }
    _write_json(output / "run.json", spec)
    if args.dataset == "visa":
        with (output / "input_visa_test.jsonl").open("x", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample.source, ensure_ascii=False, allow_nan=False) + "\n")

    if args.verify_zero:
        try:
            print(f"Verification only: {len(samples)} paired runs; no Lat/Mem benchmark or visualizations.", flush=True)
            backend, _ = load_backend_for_run(args, backend_factory, spec, output)
            method = SingleFileVisionZip(backend.model.get_model())
            verify_zero_equivalence(backend, method, samples, prompts, args.seed, output)
        except BaseException as exc:
            if not (output / "zero_equivalence.json").exists():
                _write_json(output / "zero_equivalence.json", {
                    "status": "failed", "complete": False, "checked_samples": 0,
                    "expected_samples": len(samples), "error": f"{type(exc).__name__}: {exc}"})
            raise
        print(f"All {len(samples)} samples matched exactly. Report: {output / 'zero_equivalence.json'}", flush=True)
        return output

    accuracy = Accuracy(expected_samples=len(samples))
    measurements = []
    total_memory = None
    model_load_seconds = None
    warmups_done = 0
    start = time.perf_counter()
    status, error = "running", None

    def save_report():
        report = {
            **{k: v for k, v in accuracy.result(rate, complete=status == "success").items() if k != "rule"},
            "dataset": args.dataset, "method": spec["method"], "status": status, "error": error,
            "warmup_samples": warmups_done, "capture_visualization_decisions": capture,
            "model_load_seconds": model_load_seconds, "run_wall_seconds": time.perf_counter() - start,
        }
        if measurements and total_memory:
            report.update(measurement_summary(measurements, total_memory))
        _write_json(rate_dir / "metrics.json", report)
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow({key: report.get(key) for key in SUMMARY_FIELDS})
        return report

    save_report()
    try:
        print(f"Loading model; {len(samples)} evaluation images, warmup={min(args.warmup, len(samples))}", flush=True)
        backend, model_load_seconds = load_backend_for_run(args, backend_factory, spec, output)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        method = SingleFileVisionZip(backend.model.get_model())
        for index in warmup_indices(len(samples), args.warmup):
            # Same instrumented path as evaluation/benchmark_fastv; discard ALL warmup metrics.
            generate_sample(backend, method, samples[index], prompts[index], rate, args.seed + index,
                            measure=True, capture=False)
            warmups_done += 1
            print(f"Warmup {warmups_done}/{min(args.warmup, len(samples))}", flush=True)
        print(f"Warmup finished ({warmups_done}); evaluating ALL {len(samples)} images", flush=True)
        fields = ("question_id", "image", "gt", "prediction", "prediction_label", "answer", "prune_rate",
                  *TIMING_FIELDS, *TOKEN_FIELDS)
        with (rate_dir / "predictions.jsonl").open("x", encoding="utf-8") as json_stream, \
                (rate_dir / "classification_results.csv").open("x", encoding="utf-8-sig", newline="") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=fields)
            writer.writeheader()
            for index, (sample, prompt) in enumerate(zip(samples, prompts)):
                result = generate_sample(backend, method, sample, prompt, rate, args.seed + index,
                                         measure=True, capture=capture)
                answer, stats = result["answer"], result["stats"]
                dominant, contextual = token_budget(729, rate / 100)
                if (stats.get("tokens_per_view_before"), stats.get("tokens_per_view_after"),
                    stats.get("dominant_per_view"), stats.get("contextual_per_view")) != (
                        729, dominant + contextual, dominant, contextual):
                    raise RuntimeError("Actual VisionZip budget differs from configured per-view budget")
                match = OPTION.match(answer)
                predicted = (1 if match.group(1).upper() == "A" else 0) if match else None
                timing = {key: result[key] for key in TIMING_FIELDS}
                record = {
                    "question_id": sample.sample_id, "image": str(sample.image),
                    "origin_path": sample.origin_path, "gt": sample.gt, "answer": answer,
                    "prediction": predicted, "prediction_label": {0: "Normal", 1: "Abnormal"}.get(predicted, "Unparsed"),
                    "prune_rate": rate, **timing, "pruning": stats,
                }
                # Persist classification before display-only work.
                json_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                json_stream.flush()
                writer.writerow({key: record.get(key, stats.get(key)) for key in fields})
                csv_stream.flush()
                measurements.append(timing)
                accuracy.add(sample.gt, answer)
                save_report()
                if capture:
                    save_sample_visualization(backend, method, sample, result, rate_dir / "visualizations")
                print(f"[{index+1}/{len(samples)}] id={sample.sample_id} Pred={record['prediction_label']} "
                      f"Lat={timing['lat_seconds']*1000:.2f} ms "
                      f"Mem={timing['peak_allocated_bytes']/1024**3:.3f} GiB "
                      f"LLM visual tokens={stats.get('final_llm_visual_tokens', 'original packing')}", flush=True)
        status = "success"
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = save_report()
    acc = "N/A" if report["accuracy"] is None else f"{report['accuracy']:.2%}"
    print(f"Finished {len(samples)} images. ACC={acc}, unparsed={report['unparsed']}; "
          f"Lat={report['lat_mean_ms']:.2f} ms, Mem={report['mem_peak_allocated_gib']:.3f} GiB", flush=True)
    print(f"Summary: {output / 'summary.csv'}", flush=True)
    return output


def main():
    args = build_parser().parse_args()
    if args.verify_zero and (args.self_test or args.prune_rate != 0):
        raise ValueError("--verify-zero requires --prune-rate 0 and no --self-test")
    if args.self_test:
        if args.save_prune_vis:
            self_test(args.output_dir or PROJECT / "outputs" /
                      f"visionzip_self_test_{datetime.now():%Y%m%d_%H%M%S}", args.prune_rate)
        else:
            print("Self-test skipped: --save-prune-vis False.")
        return
    if args.prune_rate:
        dominant, contextual = token_budget(729, args.prune_rate / 100)
        print(f"VisionZip: each view retains {dominant+contextual}/729 = "
              f"{dominant} dominant + {contextual} contextual; requested {args.prune_rate}% pruning.", flush=True)
    else:
        print("0%: VisionZip scoring/select-all/projector/full-grid packing; NOT a bypass to original LLaVA.", flush=True)
    if args.save_prune_vis and not args.verify_zero:
        print("Visualization capture enabled. For clean Lat/Mem comparison use --save-prune-vis False.", flush=True)
    run_experiment(args)


# Full upstream distribution license, embedded so this remains one file.
APACHE_LICENSE_TEXT = r"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


if __name__ == "__main__":
    main()

