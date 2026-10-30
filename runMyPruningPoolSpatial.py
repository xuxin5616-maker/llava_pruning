#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Haotian Liu
# Copyright 2024 Senqiao Yang
# Modified: internal pooled-cosine redundancy budgets and original-anchor packing.
"""Standalone pooled redundancy + broad center spatial crop-budget supplement.

Place beside the original project's vendor/ and llava_pruning/ folders:
CUDA_VISIBLE_DEVICES=5 python runMyPruningPoolSpatial.py --prune-rate 30 --redundancy-layers 21 --visionzip-layers 21 --spatial-mix-weight 0.02 --allocation-temperature 0.1 --save-prune-vis False

All pooled-redundancy scoring is preserved: L2-normalize each token, pool within
each Base bin / each AnyRes crop separately, normalize pooled vectors, compare
with their own tokens, reduce each branch (mean default), add branches, then
average complete scalar scores across selected feature layers (default26).

For allocation ONLY, p_R=softmax(-R/T), p_S=P/sum(P), and
p=(1-lambda)*p_R+lambda*p_S. No second softmax follows mixing. Lambda defaults
to0 and returns the ORIGINAL allocator result, preserving its exact arithmetic
and rounding. --spatial-strength is retained as an alias of --spatial-mix-weight;
its meaning is now a probability fraction [0,1], not the old score-level delta.
The global flat-top raised-cosine field, half-widths1/6 and1/2, and Base floor
bins stay unchanged in this fusion-only ablation (including the known2x2
discretization asymmetry). An all-zero prior falls back to the legacy allocator.
Mixed probabilities are fixed during capacity redistribution; log arithmetic
preserves tiny weights after larger crops saturate. If remaining mixed mass is
zero, original redundancy weights redistribute that remaining budget.
This custom center prior is inspired by spatial protection; it is NOT a
reproduction of ATP-LLaVA's multi-granularity sampling / learned SAP masks.

The existing bounded softmax, per-crop4..729, exact aggregate quotas, independent
Base budget, native VisionZip attention selection, block26 Key/feature merging,
and original-anchor packing are preserved. Nonzero rows now include protected
checkpoint newlines. Full retention uses native unpad/max9/newlines and
preserves original features/layout and 3D projection. Position IDs remain consecutive in
the packed sequence. Spatial evidence does not enter token selection/merging.

Classify all samples with original prompts and greedy decoding. Save normal
CSV/JSONL/metrics plus raw redundancy, spatial prior and the three probability
vectors. Legacy spatial_bonus now means lambda*p_S; after-score is null when
enabled because this scheme has no fused score. Visualization is optional/off.
No older method file is
imported at runtime. For ablation pass --spatial-strength 0.
Zero is supported by the method itself. Verify with --prune-rate 0 --verify-zero.
Sparse packing keeps raw row geometry; it does not unpad or max9-resize sparse holes.
"""


from __future__ import annotations

import argparse
import heapq

import csv

import json

import math

import re

import sys

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
    torch = None  # --help / --version do not require Torch or CUDA.

PROJECT = Path(__file__).resolve().parent

DEFAULT_MODEL = "/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/"

DEFAULT_INPUT = "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"

DEFAULT_DATA = "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/"

PRUNE_RATE = 70



DEFAULT_REDUNDANCY_LAYERS = (26,)
DEFAULT_REDUNDANCY_REDUCTION = "mean"
DEFAULT_VISIONZIP_LAYERS = (26,)
DEFAULT_ALLOCATION_TEMPERATURE = 0.05
DEFAULT_SPATIAL_STRENGTH = 0.0
SPATIAL_PLATEAU_HALF_WIDTH = 1 / 6
SPATIAL_SUPPORT_HALF_WIDTH = 1 / 2
MINIMUM_CROP_KEEP = 4
SOURCE_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"
SOURCE_URL = f"https://github.com/JIA-Lab-research/VisionZip/tree/{SOURCE_COMMIT}"
SCRIPT_VERSION = "1.23-pool-spatial-probability-mix"
BASELINE_SHA256 = "aa555ba378a6fe39a5ce31ff1b7e85d17cead39a0cf982783a3f2997150e5e2c"



@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Path
    category: str
    origin_path: str
    text: str | None
    gt: int | None
    mask: Path | None
    bbox: list[list[int]] | None
    source: dict[str, Any]

def _resolve(root: Path, raw: str, *, image: bool) -> Path:
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(f"Data paths must be relative to --data-root: {raw}")
    candidates = [root / relative]
    if image and len(relative.parts) == 1:
        candidates.append(root / "imgs" / relative)
    resolved_root = root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"Data path escapes --data-root: {raw}")
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Cannot find {raw} under {root}")

def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
    else:
        raise ValueError("--input-json must end in .json or .jsonl")
    if not isinstance(records, list):
        raise ValueError("Input must contain a list of JSON objects or JSONL objects")
    return records

def load_samples(input_json: str | Path, data_root: str | Path) -> list[Sample]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Data root not found: {root}")
    samples = []
    seen_ids = set()
    for index, record in enumerate(_records(Path(input_json))):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object")
        sample_id = str(record.get("question_id", record.get("id", "")))
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Record {index} has an empty or duplicate question_id: {sample_id!r}")
        seen_ids.add(sample_id)
        image_name = record.get("image")
        if not isinstance(image_name, str):
            raise ValueError(f"Record {index} needs a single image path")
        origin = str(record.get("origin_path", image_name))
        origin_parts = origin.replace("\\", "/").split("/")
        inferred_category = origin_parts[0]
        if inferred_category == root.name and len(origin_parts) > 1:
            inferred_category = origin_parts[1]
        category = str(record.get("category") or record.get("subset") or
                       record.get("sub_dataset") or inferred_category)
        mask_name = record.get("mask")
        if isinstance(mask_name, list) and len(mask_name) == 1:
            mask_name = mask_name[0]
        mask = _resolve(root, mask_name, image=False) if isinstance(mask_name, str) and mask_name else None
        if mask_name is not None and not isinstance(mask_name, str):
            raise ValueError(f"Record {index} mask must be a string path")
        bbox = record.get("bbox")
        # Legacy LLaVA records can include an outer per-image list.
        if isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list) and bbox[0] and isinstance(bbox[0][0], list):
            bbox = bbox[0]
        if bbox is not None:
            if (not isinstance(bbox, list) or any(
                not isinstance(box, list) or len(box) != 4 or
                any(not isinstance(value, int) for value in box)
                for box in bbox
            )):
                raise ValueError(f"Record {index} bbox must be a list of four-integer boxes")
        gt = record.get("gt")
        if gt is not None and gt not in (0, 1):
            raise ValueError(f"Record {index} gt must be 0 or 1")
        samples.append(Sample(
            sample_id=sample_id,
            image=_resolve(root, image_name, image=True),
            category=category,
            origin_path=origin,
            text=record.get("text"),
            gt=gt,
            mask=mask,
            bbox=bbox,
            source=record,
        ))
    return samples

OPTION = re.compile(r"^\s*([AB])\b", re.IGNORECASE)

METRIC_RULE = (
    "A=defect (1), B=no defect (0); accuracy includes all labeled samples and counts unparsed answers as incorrect; "
    "precision/recall/tnr and tp/fp/tn/fn use only parsed labeled A/B answers; zero denominators are null"
)

@dataclass
class Accuracy:
    expected_samples: int
    evaluated_samples: int = 0
    labeled_samples: int = 0
    correct: int = 0
    unparsed: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, gt: int | None, answer: str) -> None:
        self.evaluated_samples += 1
        if gt is None:
            return
        self.labeled_samples += 1
        match = OPTION.match(answer)
        if match is None:
            self.unparsed += 1
            return
        prediction = 1 if match.group(1).upper() == "A" else 0
        self.correct += prediction == gt
        if gt == 1:
            if prediction == 1:
                self.tp += 1
            else:
                self.fn += 1
        elif prediction == 1:
            self.fp += 1
        else:
            self.tn += 1

    def result(self, prune_rate: int, *, complete: bool) -> dict:
        return {
            "prune_rate": prune_rate,
            "complete": complete and self.evaluated_samples == self.expected_samples,
            "expected_samples": self.expected_samples,
            "evaluated_samples": self.evaluated_samples,
            "labeled_samples": self.labeled_samples,
            "correct": self.correct,
            "incorrect": self.labeled_samples - self.correct,
            "unparsed": self.unparsed,
            "parsed_samples": self.labeled_samples - self.unparsed,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": self.correct / self.labeled_samples if self.labeled_samples else None,
            "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else None,
            "recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else None,
            "tnr": self.tn / (self.tn + self.fp) if self.tn + self.fp else None,
            "rule": METRIC_RULE,
        }

def prediction_caption(ground_truth, answer):
    """Use the evaluation's A/B rule, never guess a class from free-form text."""
    truth = {0: "Normal", 1: "Abnormal"}.get(ground_truth, "Unknown")
    match = OPTION.match(answer) if isinstance(answer, str) else None
    predicted = ("Abnormal" if match.group(1).upper() == "A" else "Normal") if match else "Unparsed"
    return f"GT: {truth} | Pred: {predicted}"

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

def _match_feature_memory_order(selected, reference):
    """Store [B,K,F] in the encoder's axis order without changing its values.

    SigLIP can expose [B,N,F] using channel-major [B,F,N] storage. Recreate
    that order compactly for K retained tokens; do not pad back to N or cast.
    Stride magnitudes necessarily change with K. Matching the layout removes
    one source of Linear dispatch differences, not shape-dependent rounding.
    """
    if selected is reference:
        return selected
    order = sorted(range(reference.ndim), key=lambda dim: reference.stride(dim), reverse=True)
    inverse = [order.index(dim) for dim in range(reference.ndim)]
    return selected.permute(order).contiguous().permute(inverse)


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



def compress_tokens(features, scores, metric, dominant_count, contextual_count, *, audit=True, validate=True,
                    source_provenance=True, spatial_indices=None):
    """Return compressed [B,D+C,F] and JSON-safe decisions for each view.

    features [B,N,F], scores [B,N], metric [B,N,K]. Dominant top-k features
    retain spatial order. Context follows evenly spaced anchors of ascending
    non-dominant tokens; other tokens join their maximum-cosine anchor.

    Decisions use original patch indices. cluster_assignments has N entries:
    -1 for dominant/dropped tokens, otherwise the contextual output slot
    (0..C-1). Anchors and their merged tokens share that assignment.
    merged_indices excludes anchors. discarded_indices is nonempty only at
    contextual_count=0. importance contains raw received attention scores.

    Feature computation never casts or moves tensors, preserving source FP16
    arithmetic. Source has no key-norm epsilon: invalid norms fail clearly.
    """
    if validate:
        dominant_count, contextual_count = _validate_inputs(
            features, scores, metric, dominant_count, contextual_count)
    batch, token_count, feature_dim = features.shape
    ranked = scores.topk(dominant_count, dim=1).indices
    mask = torch.ones((batch, token_count), dtype=torch.bool, device=features.device)
    mask.scatter_(1, ranked, False)
    spatial = (torch.arange(token_count, device=features.device) if spatial_indices is None else spatial_indices).expand(batch, -1)
    dominant_indices = spatial[~mask].view(batch, dominant_count)
    full_selection = dominant_count == token_count and contextual_count == 0
    dominant_features = (features if full_selection else
                         features.masked_select(~mask.unsqueeze(-1)).view(batch, dominant_count, feature_dim))
    remaining_count = token_count - dominant_count
    remaining_indices = spatial[mask].view(batch, remaining_count)
    filtered_features = features.masked_select(mask.unsqueeze(-1)).view(batch, remaining_count, feature_dim)
    assignments = (torch.full((batch, token_count), -1, device=features.device, dtype=torch.long)
                   if audit or source_provenance else None)
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
        if assignments is not None:
            assignments.scatter_(1, contextual_indices, torch.arange(contextual_count, device=features.device).expand(batch, -1))
            assignments.scatter_(1, merged_indices, cluster)

    # Preserve the original tensor, layout and storage offset at full retention.
    compressed = (features if full_selection else
                  _match_feature_memory_order(torch.cat((dominant_features, contextual_features), dim=1), features))
    if not bool(torch.isfinite(compressed).all()):
        raise ValueError("VisionZip feature aggregation produced non-finite values")
    if not audit:
        # Inference-only decisions stay on the GPU. No scores/features or
        # diagnostic lists are copied to CPU just to build JSON-safe records.
        return compressed, [dict(original_token_count=token_count,
            retained_token_count=dominant_count+contextual_count,
            dominant_indices=dominant_indices[i], contextual_indices=contextual_indices[i],
            cluster_assignments=assignments[i] if assignments is not None else None, dominant_count=dominant_count,
            contextual_count=contextual_count) for i in range(batch)]
    # Transfer metadata only after processing all views together.
    ranked_cpu = ranked.detach().cpu().tolist()
    dominant_cpu = dominant_indices.detach().cpu().tolist()
    contextual_cpu = contextual_indices.detach().cpu().tolist()
    merged_cpu = merged_indices.detach().cpu().tolist()
    remaining_cpu = remaining_indices.detach().cpu().tolist()
    assignment_cpu = assignments.detach().cpu().tolist()
    importance_cpu = scores.detach().cpu().tolist()
    counts_cpu = merge_counts.detach().cpu().tolist()
    decisions = [
        {
            "original_token_count": token_count,
            "retained_token_count": dominant_count + contextual_count,
            "dominant_indices": dominant_cpu[i],
            "dominant_ranked_indices": ranked_cpu[i],
            "contextual_indices": contextual_cpu[i],
            "merged_indices": merged_cpu[i],
            "non_dominant_indices": remaining_cpu[i],
            "discarded_indices": remaining_cpu[i] if contextual_count == 0 else [],
            "cluster_assignments": assignment_cpu[i],
            "merge_counts": counts_cpu[i],
            "importance": importance_cpu[i],
        }
        for i in range(batch)
    ]
    return compressed, decisions

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



def pack_sparse_views(features, decisions, grid, dominant_count, *,
                      image_newline=None, merge_type="spatial_unpad"):
    """Uniform-budget compatibility wrapper for the all-anchor row packer."""
    if features.ndim != 3 or len(decisions) != features.shape[0]:
        raise ValueError("Sparse features/decisions must have one entry per view")
    if any(len(row["dominant_indices"]) != dominant_count for row in decisions):
        raise ValueError("Dominant counts disagree with sparse features")
    return pack_anchor_rows(list(features.unbind(0)), decisions, grid,
                            image_newline=image_newline, merge_type=merge_type)


def pack_anchor_rows(projected_views, decisions, grid, *, image_newline=None,
                     merge_type="spatial_unpad"):
    """Place both dominant/contextual rows at original anchors and protect NLs.

    Coordinates describe the RAW AnyRes grid, just as the updated VisionZip
    sparse path. Nonzero compression does not unpad or bilinearly resize holes.
    Base has no row markers. All global AnyRes rows, including empty ones,
    have their original checkpoint newline when supplied.
    """
    if (not isinstance(grid, (tuple, list)) or len(grid) != 2
            or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in grid)):
        raise ValueError("AnyRes grid dimensions must be positive integers")
    width, height = grid
    if len(projected_views) != 1 + width * height or len(decisions) != len(projected_views):
        raise ValueError("AnyRes views and decisions do not match grid")
    device, dtype, hidden = projected_views[0].device, projected_views[0].dtype, projected_views[0].shape[-1]
    indices, side = [], None
    for features, decision in zip(projected_views, decisions):
        count = decision["original_token_count"]
        current_side = math.isqrt(count)
        local = list(decision["dominant_indices"]) + list(decision["contextual_indices"])
        if (current_side < 1 or current_side ** 2 != count
                or (side is not None and side != current_side)):
            raise ValueError("All original token grids must be equal nonempty squares")
        if (features.ndim != 2 or features.shape != (len(local), hidden)
                or features.device != device or features.dtype != dtype
                or len(set(local)) != len(local)
                or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < count for v in local)):
            raise ValueError("Retained features must match unique original patch/anchor indices")
        side = current_side
        indices.append(torch.tensor(local, dtype=torch.long, device=device))
    base = projected_views[0][torch.argsort(indices[0])]
    rows, positions = [], []
    raw_width, raw_rows = width * side, height * side
    for tile, local in enumerate(indices[1:]):
        tile_row, tile_col = divmod(tile, width)
        positions.append((tile_row * side + local // side) * raw_width + tile_col * side + local % side)
        rows.append(projected_views[tile + 1])
    positions, rows = torch.cat(positions), torch.cat(rows, dim=0)
    if image_newline is not None:
        if (merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}
                or image_newline.ndim != 1 or image_newline.shape[0] != hidden):
            raise ValueError("Sparse row newlines require spatial_unpad and one hidden-size vector")
        positions = (positions // raw_width) * (raw_width + 1) + positions % raw_width
        row_ends = torch.arange(raw_rows, device=device) * (raw_width + 1) + raw_width
        newline = image_newline.to(device=device, dtype=dtype)
        positions = torch.cat((positions, row_ends))
        rows = torch.cat((rows, newline[None].expand(raw_rows, -1)), dim=0)
    pieces = [base, rows[torch.argsort(positions)]]
    if image_newline is not None and merge_type == "spatial_unpad_add_newl":
        pieces.append(newline[None])
    return torch.cat(pieces, dim=0)

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
    run_title = (f"Pooled redundancy + spatial prior + VisionZip | encoder tokens {before} -> {after} | "
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
    metadata = {"sample_id": str(sample_id), "method": "custom_pool_spatial_anchor_order_temperature_budget_visionzip",
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

VENDOR = PROJECT / "vendor" / "llava"

PruningMethod = Any

def validate_image_token_order(order, roi_mode, merge_type=None):
    if order not in {"base_first", "anyres_first"}:
        raise ValueError(f"Unknown image token order: {order}")
    if order == "anyres_first":
        if roi_mode != "anyres_max_9":
            raise ValueError("anyres_first requires --roi-mode anyres_max_9")
        if merge_type is not None and merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}:
            raise ValueError("anyres_first requires spatial_unpad or spatial_unpad_add_newl packing")

def configure_image_mode(config, roi_mode, image_token_order="base_first"):
    """Configure preprocessing/packing without changing model precision or attention."""
    if roi_mode not in {"randomroi", "randompatch", "anyres_max_9", "ex_base_copy", "anyres_only"}:
        raise ValueError(f"Unknown ROI mode: {roi_mode}")
    validate_image_token_order(image_token_order, roi_mode)
    if roi_mode == "ex_base_copy":
        config.image_aspect_ratio = "ex_base_copy"
        # 'unpad' ensures the checkpoint's learned image_newline is loaded.
        # The dedicated packing branch does NOT unpad or pool the Base copies.
        config.mm_patch_merge_type = "spatial_unpad_ex_base_copy"
    elif roi_mode in {"anyres_max_9", "anyres_only"}:
        if roi_mode == "anyres_only" and config.mm_patch_merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}:
            raise ValueError("anyres_only requires spatial_unpad or spatial_unpad_add_newl packing")
        config.image_aspect_ratio = roi_mode
        # Preserve checkpoint packing, as in reference LLaVA.
    else:
        config.image_aspect_ratio = "randomroi"
        config.mm_patch_merge_type = "spatial_avgpool_auto_unpad_add_newl"
    validate_image_token_order(image_token_order, roi_mode, config.mm_patch_merge_type)
    config.image_token_order = image_token_order

def _generate_with_timing(model, generation_options, *, include_pruning_time=True):
    """Time one generate call; optional subtraction is a profiled diagnostic.

    Synchronized pruning sections cover explicit ranking/selection/masking/drop
    routines, not all downstream cache/position handling or Python dispatch.
    Extra synchronization changes scheduling, so adjusted time is not a claim
    about uninstrumented end-to-end speed. Default timing adds no inner syncs.
    """
    import torch
    if type(include_pruning_time) is not bool:
        raise ValueError("include_pruning_time must be a boolean")
    core = model.get_model()
    timer = None
    if not include_pruning_time:
        devices = {parameter.device for layer in getattr(core, "layers", ())
                   for parameter in layer.parameters()}
        if len(devices) > 1:
            raise ValueError("Exclude-pruning timing requires decoder weights on one device; "
                             "use one visible GPU without decoder offloading")
        from llava.model.language_model.pruning_timing import PruningTimer
        timer = PruningTimer()
    previous_timer = getattr(core, "_pruning_timer", None)
    core._pruning_timer = timer
    try:
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate(**generation_options)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
    finally:
        # A failed sample or a later default-mode run must not inherit a timer.
        core._pruning_timer = previous_timer
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("Invalid synchronized generation duration")
    timing = {"generation_seconds": elapsed}
    if timer is not None:
        pruning = timer.seconds
        if not math.isfinite(pruning) or not 0 <= pruning <= elapsed:
            raise RuntimeError("Measured pruning duration is inconsistent with generation duration")
        timing.update(generation_seconds=elapsed - pruning,
                      generation_with_pruning_seconds=elapsed, pruning_seconds=pruning)
    return generated, timing

class LlavaBackend:
    def __init__(self, model_path: str | Path, roi_mode: str = "randomroi",
                 image_token_order: str = "base_first"):
        validate_image_token_order(image_token_order, roi_mode)
        # Delay heavy imports so JSON/configuration checks run without a GPU stack.
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("LLaVA inference requires a CUDA GPU")
        # Match the current original LLaVA loader. Do not override TF32 flags
        # or silently fall back to another precision/attention implementation.
        from transformers import AutoTokenizer
        from llava.constants import (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN)
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Model checkpoint not found: {path}")
        config = LlavaQwenConfig.from_pretrained(path)
        configure_image_mode(config, roi_mode, image_token_order)
        # Like reference LLaVA's --overwrite_image_aspect_ratio, anyres leaves the
        # checkpoint's merge type intact (normally spatial_unpad).
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = LlavaQwenForCausalLM.from_pretrained(
            path, config=config, low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", torch_dtype=torch.float16,
            device_map="auto",
        )
        if getattr(config, "mm_use_im_patch_token", True):
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if getattr(config, "mm_use_im_start_end", False):
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.vision_tower = self.model.get_vision_tower()
        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model(device_map="auto")
        self.processor = self.vision_tower.image_processor
        self.inference_config = {
            "model_dtype": str(self.model.dtype),
            "vision_dtype": str(self.vision_tower.dtype),
            "attention": self.model.config._attn_implementation,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "image_aspect_ratio": self.model.config.image_aspect_ratio,
            "mm_patch_merge_type": self.model.config.mm_patch_merge_type,
            "image_token_order": image_token_order,
            "attention_source": "independent_qk_at_nonzero_rates_only",
            "score_dtype": "float32",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
        if roi_mode == "ex_base_copy":
            self.inference_config.update(base_view_count=3,
                                         visual_packing="base1 + base2 + base3 + one final newline")
        elif roi_mode == "anyres_only":
            self.inference_config.update(base_view_count=0, anyres_max_patches=9,
                                         visual_packing="anyres tiles only; original unpad/downsample/newlines")
        from importlib.metadata import version
        self.inference_config.update({
            "transformers": version("transformers"),
            "flash_attn": version("flash-attn"),
        })
        print(
            f"Inference config: dtype={self.model.dtype}, "
            f"vision_dtype={self.vision_tower.dtype}, "
            f"attention={self.model.config._attn_implementation}, "
            f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn_tf32={torch.backends.cudnn.allow_tf32}, "
            f"image_aspect_ratio={self.model.config.image_aspect_ratio}, "
            f"mm_patch_merge_type={self.model.config.mm_patch_merge_type}, "
            f"image_token_order={image_token_order}",
            flush=True,
        )

    def generate(self, sample: Sample, prompt: str, method: PruningMethod,
                 rate: int, roi_mode: str, *, capture_visualization: bool,
                 capture_attention: bool,
                 random_seed: int, do_sample: bool = False,
                 include_pruning_time: bool = True) -> dict:
        import torch
        from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        from llava.mm_utils import process_images, tokenizer_image_token

        method.configure(self.model.get_model(), rate, capture_attention=capture_attention,
                         capture_visualization=capture_visualization)
        if (capture_visualization and roi_mode in {"anyres_max_9", "anyres_only"}
                and self.model.config.mm_patch_merge_type not in
                {"spatial_unpad", "spatial_unpad_add_newl"}):
            raise ValueError(
                "Anyres visualization requires spatial_unpad packing. The checkpoint's "
                "merge type is preserved to match reference; it is not silently overwritten."
            )
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        with Image.open(sample.image) as opened:
            image = opened.convert("RGB")
        # This script always uses annotation-free AnyRes preprocessing.
        roi_source, boxes_list, masks = "anyres", None, None
        pixels, crop_metadata = process_images(
            [image], self.processor, self.model.config,
            masks=masks, boxes_list=boxes_list, return_pro_data=True,
        )
        if isinstance(pixels, list):
            pixels = [item.to(self.model.device, dtype=self.model.dtype) for item in pixels]
        else:
            pixels = pixels.to(self.model.device, dtype=self.model.dtype)

        message = prompt if DEFAULT_IMAGE_TOKEN in prompt else f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        if len(prompt) > 4096:
            raise ValueError("reference LLaVA's single-image prompt limit is 4096 characters")
        if message.count(DEFAULT_IMAGE_TOKEN) != 1:
            raise ValueError("A sample prompt must contain exactly one <image> token")
        # Equivalent to the old qwen_1_5 single-turn conversation template.
        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n")
        if getattr(self.model.config, "mm_use_im_start_end", False):
            text = text.replace(DEFAULT_IMAGE_TOKEN,
                                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        tokens = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).to(self.model.device)

        # Match SimpleModelWorker's context budget, including its base-view
        # patch count approximation (not a newly chosen anyres token budget).
        max_new_tokens = min(
            512, getattr(self.model.config, "max_position_embeddings", 2048)
            - tokens.shape[-1] - self.vision_tower.num_patches,
        )
        if max_new_tokens < 1:
            raise ValueError("Prompt exceeds the original LLaVA context budget")
        generation_options = {
            "inputs": tokens, "images": pixels, "image_sizes": [image.size],
            "do_sample": do_sample,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        generated, timing = _generate_with_timing(
            self.model, generation_options, include_pruning_time=include_pruning_time)
        answer = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        core = self.model.get_model()
        result = {
            "answer": answer,
            **timing,
            "roi_source": roi_source,
            "roi_boxes": crop_metadata[0].get("roi_boxes", []),
            "stats": method.stats(core, rate),
            "allocation": method.allocation_data(core),
            "max_new_tokens": max_new_tokens,
        }
        if capture_visualization:
            result.update(method.visualization_data(core, capture_attention=capture_attention))
            result["image"] = image
            result["crop_metadata"] = crop_metadata
        return result

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
        prepared_split = getattr(adapter,"prepared_prompt_split",None) if getattr(adapter,"lean",False) else None
        if prepared_split is not None and attention_mask is None:
            split = prepared_split
        else:
            placeholders = torch.where(prompt_ids == IMAGE_TOKEN_INDEX)[0]
            if placeholders.numel() != 1:
                raise ValueError("Prompt must contain exactly one unmasked <image> placeholder")
            split = int(placeholders.item())
        # At rate=0 this still calls VisionZip scoring/selection/projector/packing.
        # No fallback to the original multimodal implementation on image prefill.
        adapter.set_image_context(image_sizes[0])
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

def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")




def boolean_argument(value):
    """Explicit True/False CLI values; bool('False') would incorrectly be True."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def validate_allocation_temperature(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("allocation_temperature must be a finite positive number")
    return float(value)


def allocation_temperature_argument(value):
    try:
        return validate_allocation_temperature(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("allocation_temperature must be a finite positive number") from exc




def _redundancy_unit_vectors(values, description, *, compact_checks=False):
    """Return true unit vectors, rejecting undefined directions explicitly."""
    if compact_checks:
        norms = torch.linalg.vector_norm(values, ord=2, dim=-1, keepdim=True)
        valid = torch.isfinite(values).all() & torch.isfinite(norms).all() & (norms > 0).all()
        if not bool(valid):
            raise ValueError(f"{description} contain non-finite or zero-norm vectors")
        return values / norms
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{description} contain non-finite values")
    norms = torch.linalg.vector_norm(values, ord=2, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all().item()):
        raise ValueError(f"{description} have non-finite L2 norms in float32")
    if bool((norms <= 0).any().item()):
        raise ValueError(f"{description} contain a zero-norm vector; its unit direction is undefined")
    return values / norms


def pooled_crop_redundancy(features, grid, reduction="mean", *, audit=True, prepared=None):
    """Score each AnyRes crop by two internal pooled-token cosine groups.

    ``features`` is [1 + width * height, N, D], ordered Base first, then
    row-major AnyRes crops. N must be square. Each input token is first L2
    normalized in float32; its region/crop mean is L2 normalized again before
    cosine similarity. With reduction="mean", each group's cosine sum is
    divided by its own token count before the two groups are added; "sum"
    preserves the unscaled sums. No similarity is shifted or clipped at zero.
    Original features
    are never modified and all tensor results stay on the input device.

    Base bins use floor(side / grid_dimension), assigning the entire remainder
    to the final row/column, exactly as the existing budget allocator does.
    Layers are scored separately by the caller, then their scores are averaged.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("features must be a torch tensor with shape [views, tokens, channels]")
    if not torch.is_floating_point(features) or torch.is_complex(features):
        raise ValueError("features must contain real floating-point values")
    views, token_count, channels = features.shape
    if token_count < 1 or channels < 1:
        raise ValueError("features must have at least one token and one channel")
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise ValueError("features must contain a square token grid per view")
    if not isinstance(grid, (tuple, list)) or len(grid) != 2:
        raise ValueError("grid must be (width, height)")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in grid):
        raise ValueError("grid dimensions must be positive integers")
    grid_width, grid_height = (int(value) for value in grid)
    if not (1 <= grid_width <= side and 1 <= grid_height <= side):
        raise ValueError("grid dimensions must be positive and no larger than the token-grid side")
    crop_count = grid_width * grid_height
    if views != 1 + crop_count:
        raise ValueError(f"features must have {1 + crop_count} views for grid {tuple(grid)}; got {views}")

    # Casting/detaching does not change the caller's input, even if it is FP32.
    token_unit = _redundancy_unit_vectors(
        features.detach().to(dtype=torch.float32), "input feature tokens", compact_checks=not audit
    )
    anyres_tokens = token_unit[1:]
    anyres_pool_unit = _redundancy_unit_vectors(
        anyres_tokens.mean(dim=1), "AnyRes pooled features", compact_checks=not audit
    )
    anyres_similarity = (anyres_tokens * anyres_pool_unit[:, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    anyres_similarity_sum = anyres_similarity.sum(dim=1)

    bin_width, bin_height = side // grid_width, side // grid_height
    base_token_indices = []
    base_pools = []
    base_sums = []
    base_similarity_map = torch.empty(token_count, dtype=torch.float32, device=features.device) if audit else None
    for crop in range(crop_count):
        crop_row, crop_col = divmod(crop, grid_width)
        row_start, col_start = crop_row * bin_height, crop_col * bin_width
        row_end = side if crop_row == grid_height - 1 else row_start + bin_height
        col_end = side if crop_col == grid_width - 1 else col_start + bin_width
        if prepared is not None:
            if prepared["grid"] != tuple(grid) or prepared["side"] != side:
                raise ValueError("Prepared Base bins disagree with actual geometry")
            indices = prepared["base_token_indices"][crop]
            index_tensor = prepared["base_index_tensors"][crop]
        else:
            indices = [row * side + col
                       for row in range(row_start, row_end)
                       for col in range(col_start, col_end)]
            index_tensor = torch.tensor(indices, dtype=torch.long, device=features.device)
        base_token_indices.append(indices)
        region_tokens = token_unit[0].index_select(0, index_tensor)
        region_pool_unit = _redundancy_unit_vectors(
            region_tokens.mean(dim=0), f"Base pooled feature for crop {crop}", compact_checks=not audit
        )
        similarity = (region_tokens * region_pool_unit).sum(dim=-1).clamp(-1.0, 1.0)
        if audit:
            base_similarity_map.index_copy_(0, index_tensor, similarity)
            base_pools.append(region_pool_unit)
        base_sums.append(similarity.sum())

    base_similarity_sum = torch.stack(base_sums)
    base_token_counts = prepared["base_token_counts"] if prepared is not None else torch.tensor(
        [len(indices) for indices in base_token_indices],
        dtype=torch.long,
        device=features.device,
    )
    if reduction == "mean":
        base_redundancy = base_similarity_sum / base_token_counts.to(dtype=torch.float32)
        anyres_redundancy = anyres_similarity_sum / token_count
    else:
        base_redundancy = base_similarity_sum
        anyres_redundancy = anyres_similarity_sum
    redundancy = base_redundancy + anyres_redundancy
    if not bool(torch.isfinite(redundancy).all().item()):
        raise ValueError("pooled-cosine redundancy produced non-finite scores")
    if not audit:
        return dict(redundancy=redundancy, base_token_counts=base_token_counts,
                    base_token_indices=base_token_indices)
    return {
        "redundancy": redundancy,
        "base_similarity_sum": base_similarity_sum,
        "anyres_similarity_sum": anyres_similarity_sum,
        "base_redundancy": base_redundancy,
        "anyres_redundancy": anyres_redundancy,
        "base_similarity_map": base_similarity_map,
        "anyres_similarity": anyres_similarity,
        "base_pool_unit": torch.stack(base_pools),
        "anyres_pool_unit": anyres_pool_unit,
        "base_token_counts": base_token_counts,
        "base_token_indices": base_token_indices,
    }


def redundancy_to_base_map(redundancy, base_token_indices, token_count):
    """Encode each negative crop score as the existing allocator's Base SUM.

    Values within each Base bin are uniform, apart from a floating-point
    correction on its final row-major token. Therefore the unchanged allocator
    sees crop score -redundancy and assigns fewer retained tokens to crops with
    greater redundancy. Returned values are np.float64 and are not Min-Max
    normalized or otherwise rescaled.
    """
    if (isinstance(token_count, bool) or not isinstance(token_count, Integral)
            or token_count < 1):
        raise ValueError("token_count must be a positive integer")
    token_count = int(token_count)
    if math.isqrt(token_count) ** 2 != token_count:
        raise ValueError("token_count must describe a square token grid")
    if isinstance(redundancy, torch.Tensor):
        if torch.is_complex(redundancy):
            raise ValueError("redundancy must contain real values")
        redundancy = redundancy.detach().to(dtype=torch.float64).cpu().numpy()
    raw = np.asarray(redundancy)
    if raw.ndim != 1 or raw.dtype.kind not in "fiu" or not len(raw):
        raise ValueError("redundancy must be a nonempty one-dimensional real numeric vector")
    scores = raw.astype(np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("redundancy must contain only finite values")
    if not isinstance(base_token_indices, (tuple, list)) or len(base_token_indices) != len(scores):
        raise ValueError("base_token_indices must contain one nonempty bin per redundancy score")

    validated_bins = []
    seen = set()
    for indices in base_token_indices:
        if not isinstance(indices, (tuple, list)) or not indices:
            raise ValueError("each Base bin must be a nonempty sequence of token indices")
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices):
            raise ValueError("Base token indices must be integers")
        indices = [int(index) for index in indices]
        if indices != sorted(indices):
            raise ValueError("each Base bin must use ascending row-major token indices")
        for index in indices:
            if not 0 <= index < token_count or index in seen:
                raise ValueError("Base bins must partition token indices without duplicates or out-of-range indices")
            seen.add(index)
        validated_bins.append(indices)
    if len(seen) != token_count:
        raise ValueError("Base bins must cover all token indices")

    base_map = np.empty(token_count, dtype=np.float64)
    for score, indices in zip(scores, validated_bins):
        target = -float(score)
        fill = target / len(indices)
        running = 0.0
        # Match allocate_crop_budgets' sequential float64 accumulation exactly.
        for index in indices[:-1]:
            base_map[index] = fill
            running += fill
        base_map[indices[-1]] = target - running
        if running + base_map[indices[-1]] != target:
            raise ValueError("unable to preserve the negative redundancy in float64 crop accumulation")
    return base_map




def validate_spatial_strength(value):
    """Spatial strength is now lambda, a probability mixture weight in [0,1]."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("spatial_strength / spatial_mix_weight must be finite in [0,1]")
    return float(value)


def spatial_strength_argument(value):
    try:
        return validate_spatial_strength(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("spatial-strength / spatial-mix-weight must be finite in [0,1]") from exc


def center_spatial_prior(token_count, base_token_indices, strength=DEFAULT_SPATIAL_STRENGTH):
    """Return one global flat-top spatial field and its per-crop mean bonuses.

    Coordinates are the centers of the original square Base token grid in
    [-1,1]. A separable raised-cosine shoulder connects the central plateau
    (half-width1/6) to zero (half-width1/2). The field is built ONCE globally,
    not once per crop. The current allocator's Base-bin indices define pooling.
    Crop means alone are divided by their maximum. No original redundancy or
    combined allocation score is normalized, clipped, or otherwise rescaled.

    Returns FP64 NumPy arrays; caller inputs are never mutated. Strength0 is an
    exact disabled spatial-bonus ablation. A tiny test grid missing the entire
    kernel support returns zero prior/bonus instead of an undefined division.
    This is a custom center prior inspired by spatial protection, not ATP-LLaVA
    SAP's multi-granularity uniform sampling score or learned-mask procedure.
    """
    strength = validate_spatial_strength(strength)
    if (isinstance(token_count, bool) or not isinstance(token_count, Integral)
            or token_count < 1 or math.isqrt(token_count) ** 2 != token_count):
        raise ValueError("token_count must describe a positive square token grid")
    token_count = int(token_count)
    if not isinstance(base_token_indices, (tuple, list)) or not base_token_indices:
        raise ValueError("base_token_indices must contain nonempty Base bins")
    bins, seen = [], set()
    for indices in base_token_indices:
        if not isinstance(indices, (tuple, list)) or not indices:
            raise ValueError("each Base bin must be a nonempty list or tuple")
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices):
            raise ValueError("Base token indices must be integers")
        values = [int(index) for index in indices]
        if values != sorted(values):
            raise ValueError("Base bins must use ascending row-major token indices")
        for index in values:
            if index < 0 or index >= token_count or index in seen:
                raise ValueError("Base bins must partition the token grid without duplicate or out-of-range indices")
            seen.add(index)
        bins.append(values)
    if len(seen) != token_count:
        raise ValueError("Base bins must cover the entire token grid")

    side = math.isqrt(token_count)
    coordinates = 2 * (np.arange(side, dtype=np.float64) + 0.5) / side - 1
    distance = np.abs(coordinates)
    plateau, support = SPATIAL_PLATEAU_HALF_WIDTH, SPATIAL_SUPPORT_HALF_WIDTH
    axis = np.zeros(side, dtype=np.float64)
    axis[distance <= plateau] = 1.0
    shoulders = (distance > plateau) & (distance < support)
    axis[shoulders] = 0.5 * (1 + np.cos(np.pi * (distance[shoulders] - plateau) / (support - plateau)))
    field = np.outer(axis, axis).reshape(-1)
    means = np.array([field[indices].mean(dtype=np.float64) for indices in bins], dtype=np.float64)
    maximum = float(means.max())
    prior = means / maximum if maximum > 0 else np.zeros_like(means)
    bonus = strength * prior
    if not np.isfinite(bonus).all():
        raise ValueError("Spatial bonus contains non-finite values")
    return {
        "spatial_field": field,
        "crop_means": means,
        "spatial_prior": prior,
        "spatial_bonus": bonus,
        "strength": strength,
        "kernel_parameters": {
            "name": "global_separable_flat_top_raised_cosine",
            "plateau_half_width": plateau,
            "support_half_width": support,
            "coordinates": "Base token centers in [-1,1] on both axes",
            "base_grid_side": side,
            "crop_pooling": "mean over unchanged allocator Base bins",
            "prior_calibration": "divide crop means by their maximum; zero if no sampled support",
            "empty_sampled_support": maximum == 0,
            "combined_score_normalization": "none",
            "scope": "crop budget only; original VisionZip token selection and merge unchanged",
        },
    }


@dataclass(frozen=True)
class MethodSettings:
    prune_rate: int = PRUNE_RATE
    allocation_temperature: float = DEFAULT_ALLOCATION_TEMPERATURE
    redundancy_layers: tuple[int, ...] = DEFAULT_REDUNDANCY_LAYERS
    redundancy_reduction: str = DEFAULT_REDUNDANCY_REDUCTION
    visionzip_layers: tuple[int, ...] = DEFAULT_VISIONZIP_LAYERS
    spatial_strength: float = DEFAULT_SPATIAL_STRENGTH

    def validate(self):
        if type(self.prune_rate) is not int or not 0 <= self.prune_rate <= 99:
            raise ValueError("prune_rate must be an integer in 0..99")
        validate_allocation_temperature(self.allocation_temperature)
        validate_spatial_strength(self.spatial_strength)
        if self.redundancy_reduction not in {"mean", "sum"}:
            raise ValueError("redundancy_reduction must be mean or sum")
        for label, layers in (("Redundancy", self.redundancy_layers),
                              ("VisionZip", self.visionzip_layers)):
            if (not layers or tuple(sorted(set(layers))) != layers
                    or any(type(layer) is not int or not 1 <= layer <= 26 for layer in layers)):
                raise ValueError(f"{label} layers must be increasing distinct SigLIP block indices in 1..26")
        return self


class CustomPruningAdapter:
    """Internal pooled cosine allocates quotas; native VisionZip compresses."""
    def __init__(self, core, settings):
        self.core = core
        self.tower = core.get_vision_tower()
        self.settings = settings.validate()
        self.enabled = False
        self.capture_visualization = False
        self.reset()

    def reset(self):
        self.decisions = []
        self.stats = {}
        self.allocation = {}
        self.redundancy_details = {}
        self.spatial_details = {}
        self.visionzip_details = {}
        self.geometry = None
        self.original_size = None

    def configure(self, rate, *, capture_visualization=True):
        if rate != self.settings.prune_rate:
            raise ValueError("Runtime rate and configured crop-allocation rate disagree")
        self.reset()
        self.capture_visualization = capture_visualization
        self.enabled = True
        blocks = self.tower.vision_tower.vision_model.encoder.layers
        if len(blocks) != 26 or self.tower.num_patches != 729:
            raise ValueError("This method targets the current 26-active-block SigLIP patch14/384 tower")
        if (self.core.config.image_aspect_ratio != "anyres_max_9"
                or self.core.config.mm_patch_merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}
                or getattr(self.core.config, "image_token_order", "base_first") != "base_first"):
            raise ValueError("Use the original spatial_unpad checkpoint with AnyRes max_9 and Base first")

    def set_image_context(self, original_size):
        from llava.mm_utils import get_anyres_image_grid_shape
        grid = get_anyres_image_grid_shape(original_size, self.core.config.image_grid_pinpoints,
                                           self.tower.image_size)
        self.original_size = tuple(map(int, original_size))
        self.geometry = ScoreGeometry.create(self.original_size, tuple(grid),
                                              int(self.tower.image_size), int(self.tower.config.patch_size))

    def allocate_view_budgets(self, raw_redundancy, first, spatial):
        """Override point for composed methods; standalone arithmetic is unchanged."""
        allocation, probability = allocate_spatial_probability_budgets(
            raw_redundancy, first["base_token_indices"], self.geometry.grid, spatial["spatial_prior"],
            prune_rate=self.settings.prune_rate, mix_weight=self.settings.spatial_strength,
            temperature=self.settings.allocation_temperature, minimum_keep=MINIMUM_CROP_KEEP)
        base_d, base_c = token_budget(729, self.settings.prune_rate / 100)
        return allocation, probability, base_d, base_c

    def encode_images(self, images):
        if self.core.training:
            raise ValueError("Custom pruning requires eval mode")
        if self.geometry is None:
            raise RuntimeError("Image geometry must be set before scoring/crop allocation")
        expected = 1 + self.geometry.tile_count
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] != expected:
            raise ValueError("Encoder input does not match Base + AnyRes crop geometry")
        blocks = self.tower.vision_tower.vision_model.encoder.layers
        attention = blocks[-1].self_attn
        captured, score_layers, redundancy_layers, handles = {}, {}, {}, []

        def capture_key(module, args, output):
            batch, tokens, _ = output.shape
            captured["metric"] = output.view(batch, tokens, attention.num_heads,
                                              attention.head_dim).permute(0, 2, 1, 3).mean(dim=1)

        def attention_hook(layer):
            def capture(module, args, kwargs, output):
                if layer in score_layers:
                    raise RuntimeError("A VisionZip scoring layer ran more than once")
                weights = output[1]
                if weights is None:
                    raise RuntimeError(f"SigLIP attention probabilities are unavailable at scoring layer {layer}")
                # Preserve the original native precision and head/query mean.
                scores = weights.mean(dim=1).mean(dim=1)
                if scores.shape != (expected, 729):
                    raise ValueError("Unexpected VisionZip scoring-layer attention shape")
                score_layers[layer] = scores
            return capture

        def redundancy_hook(layer):
            def capture(module, args, output):
                if layer in redundancy_layers:
                    raise RuntimeError("A redundancy feature layer ran more than once")
                features = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(features) or features.shape[:2] != (expected, 729):
                    raise ValueError("Unexpected redundancy-layer feature shape")
                # Unit-normalized copies are used ONLY for scoring. Every layer
                # is pooled/scored independently; native model rows are untouched.
                redundancy_layers[layer] = pooled_crop_redundancy(
                    features, self.geometry.grid, reduction=self.settings.redundancy_reduction)
            return capture

        try:
            handles.append(attention.k_proj.register_forward_hook(capture_key))
            for layer in self.settings.visionzip_layers:
                handles.append(blocks[layer - 1].self_attn.register_forward_hook(
                    attention_hook(layer), with_kwargs=True))
            for layer in self.settings.redundancy_layers:
                handles.append(blocks[layer - 1].register_forward_hook(redundancy_hook(layer)))
            hidden = self.tower(images)
        finally:
            for handle in handles:
                handle.remove()
        if (set(captured) != {"metric"}
                or set(score_layers) != set(self.settings.visionzip_layers)
                or set(redundancy_layers) != set(self.settings.redundancy_layers)
                or hidden.shape[:2] != (expected, 729)):
            raise RuntimeError("Incomplete SigLIP features / attention / Key capture")
        if len(self.settings.visionzip_layers) == 1:
            captured["attention"] = score_layers[self.settings.visionzip_layers[0]]
        else:
            captured["attention"] = torch.stack([
                score_layers[layer].float() for layer in self.settings.visionzip_layers
            ], dim=0).mean(dim=0)

        first = redundancy_layers[self.settings.redundancy_layers[0]]
        components = {}
        for name in ("redundancy", "base_redundancy", "anyres_redundancy",
                     "base_similarity_sum", "anyres_similarity_sum"):
            components[name] = torch.stack([
                redundancy_layers[layer][name] for layer in self.settings.redundancy_layers
            ], dim=0).mean(dim=0)
        # Geometry stays unchanged in this ablation. No score-level bonus.
        spatial = center_spatial_prior(729, first["base_token_indices"], strength=0.0)
        raw_redundancy = components["redundancy"].detach().to(dtype=torch.float64).cpu().numpy()
        scores_before_spatial = -raw_redundancy
        # Keep the legacy +zero operation for the exact disabled control.
        scores_after_spatial = scores_before_spatial + np.zeros_like(scores_before_spatial)
        allocation_map = redundancy_to_base_map(
            -scores_after_spatial, first["base_token_indices"], 729)
        allocation, probability, base_d, base_c = self.allocate_view_budgets(raw_redundancy, first, spatial)
        spatial["spatial_bonus"] = probability["spatial_probability_contribution"]
        # There is no fused score in the probability-mixture scheme.
        after_score_record = scores_after_spatial.tolist() if self.settings.spatial_strength == 0 else None
        base_keep = base_d + base_c
        keeps = [base_keep] + [crop["keep"] for crop in allocation["crops"]]
        if any(keep < 4 for keep in keeps):
            raise ValueError("A crop budget is too small for the fixed dominant/contextual split; lower pruning rate")
        # Allocation evidence never enters native attention ranking / Key merge.
        packed, decisions = compress_variable_views(
            hidden, captured["attention"], captured["metric"], keeps,
            self.geometry.grid, self.core.mm_projector, compress_tokens, token_budget,
            image_newline=self.core.image_newline, merge_type=self.core.config.mm_patch_merge_type,
            original_size=self.original_size)
        full_grid = decisions[0]["full_grid_identity"]
        newlines = decisions[0]["packing_newline_count"]
        before = baseline_packed_count(self.original_size, self.geometry.grid, 27, self.core.config.mm_patch_merge_type)
        expected_packed = before if full_grid else sum(keeps) + newlines
        if packed.shape[0] != expected_packed:
            raise RuntimeError("Packed sequence disagrees with content and structural budgets")
        cpu = {name: value.detach().cpu().numpy() for name, value in components.items()}
        base_counts = first["base_token_counts"].detach().cpu().tolist()
        for index, (crop, decision) in enumerate(zip(allocation["crops"], decisions[1:])):
            crop.update(dominant=decision["dominant_count"], contextual=decision["contextual_count"],
                        base_token_count=base_counts[index], anyres_token_count=729)
            crop.update({name: float(values[index]) for name, values in cpu.items()})
            crop.update(allocation_score_before_spatial=float(scores_before_spatial[index]),
                        spatial_prior=float(spatial["spatial_prior"][index]),
                        spatial_bonus=float(spatial["spatial_bonus"][index]),
                        allocation_score_after_spatial=(float(scores_after_spatial[index])
                            if self.settings.spatial_strength == 0 else None),
                        redundancy_probability=float(probability["redundancy_probabilities"][index]),
                        spatial_probability=float(probability["spatial_probabilities"][index]),
                        fused_probability=float(probability["mixed_probabilities"][index]),
                        spatial_probability_contribution=float(probability["spatial_probability_contribution"][index]))
        allocation.update(
            base_keep=base_keep, base_tokens_before=729, base_prune=729 - base_keep,
            base_dominant=base_d, base_contextual=base_c,
            redundancy_layers=list(self.settings.redundancy_layers),
            redundancy_reduction=self.settings.redundancy_reduction,
            visionzip_layers=list(self.settings.visionzip_layers),
            redundancy=[float(value) for value in cpu["redundancy"]],
            score_direction="p_R=softmax(-R/T); p=(1-lambda)*p_R+lambda*p_S; larger mixed probability keeps more",
            spatial_strength=self.settings.spatial_strength,
            spatial_mix_weight=self.settings.spatial_strength,
            spatial_fusion_rule="probability_convex_mix; lambda=0 uses original allocator unchanged",
            probability_fusion_mode=probability["mode"],
            effective_spatial_mix_weight=probability["effective_mix_weight"],
            spatial_prior_empty=probability["spatial_empty"],
            redundancy_probabilities=probability["redundancy_probabilities"].tolist(),
            spatial_probabilities=probability["spatial_probabilities"].tolist(),
            mixed_probabilities=probability["mixed_probabilities"].tolist(),
            spatial_bonus_units="probability contribution lambda*p_S; legacy field name",
            spatial_prior=spatial["spatial_prior"].tolist(),
            spatial_bonus=spatial["spatial_bonus"].tolist(),
            allocation_scores_before_spatial=scores_before_spatial.tolist(),
            allocation_scores_after_spatial=after_score_record,
            spatial_kernel=spatial["kernel_parameters"],
            pooling="L2-normalize tokens -> arithmetic mean -> L2-normalize pooled vector -> cosine with own tokens",
            layer_aggregation="equal mean of independently computed per-layer scalar scores",
            base_region_mapping="original allocator floor-grid bins; final row/column receives remainder",
            feature_layer=26, merge_key_layer=26,
            token_selection="native per-view VisionZip attention and block26 Key; no redundancy scores")
        self.allocation, self.decisions = allocation, decisions
        if self.capture_visualization:
            self.spatial_details = {
                key: spatial[key].copy() for key in
                ("spatial_field", "crop_means", "spatial_prior", "spatial_bonus")}
            self.spatial_details.update(
                spatial_strength=np.asarray(self.settings.spatial_strength, dtype=np.float64),
                allocation_scores_before_spatial=scores_before_spatial.copy(),
                redundancy_probabilities=probability["redundancy_probabilities"].copy(),
                spatial_probabilities=probability["spatial_probabilities"].copy(),
                mixed_probabilities=probability["mixed_probabilities"].copy(),
                spatial_probability_contribution=probability["spatial_probability_contribution"].copy(),
                fusion_rule=np.asarray("probability_convex_mix; no score-level addition"),
                kernel_parameters_json=np.asarray(json.dumps(spatial["kernel_parameters"], sort_keys=True)))
            if self.settings.spatial_strength == 0:
                self.spatial_details["allocation_scores_after_spatial"] = scores_after_spatial.copy()
            self.redundancy_details = {
                "layers": np.asarray(self.settings.redundancy_layers, dtype=np.int64),
                "reduction": np.asarray(self.settings.redundancy_reduction),
                "allocation_base_map": allocation_map,
                "base_token_counts": np.asarray(base_counts, dtype=np.int64),
                **{name: values.copy() for name, values in cpu.items()},
            }
            for name in ("redundancy", "base_redundancy", "anyres_redundancy",
                         "base_similarity_sum", "anyres_similarity_sum",
                         "base_similarity_map", "anyres_similarity"):
                self.redundancy_details[name + "_per_layer"] = np.stack([
                    redundancy_layers[layer][name].detach().float().cpu().numpy()
                    for layer in self.settings.redundancy_layers])
            self.visionzip_details = {
                "layers": np.asarray(self.settings.visionzip_layers, dtype=np.int64),
                "scores_per_layer": np.stack([
                    score_layers[layer].detach().float().cpu().numpy()
                    for layer in self.settings.visionzip_layers]),
                "averaged_scores": captured["attention"].detach().float().cpu().numpy(),
                "merge_key_layer": 26, "feature_layer": 26,
            }
        self.stats = {
            "mode": "pooled_internal_redundancy_spatial_anchor_order_visionzip",
            "spatial_strength": self.settings.spatial_strength,
            "spatial_mix_weight": self.settings.spatial_strength,
            "spatial_fusion_rule": "probability_convex_mix",
            "redundancy_probabilities": probability["redundancy_probabilities"].tolist(),
            "spatial_probabilities": probability["spatial_probabilities"].tolist(),
            "mixed_probabilities": probability["mixed_probabilities"].tolist(),
            "spatial_prior": spatial["spatial_prior"].tolist(),
            "spatial_bonus": spatial["spatial_bonus"].tolist(),
            "allocation_scores_before_spatial": scores_before_spatial.tolist(),
            "allocation_scores_after_spatial": after_score_record,
            "visionzip_source_commit": SOURCE_COMMIT, "globalcom2_source_commit": GLOBALCOM_SOURCE_COMMIT,
            "redundancy_layers": list(self.settings.redundancy_layers),
            "redundancy_reduction": self.settings.redundancy_reduction,
            "redundancy": cpu["redundancy"].tolist(),
            "visionzip_layers": list(self.settings.visionzip_layers),
            "feature_layer": 26, "merge_key_layer": 26,
            "allocation_rule": allocation["allocation_rule"],
            "allocation_temperature": self.settings.allocation_temperature,
            "minimum_crop_keep": MINIMUM_CROP_KEEP, "maximum_crop_keep": 729,
            "encoder_views": expected, "tokens_per_view_before": 729,
            "base_tokens_before": 729, "base_tokens_after": base_keep,
            "base_dominant": base_d, "base_contextual": base_c,
            "anyres_crops": self.geometry.tile_count, "anyres_tokens_before": 729 * self.geometry.tile_count,
            "anyres_tokens_after": sum(keeps[1:]), "anyres_target_tokens": allocation["target_anyres_keep"],
            "actual_base_prune_rate_percent": 100 * (1 - base_keep / 729),
            "actual_anyres_prune_rate_percent": 100 * (1 - sum(keeps[1:]) / (729 * self.geometry.tile_count)),
            "encoder_tokens_before": expected * 729, "encoder_tokens_after": sum(keeps),
            "actual_encoder_prune_rate_percent": 100 * (1 - sum(keeps) / (expected * 729)),
            "final_llm_visual_tokens": int(packed.shape[0]), "newline_tokens_after": newlines,
            "full_grid_identity": full_grid,
            "encoder_feature_stride": list(hidden.stride()),
            "projector_input_stride": decisions[0]["projector_input_stride"],
            "projector_input_shape": decisions[0]["projector_input_shape"],
            "packed_content_tokens": int(packed.shape[0]) - newlines,
            "sparse_row_newlines": True,
            "packing": ("complete selected grid: unpad + max9 resize + row newlines + Base first" if full_grid else
                        "Base D+C by anchors + AnyRes D+C by global anchors with original-row newlines; no unpad/max9 resize"),
            "contextual_order": "original_anchor_global_row_major",
            "position_id_policy": "consecutive compressed-sequence IDs",
        }
        return packed

    def pack(self, image_features, image_sizes, modalities):
        if len(image_features) != 1 or len(image_sizes) != 1 or list(modalities) != ["image"]:
            raise ValueError("Custom pruning supports one image per generation")
        packed = image_features[0]
        if packed.ndim != 2 or packed.shape[0] != self.stats["final_llm_visual_tokens"]:
            raise ValueError("Prepared packed feature dimensions disagree with token budgets")
        before = baseline_packed_count(self.original_size, self.geometry.grid, 27,
                                       self.core.config.mm_patch_merge_type)
        self.stats.update(baseline_packed_tokens=before,
                          actual_packed_reduction_percent=100 * (1 - packed.shape[0] / before))
        return [packed]


class CustomPruningMethod:
    name = "custom_pooled_redundancy_spatial_allocation"
    def __init__(self, core, settings):
        self.adapter = CustomPruningAdapter(core, settings)
    def configure(self, core, prune_rate, *, capture_attention=False, capture_visualization=True):
        core.configure_fastv(enabled=False)
        self.adapter.configure(prune_rate, capture_visualization=capture_visualization)
    def stats(self, core, prune_rate):
        return dict(self.adapter.stats)
    def allocation_data(self, core):
        return self.adapter.allocation
    def visualization_data(self, core, *, capture_attention=False):
        return {"decisions": self.adapter.decisions, "allocation": self.adapter.allocation,
                "redundancy_details": self.adapter.redundancy_details,
                "spatial_details": self.adapter.spatial_details,
                "visionzip_details": self.adapter.visionzip_details, "geometry": self.adapter.geometry}


def compress_variable_views(features, scores, metric, keep_counts, grid,
                            projector, compress_fn, token_budget_fn, *,
                            image_newline=None, merge_type="spatial_unpad", original_size=None):
    """Per-view VisionZip merging, original anchors, and protected row markers.

    Full retention still executes the method selector, but passes the original
    3D feature tensor to the projector and uses native full-grid packing.
    Other quotas use the existing independent pre-projector merge. Content and
    structural budgets are separate. Missing newlines are only a helper-level
    compatibility mode; the main adapter always supplies the checkpoint marker.
    """
    def count(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("features must be a [views, tokens, channels] tensor")
    if (not isinstance(scores, torch.Tensor) or scores.ndim != 2
            or not isinstance(metric, torch.Tensor) or metric.ndim != 3
            or scores.shape != features.shape[:2] or metric.shape[:2] != features.shape[:2]):
        raise ValueError("features, scores and metric must have matching view/token dimensions")
    if not features.device == scores.device == metric.device:
        raise ValueError("features, scores and metric must be on the same device")
    if not isinstance(grid, (tuple, list)) or len(grid) != 2:
        raise ValueError("grid must be (width, height)")
    width, height = (count(value, "grid dimension") for value in grid)
    views, tokens, channels = features.shape
    side = math.isqrt(tokens)
    if min(width, height, side) < 1 or side * side != tokens or views != 1 + width * height:
        raise ValueError("Expected square tokens and Base plus the AnyRes grid")
    if not isinstance(keep_counts, (tuple, list)) or len(keep_counts) != views:
        raise ValueError("keep_counts must provide one integer for every view")
    keeps = [count(value, "keep count") for value in keep_counts]
    if any(value > tokens for value in keeps):
        raise ValueError("keep count cannot exceed per-view token count")
    full_grid = all(keep == tokens for keep in keeps)
    if full_grid:
        zipped, decisions = compress_fn(features, scores, metric, tokens, 0)
        if zipped is not features or len(decisions) != views:
            raise ValueError("Full retention must preserve the original feature tensor and layout")
        for view, decision in enumerate(decisions):
            if decision["dominant_indices"] != list(range(tokens)) or decision["contextual_indices"]:
                raise ValueError("Full retention must select every patch in original order without merging")
            decision.update(view_index=view, keep_count=tokens, dominant_count=tokens, contextual_count=0)
        projected = projector(zipped)
        if (not isinstance(projected, torch.Tensor) or projected.ndim != 3
                or projected.shape[:2] != features.shape[:2] or projected.shape[-1] < 1
                or not bool(torch.isfinite(projected).all())):
            raise ValueError("Full-grid projector must return finite [views,tokens,channels]")
        if image_newline is not None:
            if original_size is None:
                raise ValueError("Full-grid native packing needs original_size")
            native_decisions = TokenDecisions(tokens, tokens,
                torch.arange(tokens, device=projected.device).expand(views, -1))
            packed, newlines = pack_full_views(projected, native_decisions, grid,
                                              original_size, image_newline, merge_type)
        else:
            packed = pack_anchor_rows(list(projected.unbind(0)), decisions, grid)
            newlines = 0
        projector_shape, projector_stride = list(zipped.shape), list(zipped.stride())
    else:
        compressed_views, decisions = [], []
        for view, keep in enumerate(keeps):
            # A full-capacity crop also needs no contextual approximation.
            requested = keep if keep == tokens else round(keep * DOMINANT_RATIO)
            dominant, contextual = token_budget_fn(tokens, dominant_count=requested,
                                                   contextual_count=keep - requested)
            dominant, contextual = count(dominant, "dominant count"), count(contextual, "contextual count")
            if dominant + contextual != keep:
                raise ValueError("Token budget function changed the requested view quota")
            compressed, decision_rows = compress_fn(features[view:view+1], scores[view:view+1],
                metric[view:view+1], dominant, contextual)
            if compressed.shape != (1, keep, channels) or len(decision_rows) != 1:
                raise ValueError("Compression result disagrees with the view quota")
            decision = dict(decision_rows[0])
            if (len(decision["dominant_indices"]) != dominant
                    or len(decision["contextual_indices"]) != contextual):
                raise ValueError("Compression decisions disagree with the view quota")
            decision.update(view_index=view, keep_count=keep, dominant_count=dominant, contextual_count=contextual)
            compressed_views.append(compressed[0])
            decisions.append(decision)
        combined = torch.cat(compressed_views, dim=0)
        projected = projector(combined)
        if (not isinstance(projected, torch.Tensor) or projected.ndim != 2
                or projected.shape[0] != sum(keeps) or projected.shape[1] < 1
                or not bool(torch.isfinite(projected).all())):
            raise ValueError("Projector must return finite [sum(keep_counts),channels]")
        packed = pack_anchor_rows(list(torch.split(projected, keeps, dim=0)), decisions, grid,
                                  image_newline=image_newline, merge_type=merge_type)
        newlines = (height * side + int(merge_type == "spatial_unpad_add_newl")
                    if image_newline is not None else 0)
        if packed.shape[0] != sum(keeps) + newlines:
            raise RuntimeError("Sparse packing changed the allocated content or structural budget")
        projector_shape, projector_stride = list(combined.shape), list(combined.stride())
    decisions[0].update(packing_newline_count=newlines, full_grid_identity=full_grid,
                        projector_input_shape=projector_shape, projector_input_stride=projector_stride)
    return packed, decisions



# ---------- Embedded globalcom_allocation.py ----------
GLOBALCOM_SOURCE_COMMIT = "de62f6674c8667d2796a2eb38ed5b50319e9f273"


GLOBALCOM_SOURCE_URL = (
    "https://github.com/xuyang-liu16/GlobalCom2/blob/"
    + GLOBALCOM_SOURCE_COMMIT + "/llava/model/llava_arch.py"
)


def _globalcom_integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bounded_largest_remainder(quotas, target, capacity):
    """Hamilton rounding with deterministic, capacity-bounded redistribution.

    Begin with each clipped quota's floor. Award the remaining units to the
    largest quota-minus-current-allocation; ties use the crop index. The first
    pass is ordinary largest-remainder rounding. If source ratio clipping has
    removed budget, subsequent awards redistribute that budget without letting
    any crop exceed capacity. Rate zero therefore correctly returns all tokens.
    """
    values = np.asarray(quotas, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Integer quotas must be a finite vector")
    capacity = _globalcom_integer(capacity, "capacity", minimum=1)
    target = _globalcom_integer(target, "target")
    if not 0 <= target <= len(values) * capacity:
        raise ValueError("Target is outside the available crop capacity")
    values = np.clip(values, 0.0, float(capacity))
    floors = np.floor(values).astype(np.int64).tolist()
    counts = floors.copy()
    deficit = target - sum(counts)
    if deficit > 0:
        queue = [(float(counts[i] - values[i]), i)
                 for i in range(len(counts)) if counts[i] < capacity]
        heapq.heapify(queue)
        for _ in range(deficit):
            if not queue:
                raise ValueError("Unable to satisfy the bounded integer budget")
            _, i = heapq.heappop(queue)
            counts[i] += 1
            if counts[i] < capacity:
                heapq.heappush(queue, (float(counts[i] - values[i]), i))
    elif deficit < 0:
        # Normally unnecessary for the source formula, but handles numerical
        # edge cases and makes the bounded apportionment complete.
        queue = [(float(values[i] - counts[i]), i)
                 for i in range(len(counts)) if counts[i] > 0]
        heapq.heapify(queue)
        for _ in range(-deficit):
            if not queue:
                raise ValueError("Unable to reduce the bounded integer budget")
            _, i = heapq.heappop(queue)
            counts[i] -= 1
            if counts[i] > 0:
                heapq.heappush(queue, (float(values[i] - counts[i]), i))
    return counts, floors


def allocate_globalcom_reference_budgets(base_scores, grid, tokens_per_crop=729,
                          prune_rate=70, temperature=10):
    """Return a JSON-safe audit and per-crop VisionZip output-token quotas.

    base_scores is one flattened square Global Base score map. Optional tensor
    inputs are detached to CPU solely for allocation/audit; no Torch dependency
    is required. Base thumbnail compression is the caller's separate budget.
    The high-resolution total is ceil(M*N*(1-prune_rate/100)), calculated with
    Decimal so percentages cannot create a spurious extra integer token.
    """
    if hasattr(base_scores, "detach"):
        base_scores = base_scores.detach().cpu().tolist()
    raw_scores = np.asarray(base_scores)
    if raw_scores.dtype.kind not in "fiu" or raw_scores.ndim != 1:
        raise ValueError("base_scores must be a one-dimensional real numeric vector")
    scores = raw_scores.astype(np.float64, copy=False)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("base_scores must be nonempty and finite")
    side = math.isqrt(len(scores))
    if side * side != len(scores):
        raise ValueError("base_scores must contain a square token grid")
    if not isinstance(grid, (tuple, list)) or len(grid) != 2:
        raise ValueError("grid must be (width, height)")
    width = _globalcom_integer(grid[0], "grid width", minimum=1)
    height = _globalcom_integer(grid[1], "grid height", minimum=1)
    capacity = _globalcom_integer(tokens_per_crop, "tokens_per_crop", minimum=1)
    if (isinstance(prune_rate, bool) or not isinstance(prune_rate, Real)
            or not math.isfinite(float(prune_rate)) or not 0 <= prune_rate < 100):
        raise ValueError("prune_rate must be a finite percentage in [0, 100)")
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature)) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    retention_decimal = (Decimal(100) - Decimal(str(prune_rate))) / Decimal(100)
    retention = float(retention_decimal)
    count = width * height
    patch_width, patch_height = max(side // width, 1), max(side // height, 1)
    crop_indices = [[] for _ in range(count)]
    score_sums = np.zeros(count)
    # Preserve the source's row-major accumulation order and float64 sums.
    for index, score in enumerate(scores):
        row, col = divmod(index, side)
        crop_row = min(row // patch_height, height - 1)
        crop_col = min(col // patch_width, width - 1)
        crop = crop_row * width + crop_col
        crop_indices[crop].append(index)
        score_sums[crop] += score.item()
    if not np.isfinite(score_sums).all():
        raise ValueError("Crop score accumulation overflowed")
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = (score_sums - np.max(score_sums)) / float(temperature)
    if not np.isfinite(shifted).all():
        raise ValueError("Shifted crop scores overflowed")
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / (np.sum(exp_scores) + 1e-8)
    if np.sum(probabilities) == 0:
        raw_ratios = np.full(count, retention)
    else:
        raw_ratios = retention * (1 + probabilities - np.mean(probabilities))
    continuous_ratios = np.clip(raw_ratios, None, 1.0)
    quotas = capacity * continuous_ratios
    total_before = count * capacity
    target = int((Decimal(total_before) * retention_decimal).to_integral_value(
        rounding=ROUND_CEILING))
    kept, floors = _bounded_largest_remainder(quotas, target, capacity)
    crops = [
        {
            "crop_index": i,
            "base_token_indices": crop_indices[i],
            "score": float(score_sums[i]),
            "globalcom_softmax_weight": float(probabilities[i]),
            "globalcom_raw_keep_ratio": float(raw_ratios[i]),
            "continuous_keep_ratio": float(continuous_ratios[i]),
            "continuous_token_quota": float(quotas[i]),
            "source_floor_keep": int(floors[i]),
            "integer_budget_correction": int(kept[i] - floors[i]),
            "keep": int(kept[i]),
            "prune": int(capacity - kept[i]),
            "tokens_before": capacity,
            "actual_prune_rate_percent": 100 * (1 - kept[i] / capacity),
        }
        for i in range(count)
    ]
    return {
        "source_commit": GLOBALCOM_SOURCE_COMMIT,
        "source_url": GLOBALCOM_SOURCE_URL,
        "allocation_function": "generate_scale_for_crop_features",
        "mapping": "source floor-grid bins; remainder assigned to final row/column; no interpolation or padding correction",
        "normalization": "none inside allocator",
        "apportionment": "bounded largest remainder; quota-minus-current priority; ties by crop index",
        "grid": [width, height],
        "base_token_count": len(scores),
        "base_grid_side": side,
        "source_bin_width": patch_width,
        "source_bin_height": patch_height,
        "temperature": float(temperature),
        "prune_rate_percent": float(prune_rate),
        "base_retention_ratio": retention,
        "tokens_per_crop": capacity,
        "crop_score_sums": score_sums.tolist(),
        "globalcom_shifted_scores": shifted.tolist(),
        "globalcom_softmax_weights": probabilities.tolist(),
        "globalcom_raw_keep_ratios": raw_ratios.tolist(),
        "globalcom_continuous_keep_ratios": continuous_ratios.tolist(),
        "continuous_token_quotas": quotas.tolist(),
        "source_floor_keeps": floors,
        "target_anyres_keep": target,
        "actual_anyres_keep": sum(kept),
        "total_before": total_before,
        "actual_anyres_prune_rate_percent": 100 * (1 - sum(kept) / total_before),
        "crops": crops,
    }

def _temperature_softmax(score_sums, temperature):
    """Stable weights; underflow to zero is safe and not an invalid probability."""
    values = np.asarray(score_sums, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        shifted = (values - np.max(values)) / float(temperature)
        weights = np.exp(shifted)
    # The maximum entry always has exp(0)=1, even when other entries underflow.
    if np.isnan(shifted).any() or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Unable to compute finite crop softmax weights")
    return weights / weights.sum(), shifted


def _bounded_temperature_extras(score_sums, temperature, total, capacity):
    """Distribute extra slots proportionally, cap, then reweight remaining crops.

    Each pass fixes at least one full crop or finishes all active crops. Scores
    are recentered among active crops every pass, so very small T still assigns
    leftover budget after the highest-scoring crops reach their upper bounds.
    """
    count = len(score_sums)
    quotas = np.zeros(count, dtype=np.float64)
    active = np.arange(count, dtype=np.int64)
    remaining = int(total)
    rounds = []
    if not 0 <= remaining <= count*capacity:
        raise ValueError("Extra budget exceeds crop capacity")
    while remaining and len(active):
        probabilities, _ = _temperature_softmax(np.asarray(score_sums)[active], temperature)
        shares = remaining*probabilities
        full = shares >= capacity
        if not full.any():
            quotas[active] = shares
            rounds.append({"remaining_budget":remaining,"active_crops":active.tolist(),"capped_crops":[]})
            remaining = 0
            break
        capped = active[full]
        quotas[capped] = capacity
        rounds.append({"remaining_budget":remaining,"active_crops":active.tolist(),"capped_crops":capped.tolist()})
        remaining -= int(len(capped))*capacity
        active = active[~full]
    if remaining:
        raise RuntimeError("Unable to assign the bounded extra budget")
    return quotas, rounds


def allocate_crop_budgets(base_scores, grid, tokens_per_crop=729,
                          prune_rate=70, temperature=DEFAULT_ALLOCATION_TEMPERATURE, minimum_keep=0, *, audit=True):
    """Temperature-only custom budget mapping with hard bounds and exact total.

    Reserve L slots for every crop, then distribute B-M*L extra slots according
    to crop-SUM softmax. Saturated crops are capped and remaining budget is
    redistributed. In continuous form K_i=L+min(c*exp(S_i/T), N-L), with c
    chosen to satisfy sum(K)=B. Final offset Hamilton rounding preserves both
    bounds and B=ceil(M*N*(1-rate/100)). The model uses L=4; no beta exists.

    This budget mapping extends GlobalCom2's crop score bins, but differs from
    its R*(1+p-mean(p)) formula. The latter remains as a reference function.
    """
    if hasattr(base_scores, "detach"):
        base_scores = base_scores.detach().cpu().tolist()
    raw_scores = np.asarray(base_scores)
    if raw_scores.dtype.kind not in "fiu" or raw_scores.ndim != 1:
        raise ValueError("base_scores must be a one-dimensional real numeric vector")
    scores = raw_scores.astype(np.float64, copy=False)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("base_scores must be nonempty and finite")
    side = math.isqrt(len(scores))
    if side * side != len(scores):
        raise ValueError("base_scores must contain a square token grid")
    if not isinstance(grid, (tuple, list)) or len(grid) != 2:
        raise ValueError("grid must be (width, height)")
    width = _globalcom_integer(grid[0], "grid width", minimum=1)
    height = _globalcom_integer(grid[1], "grid height", minimum=1)
    capacity = _globalcom_integer(tokens_per_crop, "tokens_per_crop", minimum=1)
    if (isinstance(prune_rate, bool) or not isinstance(prune_rate, Real)
            or not math.isfinite(float(prune_rate)) or not 0 <= prune_rate < 100):
        raise ValueError("prune_rate must be a finite percentage in [0, 100)")
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature)) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    minimum_keep = _globalcom_integer(minimum_keep, "minimum_keep")
    if minimum_keep > capacity:
        raise ValueError("minimum_keep cannot exceed tokens_per_crop")
    retention_decimal = (Decimal(100) - Decimal(str(prune_rate))) / Decimal(100)
    retention = float(retention_decimal)
    count = width * height
    patch_width, patch_height = max(side // width, 1), max(side // height, 1)
    crop_indices = [[] for _ in range(count)]
    score_sums = np.zeros(count)
    # Preserve the source's row-major accumulation order and float64 sums.
    for index, score in enumerate(scores):
        row, col = divmod(index, side)
        crop_row = min(row // patch_height, height - 1)
        crop_col = min(col // patch_width, width - 1)
        crop = crop_row * width + crop_col
        crop_indices[crop].append(index)
        score_sums[crop] += score.item()
    if not np.isfinite(score_sums).all():
        raise ValueError("Crop score accumulation overflowed")
    total_before = count*capacity
    target = int((Decimal(total_before)*retention_decimal).to_integral_value(rounding=ROUND_CEILING))
    reserved = count*minimum_keep
    if not reserved <= target <= total_before:
        raise ValueError(f"Total keep budget {target} cannot satisfy {count} crop bounds [{minimum_keep}, {capacity}]")
    probabilities, shifted = _temperature_softmax(score_sums, temperature)
    extra_target = target-reserved
    extra_capacity = capacity-minimum_keep
    initial_quotas = minimum_keep+extra_target*probabilities
    if extra_capacity == 0:
        extras = np.zeros(count,dtype=np.float64)
        extra_kept = extra_floors = [0]*count
        rounds = []
    else:
        extras, rounds = _bounded_temperature_extras(score_sums,temperature,extra_target,extra_capacity)
        extra_kept, extra_floors = _bounded_largest_remainder(extras,extra_target,extra_capacity)
    quotas = minimum_keep+extras
    kept = [minimum_keep+value for value in extra_kept]
    floors = [minimum_keep+value for value in extra_floors]
    if sum(kept)!=target or any(not minimum_keep<=value<=capacity for value in kept):
        raise RuntimeError("Crop allocation violated its hard bounds or aggregate budget")
    if not audit:
        return dict(target_anyres_keep=target, crops=[dict(keep=int(value)) for value in kept],
                    softmax_weights=probabilities, softmax_shifted_scores=shifted,
                    crop_score_sums=score_sums, tokens_per_crop=capacity, minimum_keep=minimum_keep,
                    extra_token_budget=extra_target)
    effective = extras/extra_target if extra_target else np.zeros(count)
    crops = []
    for i in range(count):
        lower, upper = kept[i]==minimum_keep, kept[i]==capacity
        crops.append({
            "crop_index":i,"base_token_indices":crop_indices[i],"score":float(score_sums[i]),
            "softmax_weight":float(probabilities[i]),"effective_extra_weight":float(effective[i]),
            "initial_token_quota":float(initial_quotas[i]),"continuous_token_quota":float(quotas[i]),
            "continuous_keep_ratio":float(quotas[i]/capacity),"floor_keep":int(floors[i]),
            "integer_budget_correction":int(kept[i]-floors[i]),"keep":int(kept[i]),
            "prune":int(capacity-kept[i]),"tokens_before":capacity,
            "minimum_keep":minimum_keep,"maximum_keep":capacity,
            "lower_bound_hit":bool(lower),"upper_bound_hit":bool(upper),
            "actual_prune_rate_percent":100*(1-kept[i]/capacity),
            # Legacy output-key aliases; the allocation_rule explicitly names the custom rule.
            "globalcom_softmax_weight":float(probabilities[i]),
            "globalcom_raw_keep_ratio":float(initial_quotas[i]/capacity),
            "source_floor_keep":int(floors[i]),
        })
    return {
        "source_commit":GLOBALCOM_SOURCE_COMMIT,"source_url":GLOBALCOM_SOURCE_URL,
        "source_scope":"GlobalCom2 floor-grid crop SUM only; custom temperature-only budget mapping",
        "allocation_rule":"bounded_temperature_softmax",
        "allocation_function":"reserve_minimum_then_softmax_extras_with_capacity_redistribution",
        "budget_formula":"K_i=L+min(c*exp(S_i/T),N-L); choose c so sum K_i=B; integer Hamilton rounding",
        "mapping":"source floor-grid bins; remainder assigned to final row/column; no interpolation or padding correction",
        "normalization":"crop SUM softmax; recentered/reweighted among unsaturated crops",
        "apportionment":"minimum reservation + capacity redistribution + offset bounded largest remainder; ties by crop index",
        "grid":[width,height],"base_token_count":len(scores),"base_grid_side":side,
        "source_bin_width":patch_width,"source_bin_height":patch_height,"temperature":float(temperature),
        "prune_rate_percent":float(prune_rate),"base_retention_ratio":retention,"tokens_per_crop":capacity,
        "minimum_keep":minimum_keep,"maximum_keep":capacity,"reserved_minimum_tokens":reserved,
        "extra_token_budget":extra_target,"crop_score_sums":score_sums.tolist(),
        "softmax_weights":probabilities.tolist(),"effective_extra_weights":effective.tolist(),
        "softmax_shifted_scores":[float(x) if np.isfinite(x) else "-inf" for x in shifted],
        "initial_token_quotas":initial_quotas.tolist(),"continuous_token_quotas":quotas.tolist(),
        "floor_keeps":floors,"redistribution_rounds":rounds,
        "lower_bound_crops":[i for i,value in enumerate(kept) if value==minimum_keep],
        "upper_bound_crops":[i for i,value in enumerate(kept) if value==capacity],
        "target_anyres_keep":target,"actual_anyres_keep":sum(kept),"total_before":total_before,
        "actual_anyres_prune_rate_percent":100*(1-sum(kept)/total_before),"crops":crops,
        "globalcom_softmax_weights":probabilities.tolist(),"globalcom_raw_keep_ratios":(initial_quotas/capacity).tolist(),
        "globalcom_continuous_keep_ratios":(quotas/capacity).tolist(),"source_floor_keeps":floors,
    }






def _bounded_mixed_probability_extras(log_probabilities, raw_scores, temperature, total, capacity):
    """Water-fill fixed mixed probabilities, without applying another temperature.

    Logs preserve tiny probabilities after larger crops saturate. Normalizing
    exp(log(p)-max(log(p))) is just p/sum(p) on the active set, not softmax(p).
    If the remaining set has zero representable mixed mass (including lambda=1
    with spatial-zero crops), use original redundancy ranking on that set.
    """
    active = np.arange(len(log_probabilities), dtype=np.int64)
    extras = np.zeros(len(active), dtype=np.float64)
    remaining, rounds = int(total), []
    if not 0 <= remaining <= len(active) * capacity:
        raise ValueError("Extra budget exceeds crop capacity")
    while remaining and len(active):
        local = np.asarray(log_probabilities)[active]
        maximum = float(np.max(local))
        fallback = not math.isfinite(maximum)
        if fallback:
            weights, _ = _temperature_softmax(np.asarray(raw_scores)[active], temperature)
        else:
            weights = np.exp(local - maximum)
            weights /= weights.sum()
        shares = remaining * weights
        full = shares >= capacity
        record = {"remaining_budget": remaining, "active_crops": active.tolist(),
                  "capped_crops": active[full].tolist(),
                  "zero_mass_fallback": "redundancy_softmax_on_remaining" if fallback else None}
        rounds.append(record)
        if not full.any():
            extras[active] = shares
            remaining = 0
            break
        capped = active[full]
        extras[capped] = capacity
        remaining -= int(len(capped)) * capacity
        active = active[~full]
    if remaining:
        raise RuntimeError("Unable to assign the bounded mixed-probability budget")
    return extras, rounds


def allocate_spatial_probability_budgets(raw_redundancy, base_token_indices, grid, spatial_prior,
                                       *, prune_rate=70, temperature=DEFAULT_ALLOCATION_TEMPERATURE,
                                       mix_weight=0.0, minimum_keep=MINIMUM_CROP_KEEP, tokens_per_crop=729, audit=True):
    """p=(1-lambda)*softmax(-R/T)+lambda*P/sum(P), then bounded quotas.

    lambda=0 returns the ORIGINAL allocator result, without reconstructing or
    renormalizing its probabilities/quotas. This is the strict compatibility
    path. The old allocator, score math, and model path remain unchanged.
    """
    mix_weight = validate_spatial_strength(mix_weight)
    raw = np.asarray(raw_redundancy)
    if raw.ndim != 1 or raw.dtype.kind not in "fiu" or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("raw_redundancy must be a finite real crop-score vector")
    raw = raw.astype(np.float64, copy=False)
    # Preserve old disabled transport, including the addition of positive zero.
    scores_before = -raw
    legacy_scores_after = scores_before + np.zeros_like(scores_before)
    base_map = redundancy_to_base_map(-legacy_scores_after, base_token_indices, tokens_per_crop)
    baseline = allocate_crop_budgets(base_map, grid, tokens_per_crop=tokens_per_crop,
                                    prune_rate=prune_rate, temperature=temperature, minimum_keep=minimum_keep, audit=audit)
    if not audit and mix_weight == 0:
        return baseline, {}
    p_r = np.asarray(baseline["softmax_weights"], dtype=np.float64)
    prior = np.asarray(spatial_prior)
    if (prior.ndim != 1 or prior.shape != p_r.shape or prior.dtype.kind not in "fiu"
            or not np.isfinite(prior).all() or np.any(prior < 0)):
        raise ValueError("spatial_prior must be a finite nonnegative vector with one value per crop")
    prior = prior.astype(np.float64, copy=False)
    # Scaling by max first prevents sum overflow; identical to L1 normalization.
    maximum = float(prior.max())
    empty = maximum == 0
    if empty:
        p_s = p_r.copy()
    else:
        scaled = prior / maximum
        p_s = scaled / scaled.sum()
    diagnostics = {
        "mix_weight": mix_weight, "spatial_empty": empty,
        "redundancy_probabilities": p_r.copy(), "spatial_probabilities": p_s,
        "mixed_probabilities": p_r.copy(),
        "spatial_probability_contribution": mix_weight * p_s,
        "effective_mix_weight": 0.0 if empty else mix_weight,
        "mode": "legacy_disabled" if mix_weight == 0 else "empty_prior_legacy_fallback",
    }
    # Do not enter new quota calculation at all for the disabled control.
    if mix_weight == 0 or empty:
        return baseline, diagnostics

    mixed = (1.0 - mix_weight) * p_r + mix_weight * p_s
    mixed /= mixed.sum()
    shifted = np.asarray(baseline["softmax_shifted_scores"], dtype=np.float64)
    log_p_r = shifted - np.log(np.exp(shifted).sum())
    with np.errstate(divide="ignore", invalid="raise"):
        if mix_weight == 1:
            log_mixed = np.log(p_s)
        else:
            log_mixed = np.logaddexp(np.log1p(-mix_weight) + log_p_r,
                                     math.log(mix_weight) + np.log(p_s))
    capacity, lower = baseline["tokens_per_crop"], baseline["minimum_keep"]
    target, extra_target = baseline["target_anyres_keep"], baseline["extra_token_budget"]
    if capacity == lower:
        extras = np.zeros(len(p_r), dtype=np.float64)
        extra_kept = extra_floors = [0] * len(p_r)
        rounds = []
    else:
        extras, rounds = _bounded_mixed_probability_extras(
            log_mixed, baseline["crop_score_sums"], temperature, extra_target, capacity - lower)
        extra_kept, extra_floors = _bounded_largest_remainder(extras, extra_target, capacity - lower)
    keeps = [lower + value for value in extra_kept]
    floors = [lower + value for value in extra_floors]
    quotas = lower + extras
    initial = lower + extra_target * mixed
    effective = extras / extra_target if extra_target else np.zeros(len(p_r))
    if sum(keeps) != target or any(not lower <= value <= capacity for value in keeps):
        raise RuntimeError("Probability fusion violated hard bounds or exact total")
    if not audit:
        return dict(target_anyres_keep=target, crops=[dict(keep=int(value)) for value in keeps]), {}
    result = dict(baseline)
    crops = []
    for i, old in enumerate(baseline["crops"]):
        crop = dict(old)
        crop.update(effective_extra_weight=float(effective[i]), initial_token_quota=float(initial[i]),
                    continuous_token_quota=float(quotas[i]), continuous_keep_ratio=float(quotas[i] / capacity),
                    floor_keep=floors[i], integer_budget_correction=keeps[i] - floors[i], keep=keeps[i],
                    prune=capacity - keeps[i], lower_bound_hit=keeps[i] == lower,
                    upper_bound_hit=keeps[i] == capacity,
                    actual_prune_rate_percent=100 * (1 - keeps[i] / capacity),
                    globalcom_raw_keep_ratio=float(initial[i] / capacity), source_floor_keep=floors[i])
        crops.append(crop)
    result.update(
        allocation_rule="bounded_probability_mixture",
        allocation_function="fixed_convex_mixture_then_conditional_probability_capacity_redistribution",
        budget_formula="p=(1-lambda)*softmax(-R/T)+lambda*P/sum(P); K_i=L+min(c*p_i,N-L); exact Hamilton total",
        normalization="probability L1 mixture; no second softmax or temperature after mixing",
        allocation_probabilities=mixed.tolist(), redundancy_softmax_weights=p_r.tolist(),
        spatial_probabilities=p_s.tolist(), spatial_mix_weight=mix_weight,
        effective_extra_weights=effective.tolist(), initial_token_quotas=initial.tolist(),
        continuous_token_quotas=quotas.tolist(), floor_keeps=floors, redistribution_rounds=rounds,
        lower_bound_crops=[i for i, value in enumerate(keeps) if value == lower],
        upper_bound_crops=[i for i, value in enumerate(keeps) if value == capacity],
        actual_anyres_keep=sum(keeps), actual_anyres_prune_rate_percent=100 * (1 - sum(keeps) / baseline["total_before"]),
        crops=crops, globalcom_raw_keep_ratios=(initial / capacity).tolist(),
        globalcom_continuous_keep_ratios=(quotas / capacity).tolist(), source_floor_keeps=floors,
        zero_mass_policy="original redundancy softmax on remaining crops if mixed support is exhausted")
    diagnostics.update(mixed_probabilities=mixed, mode="probability_convex_mix")
    return result, diagnostics


GLOBALCOM2_COMMIT = GLOBALCOM_SOURCE_COMMIT

"""CLI and complete inference runner for the pooled-redundancy ablation.

This fragment is assembled into the standalone pruning script after its shared
helpers, MethodSettings, adapter and backend. It does not run on import.
"""





"""CLI and complete inference runner for pooled redundancy with a spatial prior.

This fragment is assembled into the standalone pruning script after its shared
helpers, MethodSettings, adapter and backend. It does not run on import.
"""



def settings_from_args(args):
    return MethodSettings(
        prune_rate=args.prune_rate,
        allocation_temperature=args.allocation_temperature,
        redundancy_layers=tuple(args.redundancy_layers),
        redundancy_reduction=args.redundancy_reduction,
        visionzip_layers=tuple(args.visionzip_layers),
        spatial_strength=args.spatial_strength,
    ).validate()


def source_metadata(settings):
    return {
        "visionzip_source_url": SOURCE_URL,
        "visionzip_source_commit": SOURCE_COMMIT,
        "visionzip_non_cls_rule": "https://arxiv.org/html/2412.04467v1#A2 Algorithm3",
        "globalcom2_source_url": GLOBALCOM_SOURCE_URL,
        "globalcom2_source_commit": GLOBALCOM_SOURCE_COMMIT,
        "globalcom2_scope": "Source floor-grid bins and original disabled allocator retained; enabled uses bounded probability mixture; actual pruning is VisionZip",
        "allocation_pipeline": "Unchanged normalized pooled redundancy -> p_R=softmax(-R/T); p_S=P/sum(P); p=(1-lambda)*p_R+lambda*p_S -> bounded probability quotas, no second softmax; lambda0 directly uses original allocator and rounding",
        "redundancy_layers": list(settings.redundancy_layers),
        "redundancy_reduction": settings.redundancy_reduction,
        "redundancy_formula": (
            "R_c=mean_i cosine(unit(mean_j unit(f_base_j)),unit(f_base_i)) + mean_i cosine(unit(mean_j unit(f_crop_j)),unit(f_crop_i))"
            if settings.redundancy_reduction == "mean" else
            "R_c=sum_i cosine(unit(mean_j unit(f_base_j)),unit(f_base_i)) + sum_i cosine(unit(mean_j unit(f_crop_j)),unit(f_crop_i))"
        ),
        "redundancy_comparison_scope": "Base pooled vector compares only with its Base-bin tokens; AnyRes pooled vector compares only with its own AnyRes tokens; no cross-view comparisons",
        "redundancy_normalization": "L2-normalize every original token before pooling; L2-normalize pooled vector before cosine; FP32 on original device; no score Min-Max or post-reduction rescaling",
        "redundancy_zero_policy": "Reject non-finite features/norms and zero-norm token or pooled vectors with ValueError; no epsilon substitute for an undefined direction",
        "redundancy_cosine_policy": "Clamp only numerical overshoot to [-1,1]; do not shift or zero negative individual similarities",
        "redundancy_layer_aggregation": "Compute complete scores independently at each selected 1-based vision block; arithmetic mean of final per-crop scores, not concatenated or averaged feature vectors",
        "redundancy_mean_range": "Each branch mean equals the norm of its mean unit vector, mathematically in [0,1]; combined mean redundancy is in [0,2]",
        "redundancy_temperature_note": "Default T=0.05 for mean redundancy in [0,2]; 0.02 concentrates budgets more, 0.1 is gentler. Actual separation depends on observed crop score differences; no automatic temperature adjustment is applied.",
        "allocation_score": "Original -R is unchanged; spatial evidence mixes only after its softmax",
        "allocation_score_transport": "Legacy disabled path preserves -R + zero and exact float64 Base-bin SUM transport; no spatial term is added to scores",
        "spatial_strength": settings.spatial_strength,
        "spatial_mix_weight": settings.spatial_strength,
        "spatial_strength_semantics": "lambda in [0,1], an alias of spatial-mix-weight; not the historical additive score delta",
        "spatial_fusion_rule": "p=(1-lambda)*softmax(-R/T)+lambda*P/sum(P); no second softmax",
        "spatial_disabled_policy": "lambda=0 returns original allocate_crop_budgets result, preserving all quota/rounding operations",
        "spatial_empty_policy": "all-zero prior returns original redundancy allocation",
        "zero_mixed_mass_policy": "if remaining mixed support is exhausted, use original redundancy softmax on remaining crops",
        "spatial_prior_source": "Custom center spatial prior inspired by ATP-LLaVA spatial protection; not an ATP-LLaVA or Spatial Augmented Pruning reproduction",
        "spatial_prior_formula": "Existing P_c=mean(field in Base bin)/max(crop means), then p_S=P/sum(P); current floor-bin geometry unchanged in this fusion-only ablation",
        "spatial_kernel": {
            "name": "global_separable_flat_top_raised_cosine",
            "plateau_half_width": SPATIAL_PLATEAU_HALF_WIDTH,
            "support_half_width": SPATIAL_SUPPORT_HALF_WIDTH,
            "coordinates": "Base token centers in [-1,1] on both axes",
            "crop_pooling": "mean over unchanged allocator Base bins",
            "prior_calibration": "divide crop means by their maximum; zero if no sampled support",
            "combined_score_normalization": "none",
            "scope": "crop budget only; original VisionZip token selection and merge unchanged",
        },
        "spatial_prior_scope": "Only AnyRes crop budget allocation; original VisionZip token attention selection, Key/features, merging, Base budget, anchor spatial order and continuous position IDs remain unchanged",
        "visionzip_layers": list(settings.visionzip_layers),
        "visionzip_score_aggregation": "Actual selection: equal arithmetic mean of native per-layer received attention; native singleton, float32 multi-layer",
        "merge_key_layer": 26,
        "feature_layer": 26,
        "allocation_rule": "bounded_temperature_softmax" if settings.spatial_strength == 0 else "bounded_probability_mixture",
        "allocation_formula": "Unchanged B=ceil(M*729*(1-rate/100)),L=4; disabled: legacy allocator; enabled: K_i=L+min(c*p_i,729-L), p is the fixed convex mixture; exact Hamilton integer total",
        "allocation_bounds": {"minimum_keep": MINIMUM_CROP_KEEP, "maximum_keep": 729},
        "allocation_temperature": settings.allocation_temperature,
        "allocation_bins": "Official floor(27/grid) Base bins; leftover tokens go to the last row/column. These can differ from exact AnyRes pixel boundaries.",
        "integer_extension": "Reserve each crop minimum; legacy softmax extras when disabled, fixed mixed-probability extras when enabled; capacity redistribution and bounded Hamilton preserve exact total",
        "prune_rate_percent": settings.prune_rate,
        "base_budget": "Separate ceil(729*(1-prune_rate/100)), unaffected by crop redundancy",
        "pruning": "Each view's own selected-layer mean received attention; raw block26 K head mean then L2; contextual anchor + mean assigned OTHER features",
        "dominant_contextual_split": "D=round(K*54/64), C=K-D when K<N; full K=N uses D=N,C=0 without contextual approximation",
        "packing": "Full grid: native unpad/max9/row newlines/Base first. Nonzero: Base D+C by anchors + all AnyRes D+C by global anchors with every raw-row newline; no sparse unpad/max9 resize",
        "packing_reference_version": "runVisionZip.py v3.5-padding-grid-audit",
        "packing_reference_sha256": BASELINE_SHA256,
        "zero_policy": "Run pooled scoring, spatial scoring and allocation, then select all; keep original feature tensor/stride and original 3D projector dispatch; native full-grid packing; no baseline bypass",
        "newline_policy": "Checkpoint image_newline after each raw global AnyRes anchor row including empty rows; all D+C participate; optional final marker for spatial_unpad_add_newl; Base has no row markers",
        "contextual_order": "original_anchor_global_row_major",
        "position_id_policy": "consecutive compressed-sequence IDs; original patch RoPE IDs are not restored",
        "visualization": {"visibility_over_white": VISIBILITY, "merged_sources": "contribute information to contextual tokens", "allocation_heatmaps": False},
    }


TRACE_FIELDS = ("pixels", "input_ids", "selected_features", "projected_features",
                "llm_inputs_embeds", "position_ids", "attention_mask", "generated_token_ids")

GEOMETRY_FIELDS = ("original_width", "original_height", "anyres_grid_width", "anyres_grid_height",
    "anyres_tiles", "canvas_width", "canvas_height", "padding_left", "padding_top", "padding_right",
    "padding_bottom", "padding_required", "padding_call_observed", "padding_output_size_matches",
    "padding_missing_or_mismatch", "baseline_unpad_tokens", "unpad_executed", "unpad_required_but_skipped")

def generate_sample(backend, method, sample, prompt, rate, seed, *, measure=False, capture=False, trace=None):
    """Use the main-method prefill; diagnostics never count as a benchmark."""
    with observe_anyres_padding() as calls, visionzip_generation(backend.model, method.adapter):
        with capture_generation_trace(backend.model, trace) if trace is not None else nullcontext():
            result = backend.generate(sample, prompt, method, rate, "anyres_max_9",
                capture_visualization=bool(capture), capture_attention=False,
                random_seed=seed, do_sample=False, include_pruning_time=True)
    adapter = method.adapter
    if adapter.geometry is not None and adapter.original_size is not None:
        size = int(adapter.tower.image_size)
        metadata = {"original_size": list(adapter.original_size), "grid_patches": list(adapter.geometry.grid),
                    "processed_view_sizes": [[size,size]]}
        result["stats"]["geometry_audit"] = audit_anyres_geometry(metadata, adapter.tower.num_patches_per_side,
            result["stats"].get("full_grid_identity", False), calls)
    return result

@contextmanager
def observe_anyres_padding():
    """Observe the existing PIL padding call; do not replace its computation.

    This standalone runner is single-threaded. Restore the module function even
    on failure. No image copies, pixel scans, GPU work or extra encoder passes.
    """
    from llava import mm_utils
    original = mm_utils.resize_and_pad_image
    calls = []

    def observe(image, target_resolution):
        result = original(image, target_resolution)
        calls.append((tuple(image.size), tuple(target_resolution), tuple(result.size)))
        return result

    mm_utils.resize_and_pad_image = observe
    try:
        yield calls
    finally:
        mm_utils.resize_and_pad_image = original


def audit_anyres_geometry(metadata, token_side, full_grid, padding_calls):
    """Separate image padding from feature unpadding; count no GPU tensors.

    padding_* is an operation/size audit of the checked-in preprocessing, not a
    pixel-content test. baseline_unpad_tokens counts RAW grid positions removed
    by the baseline unpad operation, not padding tokens surviving sparse merging.
    """
    original = tuple(metadata["original_size"])
    grid = tuple(metadata["grid_patches"])
    size = metadata["processed_view_sizes"][0][0]
    geometry = ScoreGeometry.create(original, grid, size, 1)
    target = (grid[0] * size, grid[1] * size)
    left, top = geometry.paste_xy
    right, bottom = (target[0] - left - geometry.resized_size[0],
                     target[1] - top - geometry.resized_size[1])
    required = any((left, top, right, bottom))
    matching = [output for source, canvas, output in padding_calls
                if source == original and canvas == target]
    output_matches = (all(output == target for output in matching) if matching else None)
    # Match vendor unpad_image's int/floor behavior, including odd remainders.
    rows, cols = grid[1] * token_side, grid[0] * token_side
    if original[0] / original[1] > cols / rows:
        removed = 2 * ((rows - int(original[1] * (cols / original[0]))) // 2) * cols
    else:
        removed = 2 * ((cols - int(original[0] * (rows / original[1]))) // 2) * rows
    return dict(
        original_width=original[0], original_height=original[1],
        anyres_grid_width=grid[0], anyres_grid_height=grid[1], anyres_tiles=grid[0] * grid[1],
        canvas_width=target[0], canvas_height=target[1],
        padding_left=left, padding_top=top, padding_right=right, padding_bottom=bottom,
        padding_required=required, padding_call_observed=bool(matching),
        padding_output_size_matches=output_matches,
        padding_missing_or_mismatch=bool(required and not output_matches),
        baseline_unpad_tokens=removed, unpad_executed=bool(full_grid),
        unpad_required_but_skipped=bool(removed > 0 and not full_grid),
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

    def layout(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [layout(item) for item in value]
        if not torch.is_tensor(value):
            raise TypeError("Verification expected a tensor/None or list of tensors")
        return {"stride": list(value.stride()), "is_contiguous": value.is_contiguous(),
                "storage_offset": value.storage_offset(), "device": str(value.device)}

    def record(name, value):
        # Snapshot copies can change strides/offsets. Capture the live tensor's
        # layout first; these small metadata records are diagnostic only.
        trace.setdefault("tensor_layouts", {})[name] = layout(value)
        trace[name] = snapshot(value)

    original_prepare = model.prepare_inputs_labels_for_multimodal
    original_generate = model.generate
    names = ("prepare_inputs_labels_for_multimodal", "generate")
    saved = {name: (name in model.__dict__, model.__dict__.get(name)) for name in names}
    handles = []

    def observe_projector(module, args, output):
        if "selected_features" in trace:
            raise RuntimeError("Verification expects exactly one projector call per image")
        record("selected_features", args[0])
        record("projected_features", output)

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        if result[4] is not None:
            if "llm_inputs_embeds" in trace:
                raise RuntimeError("Verification expects exactly one image prefill")
            record("input_ids", args[0])
            record("pixels", args[5] if len(args) > 5 else kwargs["images"])
            record("position_ids", result[1])
            record("attention_mask", result[2])
            record("llm_inputs_embeds", result[4])
        return result

    def generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        record("generated_token_ids", result)
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

    stages = {name: (compare(reference[name], candidate[name])
                    if name in reference and name in candidate else {"equal": False, "reason": "missing capture"})
              for name in TRACE_FIELDS}
    reference_layouts, candidate_layouts = reference.get("tensor_layouts", {}), candidate.get("tensor_layouts", {})
    for name, stage in stages.items():
        if name in reference_layouts and name in candidate_layouts:
            stage.update(reference_layout=reference_layouts[name], candidate_layout=candidate_layouts[name],
                         layout_equal=reference_layouts[name] == candidate_layouts[name])
    return stages


class ReferenceOnlyMethod:
    """Only the --verify-zero comparator uses this; never a replacement candidate."""
    def configure(self, core, rate, **kwargs):
        core.configure_fastv(enabled=False)

    def stats(self, core, rate):
        return {"mode": "original_reference_for_verification_only"}

    def allocation_data(self, core):
        return {}


def verify_zero_equivalence(backend, method, samples, prompts, seed, output):
    """Run both paths for every selected sample; stop/report on first mismatch."""
    summary = {
        "script_version": SCRIPT_VERSION,
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
                    raise RuntimeError("Candidate did not execute the main method full-grid packing")
                with capture_generation_trace(backend.model, reference_trace):
                    reference = backend.generate(
                        sample, prompt, ReferenceOnlyMethod(), 0, "anyres_max_9",
                        capture_visualization=False, capture_attention=False,
                        random_seed=seed + index, do_sample=False, include_pruning_time=True,
                        )
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
                    failed = [name for name, stage in stages.items() if not stage["equal"]]
                    if candidate["answer"] != reference["answer"]:
                        failed.append("answer")
                    raise RuntimeError(f"Zero-rate mismatch at {sample.sample_id}; "
                                       f"mismatched stages: {', '.join(failed)}; "
                                       f"see {output}/zero_equivalence.jsonl")
                print(f"[{index+1}/{len(samples)}] id={sample.sample_id}: EXACT MATCH (features, inputs, token IDs, answer)", flush=True)
        summary.update(status="passed", complete=summary["checked_samples"] == len(samples))
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write_json(report_path, summary)
    return summary



def run_experiment(args, settings=None, *, backend_factory=LlavaBackend):
    """Run complete classification, metrics and exact per-crop budget logging."""
    settings = (settings or settings_from_args(args)).validate()
    verify_zero = bool(getattr(args, "verify_zero", False))
    if verify_zero and (settings.prune_rate != 0 or getattr(args, "self_test", False)):
        raise ValueError("--verify-zero requires --prune-rate 0 and cannot use --self-test")
    save_prune_vis = boolean_argument(getattr(args, "save_prune_vis", False))
    vis_abnormal_only = boolean_argument(getattr(args, "vis_abnormal_only", True))
    if torch is None:
        raise RuntimeError("Use the original server Torch/CUDA/FlashAttention2 environment for inference")
    samples = load_samples(args.input_json, args.data_root)
    if not samples:
        raise ValueError("Input JSON has no samples")
    if not 0 <= args.seed <= 2**32 - len(samples):
        raise ValueError("seed + sample index must remain within NumPy's 32-bit range")
    from llava_pruning.prompts import resolve_prompt
    prompts = [resolve_prompt(args.prompt_version, sample.category, sample.text)[0] for sample in samples]
    default_output = PROJECT / "outputs" / ("my_pruning_pool_spatial_anchor_order_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output = Path(args.output_dir if args.output_dir is not None else default_output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {output}")
    backend = backend_factory(args.model_path, roi_mode="anyres_max_9", image_token_order="base_first")
    method = CustomPruningMethod(backend.model.get_model(), settings)
    rate = settings.prune_rate
    output.mkdir(parents=True, exist_ok=True)
    rate_dir = output / f"prune_{rate:02d}"
    rate_dir.mkdir()
    inference = dict(getattr(backend, "inference_config", {}))
    inference.update(
        attention_source=f"SigLIP scoring layers {list(settings.visionzip_layers)} received-attention mean",
        score_dtype=(inference.get("vision_dtype", "vision feature dtype") if len(settings.visionzip_layers) == 1 else "torch.float32"),
        merge_key_layer=26, feature_layer=26,
        budget_feature_layers=list(settings.redundancy_layers),
        budget_redundancy_reduction=settings.redundancy_reduction,
        budget_feature_dtype="torch.float32",
        budget_feature_normalization="token L2 before pooling; pooled-vector L2 before internal cosine",
        spatial_strength=settings.spatial_strength,
        budget_allocation_score="p=(1-lambda)*softmax(-R/T)+lambda*P/sum(P); disabled uses original allocator",
        encoder_passes_per_image=1,
    )
    _write_json(output / "run.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_version": SCRIPT_VERSION,
        "method": "custom_pool_spatial_anchor_order_temperature_budget_visionzip", "method_details": source_metadata(settings),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "input_json": str(Path(args.input_json).resolve()), "data_root": str(Path(args.data_root).resolve()),
        "samples": len(samples), "prune_rate": rate, "roi_mode": "anyres_max_9", "image_token_order": "base_first",
        "prompt_version": args.prompt_version, "seed": args.seed, "do_sample": False,
        "decoding": "greedy, max_new_tokens<=512, original LLaVA context budget",
        "save_prune_vis": save_prune_vis, "vis_abnormal_only": vis_abnormal_only,
        "verification_only": verify_zero, "sparse_row_newlines": True,
        "rate_definition": "Base separate / aggregate raw AnyRes encoder content quotas; structural newlines protected; final packed reduction reported separately",
        "visualization_policy": "Ground-truth abnormal only (gt=1)" if vis_abnormal_only else "All samples",
        "visualization_gate": "save_prune_vis and (not vis_abnormal_only or sample.gt == 1); capture disabled before inference for excluded samples",
        "redundancy_layers": list(settings.redundancy_layers), "redundancy_reduction": settings.redundancy_reduction,
        "visionzip_layers": list(settings.visionzip_layers), "inference": inference, "metric_rule": METRIC_RULE,
        "spatial_strength": settings.spatial_strength,
        "spatial_mix_weight": settings.spatial_strength,
        "spatial_fusion_rule": "probability_convex_mix",
        "spatial_strength_semantics": "lambda in [0,1], not the historical additive delta",
    })
    if verify_zero:
        verify_zero_equivalence(backend, method, samples, prompts, args.seed, output)
        print(f"Zero-rate exact comparison passed. Report: {output / 'zero_equivalence.json'}", flush=True)
        return output
    classification_fields = (
        "question_id", "image", "gt", "prediction", "prediction_label", "answer", "prune_rate",
        "redundancy_layers", "redundancy_reduction", "visionzip_layers", "allocation_temperature", "spatial_strength",
        "encoder_views", "base_tokens_before", "base_tokens_after", "actual_base_prune_rate_percent",
        "anyres_crops", "anyres_tokens_before", "anyres_tokens_after", "anyres_target_tokens",
        "actual_anyres_prune_rate_percent", "encoder_tokens_before", "encoder_tokens_after",
        "baseline_packed_tokens", "final_llm_visual_tokens", "actual_packed_reduction_percent", "generation_seconds",
        "newline_tokens_after", "packed_content_tokens", "full_grid_identity",
    )
    crop_fields = (
        "question_id", "crop_index", "score", "softmax_weight", "effective_extra_weight", "continuous_keep_ratio",
        "initial_token_quota", "continuous_token_quota", "floor_keep", "integer_budget_correction", "keep", "prune",
        "tokens_before", "minimum_keep", "maximum_keep", "lower_bound_hit", "upper_bound_hit",
        "actual_prune_rate_percent", "dominant", "contextual",
        "redundancy", "base_redundancy", "anyres_redundancy", "base_similarity_sum", "anyres_similarity_sum",
        "base_token_count", "anyres_token_count",
        "allocation_score_before_spatial", "spatial_prior", "spatial_bonus", "allocation_score_after_spatial",
        "redundancy_probability", "spatial_probability", "fused_probability", "spatial_probability_contribution",
    )
    accuracy = Accuracy(expected_samples=len(samples))

    def save_metrics(complete):
        value = accuracy.result(rate, complete=complete)
        value.update(
            redundancy_layers=list(settings.redundancy_layers),
            redundancy_reduction=settings.redundancy_reduction,
            visionzip_layers=list(settings.visionzip_layers),
            allocation_temperature=settings.allocation_temperature,
            spatial_strength=settings.spatial_strength,
            spatial_mix_weight=settings.spatial_strength,
            spatial_fusion_rule="probability_convex_mix",
        )
        _write_json(rate_dir / "metrics.json", {key: item for key, item in value.items() if key != "rule"})
        return value

    def csv_value(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (tuple, list)) else value

    save_metrics(False)
    with             (rate_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
            (rate_dir / "classification_results.csv").open("w", encoding="utf-8-sig", newline="") as classes, \
            (rate_dir / "crop_budgets.csv").open("w", encoding="utf-8-sig", newline="") as crop_stream, \
            (rate_dir / "geometry_audit.csv").open("w", encoding="utf-8-sig", newline="") as geometry_stream:
        class_writer = csv.DictWriter(classes, fieldnames=classification_fields)
        crop_writer = csv.DictWriter(crop_stream, fieldnames=crop_fields)
        geometry_writer = csv.DictWriter(geometry_stream, fieldnames=("question_id", "image", *GEOMETRY_FIELDS))
        geometry_writer.writeheader()
        class_writer.writeheader()
        crop_writer.writeheader()
        for index, (sample, prompt) in enumerate(zip(samples, prompts)):
            visualize_sample = save_prune_vis and (not vis_abnormal_only or sample.gt == 1)
            result = generate_sample(backend, method, sample, prompt, rate, args.seed + index,
                                     capture=visualize_sample)
            stats, allocation = result["stats"], result["allocation"]
            actual = sum(crop["keep"] for crop in allocation["crops"])
            expected = allocation["target_anyres_keep"]
            base_d, base_c = token_budget(729, rate / 100)
            if (stats.get("base_tokens_after") != base_d + base_c or actual != expected
                    or stats.get("anyres_tokens_after") != expected
                    or stats.get("final_llm_visual_tokens") != (stats["baseline_packed_tokens"] if stats.get("full_grid_identity")
                        else base_d + base_c + expected + stats["newline_tokens_after"])):
                raise RuntimeError("Actual Base/AnyRes/LLM counts differ from the allocated exact integer budgets")
            answer = result["answer"]
            match = OPTION.match(answer)
            prediction = (1 if match.group(1).upper() == "A" else 0) if match else None
            record = {
                "question_id": sample.sample_id, "image": str(sample.image), "origin_path": sample.origin_path,
                "gt": sample.gt, "answer": answer, "prediction": prediction,
                "prediction_label": {0: "Normal", 1: "Abnormal"}.get(prediction, "Unparsed"),
                "prune_rate": rate, "method": "custom_pool_spatial_allocation_visionzip",
                "redundancy_layers": list(settings.redundancy_layers), "redundancy_reduction": settings.redundancy_reduction,
                "visionzip_layers": list(settings.visionzip_layers),
                "allocation_temperature": settings.allocation_temperature, "allocation_rule": allocation["allocation_rule"],
                "spatial_strength": settings.spatial_strength,
                "generation_seconds": result["generation_seconds"], "max_new_tokens": result.get("max_new_tokens"),
                "pruning": stats, "crop_keep_counts": [crop["keep"] for crop in allocation["crops"]],
            }
            predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.flush()
            class_writer.writerow({key: csv_value(record.get(key, stats.get(key))) for key in classification_fields})
            classes.flush()
            for crop in allocation["crops"]:
                crop_writer.writerow({key: sample.sample_id if key == "question_id" else crop.get(key) for key in crop_fields})
            crop_stream.flush()
            geometry_writer.writerow({"question_id": sample.sample_id, "image": str(sample.image),
                                      **stats.get("geometry_audit", {})})
            geometry_stream.flush()
            accuracy.add(sample.gt, answer)
            save_metrics(False)
            if visualize_sample:
                saved = save_visionzip_visualizations(
                    result["image"], result["crop_metadata"][0], result["decisions"], stats,
                    int(backend.vision_tower.num_patches_per_side), int(backend.vision_tower.config.patch_size),
                    rate_dir / "visualizations", sample.sample_id, answer, sample.gt,
                )
                folder = Path(saved[0]["directory"])
                _write_json(folder / "allocation.json", {
                    "question_id": sample.sample_id, "answer": answer, "prediction": prediction, "gt": sample.gt,
                    "allocation": allocation, "settings": source_metadata(settings),
                    "visualization": {"pruning_images": True, "allocation_heatmaps": False},
                })
                np.savez_compressed(folder / "redundancy_details.npz", **result["redundancy_details"])
                np.savez_compressed(folder / "visionzip_details.npz", **result["visionzip_details"])
                np.savez_compressed(folder / "spatial_details.npz", **result["spatial_details"])
            print(
                f"[{index+1}/{len(samples)}] id={sample.sample_id} Pred={record['prediction_label']} "
                f"Base=729->{stats['base_tokens_after']} AnyRes={stats['anyres_tokens_before']}->{expected} "
                f"crop_keep={record['crop_keep_counts']}", flush=True,
            )
    metrics = save_metrics(True)
    summary_fields = (
        "prune_rate", "complete", "expected_samples", "evaluated_samples", "labeled_samples",
        "correct", "incorrect", "unparsed", "accuracy", "parsed_samples", "tp", "fp", "tn", "fn",
        "precision", "recall", "tnr", "redundancy_layers", "redundancy_reduction", "visionzip_layers", "allocation_temperature", "spatial_strength",
        "spatial_mix_weight", "spatial_fusion_rule",
    )
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({key: csv_value(metrics[key]) for key in summary_fields})
    value = "N/A" if metrics["accuracy"] is None else f"{metrics['accuracy']:.2%}"
    print(f"Finished {len(samples)} images. ACC={value}, unparsed={metrics['unparsed']}", flush=True)
    print(f"Final classifications: {rate_dir/'classification_results.csv'}", flush=True)
    print(f"Per-crop allocations: {rate_dir/'crop_budgets.csv'}", flush=True)
    if save_prune_vis:
        print(f"Pruning visualizations and score diagnostics ({'GT abnormal only' if vis_abnormal_only else 'all samples'}): {rate_dir/'visualizations'}", flush=True)
    return output


def self_test(output_dir, prune_rate=PRUNE_RATE,
              allocation_temperature=DEFAULT_ALLOCATION_TEMPERATURE,
              redundancy_layers=(26,), redundancy_reduction="mean", visionzip_layers=(26,),
              spatial_strength=DEFAULT_SPATIAL_STRENGTH):
    """Synthetic normalized-pooling, spatial bonus and exact-quota checks."""
    if torch is None:
        raise RuntimeError("Synthetic pooled-feature checks require Torch; no model weights are needed")
    settings = MethodSettings(
        prune_rate=prune_rate, allocation_temperature=allocation_temperature,
        redundancy_layers=tuple(redundancy_layers), redundancy_reduction=redundancy_reduction,
        visionzip_layers=tuple(visionzip_layers),
        spatial_strength=spatial_strength,
    ).validate()
    generator = torch.Generator(device="cpu").manual_seed(8126)
    features = torch.randn(3, 729, 8, generator=generator, dtype=torch.float32)
    original = features.clone()
    detail = pooled_crop_redundancy(features, (2, 1), reduction=settings.redundancy_reduction)
    assert torch.equal(features, original), "Scoring modified the original feature tensor"
    assert detail["base_token_counts"].tolist() == [351, 378]
    for key in ("base_pool_unit", "anyres_pool_unit"):
        norms = torch.linalg.vector_norm(detail[key], dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6)
    if settings.redundancy_reduction == "mean":
        torch.testing.assert_close(detail["base_redundancy"], detail["base_similarity_sum"] / detail["base_token_counts"])
        torch.testing.assert_close(detail["anyres_redundancy"], detail["anyres_similarity_sum"] / 729)
        assert bool(((detail["redundancy"] >= -1e-5) & (detail["redundancy"] <= 2 + 1e-5)).all())
    else:
        torch.testing.assert_close(detail["base_redundancy"], detail["base_similarity_sum"])
        torch.testing.assert_close(detail["anyres_redundancy"], detail["anyres_similarity_sum"])
    spatial = center_spatial_prior(729, detail["base_token_indices"], strength=0.0)
    assert np.asarray(spatial["spatial_field"]).shape == (729,)
    assert np.asarray(spatial["spatial_prior"]).shape == (2,)
    assert np.max(spatial["spatial_prior"]) == 1.0
    np.testing.assert_array_equal(spatial["spatial_bonus"], np.zeros(2, dtype=np.float64))
    zero_spatial = center_spatial_prior(729, detail["base_token_indices"], strength=0.0)
    np.testing.assert_array_equal(zero_spatial["spatial_bonus"], np.zeros(2, dtype=np.float64))
    redundancy = detail["redundancy"].to(dtype=torch.float64).cpu().numpy()
    expected_scores = -redundancy + np.zeros_like(redundancy)
    allocation, probability = allocate_spatial_probability_budgets(
        redundancy, detail["base_token_indices"], (2, 1), spatial["spatial_prior"],
        prune_rate=settings.prune_rate, temperature=settings.allocation_temperature,
        mix_weight=settings.spatial_strength, minimum_keep=MINIMUM_CROP_KEEP)
    p_r = probability["redundancy_probabilities"]
    p_s = probability["spatial_probabilities"]
    expected_probability = (1 - settings.spatial_strength) * p_r + settings.spatial_strength * p_s
    np.testing.assert_allclose(probability["mixed_probabilities"], expected_probability, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(allocation["crop_score_sums"]), expected_scores)
    keeps = [crop["keep"] for crop in allocation["crops"]]
    target = (2 * 729 * (100 - settings.prune_rate) + 99) // 100
    assert sum(keeps) == target
    assert all(MINIMUM_CROP_KEEP <= keep <= 729 for keep in keeps)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = output / "self_test.json"
    if report.exists():
        raise FileExistsError(f"Self-test report already exists: {report}")
    _write_json(report, {
        "synthetic": True, "model_inference": False, "visualization": False, "version": SCRIPT_VERSION,
        "redundancy_layers_recorded_only": list(settings.redundancy_layers),
        "visionzip_layers_recorded_only": list(settings.visionzip_layers),
        "redundancy_reduction": settings.redundancy_reduction,
        "redundancy": detail["redundancy"].cpu().tolist(), "allocation_scores": expected_scores.tolist(),
        "prune_rate": settings.prune_rate, "allocation_temperature": settings.allocation_temperature,
        "spatial_strength": settings.spatial_strength,
        "spatial_prior": spatial["spatial_prior"].tolist(),
        "redundancy_probabilities": p_r.tolist(), "spatial_probabilities": p_s.tolist(),
        "mixed_probabilities": probability["mixed_probabilities"].tolist(),
        "spatial_probability_contribution": probability["spatial_probability_contribution"].tolist(),
        "spatial_fusion_rule": "probability_convex_mix; lambda0 legacy exact",
        "spatial_kernel": spatial["kernel_parameters"],
        "anyres_keep": keeps, "anyres_total": sum(keeps), "target_anyres_keep": target, "passed": True,
    })
    print("Synthetic normalized-pooling, probability mixture and exact-budget checks passed; no model inference or images.")
    print(report)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Normalized pooled-token redundancy plus custom center spatial prior allocation, with VisionZip and contextual anchors restored to spatial order")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--input-json", default=DEFAULT_INPUT)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    parser.add_argument("--prune-rate", type=int, choices=range(100), default=PRUNE_RATE, metavar="0..99",
                        help="Percentage of visual tokens to prune (default 70); Base separately, all AnyRes crops combined")
    parser.add_argument("--redundancy-layers", type=int, nargs="+", default=(26,),
                        help="1-based SigLIP feature blocks for redundancy (default 26); independently score layers, then average crop scores")
    parser.add_argument("--redundancy-reduction", choices=("mean", "sum"), default="mean",
                        help="Within each branch use token mean (default, balanced Base/AnyRes scales) or raw token sum; then add branches")
    parser.add_argument("--visionzip-layers", type=int, nargs="+", default=(26,),
                        help="1-based SigLIP native-attention scoring blocks; multi-layer scores averaged, merge Key/features stay26")
    parser.add_argument("--allocation-temperature", type=allocation_temperature_argument, default=DEFAULT_ALLOCATION_TEMPERATURE,
                        help="Positive T (default %(default)g); 0.02 concentrates budgets more, 0.1 is gentler; no automatic rescaling")
    parser.add_argument("--spatial-strength", "--spatial-mix-weight", dest="spatial_strength",
                        type=spatial_strength_argument, default=DEFAULT_SPATIAL_STRENGTH,
                        help="Probability mixture lambda in [0,1] (default 0); p=(1-lambda)*softmax(-R/T)+lambda*P/sum(P); 0 uses exact original allocator; not the historical additive score delta")
    parser.add_argument("--save-prune-vis", type=boolean_argument, nargs="?", const=True, default=False, metavar="True/False",
                        help="Save pruning images and NPZ/JSON diagnostics (default False); no old attention/residual allocation heatmaps")
    parser.add_argument("--vis-abnormal-only", type=boolean_argument, nargs="?", const=True, default=True, metavar="True/False",
                        help="Save visual diagnostics only for GT abnormal samples (default True); False includes all samples; classification/metrics always include all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New output directory; default includes pool_spatial_anchor_order and timestamp")
    parser.add_argument("--self-test", action="store_true", help="Synthetic feature/cosine/quota QA only; no weights or model inference")
    parser.add_argument("--verify-zero", action="store_true",
                        help="With --prune-rate 0: exact paired comparison to native LLaVA-OV; diagnostic only")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser


def main():
    args = build_parser().parse_args()
    if args.verify_zero and (args.prune_rate != 0 or args.self_test):
        raise ValueError("--verify-zero requires --prune-rate 0 and no --self-test")
    if args.output_dir is None:
        args.output_dir = PROJECT / "outputs" / ("my_pruning_pool_spatial_anchor_order_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    settings = settings_from_args(args)
    if args.self_test:
        self_test(
            args.output_dir, prune_rate=settings.prune_rate, allocation_temperature=settings.allocation_temperature,
            redundancy_layers=settings.redundancy_layers, redundancy_reduction=settings.redundancy_reduction,
            visionzip_layers=settings.visionzip_layers,
            spatial_strength=settings.spatial_strength,
        )
        return
    if not (PROJECT / "vendor" / "llava").is_dir() or not (PROJECT / "llava_pruning" / "prompts.py").is_file():
        raise FileNotFoundError("Place runMyPruningPoolSpatial.py in the ORIGINAL llava_pruning-ex project root")
    print(f"Custom pruning {SCRIPT_VERSION}: {settings.prune_rate}% Base separately / AnyRes aggregate.", flush=True)
    print(f"Redundancy feature layers={settings.redundancy_layers}; branch reduction={settings.redundancy_reduction}; equal layer score mean.", flush=True)
    print("Unit-normalize tokens before mean pooling, then unit-normalize pooled tokens before within-region/crop cosine comparisons.", flush=True)
    print(f"Spatial probability mixture lambda={settings.spatial_strength}; p_R=softmax(-R/T), p_S=P/sum(P), p=(1-lambda)*p_R+lambda*p_S; no second softmax. Lambda0 uses original allocator exactly.", flush=True)
    print(f"Bounded-softmax temperature={settings.allocation_temperature}; crop bounds=4..729; exact aggregate token total.", flush=True)
    # Mean cosine-to-unit-pool equals ||mean(unit tokens)||, so each branch is
    # in [0,1]. Default T=0.05 replaces the old allocation score's T=8.
    if settings.redundancy_reduction == "mean":
        print("Mean redundancy is in [0,2]; default T=0.05, 0.02 concentrates budgets more, 0.1 is gentler. No automatic score/temperature rescaling.", flush=True)
    print(f"Native VisionZip scoring layers={settings.visionzip_layers}; merge Key/features=26.", flush=True)
    print("Packing: 0% native full-grid unpad/max9/newlines; nonzero D+C at original anchors with protected original-row newlines; Base first; continuous position IDs.", flush=True)
    print(f"Classify all images, greedy decoding. Visualization={args.save_prune_vis}; scope={'GT abnormal only (gt=1)' if args.vis_abnormal_only else 'all samples'}.", flush=True)
    run_experiment(args, settings)


# VisionZip and GlobalCom2 both use Apache-2.0; full license embedded for one-file distribution.
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
