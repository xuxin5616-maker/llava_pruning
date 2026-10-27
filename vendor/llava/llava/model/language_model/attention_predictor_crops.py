"""Inference-only Base-to-crop budgets; native decoder/student math is untouched.

Crop IDs refer to ORIGINAL encoder tiles. Packed token centers are mapped through
native unpadding/max9 resizing, never by dividing the remaining sequence evenly.
The provenance schema reserves source IDs/weights for a future pre-LLM adapter;
this change does not enable pre-LLM pruning or infer missing merge histories.
"""

import math

import torch

from .attention_predictor_selection import named_spatial_grids, base_guided_budget


DEFAULT_CROP_TEMPERATURE = 0.05


def validate_crop_options(temperature, minimum_keep):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("crop_temperature must be finite and positive")
    if type(minimum_keep) is not int or minimum_keep < 0:
        raise ValueError("crop_min_keep must be a nonnegative integer")


def dense_crop_provenance(layout, raw_grid, raw_side, unpadded_shape):
    """Describe canonical packed anchors, not exact pixel receptive fields.

    Identity source IDs address the baseline PACKED sequence (after native
    resizing), not raw encoder tokens. A future merger must supply its actual
    source IDs/coefficients explicitly; coefficients need not sum to one.
    """
    named = named_spatial_grids(layout)
    if (len(raw_grid) != 2 or any(type(v) is not int or v < 1 for v in raw_grid)
            or type(raw_side) is not int or raw_side < 1
            or len(unpadded_shape) != 2 or any(type(v) is not int or v < 1 for v in unpadded_shape)):
        raise ValueError("Invalid original crop/unpadding geometry")
    width, height = raw_grid
    raw_h, raw_w = height * raw_side, width * raw_side
    unpad_h, unpad_w = unpadded_shape
    if unpad_h > raw_h or unpad_w > raw_w:
        raise ValueError("Unpadded geometry exceeds original AnyRes grid")
    top, left = (raw_h - unpad_h) // 2, (raw_w - unpad_w) // 2
    crops = [dict(crop_id=i, indices=[], base_indices=[]) for i in range(width * height)]
    tokens = []
    for name in ("base", "anyres"):
        grid = named[name]
        for slot, index in enumerate(grid["indices"]):
            row, col = divmod(slot, grid["cols"])
            x, y = (col + .5) / grid["cols"], (row + .5) / grid["rows"]
            crop_col = min(width - 1, int((left + x * unpad_w) / raw_side))
            crop_row = min(height - 1, int((top + y * unpad_h) / raw_side))
            crop_id = crop_row * width + crop_col
            crops[crop_id]["base_indices" if name == "base" else "indices"].append(index)
            tokens.append(dict(index=index, group=name, crop_id=crop_id, xy=[x, y],
                               source_ids=[index], source_weights=[1.]))
    return dict(version=1, pre_pruning_applied=False,
                baseline_packed_tokens=layout["packed_tokens"],
                source_id_space="baseline_packed_anchor", raw_crop_grid=list(raw_grid),
                raw_token_side=raw_side, unpadded_shape=list(unpadded_shape),
                unpad_offset=[top, left], crops=crops, tokens=tokens)


def validate_crop_provenance(layout, provenance):
    """Validate identities before scores are used; reject invented dense shapes."""
    named = named_spatial_grids(layout)
    if not isinstance(provenance, dict) or provenance.get("version") != 1:
        raise ValueError("Base-crop selection requires versioned crop provenance")
    if provenance.get("pre_pruning_applied") is not False:
        raise NotImplementedError("Pre-LLM pruning is not connected: baseline layer-average accounting and sparse layout need an explicit adapter")
    if provenance.get("baseline_packed_tokens") != layout["packed_tokens"]:
        raise ValueError("Crop baseline token count does not match packed image")
    crops = provenance.get("crops", [])
    if not crops or [c.get("crop_id") for c in crops] != list(range(len(crops))):
        raise ValueError("Crop IDs must be distinct original row-major IDs")
    memberships = {}
    for name, key in (("base", "base_indices"), ("anyres", "indices")):
        actual = [i for crop in crops for i in crop.get(key, [])]
        if (any(type(i) is not int for i in actual) or len(actual) != len(set(actual))
                or set(actual) != set(named[name]["indices"])):
            raise ValueError("Crop membership must partition actual Base/AnyRes candidates exactly")
        memberships.update({(name, crop["crop_id"]): set(crop[key]) for crop in crops})
    if any(crop["indices"] and not crop["base_indices"] for crop in crops):
        raise ValueError("A nonempty AnyRes crop has no Base reference tokens; supply explicit sparse reference provenance")
    records = provenance.get("tokens", [])
    expected = {i for grid in named.values() for i in grid["indices"]}
    if (len(records) != len(expected) or {r.get("index") for r in records} != expected):
        raise ValueError("Missing or duplicate spatial token provenance")
    for record in records:
        ids, weights, xy = record.get("source_ids", []), record.get("source_weights", []), record.get("xy", [])
        if (not ids or len(ids) != len(weights) or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < provenance["baseline_packed_tokens"] for i in ids)
                or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights)
                or len(xy) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in xy)):
            raise ValueError("Invalid token coordinates/source IDs/merge coefficients")
        group, crop_id = record.get("group"), record.get("crop_id")
        if group not in named or type(crop_id) is not int or not 0 <= crop_id < len(crops):
            raise ValueError("Invalid source group/crop ID")
        if record["index"] not in memberships[(group, crop_id)]:
            raise ValueError("Source token and crop membership disagree")
    return crops


def bounded_crop_quotas(means, capacities, total, *, temperature=DEFAULT_CROP_TEMPERATURE, minimum_keep=0):
    """Softmax(positive mean/T), reserve minima, cap/reweight, Hamilton round.

    Unlike VisionZip, LLM top-k does not require four contextual/anchor tokens;
    minimum_keep defaults to zero. Each crop's capacity is its CURRENT count.
    """
    validate_crop_options(temperature, minimum_keep)
    if not torch.is_tensor(means) or means.ndim != 1 or means.numel() != len(capacities):
        raise ValueError("One mean score is required per crop")
    if any(type(n) is not int or n < 0 for n in capacities):
        raise ValueError("Crop capacities must be nonnegative integer survivor counts")
    if type(total) is not int or not 0 <= total <= sum(capacities):
        raise ValueError("AnyRes budget exceeds current crop capacities")
    if not torch.isfinite(means).all() or (means < 0).any():
        raise ValueError("Base attention crop means must be finite and nonnegative")
    lower = [min(minimum_keep, n) for n in capacities]
    if total < sum(lower):
        raise ValueError("AnyRes budget cannot satisfy crop_min_keep; lower the minimum or pruning target")
    # Only tiny crop vectors synchronize; no token score array leaves the GPU.
    score = means.detach().double()
    probability = score.new_zeros(len(capacities))
    active = [i for i, n in enumerate(capacities) if n > 0]
    if active:
        probability[active] = torch.softmax((score[active] - score[active].max()) / temperature, dim=0)
    extra = [n - l for n, l in zip(capacities, lower)]
    quota = [float(n) for n in lower]
    active = [i for i, n in enumerate(extra) if n > 0]
    remaining = total - sum(lower)
    while remaining and active:
        local = score[active]
        weights = torch.softmax((local - local.max()) / temperature, dim=0)
        shares = (weights * remaining).cpu().tolist()
        capped = [i for i, share in zip(active, shares) if share >= extra[i]]
        if not capped:
            for i, share in zip(active, shares):
                quota[i] += share
            break
        for i in capped:
            quota[i] = float(capacities[i])
            remaining -= extra[i]
        active = [i for i in active if i not in capped]
    keep = [min(n, max(l, math.floor(q))) for n, l, q in zip(capacities, lower, quota)]
    leftover = total - sum(keep)
    order = sorted((i for i in range(len(keep)) if keep[i] < capacities[i]),
                   key=lambda i: (-(quota[i] - math.floor(quota[i])), i))
    if not 0 <= leftover <= len(order):
        raise RuntimeError("Crop quota rounding lost the exact integer budget")
    for i in order[:leftover]:
        keep[i] += 1
    if sum(keep) != total:
        raise RuntimeError("Crop allocation changed the AnyRes budget")
    return keep, probability.cpu().tolist()


def select_base_crop_tokens(scores, spatial_indices, original, keep_ratio, layout, *,
                            temperature=DEFAULT_CROP_TEMPERATURE, minimum_keep=0):
    """Base score means set crop quotas; own predicted AnyRes scores set top-k."""
    plan = base_guided_budget(layout, keep_ratio)
    crops = validate_crop_provenance(layout, layout.get("predictor_crops"))
    expected = [i for grid in layout["grids"] for i in grid["indices"]]
    if (original != layout["packed_tokens"] or list(spatial_indices) != expected
            or scores.shape != (len(expected),) or not torch.isfinite(scores).all()):
        raise ValueError("Base-crop scores/candidates disagree with packed provenance")
    packed = scores.new_full((original,), float("-inf"), dtype=torch.float32)
    packed[torch.tensor(expected, device=scores.device)] = scores.float()
    # Use the SAME jointly visual-normalized predicted probabilities as training.
    # No extra Base-only normalization, per-crop minmax, spatial prior, or sign flip.
    log_probability = packed - torch.logsumexp(scores.float(), dim=0)
    means = []
    for crop in crops:
        if crop["indices"] and not crop["base_indices"]:
            raise ValueError("A nonempty AnyRes crop has no Base reference tokens; supply explicit sparse reference provenance")
        ids = torch.tensor(crop["base_indices"], dtype=torch.long, device=scores.device)
        means.append(log_probability[ids].exp().mean() if ids.numel() else scores.new_zeros(()))
    means = torch.stack(means)
    capacities = [len(crop["indices"]) for crop in crops]
    anyres_keep = next(g["kept_tokens"] for g in plan["groups"] if g["name"] == "anyres")
    keeps, probabilities = bounded_crop_quotas(means, capacities, anyres_keep,
                                              temperature=temperature, minimum_keep=minimum_keep)
    named = named_spatial_grids(layout)
    base_keep = next(g["kept_tokens"] for g in plan["groups"] if g["name"] == "base")
    base = torch.tensor(named["base"]["indices"], device=scores.device)
    selected = [torch.tensor(plan["protected_indices"], device=scores.device, dtype=torch.long),
                base[torch.argsort(packed[base], descending=True, stable=True)[:base_keep]]]
    mean_values = means.detach().cpu().tolist()
    rows = []
    for crop, capacity, keep, mean, probability in zip(crops, capacities, keeps, mean_values, probabilities):
        indices = torch.tensor(sorted(crop["indices"]), device=scores.device, dtype=torch.long)
        selected.append(indices[torch.argsort(packed[indices], descending=True, stable=True)[:keep]])
        rows.append(dict(crop_id=crop["crop_id"], candidate_tokens=capacity, kept_tokens=keep,
                         masked_tokens=capacity - keep, actual_prune_rate=100 * (1 - keep / capacity) if capacity else None,
                         base_reference_tokens=len(crop["base_indices"]), base_mean_attention=mean,
                         softmax_weight=probability, crop_temperature=temperature, crop_min_keep=minimum_keep))
    plan["crops"] = rows
    return torch.cat(selected).sort().values, plan
