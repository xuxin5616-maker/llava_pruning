"""Base-guided AnyRes ranking and equal-rate group budgets, inference only.

Geometry is the actual complete packed Base/AnyRes grids AFTER native unpad and
max9 resizing. No crop count, raw tile grid, or fixed 27x27 shape is assumed.
"""

import math

import torch
from torch.nn import functional as F


def named_spatial_grids(layout):
    if not isinstance(layout, dict) or type(layout.get("packed_tokens")) is not int or layout["packed_tokens"] < 1:
        raise ValueError("Base-guided selection requires actual packed image geometry")
    grids = layout.get("grids", [])
    if len(grids) != 2 or {grid.get("name") for grid in grids} != {"base", "anyres"}:
        raise ValueError("Base-guided selection requires exactly one Base and one whole AnyRes grid")
    seen = set()
    for grid in grids:
        if (any(type(grid.get(key)) is not int or grid[key] < 1 for key in ("rows", "cols"))
                or len(grid.get("indices", [])) != grid["rows"] * grid["cols"]):
            raise ValueError("Base/AnyRes spatial indices must match their actual grid shape")
        for index in grid["indices"]:
            if type(index) is not int or not 0 <= index < layout["packed_tokens"] or index in seen:
                raise ValueError("Base/AnyRes grids must contain distinct valid packed spatial indices")
            seen.add(index)
    return {grid["name"]: grid for grid in grids}


def base_guided_log_scores(base_log_scores, layout):
    """Resize Base PROBABILITIES, not log scores, onto the whole AnyRes grid.

    The student's own AnyRes ranking scores are not an input. Normalize within
    Base to remove the irrelevant global visual mass. Bilinear interpolation
    uses align_corners=False and FP32. Log ranking preserves tiny Base scores;
    a float32-tiny floor only prevents log(0) in interpolated AnyRes cells.
    Return in layout['grids'] order, exactly as the shared engine's candidates.
    A common log normalization makes the visualization sum to one; it cannot
    affect either group's independent ordering.
    """
    grids = named_spatial_grids(layout)
    base, anyres = grids["base"], grids["anyres"]
    if (base_log_scores.ndim != 1 or base_log_scores.numel() != len(base["indices"])
            or not torch.isfinite(base_log_scores).all()):
        raise ValueError("Base-guided mapping needs finite scores for every Base spatial token")
    with torch.autocast(device_type=base_log_scores.device.type, enabled=False):
        base_log = F.log_softmax(base_log_scores.float(), dim=0)
        probability = base_log.exp().reshape(1, 1, base["rows"], base["cols"])
        mapped = F.interpolate(probability, size=(anyres["rows"], anyres["cols"]),
                               mode="bilinear", align_corners=False).flatten()
        anyres_log = mapped.clamp_min(torch.finfo(torch.float32).tiny).log()
        canonical = torch.cat((base_log, anyres_log))
        joined = (canonical if layout["grids"][0]["name"] == "base"
                  else torch.cat((anyres_log, base_log)))
        return joined - torch.logsumexp(canonical, dim=0)


def base_guided_budget(layout, keep_ratio):
    """Keep the old packed target, protect newlines, apportion spatial slots.

    Both groups use the SAME continuous spatial retention ratio. Integer
    largest-remainder apportionment preserves the exact global packed target;
    per-group errors are less than one token. No transfer between groups based
    on attention mass, no fixed per-tile budget and no silent safety clamp.
    """
    grids = named_spatial_grids(layout)
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError("Base-guided keep_ratio must be in (0,1]")
    original = layout["packed_tokens"]
    candidates = {name: len(grids[name]["indices"]) for name in ("base", "anyres")}
    spatial = sum(candidates.values())
    protected = sorted(set(range(original)) - {i for grid in grids.values() for i in grid["indices"]})
    target = min(original, max(1, math.floor(original * keep_ratio + 1e-10)))
    available = target - len(protected)
    if available < 2:
        raise ValueError(f"Base-guided packed budget keeps {target}/{original} tokens but needs "
                         f"{len(protected)} protected newlines plus at least one spatial token in each group; "
                         "lower the pruning target")
    # Integer arithmetic avoids rounding a mathematically integral quota down.
    keeps, remainders = {}, {}
    for name in ("base", "anyres"):
        keeps[name], remainders[name] = divmod(available * candidates[name], spatial)
    leftover = available - sum(keeps.values())
    order = sorted(("base", "anyres"), key=lambda name: (-remainders[name], 0 if name == "base" else 1))
    for name in order[:leftover]:
        keeps[name] += 1
    groups = []
    for name in ("base", "anyres"):
        count, keep = candidates[name], keeps[name]
        if not 1 <= keep <= count:
            raise ValueError(f"Base-guided proportional {name} budget keeps {keep}/{count}; "
                             "at least one spatial token per group is required; lower the pruning target")
        groups.append(dict(name=name, candidate_tokens=count, kept_tokens=keep,
                           masked_tokens=count - keep, actual_prune_rate=100 * (1 - keep / count)))
    return dict(target=target, protected_indices=protected,
                spatial_keep_ratio=available / spatial, groups=groups)


def select_base_guided_tokens(scores, spatial_indices, original, keep_ratio, layout):
    plan = base_guided_budget(layout, keep_ratio)
    if original != layout["packed_tokens"]:
        raise ValueError("Base-guided budget does not match actual packed image length")
    expected = [index for grid in layout["grids"] for index in grid["indices"]]
    if list(spatial_indices) != expected or scores.shape != (len(expected),) or not torch.isfinite(scores).all():
        raise ValueError("Base-guided scores/candidates disagree with actual spatial packing")
    packed_scores = scores.new_zeros(original)
    packed_scores[torch.tensor(expected, device=scores.device)] = scores
    named = named_spatial_grids(layout)
    selected = [torch.tensor(plan["protected_indices"], dtype=torch.long, device=scores.device)]
    for group in plan["groups"]:
        indices = torch.tensor(named[group["name"]]["indices"], device=scores.device)
        # Equal mapped scores are common; break ties by original row-major order.
        order = torch.argsort(packed_scores[indices], descending=True, stable=True)
        selected.append(indices[order[:group["kept_tokens"]]])
    return torch.cat(selected).sort().values, plan


def base_full_count_budget(base_count, anyres_count, protected_count, keep_ratio):
    """Keep the ORIGINAL packed target; all deletions are from AnyRes.

    The public method converts layer-average rates to this boundary keep_ratio.
    Never clamp an infeasible target to Base size or change its denominator.
    Zero retained AnyRes tokens is permitted at the exact feasible boundary.
    """
    if (any(type(n) is not int or n < 0 for n in (base_count, anyres_count, protected_count))
            or min(base_count, anyres_count) < 1
            or isinstance(keep_ratio, bool) or not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1):
        raise ValueError("Invalid Base-full counts or keep ratio")
    original = base_count + anyres_count + protected_count
    target = min(original, max(1, math.floor(original * keep_ratio + 1e-10)))
    anyres_keep = target - base_count - protected_count
    if not 0 <= anyres_keep <= anyres_count:
        maximum = 100 * anyres_count / original
        raise ValueError(
            f"Base-full packed target {target}/{original} cannot keep all {base_count} Base tokens "
            f"and {protected_count} protected newlines; AnyRes keep would be {anyres_keep}/{anyres_count}. "
            f"Maximum boundary pruning is {maximum:.6f}%; lower the requested WHOLE-image layer-average rate.")
    return dict(target=target, anyres_spatial_keep_ratio=anyres_keep / anyres_count,
                common_spatial_keep_ratio=None,
                groups=[dict(name=name, candidate_tokens=count, kept_tokens=keep,
                             masked_tokens=count - keep, actual_prune_rate=100 * (1 - keep / count))
                        for name, count, keep in (("base", base_count, base_count),
                                                 ("anyres", anyres_count, anyres_keep))])


def base_full_budget(layout, keep_ratio):
    named = named_spatial_grids(layout)
    spatial = {i for grid in named.values() for i in grid["indices"]}
    protected = sorted(set(range(layout["packed_tokens"])) - spatial)
    plan = base_full_count_budget(len(named["base"]["indices"]), len(named["anyres"]["indices"]),
                                  len(protected), keep_ratio)
    plan["protected_indices"] = protected
    return plan


def select_base_full_tokens(scores, spatial_indices, original, keep_ratio, layout):
    """Keep Base verbatim; ONE global AnyRes top-k of Base-mapped scores."""
    plan = base_full_budget(layout, keep_ratio)
    expected = [i for grid in layout["grids"] for i in grid["indices"]]
    if (original != layout["packed_tokens"] or list(spatial_indices) != expected
            or scores.shape != (len(expected),) or not torch.isfinite(scores).all()):
        raise ValueError("Base-full mapped scores/candidates disagree with actual packing")
    packed = scores.new_zeros(original)
    packed[torch.tensor(expected, device=scores.device)] = scores
    named = named_spatial_grids(layout)
    base = torch.tensor(named["base"]["indices"], dtype=torch.long, device=scores.device)
    anyres = torch.tensor(named["anyres"]["indices"], dtype=torch.long, device=scores.device)
    keep = plan["groups"][1]["kept_tokens"]
    order = torch.argsort(packed[anyres], descending=True, stable=True)
    protected = torch.tensor(plan["protected_indices"], dtype=torch.long, device=scores.device)
    return torch.cat((base, protected, anyres[order[:keep]])).sort().values, plan


def native_anyres_counts(base_side, raw_grid, original_size, *, final_newline=False, max_patches=9):
    """CPU count-only counterpart of native unpad -> conditional max9 -> newlines."""
    raw_h, raw_w = raw_grid[1] * base_side, raw_grid[0] * base_side
    original_w, original_h = original_size
    if min(base_side, raw_h, raw_w, original_w, original_h) < 1:
        raise ValueError("Invalid native image geometry")
    h, w = raw_h, raw_w
    if original_w / original_h > raw_w / raw_h:
        h -= 2 * ((raw_h - int(original_h * raw_w / original_w)) // 2)
    else:
        w -= 2 * ((raw_w - int(original_w * raw_h / original_h)) // 2)
    times = math.sqrt(h * w / (max_patches * base_side ** 2))
    if times > 1.1:
        h, w = int(h // times), int(w // times)
    return base_side ** 2, h * w, h + int(final_newline)
