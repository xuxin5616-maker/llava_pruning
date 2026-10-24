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
