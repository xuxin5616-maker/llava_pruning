"""Whole packed-image layer-average accounting for pre-LLM + block2 pruning.

No Torch dependency. Rates are percentages, share is a fraction. Newlines are
in the SAME native unpruned denominator as Base and AnyRes spatial tokens.
"""

from decimal import Decimal, ROUND_FLOOR
import math


def hybrid_budget(original, rate, share, depth=28, boundary=2):
    if (type(original) is not int or original < 1 or type(depth) is not int
            or type(boundary) is not int or not 0 < boundary < depth):
        raise ValueError("Invalid original packed count / model depth / boundary")
    for value, lower, upper, name in ((rate, 0, 100, "rate"), (share, 0, 1, "lambda")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Invalid {name}")
    if rate == 100:
        raise ValueError("Average pruning must be below 100 percent")
    r, weight = Decimal(str(rate)) / 100, Decimal(str(share))
    n, d, k = Decimal(original), Decimal(depth), Decimal(boundary)
    pre = int((n * (1 - (1 - weight) * r)).to_integral_value(rounding=ROUND_FLOOR))
    final = min(pre, int(((d * n * (1 - r) - k * pre) / (d - k)).to_integral_value(rounding=ROUND_FLOOR)))
    if share == 0:
        final = pre  # Integer rounding alone must not activate the disabled side.
    if not 1 <= final <= pre <= original:
        raise ValueError("Unreachable hybrid packed budget; lower pruning target")
    actual_pre = 100 * (1 - pre / original)
    actual_side = 100 * (depth - boundary) / depth * (pre - final) / original
    return dict(baseline_packed_tokens=original, pre_llm_tokens=pre, final_llm_tokens=final,
                target_average_prune_rate=rate, distill_share=share,
                target_pre_contribution=(1 - share) * rate, target_distill_contribution=share * rate,
                actual_pre_contribution=actual_pre, actual_distill_contribution=actual_side,
                actual_average_prune_rate=actual_pre + actual_side,
                stage2_remaining_prune_rate=100 * (1 - final / pre), depth=depth, pruning_layer=boundary)


def pre_view_budget(plan, crop_count, side=27, final_newline=False, minimum=4):
    """Split the exact packed pre-budget proportionally between Base / AnyRes.

    MyPool's sparse row newlines are reserved BEFORE content apportionment.
    Hamilton rounding, ties Base first; MyPool then distributes the AnyRes sum.
    """
    if type(crop_count) is not int or crop_count < 1 or type(side) is not int or side < 1:
        raise ValueError("Invalid crop geometry")
    # caller supplies actual grid HEIGHT in plan, never assumes a square grid.
    rows = plan["raw_grid_height"] * side
    protected = rows + int(final_newline)
    content = plan["pre_llm_tokens"] - protected
    capacities = (side * side, crop_count * side * side)
    total = sum(capacities)
    if not (crop_count + 1) * minimum <= content <= total:
        raise ValueError("Pre-budget cannot preserve MyPool crop minima and raw row newlines")
    keeps, remainders = zip(*(divmod(content * c, total) for c in capacities))
    keeps = list(keeps)
    for i in sorted(range(2), key=lambda i: (-remainders[i], i))[:content - sum(keeps)]:
        keeps[i] += 1
    if keeps[0] < minimum or keeps[1] < crop_count * minimum:
        raise ValueError("Pre-budget cannot preserve independent Base / AnyRes minima")
    if plan["final_llm_tokens"] < keeps[0] + protected:
        raise ValueError("Distillation budget cannot protect the remaining Base and newlines")
    return dict(base_keep=keeps[0], anyres_keep=keeps[1], protected_newlines=protected)
