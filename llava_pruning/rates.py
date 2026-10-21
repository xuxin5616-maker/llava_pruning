"""Percentage validation and lossless labels shared by sweeps and plots."""

from __future__ import annotations

import math


def is_prune_rate(value) -> bool:
    """Accept finite JSON numbers in [0, 100), excluding booleans."""
    return (type(value) in (int, float) and 0 <= value < 100
            and math.isfinite(value))


def format_prune_rate(rate: int | float) -> str:
    """Keep old integer labels (00/10), without rounding fractional rates."""
    if not is_prune_rate(rate):
        raise ValueError(f"Invalid pruning rate: {rate!r}; expected a finite percentage in [0, 100)")
    if rate == int(rate):
        return f"{int(rate):02d}"
    return str(rate)
