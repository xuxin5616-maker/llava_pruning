"""Optimizer-update counts for single-image attention distillation (no Torch/GPU)."""

import math


def training_step_plan(samples_per_epoch, epochs, accumulation, warmup_ratio):
    if any(type(value) is not int or value < 1 for value in (samples_per_epoch, epochs, accumulation)):
        raise ValueError("Training samples, epochs and gradient accumulation must be positive integers")
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError("warmup-ratio must be finite and in [0,1)")
    per_epoch = math.ceil(samples_per_epoch / accumulation)
    total = epochs * per_epoch
    # Same rounding as Transformers TrainingArguments.get_warmup_steps.
    warmup = math.ceil(total * warmup_ratio)
    if warmup >= total:
        raise ValueError("Warmup covers all optimizer updates (a one-update run would use zero LR). "
                         "For a tiny smoke test, reduce --gradient-accumulation or use --warmup-ratio 0")
    return dict(type="cosine", warmup="linear", warmup_ratio=warmup_ratio,
                warmup_steps=warmup, steps_per_epoch=per_epoch, total_steps=total,
                micro_batch_size=1, effective_batch_size=accumulation,
                last_batch_size=samples_per_epoch % accumulation or accumulation,
                step_unit="optimizer updates only; epoch tail is an actual-size mean; validation does not step")
