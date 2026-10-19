"""Opt-in, passive first-token timing; no changes to logits or token budget."""

from __future__ import annotations

import math
import time


class FirstTokenObserver:
    """HF calls stopping criteria after appending each selected token, even EOS.

    Always returns False: this observer must never stop generation. It only
    synchronizes the first token. Do not use with beams, batching or speculation.
    """

    def __init__(self, synchronize, clock):
        self.synchronize = synchronize
        self.clock = clock
        self.start = None
        self.latency = None
        self.tokens = 0
        self.last_length = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start is None:
            raise RuntimeError("First-token observer was not started")
        if input_ids.shape[0] != 1:
            raise ValueError("First-token benchmark requires batch size 1")
        length = input_ids.shape[-1]
        if self.last_length is not None and length != self.last_length + 1:
            raise RuntimeError("First-token benchmark requires ordinary autoregressive decoding")
        if self.tokens == 0:
            self.synchronize()
            self.latency = self.clock() - self.start
        self.tokens += 1
        self.last_length = length
        return False


def generate_with_benchmark(model, generation_options):
    """Measure the SAME full-answer call used for accuracy, on logical cuda:0.

    Start: all input tensors are resident and synchronized, immediately before
    model.generate (including LLaVA vision encoding/projector/packing).
    Lat end: first selected token has been appended and GPU work synchronized.
    Mem: maximum PyTorch allocated/reserved during the full generate call,
    including resident weights/inputs, excluding loading/preprocessing peaks.
    Reserved includes allocator cache; this is not whole-device/NVML usage.
    """
    import torch
    from transformers import StoppingCriteriaList

    config = model.generation_config
    if (generation_options.get("do_sample", config.do_sample)
            or generation_options.get("num_beams", config.num_beams) != 1
            or generation_options.get("num_return_sequences", config.num_return_sequences) != 1
            or generation_options.get("assistant_model") is not None
            or getattr(config, "prompt_lookup_num_tokens", None)
            or getattr(config, "dola_layers", None)):
        raise ValueError("Benchmark supports only batch-1 greedy autoregressive generation")
    if generation_options["inputs"].shape[0] != 1:
        raise ValueError("Benchmark requires batch size 1")
    observer = FirstTokenObserver(lambda: torch.cuda.synchronize(0), time.perf_counter)
    criteria = StoppingCriteriaList(list(generation_options.get("stopping_criteria") or []))
    criteria.append(observer)
    options = dict(generation_options, stopping_criteria=criteria)
    core = model.get_model()
    previous_timer = getattr(core, "_pruning_timer", None)
    core._pruning_timer = None  # Include ALL pruning work; never subtract it.
    try:
        with torch.inference_mode():
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            observer.start = time.perf_counter()
            generated = model.generate(**options)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - observer.start
            allocated = torch.cuda.max_memory_allocated(0)
            reserved = torch.cuda.max_memory_reserved(0)
    finally:
        core._pruning_timer = previous_timer
    if observer.latency is None:
        raise RuntimeError("Generation never called first-token observer; Lat is unavailable")
    if (not math.isfinite(elapsed) or not math.isfinite(observer.latency)
            or not 0 <= observer.latency <= elapsed):
        raise RuntimeError("Invalid first-token/full-generation duration")
    return generated, {
        "lat_seconds": observer.latency,
        "generation_seconds": elapsed,
        "generated_tokens": observer.tokens,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }
