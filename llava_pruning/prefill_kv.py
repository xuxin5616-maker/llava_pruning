"""Observe the logical K/V tensor size after the first decoder forward."""

from contextlib import contextmanager


KV_BYTES = "post_prefill_kv_bytes"
KV_MIB = "post_prefill_kv_mib_per_sample"


def cache_bytes(cache, layer_count):
    if isinstance(cache, (tuple, list)):
        pairs = list(cache)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) != len(cache.value_cache):
            raise RuntimeError("K/V layer counts differ")
        pairs = list(zip(cache.key_cache, cache.value_cache))
    elif hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    else:
        raise RuntimeError("No supported decoder K/V cache at the prefill boundary")
    if len(pairs) != layer_count:
        raise RuntimeError(f"Expected {layer_count} K/V cache layers after prefill, got {len(pairs)}")
    if any(key is None or value is None or key.ndim != 4 or key.shape != value.shape
           or key.shape[0] != 1 or key.shape[2] == 0 for key, value in pairs):
        raise RuntimeError("Post-prefill measurement requires one sample and nonempty K/V in every layer")
    return sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in pairs)


@contextmanager
def observe_post_prefill_kv(core):
    """Keep a byte count only. Do not clone, retain, or synchronize cache tensors."""
    measured = []

    def first_forward(_module, _inputs, result):
        handle.remove()
        measured.append(cache_bytes(result.past_key_values, len(core.layers)))

    handle = core.register_forward_hook(first_forward)
    try:
        yield measured
        if not measured:
            raise RuntimeError("No decoder prefill observed during generation")
    finally:
        handle.remove()
