"""Read-only SigLIP probes; never run the projector or the language decoder."""
from __future__ import annotations

import numpy as np

from feature_residual_math import stage_residuals

LAYERS = (7, 14, 21, 26)
STAGES = tuple(f"siglip_{layer}" for layer in LAYERS)


def capture_stage_scores(tower, pixels, geometry):
    """Capture 1-based block outputs before post_layernorm without changing them.

    The existing tower's weights, dtype and attention implementation are used
    unchanged. Only embeddings and the encoder execute. Hooks immediately
    reduce one layer on CPU and retain scalar scores, not full GPU features.
    """
    import torch

    if any(module.training for module in tower.modules()):
        raise ValueError("SigLIP tower must be in eval mode")
    vision = tower.vision_tower.vision_model
    if len(vision.encoder.layers) != 26:
        raise ValueError("Expected the existing 26-block SigLIP tower")
    expected = (1 + geometry.tile_count, 3, geometry.tile_size, geometry.tile_size)
    if tuple(pixels.shape) != expected:
        raise ValueError(f"Processed pixels must have shape {expected}")
    captured, shapes, handles = {}, {}, []

    def hook(stage):
        def capture(module, args, output):
            if stage in captured:
                raise ValueError(f"Layer executed more than once: {stage}")
            features = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(features, torch.Tensor):
                raise ValueError("Unexpected SigLIP block output")
            shapes[stage] = list(features.shape)
            # FP16/BF16 -> FP32 is exact. Preserve FP64 if a test tower uses it.
            cpu_dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
            raw = features.detach().to(device="cpu", dtype=cpu_dtype).numpy()
            captured[stage] = stage_residuals(raw, geometry)
            # Return None: never replace the original forward output.
        return capture

    try:
        for layer, stage in zip(LAYERS, STAGES):
            handles.append(vision.encoder.layers[layer - 1].register_forward_hook(hook(stage)))
        with torch.inference_mode():
            hidden = vision.embeddings(pixels)
            vision.encoder(inputs_embeds=hidden, output_attentions=False,
                           output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(STAGES):
        raise RuntimeError("Not all requested SigLIP stages were captured")
    return {key: np.stack([captured[stage][key] for stage in STAGES])
            for key in captured[STAGES[0]]}, shapes
