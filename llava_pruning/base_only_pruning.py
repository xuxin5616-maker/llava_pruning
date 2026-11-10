"""Experiment adapter for matched FastV and distilled pruning of one Base view."""

from contextlib import contextmanager

from .visa_comparison_methods import single_drop_keep


RATE = 70
BOUNDARY = 2
MODES = ("fastv", "distilled")


class BaseOnlyPruning:
    def __init__(self, core, mode, checkpoint):
        if mode not in MODES or len(core.layers) != 28:
            raise ValueError("Expected FastV or distilled pruning on the 28-block model")
        self.core, self.mode, self.checkpoint = core, mode, checkpoint
        self.side = core.get_vision_tower().num_patches_per_side
        self.original = self.side ** 2
        self.target = single_drop_keep(self.original, RATE, len(core.layers), BOUNDARY)

    def configure(self, core, rate, *, capture_attention, capture_visualization):
        # Use the legacy Base-only backend's input path; this adapter owns the budget and capture.
        if (core is not self.core or rate != 0 or capture_attention or capture_visualization
                or core.config.image_aspect_ratio != "base_only"):
            raise ValueError("Expected the dedicated Base-only pruning experiment")
        if self.mode == "fastv":
            core.configure_fastv(enabled=True, layer=BOUNDARY, keep_ratio=(self.target - 0.25) / self.original,
                                 preserve_image_newline=False)
            return

        # Check the checkpoint against its native training layout. The actual
        # preprocessor and inference input use Base-only packing immediately after.
        config = core.config
        config.image_aspect_ratio = "anyres_max_9"
        try:
            core.configure_attention_predictor(
                enabled=True, keep_ratio=self.target / self.original, checkpoint=self.checkpoint,
                capture_visualization=True, selection_mode="joint",
            )
        finally:
            config.image_aspect_ratio = "base_only"

    @contextmanager
    def inference_context(self, model):
        if self.mode == "fastv":
            yield
            return
        core = self.core
        name = "prepare_inputs_labels_for_multimodal"
        native = getattr(model, name)
        had_override = name in model.__dict__
        previous = model.__dict__.get(name)

        def prepare(*args, **kwargs):
            # The native packer records the actual span through its ordinary
            # setter. Its two-grid guard is only relevant to Base+AnyRes.
            enabled = core.clse_enabled
            core.clse_enabled = False
            try:
                result = native(*args, **kwargs)
            finally:
                core.clse_enabled = enabled
            if result[4] is not None:
                spans = core._clse_spans
                if len(spans) != 1 or len(spans[0]) != 1 or spans[0][0][1] - spans[0][0][0] != self.original:
                    raise RuntimeError("Native packing did not produce one complete Base grid")
                core._clse_layout = {
                    "packed_tokens": self.original,
                    "grids": [dict(name="base", rows=self.side, cols=self.side, indices=list(range(self.original)))],
                }
            return result

        setattr(model, name, prepare)
        try:
            yield
        finally:
            if had_override:
                setattr(model, name, previous)
            else:
                delattr(model, name)

    def stats(self, core, rate):
        if self.mode == "fastv":
            masks = core.get_fastv_image_masks()
            if len(masks) != 1 or len(masks[0]) != 1:
                raise RuntimeError("Expected exactly one FastV Base mask")
            keep = masks[0][0]["keep"]
        else:
            stages = core.get_clse_stages()
            if len(stages) != 1 or stages[0]["after_layer"] != BOUNDARY:
                raise RuntimeError("Missing the distilled pruning decision after block 2")
            keep = stages[0]["mask"]["keep"]
        if len(keep) != self.original or sum(keep) != self.target:
            raise RuntimeError("Actual mask differs from the matched Base token budget")
        depth = len(core.layers)
        return dict(keep=keep, image_tokens=self.original, kept_tokens=self.target,
                    actual_average_prune_rate=100 * (depth - BOUNDARY) / depth * (1 - self.target / self.original))
