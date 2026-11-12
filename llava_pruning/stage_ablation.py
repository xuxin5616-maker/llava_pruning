"""Budget-matched replacements for the two hybrid pruning stages."""

from contextlib import contextmanager

from runMyPruningPoolSpatial import MethodSettings
from .pool_distill import PoolDistillAdapter, PoolDistillMethod
from .pool_lean import compress_views, encode


RATE_SHARE = .7
PRUNE_AFTER = 2
TAIL = 8


def compress_visionzip(adapter, features, scores, metric):
    # Match MyPool's Base/AnyRes totals. VisionZip distributes the AnyRes
    # budget uniformly among its raw views instead of computing crop scores.
    each, extra = divmod(adapter.view_plan["anyres_keep"], adapter.geometry.tile_count)
    keeps = [adapter.view_plan["base_keep"],
             *(each + (i < extra) for i in range(adapter.geometry.tile_count))]
    # This shared kernel implements VisionZip attention top-k and Key-based
    # contextual merging. Use identical anchor order and newline handling.
    return compress_views(features, scores, metric, keeps, adapter.geometry.grid,
                          adapter.core.mm_projector, adapter.core.image_newline,
                          adapter.core.config.mm_patch_merge_type,
                          source_provenance=adapter.mode == "base_full", prepared=adapter.prepared)


class StageAblationAdapter(PoolDistillAdapter):
    def encode_lean(self, images):
        if self.pre_stage == "visionzip":
            return encode(self, images, compress_global=compress_visionzip)
        return super().encode_lean(images)


class StageAblationMethod(PoolDistillMethod):
    adapter_class = StageAblationAdapter

    def __init__(self, core, checkpoint, job, mode):
        self.pre_stage, self.post_stage = job["pre_stage"], job["post_stage"]
        self.fastv = self.post_stage == "fastv_score_physical"
        base_guided = mode == "mvtecmode"
        mapped = base_guided and not self.fastv
        super().__init__(core, checkpoint, rate=job["prune_rate"], share=RATE_SHARE,
                         mode="base_full" if mapped else "anyres_self", settings=MethodSettings(prune_rate=0),
                         tail_drop_layers=TAIL, student_layer=PRUNE_AFTER,
                         predictor_input="base_text" if mapped else "full",
                         base_mapping="fast" if mapped else "reference",
                         base_policy="anyres_first" if base_guided else "joint")
        self.adapter.pre_stage = self.pre_stage

    def configure(self, core, rate, **capture):
        if not self.fastv:
            return super().configure(core, rate, **capture)
        if rate != self.rate or any(capture.values()):
            raise ValueError("Physical FastV ablation requires the configured rate and results-only inference")
        self.adapter.prepared_prompt_split = None
        self.adapter.stage2_counts = None
        core.configure_clse(enabled=True, keep_ratio=.5)
        self.adapter.configure(self.adapter.settings.prune_rate, capture_visualization=False)

    @contextmanager
    def generation_context(self, model):
        with super().generation_context(model):
            core = model.get_model()
            if self.fastv:
                core._pool_fastv_adapter = self.adapter
            try:
                yield
            finally:
                if self.fastv:
                    del core._pool_fastv_adapter

    def stats(self, core, rate):
        stats = super().stats(core, rate)
        stats.update(pre_stage=self.pre_stage, post_stage=self.post_stage)
        if self.pre_stage == "visionzip":
            stats["base_policy"] = stats["base_policy"].replace("MyPool", "VisionZip")
            stats["position_policy"] = stats["position_policy"].replace("MyPool", "VisionZip")
        if self.fastv:
            stats.update(predictor_input=None, selection_mode="fastv_backbone_attention", source_policy=None)
        return stats
