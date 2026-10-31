"""Compose the supplied MyPool allocator/merger with the existing side predictor.

The original standalone method and native dense predictor paths stay usable.
Only this adapter accepts sparse merged tokens, with explicit source provenance.
"""

from contextlib import contextmanager
import math
import types

import torch
from torch.nn import functional as F

import runMyPruningPoolSpatial as pool
from .pool_distill_budget import hybrid_budget, pre_view_budget


def source_records(decisions, grid, side=27, final_newline=False, *, fast=False):
    """Mirror pack_anchor_rows exactly, recording pre-projector linear sources.

    A contextual feature is anchor + mean(OTHER), not a convex average. Keep
    those coefficients here; normalize only when averaging scalar scores.
    """
    width, height = grid
    if len(decisions) != width * height + 1:
        raise ValueError("Missing view decisions")
    base, anyres = [], []
    for view, decision in enumerate(decisions):
        dominant, contextual = decision["dominant_indices"], decision["contextual_indices"]
        assignments = decision["cluster_assignments"]
        if fast:
            cluster_of = {anchor:cluster for cluster,anchor in enumerate(contextual)}
            members = [[] for _ in contextual]
            for source,cluster in enumerate(assignments):
                if cluster >= 0:
                    if cluster >= len(contextual):
                        raise ValueError("Invalid contextual cluster assignment")
                    members[cluster].append(source)
        for anchor in dominant + contextual:
            sources, weights = [anchor], [1.]
            if anchor in (cluster_of if fast else contextual):
                cluster = cluster_of[anchor] if fast else contextual.index(anchor)
                others = ([i for i in members[cluster] if i != anchor] if fast else
                          [i for i, slot in enumerate(assignments) if slot == cluster and i != anchor])
                if others:
                    sources += others
                    weights += [1. / len(others)] * len(others)
            record = dict(view=view, anchor=anchor, sources=sources, weights=weights)
            if view == 0:
                base.append((anchor, record))
            else:
                tile_y, tile_x = divmod(view - 1, width)
                y, x = divmod(anchor, side)
                position = (tile_y * side + y) * (width * side + 1) + tile_x * side + x
                anyres.append((position, record))
    for y in range(height * side):
        anyres.append((y * (width * side + 1) + width * side, None))
    result = [record for _, record in sorted(base)] + [record for _, record in sorted(anyres)]
    if final_newline:
        result.append(None)
    return result


def weighted_sources(records, *, view_offset=False, device):
    owners, sources, weights = [], [], []
    for j, record in enumerate(records):
        total = sum(record["weights"])
        if total <= 0 or len(record["sources"]) != len(record["weights"]):
            raise ValueError("Invalid merge provenance")
        for source, coefficient in zip(record["sources"], record["weights"]):
            if not 0 <= source < 729 or coefficient <= 0 or not math.isfinite(coefficient):
                raise ValueError("Invalid source / merge coefficient")
            owners.append(j)
            sources.append(source + (record["view"] - 1) * 729 if view_offset else source)
            weights.append(coefficient / total)
    return (torch.tensor(owners, device=device, dtype=torch.long),
            torch.tensor(sources, device=device, dtype=torch.long),
            torch.tensor(weights, device=device, dtype=torch.float32))


def mapped_scores(base_log, base_records, anyres_records, geometry):
    """Mass-preserving Base backprojection, then source-weighted AnyRes averages.

    Base token probabilities are distributed to their original sources using
    normalized merge coefficients (not anchor-only interpolation). Bilinear
    sampling uses actual resize/paste geometry and 14px token centers. Padding
    outside image content scores zero. No min-max or hidden temperature.
    """
    with torch.autocast(device_type=base_log.device.type, enabled=False):
        probability = F.softmax(base_log.float(), dim=0)
        owner, source, weight = weighted_sources(base_records, device=base_log.device)
        dense = probability.new_zeros(729).index_add(0, source, probability[owner] * weight)
        if not torch.isclose(dense.sum(), probability.new_tensor(1.), atol=1e-5):
            raise RuntimeError("Base source backprojection lost probability mass")
        side, patch, tile = geometry.token_side, geometry.patch_size, geometry.tile_size
        if side != 27:
            raise ValueError("Current MyPool adapter requires 27x27 SigLIP tokens")
        local = (torch.arange(side, device=base_log.device, dtype=torch.float32) + .5) * patch
        yy, xx = torch.meshgrid(local, local, indexing="ij")
        maps = []
        for crop in range(geometry.tile_count):
            row, col = divmod(crop, geometry.grid[0])
            x = (xx + col * tile - geometry.paste_xy[0]) / geometry.resized_size[0]
            y = (yy + row * tile - geometry.paste_xy[1]) / geometry.resized_size[1]
            # Base pixel -> SigLIP token lattice (27*14 support inside 384).
            coordinates = torch.stack((2 * x * tile / (side * patch) - 1,
                                       2 * y * tile / (side * patch) - 1), dim=-1)[None]
            sampled = F.grid_sample(dense.reshape(1, 1, side, side), coordinates,
                                    mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
            maps.append(sampled.masked_fill((x < 0) | (x >= 1) | (y < 0) | (y >= 1), 0).flatten())
        owner, source, weight = weighted_sources(anyres_records, view_offset=True, device=base_log.device)
        scores = probability.new_zeros(len(anyres_records)).index_add(0, owner, torch.cat(maps)[source] * weight)
        joined = torch.cat((probability, scores)).clamp_min(torch.finfo(torch.float32).tiny).log()
        return joined - torch.logsumexp(joined, dim=0)


class PoolDistillAdapter(pool.CustomPruningAdapter):
    def __init__(self, core, settings, *, rate, share, mode, lean=False, predictor_input="full"):
        super().__init__(core, settings)
        self.rate, self.share, self.mode = rate, share, mode
        self.plan, self.records = None, None
        self.lean = lean
        self.predictor_input = predictor_input
        self._geometry_cache = {}
        self._prepared_shapes = {}
        self.prepared = None
        self.prepared_prompt_split = None

    def set_image_context(self, original_size):
        key = tuple(map(int,original_size))
        if self.lean and key in self._geometry_cache:
            self.original_size,self.geometry,self.plan,self.view_plan,self.prepared = self._geometry_cache[key]
            return
        super().set_image_context(original_size)
        original = pool.baseline_packed_count(self.original_size, self.geometry.grid, 27,
                                             self.core.config.mm_patch_merge_type)
        self.plan = hybrid_budget(original, self.rate, self.share, len(self.core.layers))
        self.plan["raw_grid_height"] = self.geometry.grid[1]
        self.view_plan = pre_view_budget(self.plan, self.geometry.tile_count,
                                        final_newline=self.core.config.mm_patch_merge_type.endswith("_add_newl"))
        if self.lean:
            from .pool_lean import prepare_geometry
            shape_key=(tuple(self.geometry.grid),self.geometry.token_side)
            if shape_key not in self._prepared_shapes:
                prepared = prepare_geometry(self.geometry,self.core.image_newline.device)
                # No spatial-prior template/normalization when its weight is zero.
                prepared["spatial_prior"] = (pool.center_spatial_prior(729,prepared["base_token_indices"],0.)["spatial_prior"]
                    if self.settings.spatial_strength else None)
                self._prepared_shapes[shape_key]=prepared
            self.prepared=self._prepared_shapes[shape_key]
            self._geometry_cache[key] = (self.original_size,self.geometry,self.plan,self.view_plan,self.prepared)

    def allocate_view_budgets(self, raw_redundancy, first, spatial):
        # Choose an interior point of the ceil interval to avoid FP/Decimal
        # boundary errors. This rate only steers the legacy exact allocator;
        # it is NOT the public experiment rate.
        target = self.view_plan["anyres_keep"]
        raw = self.geometry.tile_count * 729
        allocator_rate = 100 * (1 - (target - .25) / raw)
        allocation, probability = pool.allocate_spatial_probability_budgets(
            raw_redundancy, first["base_token_indices"], self.geometry.grid, spatial["spatial_prior"],
            prune_rate=allocator_rate, mix_weight=self.settings.spatial_strength,
            temperature=self.settings.allocation_temperature, minimum_keep=pool.MINIMUM_CROP_KEEP,audit=not self.lean)
        if allocation["target_anyres_keep"] != target:
            raise RuntimeError("MyPool allocator changed the exact hybrid AnyRes target")
        keep = self.view_plan["base_keep"]
        dominant = keep if keep == 729 else round(keep * pool.DOMINANT_RATIO)
        return allocation, probability, dominant, keep - dominant

    def encode_images(self, images):
        if self.lean:
            from .pool_lean import encode,export_source_decisions
            packed = encode(self,images)
            if self.mode == "base_full":
                self.decisions = export_source_decisions(self.device_decisions)
                self.records = source_records(self.decisions,self.geometry.grid,
                    final_newline=self.core.config.mm_patch_merge_type.endswith("_add_newl"),fast=True)
            else:
                # Only roles are needed for own-AnyRes scoring; do not construct
                # unused original-source lists/weights for every merged token.
                self.records = [None if role==2 else dict(view=role) for role in self.roles.cpu().tolist()]
                self.decisions = []
        else:
            packed = super().encode_images(images)
            self.records = source_records(self.decisions, self.geometry.grid,
                                          final_newline=self.core.config.mm_patch_merge_type.endswith("_add_newl"))
        if len(self.records) != packed.shape[0] or packed.shape[0] != self.plan["pre_llm_tokens"]:
            raise RuntimeError("Sparse provenance / pre-LLM budget / packed features disagree")
        # The shared CLSE engine physically drops at this boundary keep ratio.
        self.core.clse_keep_ratio = self.plan["final_llm_tokens"] / self.plan["pre_llm_tokens"]
        return packed

    def pack(self,image_features,image_sizes,modalities):
        if not self.lean:
            return super().pack(image_features,image_sizes,modalities)
        if len(image_features)!=1 or len(image_sizes)!=1 or list(modalities)!=["image"]:
            raise ValueError("Lean hybrid requires one image")
        if image_features[0].ndim!=2 or image_features[0].shape[0]!=self.plan["pre_llm_tokens"]:
            raise RuntimeError("Lean packed image violates the exact pre-LLM budget")
        return image_features

    def attach_layout(self, input_ids):
        from llava.constants import IMAGE_TOKEN_INDEX
        if self.lean and self.prepared_prompt_split is not None:
            start = self.prepared_prompt_split
        else:
            placeholders = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
            if len(placeholders) != 1:
                raise ValueError("Hybrid requires one image placeholder")
            start = int(placeholders[0])
        self.spatial = [i for i, record in enumerate(self.records) if record is not None]
        self.base = [i for i in self.spatial if self.records[i]["view"] == 0]
        self.anyres = [i for i in self.spatial if self.records[i]["view"] != 0]
        self.protected = [i for i, record in enumerate(self.records) if record is None]
        self.core._clse_spans = [[(start, start + len(self.records))]]
        self.core._clse_layout = dict(packed_tokens=len(self.records), grids=[
            dict(name="base", indices=self.base), dict(name="anyres", indices=self.anyres)],
            sparse=True, pre_pruning_applied=True)

    def reference(self, hidden, start, end):
        if end - start != len(self.records):
            raise ValueError("Hybrid image span does not match provenance")
        if self.plan["final_llm_tokens"] < len(self.base) + len(self.protected):
            raise ValueError("Hybrid cannot preserve remaining Base and structural tokens")
        return None, self.spatial

    def rank(self, hidden, start, ids):
        from llava.model.language_model.attention_predictor import student_autocast, base_text_predictor_inputs
        positions = self.base if self.mode == "base_full" else self.spatial
        indices = [start + i for i in positions]
        if self.predictor_input == "base_text":
            hidden, ids, indices = base_text_predictor_inputs(hidden, ids, start, start+len(self.records), self.base)
        with student_autocast(hidden.device, self.core._predictor_precision):
            scores = self.core.attention_predictor(hidden, ids, indices)
        if self.mode == "base_full":
            scores = mapped_scores(scores, [self.records[i] for i in self.base],
                                   [self.records[i] for i in self.anyres], self.geometry)
        if scores.shape != (len(self.spatial),) or not torch.isfinite(scores).all():
            raise FloatingPointError("Invalid sparse predictor scores")
        return scores

    def select(self, ranking, spatial, original):
        if spatial != self.spatial or original != len(self.records):
            raise ValueError("Hybrid selection mismatches sparse token order")
        full = ranking.new_zeros(original)
        full[torch.tensor(spatial, device=ranking.device)] = ranking
        anyres = torch.tensor(self.anyres, device=ranking.device)
        count = self.plan["final_llm_tokens"] - len(self.base) - len(self.protected)
        order = torch.argsort(full[anyres], descending=True, stable=True)
        selected = torch.cat((torch.tensor(self.base + self.protected, device=ranking.device), anyres[order[:count]])).long().sort().values
        return selected, self.plan["final_llm_tokens"], len(self.protected)


class PoolDistillMethod:
    name = "pool_distill"

    def __init__(self, core, checkpoint, *, rate, share, mode, settings, pruning_audit="lean",
                 tail_drop_layers=0, predictor_input="full"):
        if mode not in {"base_full", "anyres_self"}:
            raise ValueError("Choose base_full or anyres_self")
        from .attention_predictor import AttentionPredictorMethod
        self.native = AttentionPredictorMethod(checkpoint, (rate,), frozenset(), selection_mode=mode)
        if pruning_audit not in {"lean","full"}:
            raise ValueError("pruning_audit must be lean or full")
        if (type(tail_drop_layers) is not int or tail_drop_layers not in {0,2,3,4}
                or (tail_drop_layers and tail_drop_layers >= len(core.layers)-2)):
            raise ValueError("Tail removal must be 0/2/3/4 blocks and begin after block2")
        if predictor_input not in {"full","base_text"} or (predictor_input=="base_text" and mode!="base_full"):
            raise ValueError("Base+Text predictor input requires base_full mode")
        if (tail_drop_layers or predictor_input=="base_text") and (share==0 or rate==0):
            raise ValueError("Ablations require an active distilled pruning stage")
        self.tail_drop_layers, self.predictor_input = tail_drop_layers, predictor_input
        self.adapter = PoolDistillAdapter(core, settings, rate=rate, share=share, mode=mode,
            lean=pruning_audit=="lean", predictor_input=predictor_input)
        self.rate, self.share, self.mode = rate, share, mode

    def prepare_inputs(self, original_size, tokens):
        """Static geometry/prompt metadata belongs to input preparation, before Lat."""
        if not self.adapter.lean or self.share == 1 or self.rate == 0:
            return
        from llava.constants import IMAGE_TOKEN_INDEX
        positions = (tokens[0]==IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        if len(positions)!=1:
            raise ValueError("Hybrid requires one image placeholder")
        self.adapter.prepared_prompt_split = int(positions[0])
        self.adapter.set_image_context(original_size)

    def configure(self, core, rate, **capture):
        self.adapter.prepared_prompt_split = None
        if rate != self.rate:
            raise ValueError("Wrong runtime hybrid rate")
        if any(capture.values()):
            raise ValueError("Hybrid first experiment is results-only; visualization requires sparse-source rendering")
        if self.share == 1 or rate == 0:
            self.native.configure(core, rate, **capture)
        else:
            # Dense configuration treats keep_ratio=1 as disabled, including
            # skipping checkpoint loading. Actual ratio is assigned after the
            # pre-LLM adapter knows this image's original and sparse lengths.
            core.configure_attention_predictor(enabled=self.share > 0, keep_ratio=.5 if self.share > 0 else 1.,
                checkpoint=self.native.checkpoint, selection_mode=self.mode)
            self.adapter.configure(self.adapter.settings.prune_rate, capture_visualization=False)

    @contextmanager
    def generation_context(self, model):
        core = model.get_model()
        options = {"_pool_tail_drop_layers":self.tail_drop_layers, "_pool_predictor_input":self.predictor_input}
        previous = {name:getattr(core,name,None) for name in options}
        present = {name:hasattr(core,name) for name in options}
        try:
            for name,value in options.items():
                setattr(core,name,value)
            with self._generation_context(model):
                yield
        finally:
            for name in options:
                if present[name]:
                    setattr(core,name,previous[name])
                else:
                    delattr(core,name)

    @contextmanager
    def _generation_context(self, model):
        if self.share == 1 or self.rate == 0:
            yield
            return
        core = model.get_model()
        previous = getattr(core, "_pool_distill_adapter", None)
        core._pool_distill_adapter = self.adapter
        try:
            with pool.visionzip_generation(model, self.adapter):
                prepare = model.prepare_inputs_labels_for_multimodal
                def wrapped(this, *args, **kwargs):
                    result = prepare(*args, **kwargs)
                    if result[4] is not None:
                        self.adapter.attach_layout(args[0] if args else kwargs["input_ids"])
                    return result
                model.prepare_inputs_labels_for_multimodal = types.MethodType(wrapped, model)
                yield
        finally:
            if previous is None:
                delattr(core, "_pool_distill_adapter")
            else:
                core._pool_distill_adapter = previous

    def stats(self, core, rate):
        if self.share == 1 or rate == 0:
            raw = self.native.stats(core, rate)
            plan = hybrid_budget(raw.get("original_image_tokens") or raw["final_llm_visual_tokens"],
                                 rate, self.share, len(core.layers))
            original, pre = plan["baseline_packed_tokens"], plan["pre_llm_tokens"]
            final = raw.get("stage2_llm_visual_tokens", raw["final_llm_visual_tokens"])
        else:
            plan = self.adapter.plan
            raw = dict(core.get_clse_stats()) if self.share > 0 else {}
            original, pre = plan["baseline_packed_tokens"], plan["pre_llm_tokens"]
            final = raw.get("stage2_llm_visual_tokens", raw["final_llm_visual_tokens"]) if self.share > 0 else pre
        if final != plan["final_llm_tokens"]:
            raise RuntimeError(f"Final native generation {final} differs from hybrid plan {plan['final_llm_tokens']} (lambda={self.share})")
        depth = len(core.layers)
        rows = raw.get("layers")
        observed = sum(row["image_tokens_in"] for row in rows) if rows else depth * pre
        expected = 2 * pre + (depth - 2 - self.tail_drop_layers) * final
        if observed != expected:
            raise RuntimeError("Actual per-layer token accounting differs from hybrid budget")
        if self.share == 1 or rate == 0:
            grids = {g["name"]: g for g in core._clse_layout["grids"]}
            base, anyres = len(grids["base"]["indices"]), len(grids["anyres"]["indices"])
            protected = pre - base - anyres
        else:
            base, protected = self.adapter.view_plan["base_keep"], self.adapter.view_plan["protected_newlines"]
            anyres = pre - base - protected
        tail = self.tail_drop_layers
        last_visual = raw["final_llm_visual_tokens"] if self.share > 0 else pre
        if last_visual != (0 if tail else final):
            raise RuntimeError("Tail image-drop result differs from the requested ablation")
        return dict(plan, selection_mode=self.mode, predictor_input=self.predictor_input,
                    tail_drop_layers=tail, tail_drop_first_layer=depth-tail+1 if tail else None,
                    two_stage_average_prune_rate=plan["actual_average_prune_rate"],
                    actual_tail_contribution=100*tail*final/(depth*original),
                    observed_layer_token_sum=observed,
                    budget_source="observed_generation", planned_layer_token_sum=expected,
                    pre_base_tokens=base, pre_anyres_tokens=anyres, protected_newlines=protected,
                    stage2_llm_visual_tokens=final, stage2_anyres_tokens=final-base-protected,
                    final_base_tokens=0 if tail else base, final_anyres_tokens=0 if tail else final-base-protected,
                    final_llm_visual_tokens=last_visual,
                    actual_average_prune_rate=100 * (1 - observed / (depth * original)),
                    base_policy="MyPool compresses Base; stage2 protects remaining Base",
                    position_policy="MyPool consecutive prefill IDs; stage2 preserves these IDs",
                    source_policy="normalized original merge coefficients; Base backprojection preserves probability mass")

    def recover_legacy_budget(self, original_size):
        """Reconstruct deterministic integer counts, not scores or observed rows.

        Old workers checked stats before saving each prediction, but did not
        save per-image budgets until the entire trial finished. Mark these
        reconstructed counts explicitly; original Lat/Mem are never recomputed.
        """
        core = self.adapter.core
        if self.share == 1:
            from llava.mm_utils import get_anyres_image_grid_shape
            from llava.model.language_model.attention_predictor_selection import native_anyres_counts
            tower = core.get_vision_tower()
            grid = get_anyres_image_grid_shape(original_size, core.config.image_grid_pinpoints, tower.image_size)
            base, anyres, protected = native_anyres_counts(int(tower.num_patches_per_side), grid, original_size,
                final_newline=core.config.mm_patch_merge_type.endswith("_add_newl"))
            plan = hybrid_budget(base+anyres+protected, self.rate, self.share, len(core.layers))
        else:
            self.adapter.set_image_context(original_size)
            plan = dict(self.adapter.plan)
            base = self.adapter.view_plan["base_keep"]
            protected = self.adapter.view_plan["protected_newlines"]
            anyres = plan["pre_llm_tokens"] - base - protected
        final = plan["final_llm_tokens"]
        if final < base + protected:
            raise ValueError("Legacy counts cannot preserve Base and newlines")
        tail, depth = self.tail_drop_layers, len(core.layers)
        expected = 2*plan["pre_llm_tokens"]+(depth-2-tail)*final
        return dict(plan, selection_mode=self.mode, predictor_input=self.predictor_input,
                    tail_drop_layers=tail, tail_drop_first_layer=depth-tail+1 if tail else None,
                    two_stage_average_prune_rate=plan["actual_average_prune_rate"],
                    actual_tail_contribution=100*tail*final/(depth*plan["baseline_packed_tokens"]),
                    actual_average_prune_rate=100*(1-expected/(depth*plan["baseline_packed_tokens"])),
                    observed_layer_token_sum=None, planned_layer_token_sum=expected,
                    budget_source="legacy_reconstructed_counts",
                    pre_base_tokens=base, pre_anyres_tokens=anyres, protected_newlines=protected,
                    stage2_llm_visual_tokens=final, stage2_anyres_tokens=final-base-protected,
                    final_base_tokens=0 if tail else base, final_anyres_tokens=0 if tail else final-base-protected,
                    final_llm_visual_tokens=0 if tail else final,
                    base_policy="MyPool compresses Base; stage2 protects remaining Base",
                    position_policy="MyPool consecutive prefill IDs; stage2 preserves these IDs",
                    source_policy="normalized original merge coefficients; Base backprojection preserves probability mass")
