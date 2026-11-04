"""AnchorPrune independent OneVision/SigLIP adaptation, before all LLM blocks.

The pinned official selector is unchanged. Native visual packing completes
before selection; image newlines are structural, protected budget members.
"""

from contextlib import contextmanager
from decimal import Decimal, ROUND_FLOOR
import json
import math
from pathlib import Path

from .rates import is_prune_rate


SOURCE = Path(__file__).resolve().parents[1] / "third_party/anchorprune/SOURCE.json"


def packed_budget(original, protected, rate):
    if (type(original) is not int or type(protected) is not int
            or not 0 <= protected < original or not is_prune_rate(rate)):
        raise ValueError("Invalid AnchorPrune packed budget")
    target = int((Decimal(original) * (1 - Decimal(str(rate)) / 100)).to_integral_value(rounding=ROUND_FLOOR))
    if target <= protected:
        raise ValueError(f"AnchorPrune {rate}% keeps {target}/{original} tokens, but must protect "
                         f"{protected} newlines and at least one spatial token; lower the rate")
    return target, target - protected


def received_importance(query, key, heads, scale, chunk=64):
    """Independent noncausal FP32 QK, key-softmax, mean over heads/queries."""
    import torch
    with torch.autocast(device_type=query.device.type, enabled=False):
        batch, count, channels = query.shape
        if key.shape != query.shape or channels % heads or chunk < 1:
            raise ValueError("Invalid SigLIP Q/K shapes")
        q = query.float().view(batch, count, heads, channels // heads).transpose(1, 2)
        k = key.float().view(batch, count, heads, channels // heads).transpose(1, 2)
        received = q.new_zeros(batch, heads, count)
        for start in range(0, count, chunk):
            weights = (q[:, :, start:start+chunk] @ k.transpose(-2, -1) * scale).softmax(-1)
            received += weights.sum(-2)
        return (received / count).mean(1)


def pack_importance(received, grid, original_size, merge_type, order):
    """Apply the same native unpad/max9 operations to scores; label visual units.

    Only unit membership uses nearest-exact interpolation at a max9 resize;
    continuous importance uses native bilinear interpolation, as features do.
    """
    import torch
    from torch.nn import functional as F
    from llava.model.llava_arch import unpad_image
    from llava.model.language_model.fastv_budget import base_anyres_groups
    views, count = received.shape
    side = math.isqrt(count)
    if side * side != count or views != 1 + grid[0] * grid[1]:
        raise ValueError("AnchorPrune received attention disagrees with native crop geometry")
    tiles = received[1:].view(grid[1], grid[0], side, side).permute(0, 2, 1, 3).reshape(1, grid[1]*side, grid[0]*side)
    units = torch.arange(1, views, device=received.device, dtype=torch.float32).view(grid[1], grid[0])
    units = units.repeat_interleave(side, 0).repeat_interleave(side, 1)[None]
    tiles, units = unpad_image(tiles, original_size), unpad_image(units, original_size)
    _, height, width = tiles.shape
    times = math.sqrt(height * width / (9 * side ** 2))
    if times > 1.1:
        size = [int(height // times), int(width // times)]
        tiles = F.interpolate(tiles[None], size, mode="bilinear")[0]
        units = F.interpolate(units[None], size, mode="nearest-exact")[0]
    height, width = tiles.shape[-2:]
    layout = base_anyres_groups(count, height, width, order=order,
                               final_newline=merge_type.endswith("_add_newl"))
    spatial = [i for group in layout["groups"] for i in group["indices"]]
    # The layout groups are canonical Base/AnyRes, even for anyres_first packing.
    scores = torch.cat((received[0], tiles.flatten()))
    unit_ids = torch.cat((torch.zeros(count, device=received.device, dtype=torch.long), units.flatten().long()))
    return scores, unit_ids, spatial, layout["packed_tokens"]


def select_units(relevance, features, importance, units, keep, *, min_ratio=5/32, tau=.2, patience=3):
    """Paper A.4: per-unit adaptive anchors, then ONE global context expansion.

    For dynamic OneVision budgets, apportion unit quotas by candidate count
    using largest remainders; extend the paper's K_min/K=5/32 schedule.
    Context expansion is free to reallocate the remaining slots globally.
    """
    import torch
    from third_party.anchorprune.selection import adaptive_relevance_anchor, importance_weighted_expansion
    count = features.shape[0]
    if (relevance.shape != (count,) or importance.shape != (count,) or units.shape != (count,)
            or type(keep) is not int or not 1 <= keep <= count
            or not torch.isfinite(features).all() or not torch.isfinite(relevance).all()
            or not torch.isfinite(importance).all() or (importance < 0).any()):
        raise ValueError("Invalid AnchorPrune candidate scores/features/budget")
    if keep == count:
        return torch.arange(count, device=features.device), 0
    unit_values, counts = torch.unique(units, sorted=True, return_counts=True)
    sizes = counts.tolist()
    quotas = [keep * size // count for size in sizes]
    rest = keep - sum(quotas)
    for index in sorted(range(len(sizes)), key=lambda j: (-(keep * sizes[j] % count), j))[:rest]:
        quotas[index] += 1
    anchors = []
    for value, quota in zip(unit_values, quotas):
        if not quota:
            continue
        indices = (units == value).nonzero(as_tuple=False).flatten()
        minimum = min(max(1, int(math.ceil(quota * min_ratio))), max(1, quota // 2))
        local = adaptive_relevance_anchor(relevance[indices], features[indices], quota, minimum,
                                          tau=tau, patience=patience, kmax_ratio=.5)
        anchors.append(indices[local])
    anchor = torch.cat(anchors) if anchors else torch.empty(0, dtype=torch.long, device=features.device)
    selected = importance_weighted_expansion(features, importance, keep, preselected=anchor)
    if selected.numel() != keep or selected.unique().numel() != keep:
        raise RuntimeError("Official AnchorPrune selector violated the exact budget")
    return selected, anchor.numel()


class AnchorPruneMethod:
    name = "anchorprune"
    layer = 0

    def __init__(self, rates=(1, 50, 70, 90), *, vision_layer=26, anchor_min_ratio=5/32,
                 tau=.2, patience=3, query_chunk=64):
        self.rates, self.visualize_rates = tuple(rates), frozenset()
        if (not self.rates or len(set(self.rates)) != len(self.rates)
                or any(not is_prune_rate(r) for r in self.rates)):
            raise ValueError("AnchorPrune rates must be distinct percentages in [0,100)")
        if (type(vision_layer) is not int or vision_layer < 1 or type(patience) is not int or patience < 1
                or type(query_chunk) is not int or query_chunk < 1
                or type(tau) not in (int, float) or not math.isfinite(tau) or tau <= 0
                or type(anchor_min_ratio) not in (int, float) or not math.isfinite(anchor_min_ratio)
                or not 0 < anchor_min_ratio <= .5):
            raise ValueError("Invalid AnchorPrune selector/vision settings")
        self.vision_layer, self.anchor_min_ratio = vision_layer, anchor_min_ratio
        self.tau, self.patience, self.query_chunk = tau, patience, query_chunk
        self._rate, self._stats, self.selected = 0, {}, None

    @classmethod
    def from_config(cls, path):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {"prune_rates", "rate_semantics", "visualize_rates", "vision_layer", "anchor_min_ratio",
                   "tau", "patience", "query_chunk"}
        if set(config)-allowed or config.get("rate_semantics", "layer_average") != "layer_average":
            raise ValueError("AnchorPrune requires layer_average rates and recognized config keys")
        if config.get("visualize_rates", []):
            raise ValueError("This independent AnchorPrune adapter currently exports results without sample PNGs")
        return cls(config["prune_rates"], **{k:v for k,v in config.items() if k not in
                   {"prune_rates", "rate_semantics", "visualize_rates"}})

    def resolve_budgets(self, model_depth):
        return [dict(prune_rate=r, rate_semantics="layer_average", pruning_layer=0,
                     target_average_prune_rate=r, requested_layer_prune_rate=r) for r in self.rates]

    def configure(self, core, prune_rate, *, capture_attention=False, capture_visualization=False):
        if prune_rate not in self.rates or capture_attention or capture_visualization:
            raise ValueError("AnchorPrune expects a configured rate and no sample visualization")
        if (core.training or core.config.image_aspect_ratio != "anyres_max_9"
                or core.config.mm_patch_merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}):
            raise ValueError("AnchorPrune requires eval(), anyres_max_9 and native spatial_unpad packing")
        core.configure_fastv(enabled=False)
        self._rate, self._stats, self.selected = prune_rate, {}, None

    def prepare_prompt(self, prompt, tokenizer):
        # Tokenization is input preparation. The embedding lookup and relevance
        # calculation stay inside the measured model.generate call.
        self.query_ids = tokenizer(prompt.replace("<image>", ""), add_special_tokens=False,
                                   return_tensors="pt").input_ids
        if self.query_ids.ndim != 2 or self.query_ids.shape[0] != 1 or not self.query_ids.shape[1]:
            raise ValueError("AnchorPrune needs a nonempty instruction")

    def prepare_inputs(self, original_size, tokens):
        self.original_size, self.tokens = tuple(original_size), tokens

    def validate_input_budgets(self, backend, samples, resolved_budgets):
        from PIL import Image
        from llava.mm_utils import get_anyres_image_grid_shape
        from llava.model.language_model.attention_predictor_selection import native_anyres_counts
        core, tower = backend.model.get_model(), backend.vision_tower
        seen = set()
        for sample in samples:
            with Image.open(sample.image) as image:
                size = image.size
            if size in seen:
                continue
            seen.add(size)
            grid = get_anyres_image_grid_shape(size, core.config.image_grid_pinpoints, tower.image_size)
            counts = native_anyres_counts(tower.num_patches_per_side, grid, size,
                                         final_newline=core.config.mm_patch_merge_type.endswith("_add_newl"))
            for rate in self.rates:
                packed_budget(sum(counts), counts[2], rate)
        return dict(unique_image_sizes=len(seen), checked_rates=list(self.rates),
                    denominator="native packed Base+AnyRes+newlines; protected newlines count toward target")

    @contextmanager
    def inference_context(self, model):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import get_anyres_image_grid_shape
        from llava.model.language_model.attention_predictor_selection import native_anyres_counts
        core, handles, captured = model.get_model(), [], {}
        tower = core.get_vision_tower()
        config = core.config
        grid = get_anyres_image_grid_shape(self.original_size, config.image_grid_pinpoints, tower.image_size)
        base, anyres, protected = native_anyres_counts(tower.num_patches_per_side, grid, self.original_size,
                                                      final_newline=config.mm_patch_merge_type.endswith("_add_newl"))
        original = base+anyres+protected
        target, keep = packed_budget(original, protected, self._rate)
        # Placeholder lookup precedes the timer; all dynamic selection stays in it.
        locations = (self.tokens[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        if len(locations) != 1 or self.tokens.shape[0] != 1:
            raise ValueError("AnchorPrune supports one image and batch size one")
        start = int(locations[0])
        text_count = self.tokens.shape[1]-1
        observed = {}
        for index, layer in enumerate(core.layers):
            def observe(module, args, kwargs, index=index):
                if index not in observed:
                    hidden = args[0] if args else kwargs["hidden_states"]
                    observed[index] = hidden.shape[1]-text_count
                    if observed[index] != target:
                        raise RuntimeError("Observed LLM visual length disagrees with AnchorPrune budget")
            handles.append(layer.register_forward_pre_hook(observe, with_kwargs=True))
        original_prepare = model.prepare_inputs_labels_for_multimodal
        had_override = "prepare_inputs_labels_for_multimodal" in model.__dict__
        previous_override = model.__dict__.get("prepare_inputs_labels_for_multimodal")

        def prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            hidden = result[4]
            # Cached autoregressive decode re-enters native preparation with
            # one text token and no images; prune only the initial prefill.
            if hidden is None:
                return result
            if hidden.shape[:2] != (1, text_count+original):
                raise RuntimeError("Native packed embeddings disagree with AnchorPrune geometry")
            if set(captured) != {"q", "k"}:
                raise RuntimeError("Missing selected SigLIP Q/K; no importance fallback is allowed")
            importance = received_importance(captured["q"], captured["k"], attention.num_heads,
                                              attention.scale, self.query_chunk)
            captured.clear()
            importance, units, spatial, packed = pack_importance(importance, grid, self.original_size,
                config.mm_patch_merge_type, getattr(config, "image_token_order", "base_first"))
            if packed != original:
                raise RuntimeError("Importance packing differs from native visual packing")
            indices = torch.tensor(spatial, device=hidden.device, dtype=torch.long)
            features = hidden[0, start+indices]
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                query = core.embed_tokens(self.query_ids.to(hidden.device))[0].float()
                f = torch.nn.functional.normalize(features.float(), dim=-1)
                q = torch.nn.functional.normalize(query, dim=-1)
                relevance = (f @ q.t()).max(-1).values
                chosen, anchor_count = select_units(relevance, features, importance, units, keep,
                    min_ratio=self.anchor_min_ratio, tau=self.tau, patience=self.patience)
            all_indices = torch.arange(original, device=hidden.device)
            structural_mask = torch.ones(original, dtype=torch.bool, device=hidden.device)
            structural_mask[indices] = False
            image_selected = torch.cat((indices[chosen], all_indices[structural_mask])).sort().values
            if image_selected.numel() != target:
                raise RuntimeError("AnchorPrune selection changed the protected packed target")
            self.selected = image_selected.detach()
            sequence = torch.cat((torch.arange(start, device=hidden.device), start+image_selected,
                                  torch.arange(start+original, hidden.shape[1], device=hidden.device)))
            self._stats["anchor_tokens"] = anchor_count
            result = list(result)
            result[4] = hidden.index_select(1, sequence)
            if result[1] is not None:
                result[1] = torch.arange(len(sequence), device=hidden.device, dtype=result[1].dtype)[None]
            if result[2] is not None:
                result[2] = result[2].index_select(-1, sequence)
            if result[5] is not None:
                result[5] = result[5].index_select(1, sequence)
            return tuple(result)

        try:
            if self._rate:
                blocks = tower.vision_tower.vision_model.encoder.layers
                if not 1 <= self.vision_layer <= len(blocks):
                    raise ValueError("AnchorPrune vision_layer is outside the active SigLIP blocks")
                attention = blocks[self.vision_layer-1].self_attn
                def capture(name):
                    def hook(module, args, output): captured[name] = output.detach()
                    return hook
                handles.append(attention.q_proj.register_forward_hook(capture("q")))
                handles.append(attention.k_proj.register_forward_hook(capture("k")))
                model.prepare_inputs_labels_for_multimodal = prepare
            yield
            if len(observed) != len(core.layers):
                raise RuntimeError("Incomplete AnchorPrune LLM-layer accounting")
            self._stats.update(original_image_tokens=original, final_llm_visual_tokens=target,
                protected_newline_tokens=protected, observed_layer_token_sum=sum(observed.values()),
                actual_layer_prune_rate=100*(1-target/original),
                actual_average_prune_rate=100*(1-sum(observed.values())/(len(core.layers)*original)))
        finally:
            if self._rate:
                if had_override:
                    model.prepare_inputs_labels_for_multimodal = previous_override
                else:
                    delattr(model, "prepare_inputs_labels_for_multimodal")
            for handle in handles: handle.remove()
            captured.clear()

    def stats(self, core, prune_rate):
        return dict(self.resolve_budgets(len(core.layers))[self.rates.index(prune_rate)], **self._stats)

    def details(self):
        return dict(source=json.loads(SOURCE.read_text(encoding="utf-8")),
            adaptation="OneVision/SigLIP; paper non-CLIP post-projector token-wise maximum relevance",
            importance=f"independent FP32 received QK attention from active SigLIP block {self.vision_layer}",
            selector="official per-unit adaptive anchors, then importance-weighted global context expansion",
            anchor_min_ratio=self.anchor_min_ratio, tau=self.tau, patience=self.patience, kmax_ratio=.5,
            dynamic_unit_budget="candidate-count proportional quotas with exact largest-remainder rounding",
            rate_semantics="layer_average; native packed Base+AnyRes+newlines denominator",
            budget_rounding="floor original*(1-rate/100); reserve ALL native newlines before spatial selection",
            packing="native unpad/max9/newlines first; select existing projected features without merging",
            position_policy="native retained visual order; consecutive LLM input IDs after pre-LLM compaction",
            zero_policy="native preparation and encoder; no QK capture, scoring or selection at rate 0")
