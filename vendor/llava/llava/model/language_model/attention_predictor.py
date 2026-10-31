"""Pretrained Qwen2 blocks 3/4/5 as an independent attention-only side branch.

The original small architecture is kept ONLY for loading version-1 checkpoints.
New training copies full-width native blocks, not the backbone's parameters by
reference. Only block5's input norm and Q/K affect its attention loss; its other
copied parameters are retained but frozen and never executed. No LM head.
"""

import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_VERSION = 2
LEGACY_CHECKPOINT_VERSION = 1

# Only decoder architecture fields, not vision paths/runtime pruning state.
QWEN_FIELDS = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
               "num_key_value_heads", "hidden_act", "max_position_embeddings", "rms_norm_eps",
               "rope_theta", "rope_scaling", "attention_dropout", "use_sliding_window",
               "sliding_window", "max_window_layers")


def copied_qwen_config(config):
    return {key: copy.deepcopy(getattr(config, key, None)) for key in QWEN_FIELDS}


def backbone_signature(config):
    result = {key: getattr(config, key, None) for key in (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "rope_theta", "vocab_size", "image_aspect_ratio", "mm_patch_merge_type", "image_grid_pinpoints")}
    result["image_token_order"] = getattr(config, "image_token_order", "base_first")
    return result


@dataclass(frozen=True)
class PredictorConfig:
    input_size: int
    hidden_size: int = 512
    num_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.
    rope_theta: float = 1000000.
    num_layers: int = 3

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in
               (self.input_size, self.hidden_size, self.num_heads, self.intermediate_size)):
            raise ValueError("Predictor dimensions must be positive integers")
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError("Predictor head dimension must be even and divide hidden_size")
        if self.num_layers != 3 or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("Predictor requires exactly three layers and dropout in [0,1)")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("Predictor rope_theta must be positive")


def _rope(x, positions, theta):
    dimension = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, dimension, 2, device=x.device).float() / dimension)
    angles = positions.float()[..., None] * frequencies
    angles = torch.cat((angles, angles), -1)[:, None]
    a, b = x.chunk(2, -1)
    return x * angles.cos().to(x.dtype) + torch.cat((-b, a), -1) * angles.sin().to(x.dtype)


class SideBlock(nn.Module):
    def __init__(self, config, score_only=False):
        super().__init__()
        self.config, self.score_only = config, score_only
        self.norm = nn.LayerNorm(config.hidden_size)
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if not score_only:
            self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.ff_norm = nn.LayerNorm(config.hidden_size)
            self.ff = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.GELU(),
                                    nn.Linear(config.intermediate_size, config.hidden_size))
            self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, positions, spatial_indices=None):
        n, heads = hidden.shape[1], self.config.num_heads
        dimension = self.config.hidden_size // heads
        normalized = self.norm(hidden)
        query_input = normalized[:, -1:] if self.score_only else normalized
        q = self.q(query_input).view(1, -1, heads, dimension).transpose(1, 2)
        k = self.k(normalized).view(1, n, heads, dimension).transpose(1, 2)
        q = _rope(q, positions[:, -1:] if self.score_only else positions, self.config.rope_theta)
        k = _rope(k, positions, self.config.rope_theta)
        if self.score_only:
            # Full-key softmax -> head mean -> spatial selection -> L1 normalization,
            # exactly the teacher aggregation order. Log-space avoids underflow.
            # Explicit .float() alone is insufficient inside autocast: matmul
            # would be downcast again. Keep this final tiny row truly FP32.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(dimension)
                log_rows = F.log_softmax(logits, dim=-1)[0, :, 0]
                log_mean = torch.logsumexp(log_rows, dim=0) - math.log(heads)
                log_spatial = log_mean[spatial_indices]
                return log_spatial - torch.logsumexp(log_spatial, dim=0)
        v = self.v(normalized).view(1, n, heads, dimension).transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.config.dropout if self.training else 0.)
        output = output.transpose(1, 2).reshape(1, n, self.config.hidden_size)
        hidden = hidden + self.dropout(self.o(output))
        return hidden + self.dropout(self.ff(self.ff_norm(hidden)))


class AttentionPredictor(nn.Module):
    """Legacy 512-wide predictor, for version-1 checkpoint compatibility only."""
    checkpoint_version = LEGACY_CHECKPOINT_VERSION

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_size)
        self.project = nn.Linear(config.input_size, config.hidden_size)
        self.layers = nn.ModuleList([SideBlock(config), SideBlock(config), SideBlock(config, score_only=True)])

    def forward(self, hidden, position_ids, spatial_indices):
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != self.config.input_size:
            raise ValueError("Predictor requires one unpadded full multimodal sequence")
        if position_ids.shape != hidden.shape[:2]:
            raise ValueError("Predictor position IDs must match the original sequence")
        indices = torch.as_tensor(spatial_indices, dtype=torch.long, device=hidden.device)
        if (indices.ndim != 1 or indices.numel() == 0 or int(indices.min()) < 0
                or int(indices.max()) >= hidden.shape[1] or len(indices.unique()) != len(indices)):
            raise ValueError("Predictor needs distinct original spatial token indices")
        value = self.project(self.input_norm(hidden.float()))
        for layer in self.layers[:2]:
            value = layer(value, position_ids)
        return self.layers[2](value, position_ids, indices)


@dataclass(frozen=True)
class CopiedPredictorConfig:
    qwen_config: dict
    attention_implementation: str
    gradient_checkpointing: bool = True
    source_layers: tuple = (3, 4, 5)

    def __post_init__(self):
        if tuple(self.source_layers) != (3, 4, 5):
            raise ValueError("Copied predictor must initialize from original blocks 3, 4, 5")
        if self.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported copied Qwen2 attention implementation")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if set(self.qwen_config) != set(QWEN_FIELDS):
            raise ValueError("Incomplete copied Qwen2 architecture")
        if self.qwen_config["num_hidden_layers"] < 5:
            raise ValueError("Copied predictor requires backbone blocks 3, 4, 5")
        if self.qwen_config["use_sliding_window"]:
            raise ValueError("Copied predictor currently requires non-sliding attention")

    @property
    def input_size(self):
        return self.qwen_config["hidden_size"]


class CopiedAttentionPredictor(nn.Module):
    """Native full-width blocks; differentiable, independently calculated Q/K score."""
    checkpoint_version = CHECKPOINT_VERSION

    def __init__(self, config):
        super().__init__()
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        self.config = config
        native_config = Qwen2Config(**config.qwen_config)
        native_config._attn_implementation = config.attention_implementation
        # Keep original layer_idx for RoPE/attention policy. No KV is ever used.
        self.layers = nn.ModuleList([Qwen2DecoderLayer(native_config, layer_idx=i - 1)
                                     for i in config.source_layers])
        self.rotary_emb = Qwen2RotaryEmbedding(config=native_config)
        self.requires_grad_(True)
        last = self.layers[2]
        last.requires_grad_(False)
        for module in (last.input_layernorm, last.self_attn.q_proj, last.self_attn.k_proj):
            module.requires_grad_(True)

    @classmethod
    def from_backbone(cls, core, *, gradient_checkpointing=True):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        if len(core.layers) < 5 or not isinstance(core.rotary_emb, Qwen2RotaryEmbedding):
            raise ValueError("Copied predictor requires native Qwen2 blocks 3, 4, 5 and RoPE")
        config = CopiedPredictorConfig(copied_qwen_config(core.config), core.config._attn_implementation,
                                       gradient_checkpointing=gradient_checkpointing)
        # Allocate independent FP32 masters on CPU, then copy one block at a time.
        # Never deepcopy the resident GPU modules/hooks or change the main model.
        result = cls(config).float()
        for destination, index in zip(result.layers, config.source_layers):
            source = core.layers[index - 1]
            if type(source) is not Qwen2DecoderLayer or type(source.self_attn) is not type(destination.self_attn):
                raise ValueError("Source must be unmodified native Qwen2 decoder/attention blocks")
            if any(parameter.is_meta for parameter in source.parameters()):
                raise ValueError("Cannot initialize side branch from unmaterialized meta weights")
            destination.load_state_dict({key: value.detach().cpu() for key, value in source.state_dict().items()}, strict=True)
        return result

    def _run_block(self, layer, hidden, position_ids):
        rotary = self.rotary_emb(hidden, position_ids)
        bias = None
        if self.config.attention_implementation == "eager":
            n = hidden.shape[1]
            future = torch.arange(n, device=hidden.device)[None, :] > torch.arange(n, device=hidden.device)[:, None]
            bias = hidden.new_zeros((1, 1, n, n)).masked_fill(future, torch.finfo(hidden.dtype).min)
        # Native SDPA uses is_causal when bias=None; FlashAttention2 is causal too.
        return layer(hidden, attention_mask=bias, position_ids=position_ids, past_key_value=None,
                     output_attentions=False, use_cache=False, cache_position=position_ids[0],
                     position_embeddings=rotary)[0]

    def forward(self, hidden, position_ids, spatial_indices):
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != self.config.input_size:
            raise ValueError("Copied predictor requires one unpadded full multimodal sequence")
        if position_ids.shape != hidden.shape[:2]:
            raise ValueError("Predictor position IDs must match the original sequence")
        indices = torch.as_tensor(spatial_indices, dtype=torch.long, device=hidden.device)
        if (indices.ndim != 1 or indices.numel() == 0 or int(indices.min()) < 0
                or int(indices.max()) >= hidden.shape[1] or len(indices.unique()) != len(indices)):
            raise ValueError("Predictor needs distinct original spatial token indices")
        # Detach is intentional: training must never backpropagate into H2/main.
        value = hidden.detach().to(self.layers[0].input_layernorm.weight.dtype)
        for layer in self.layers[:2]:
            if self.training and self.config.gradient_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                # Bind layer now; a loop closure would recompute the wrong layer.
                def run_block(x, ids, block=layer):
                    return self._run_block(block, x, ids)
                value = checkpoint(run_block, value, position_ids, use_reentrant=False)
            else:
                value = self._run_block(layer, value, position_ids)
        last = self.layers[2]
        attention = last.self_attn
        normalized = last.input_layernorm(value)
        query = attention.q_proj(normalized[:, -1:]).view(1, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
        key = attention.k_proj(normalized).view(1, -1, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value, position_ids)
        from .fastv_attention import _rotate_half
        query = query * cos[:, None, -1:] + _rotate_half(query) * sin[:, None, -1:]
        key = key * cos[:, None] + _rotate_half(key) * sin[:, None]
        key = key.repeat_interleave(attention.num_key_value_groups, dim=1)
        # No @no_grad or native output_attentions=True: this row trains Q/K and
        # the preceding copied blocks, with stable FP32 log-softmax/head mean.
        with torch.autocast(device_type=value.device.type, enabled=False):
            logits = (query.float() @ key.float().transpose(-2, -1)) / math.sqrt(attention.head_dim)
            rows = F.log_softmax(logits, dim=-1)[0, :, 0]
            mean = torch.logsumexp(rows, dim=0) - math.log(attention.num_heads)
            spatial = mean[indices]
            return spatial - torch.logsumexp(spatial, dim=0)


def base_text_predictor_inputs(hidden, position_ids, start, end, base_indices):
    """Select H2 Base/text rows, dropping the rest of the packed image span.

    Positions retain their original RoPE IDs. Only the side input is shortened;
    the backbone hidden states/cache are not changed. Image structural markers
    are excluded together with AnyRes, because they are not Base spatial rows.
    """
    base = torch.as_tensor(base_indices, dtype=torch.long, device=hidden.device)
    if (not 0 <= start < end < hidden.shape[1] or base.ndim != 1 or not len(base)
            or int(base.min()) < 0 or int(base.max()) >= end-start
            or (len(base) > 1 and not bool((base[1:] > base[:-1]).all()))):
        raise ValueError("Base+Text side input requires ordered Base indices and trailing prompt text")
    selected = torch.cat((torch.arange(start, device=hidden.device), base+start,
                          torch.arange(end, hidden.shape[1], device=hidden.device)))
    spatial = torch.arange(start, start+len(base), device=hidden.device)
    return hidden.index_select(1, selected), position_ids.index_select(1, selected), spatial


def student_autocast(device, precision):
    if precision == "float32":
        return nullcontext()
    if precision != "bfloat16" or torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("Student bfloat16 requires a BF16-capable CUDA GPU; choose float32 otherwise")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def load_predictor(path, device, input_size):
    # weights_only avoids executing arbitrary pickle objects; all payload fields
    # are tensors or primitive containers, including the training provenance.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") == LEGACY_CHECKPOINT_VERSION:
        config = PredictorConfig(**saved["architecture"])
        model = AttentionPredictor(config)
    elif saved.get("version") == CHECKPOINT_VERSION:
        config = CopiedPredictorConfig(**saved["architecture"])
        model = CopiedAttentionPredictor(config)
    else:
        raise ValueError("Unsupported attention predictor checkpoint version")
    if config.input_size != input_size or saved["student_layer"] != 2:
        raise ValueError("Predictor backbone dimension or source-layer mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device=device).eval().requires_grad_(False)
    precision = saved.get("student_precision", "float32")
    if (isinstance(model, CopiedAttentionPredictor) and config.attention_implementation == "flash_attention_2"
            and precision != "bfloat16"):
        raise ValueError("Copied FlashAttention2 student requires bfloat16 autocast; main model stays FP16")
    # Validate precision before a long evaluation starts.
    with student_autocast(device, precision):
        pass
    return model, precision, saved


class AttentionPredictorMixin:
    """Learned score hooks into the existing CLSE physical-drop/cache engine."""
    def init_attention_predictor(self):
        self.attention_predictor_enabled = False
        self.attention_predictor = None
        self._predictor_checkpoint = None

    def clear_attention_predictor(self):
        self.attention_predictor_enabled = False
        self.attention_predictor = None
        self._predictor_checkpoint = None

    def configure_clse(self, **kwargs):
        self.attention_predictor_enabled = False
        return super().configure_clse(**kwargs)

    def configure_priortr(self, **kwargs):
        self.clear_attention_predictor()
        return super().configure_priortr(**kwargs)

    def configure_vico(self, **kwargs):
        self.clear_attention_predictor()
        return super().configure_vico(**kwargs)

    def configure_pyramiddrop(self, **kwargs):
        self.clear_attention_predictor()
        return super().configure_pyramiddrop(**kwargs)

    def configure_attention_predictor(self, *, enabled, keep_ratio, checkpoint=None,
                                      capture_attention=False, capture_visualization=False, selection_mode="base_full",
                                      crop_temperature=0.05, crop_min_keep=0):
        # Old training method.json files explicitly used base_guided. Upgrade
        # that alias to the current policy; explicit old modes remain available.
        selection_mode = "base_full" if selection_mode == "base_guided" else selection_mode
        if selection_mode not in {"base_full", "anyres_self", "base_crop", "base_interpolated", "joint"}:
            raise ValueError("Predictor selection_mode must be base_full, anyres_self, base_crop, base_interpolated or joint")
        from .attention_predictor_crops import validate_crop_options
        validate_crop_options(crop_temperature, crop_min_keep)
        self._predictor_crop_temperature, self._predictor_crop_min_keep = crop_temperature, crop_min_keep
        self._predictor_selection_mode = selection_mode
        if enabled and len(self.layers) <= 2:
            raise ValueError("Attention predictor needs a decoder deeper than block2")
        # Preserve already loaded side weights across rate/sample resets, but not
        # when switching to an unrelated method or the zero-rate baseline.
        self._preserve_predictor_weights = bool(enabled)
        try:
            self.configure_clse(enabled=enabled, keep_ratio=keep_ratio,
                                capture_attention=capture_attention, capture_visualization=capture_visualization)
        finally:
            self._preserve_predictor_weights = False
        self.attention_predictor_enabled = self.clse_enabled
        if not self.attention_predictor_enabled:
            return
        from pathlib import Path
        path = Path(checkpoint).resolve()
        identity = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        if identity != self._predictor_checkpoint:
            model, precision, saved = load_predictor(path, self.embed_tokens.weight.device, self.config.hidden_size)
            if saved.get("backbone_depth") != len(self.layers):
                raise ValueError("Predictor was trained for a different backbone depth")
            if type(saved.get("teacher_layer")) is not int or not 2 < saved["teacher_layer"] <= len(self.layers):
                raise ValueError("Predictor checkpoint has an invalid teacher layer")
            if saved.get("backbone_signature") != backbone_signature(self.config):
                raise ValueError("Predictor backbone architecture/AnyRes packing/order does not match training")
            if isinstance(model, CopiedAttentionPredictor) and (
                    model.config.qwen_config != copied_qwen_config(self.config)
                    or model.config.attention_implementation != self.config._attn_implementation):
                raise ValueError("Copied predictor Qwen2 architecture/attention backend does not match backbone")
            self.attention_predictor, self._predictor_precision = model, precision
            self._predictor_teacher_layer = saved["teacher_layer"]
            self._predictor_checkpoint = identity

    def _clse_boundary(self):
        return 2 if self.attention_predictor_enabled else super()._clse_boundary()

    def _clse_reference(self, hidden, start, end):
        if not self.attention_predictor_enabled:
            return super()._clse_reference(hidden, start, end)
        hybrid = getattr(self, "_pool_distill_adapter", None)
        if hybrid is not None:
            return hybrid.reference(hidden, start, end)
        if self._predictor_selection_mode in {"base_full", "anyres_self"}:
            from .attention_predictor_selection import base_full_budget
            base_full_budget(self._clse_layout, self.clse_keep_ratio)
        elif self._predictor_selection_mode in {"base_crop", "base_interpolated"}:
            from .attention_predictor_selection import base_guided_budget
            # Validate both group quotas before any decoder computation.
            base_guided_budget(self._clse_layout, self.clse_keep_ratio)
            if self._predictor_selection_mode == "base_crop":
                from .attention_predictor_crops import validate_crop_provenance
                validate_crop_provenance(self._clse_layout, self._clse_layout.get("predictor_crops"))
        spatial = [i for grid in self._clse_layout["grids"] for i in grid["indices"]]
        return None, spatial

    def _clse_rank(self, hidden, start, end, ids, reference):
        if not self.attention_predictor_enabled:
            return super()._clse_rank(hidden, start, end, ids, reference)
        hybrid = getattr(self, "_pool_distill_adapter", None)
        if hybrid is not None:
            return hybrid.rank(hidden, start, ids)
        guided = self._predictor_selection_mode in {"base_full", "base_interpolated"}
        grids = self._clse_layout["grids"]
        if guided:
            from .attention_predictor_selection import named_spatial_grids, base_guided_log_scores
            base = named_spatial_grids(self._clse_layout)["base"]
            # Score only Base positions for ranking; side input is still the
            # FULL original H2. Training targets/checkpoint architecture do not change.
            indices = [start + i for i in base["indices"]]
        else:
            indices = [start + i for grid in grids for i in grid["indices"]]
        if getattr(self, "_pool_predictor_input", "full") == "base_text":
            if self._predictor_selection_mode != "base_full":
                raise ValueError("Base+Text side input requires Base-mapped ranking")
            hidden, ids, indices = base_text_predictor_inputs(hidden, ids, start, end, base["indices"])
        with student_autocast(hidden.device, self._predictor_precision):
            log_prob = self.attention_predictor(hidden, ids, indices)
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("Non-finite predictor scores; no precision/backend fallback")
        if guided:
            log_prob = base_guided_log_scores(log_prob, self._clse_layout)
        # Ranking log probabilities avoids exp underflow; capture probabilities
        # separately so the public overlay remains easy to interpret.
        return log_prob

    def _clse_select(self, ranking, spatial_indices, original):
        hybrid = getattr(self, "_pool_distill_adapter", None)
        if self.attention_predictor_enabled and hybrid is not None:
            return hybrid.select(ranking, spatial_indices, original)
        if not self.attention_predictor_enabled or self._predictor_selection_mode == "joint":
            return super()._clse_select(ranking, spatial_indices, original)
        if self._predictor_selection_mode in {"base_full", "anyres_self"}:
            from .attention_predictor_selection import select_base_full_tokens
            selected, plan = select_base_full_tokens(ranking, spatial_indices, original,
                                                     self.clse_keep_ratio, self._clse_layout)
            self._clse_stats.update(anyres_spatial_keep_ratio=plan["anyres_spatial_keep_ratio"],
                                    common_spatial_keep_ratio=None)
        elif self._predictor_selection_mode == "base_crop":
            from .attention_predictor_crops import select_base_crop_tokens
            selected, plan = select_base_crop_tokens(
                ranking, spatial_indices, original, self.clse_keep_ratio, self._clse_layout,
                temperature=self._predictor_crop_temperature, minimum_keep=self._predictor_crop_min_keep)
            self._clse_stats["crop_budgets"] = plan["crops"]
        else:
            from .attention_predictor_selection import select_base_guided_tokens
            selected, plan = select_base_guided_tokens(ranking, spatial_indices, original,
                                                       self.clse_keep_ratio, self._clse_layout)
        self._clse_stats.update(selection_groups=plan["groups"],
                                common_spatial_keep_ratio=plan.get("spatial_keep_ratio"))
        return selected, plan["target"], len(plan["protected_indices"])

    def _clse_extra_stats(self):
        if not self.attention_predictor_enabled:
            return super()._clse_extra_stats()
        copied = isinstance(self.attention_predictor, CopiedAttentionPredictor)
        guided = self._predictor_selection_mode == "base_interpolated"
        crop = self._predictor_selection_mode == "base_crop"
        full = self._predictor_selection_mode in {"base_full", "anyres_self"}
        mapped = self._predictor_selection_mode == "base_full"
        return dict(score="predicted_layer_attention", attention_source=("copied_qwen2_blocks_3_4_5" if copied else "legacy_small_transformer"),
                    pruning_layer=2, teacher_layer=self._predictor_teacher_layer,
                    selection_mode=self._predictor_selection_mode,
                    budget_scope=("Base fully retained; all original packed-budget deletions transferred to ONE global AnyRes top-k"
                                  if full else "equal-rate Base/whole AnyRes; Base region means -> bounded crop softmax; own AnyRes top-k"
                                  if crop else "separate Base and whole AnyRes; equal proportional spatial quotas after newline reservation"
                                  if guided else "joint Base + AnyRes; structural newlines protected"),
                    ranking_source=("predicted Base attention mapped to whole AnyRes; own AnyRes scores not requested" if mapped else
                                    "own predicted AnyRes attention; Base protected and its scores not used for selection" if full else
                                    "own predicted attention; Base crop means allocate AnyRes quotas" if crop else
                                    "predicted Base attention; FP32 bilinear probabilities on actual packed AnyRes grid"
                                    if guided else "predicted own attention for all spatial tokens"),
                    rounding=("floor ORIGINAL packed target; reserve ALL Base + newlines; remaining slots go to AnyRes; fail if infeasible"
                              if full else "floor packed target, reserve newlines, proportional largest remainder; ties Base first"
                              if guided or crop else "floor packed target, reserve newlines"),
                    crop_temperature=self._predictor_crop_temperature if crop else None,
                    crop_min_keep=self._predictor_crop_min_keep if crop else None,
                    pre_llm_pruning_enabled=False, baseline_reference="unpruned native packed sequence",
                    student_layers=3, student_precision=self._predictor_precision,
                    student_hidden_size=self.attention_predictor.config.input_size if copied else self.attention_predictor.config.hidden_size,
                    student_source_layers=[3, 4, 5] if copied else None)

    def _clse_score_kind(self):
        if not self.attention_predictor_enabled:
            return super()._clse_score_kind()
        return ("attention_predictor_base_full" if self._predictor_selection_mode == "base_full" else
                "attention_predictor_anyres_self" if self._predictor_selection_mode == "anyres_self" else
                "attention_predictor_base_crop" if self._predictor_selection_mode == "base_crop" else
                "attention_predictor_base_guided" if self._predictor_selection_mode == "base_interpolated" else "attention_predictor")

    def get_clse_stages(self):
        stages = super().get_clse_stages()
        if self.attention_predictor_enabled:
            for stage in stages:
                stage["selection_mode"] = self._predictor_selection_mode
                if "selection_groups" in self._clse_stats:
                    stage["selection_groups"] = self._clse_stats["selection_groups"]
                if "crop_budgets" in self._clse_stats:
                    stage["crop_budgets"] = self._clse_stats["crop_budgets"]
                if "attention" in stage:
                    # Newlines are non-spatial; zero is correct for their display.
                    scores = torch.tensor(stage["attention"]["scores"])
                    spatial = [i for grid in self._clse_layout["grids"] for i in grid["indices"]]
                    result = torch.zeros_like(scores)
                    result[spatial] = scores[spatial].exp()
                    stage["attention"]["scores"] = result.tolist()
        return stages
