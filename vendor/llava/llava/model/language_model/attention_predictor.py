"""Three-layer causal side Transformer; third-layer attention is the score.

Project implementation, not MAP reproduction: two complete small Transformer
blocks followed by the Q/K attention path of block 3. Its V/O/MLP cannot affect
block-3 attention and are intentionally not instantiated. No backbone edits.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_VERSION = 1


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
    if saved.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported attention predictor checkpoint version")
    config = PredictorConfig(**saved["architecture"])
    if config.input_size != input_size or saved["student_layer"] != 2:
        raise ValueError("Predictor backbone dimension or source-layer mismatch")
    model = AttentionPredictor(config)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device=device).eval().requires_grad_(False)
    precision = saved.get("student_precision", "float32")
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
                                      capture_attention=False, capture_visualization=False):
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
            self.attention_predictor, self._predictor_precision = model, precision
            self._predictor_teacher_layer = saved["teacher_layer"]
            self._predictor_checkpoint = identity

    def _clse_boundary(self):
        return 2 if self.attention_predictor_enabled else super()._clse_boundary()

    def _clse_reference(self, hidden, start, end):
        if not self.attention_predictor_enabled:
            return super()._clse_reference(hidden, start, end)
        spatial = [i for grid in self._clse_layout["grids"] for i in grid["indices"]]
        return None, spatial

    def _clse_rank(self, hidden, start, end, ids, reference):
        if not self.attention_predictor_enabled:
            return super()._clse_rank(hidden, start, end, ids, reference)
        indices = [start + i for grid in self._clse_layout["grids"] for i in grid["indices"]]
        with student_autocast(hidden.device, self._predictor_precision):
            log_prob = self.attention_predictor(hidden, ids, indices)
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("Non-finite predictor scores; no precision/backend fallback")
        # Ranking log probabilities avoids exp underflow; capture probabilities
        # separately so the public overlay remains easy to interpret.
        return log_prob

    def _clse_extra_stats(self):
        if not self.attention_predictor_enabled:
            return super()._clse_extra_stats()
        return dict(score="predicted_layer_attention", attention_source="three_layer_side_transformer",
                    pruning_layer=2, teacher_layer=self._predictor_teacher_layer,
                    student_layers=3, student_precision=self._predictor_precision)

    def _clse_score_kind(self):
        return "attention_predictor" if self.attention_predictor_enabled else super()._clse_score_kind()

    def get_clse_stages(self):
        stages = super().get_clse_stages()
        if self.attention_predictor_enabled:
            for stage in stages:
                if "attention" in stage:
                    # Newlines are non-spatial; zero is correct for their display.
                    scores = torch.tensor(stage["attention"]["scores"])
                    spatial = [i for grid in self._clse_layout["grids"] for i in grid["indices"]]
                    result = torch.zeros_like(scores)
                    result[spatial] = scores[spatial].exp()
                    stage["attention"]["scores"] = result.tolist()
        return stages
