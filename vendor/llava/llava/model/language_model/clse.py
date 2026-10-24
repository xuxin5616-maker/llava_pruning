"""CLSE single-stage OneVision/Qwen2 adaptation (inference only).

Modified from zjubinchen/CLSE cfc9710abe4a1bd5480e521f3acd86ded4855ff3,
LLaVA1.5 tools.py/clse_model.py; Apache-2.0 snapshots in third_party/clse.
Pure spectral scoring: input embeddings -> first block output. Native decoder
precision/backend remain unchanged; no attention outputs or Q/K recomputation.
Base and actual packed AnyRes grids have separate spatial FFTs, but ONE joint
evolution calibration and top-k budget. Structural newlines are protected.
"""

import math

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .pruning_timing import measure_pruning


@torch.no_grad()
def spatial_spectral_energy(hidden, indices, rows, cols, *, cutoff_ratio=.1, channel_chunk=128):
    """Pinned Gaussian high-pass FFT, channel-chunked to bound scratch memory.

    hidden: [1, packed image tokens, channels]; indices: row-major spatial cells.
    No L2 normalization and no dtype/layout modification of the native features.
    """
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or min(rows, cols) < 2
            or len(indices) != rows * cols or type(channel_chunk) is not int or channel_chunk < 1):
        raise ValueError("CLSE needs a complete spatial grid with both dimensions >= 2")
    if not math.isfinite(cutoff_ratio) or cutoff_ratio <= 0:
        raise ValueError("CLSE cutoff_ratio must be finite and positive")
    device = hidden.device
    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    if int(indices.min()) < 0 or int(indices.max()) >= hidden.shape[1] or len(indices.unique()) != len(indices):
        raise ValueError("Invalid CLSE spatial indices")
    # Match upstream's float32 mask (sqrt then square is intentional).
    y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing="ij")
    distance = torch.sqrt((y - rows // 2) ** 2 + (x - cols // 2) ** 2)
    mask = 1 - torch.exp(-(distance ** 2) / (2 * (min(rows // 2, cols // 2) * cutoff_ratio) ** 2))
    total = torch.zeros(rows, cols, device=device, dtype=torch.float32)
    for first in range(0, hidden.shape[-1], channel_chunk):
        feature = hidden[0, indices, first:first + channel_chunk].transpose(0, 1).reshape(-1, rows, cols).float()
        spectrum = torch.fft.fftshift(torch.fft.fft2(feature), dim=(-2, -1))
        residual = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1))).abs()
        total += residual.sum(dim=0)
    score = (total / hidden.shape[-1]).flatten()
    if not torch.isfinite(score).all():
        raise FloatingPointError("Non-finite CLSE spectral energy")
    return score


@torch.no_grad()
def evolution_scores(reference, current, *, epsilon=1e-6, temperature=.1):
    """Pinned pure-CLSE evolution factor; calibrate all spatial tokens jointly."""
    intensity = (current - reference).abs() / (reference.mean() + reference + epsilon)
    intensity = intensity.clamp(max=1)
    standardized = (intensity - intensity.mean()) / (intensity.std() + epsilon)
    scores = torch.sigmoid(standardized / temperature)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite CLSE evolution scores")
    return scores


def select_joint_tokens(scores, spatial_indices, original, keep_ratio):
    """Floor the TOTAL packed budget, then reserve all structural newlines."""
    target = min(original, max(1, math.floor(original * keep_ratio + 1e-10)))
    spatial_indices = torch.as_tensor(spatial_indices, dtype=torch.long, device=scores.device)
    spatial_mask = torch.zeros(original, dtype=torch.bool, device=scores.device)
    spatial_mask[spatial_indices] = True
    protected = (~spatial_mask).nonzero(as_tuple=False).flatten()
    candidate_keep = target - len(protected)
    if candidate_keep < 1:
        raise ValueError(f"CLSE packed budget keeps {target}/{original} tokens but {len(protected)} structural "
                         "newlines must be protected plus at least one spatial token; lower the pruning target")
    selected = spatial_indices[scores.topk(candidate_keep).indices]
    return torch.cat((selected, protected)).sort().values, target, len(protected)


class CLSEMixin:
    # Shared physical-drop engine hooks. The learned scorer overrides only these;
    # existing pure-CLSE scoring/precision/geometry remain unchanged.
    def _clse_boundary(self):
        return 1

    def _clse_reference(self, hidden, start, end):
        return self._clse_energy(hidden[:, start:end])

    def _clse_rank(self, hidden, start, end, ids, reference):
        energy, _ = self._clse_energy(hidden[:, start:end])
        return evolution_scores(reference, energy, epsilon=self.clse_epsilon, temperature=self.clse_temperature)

    def _clse_extra_stats(self):
        return {}

    def _clse_score_kind(self):
        return "clse"

    def _clse_select(self, ranking, spatial_indices, original):
        return select_joint_tokens(ranking, spatial_indices, original, self.clse_keep_ratio)

    def init_clse(self):
        self.clse_enabled = self.clse_configured = False
        self.reset_clse_state()

    def reset_clse_state(self):
        self._clse_spans = []
        self._clse_prompt_length = None
        self._clse_layout = None
        self._clse_stats = {}
        self._clse_stage = None
        self._clse_next_position = None

    def configure_clse(self, *, enabled, keep_ratio, cutoff_ratio=.1, epsilon=1e-6,
                       temperature=.1, channel_chunk=128, capture_visualization=False, capture_attention=False):
        if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
            raise ValueError("CLSE keep_ratio must be in (0,1]")
        if any(not math.isfinite(v) or v <= 0 for v in (cutoff_ratio, epsilon, temperature)):
            raise ValueError("CLSE spectral parameters must be positive and finite")
        if type(channel_chunk) is not int or channel_chunk < 1:
            raise ValueError("CLSE channel_chunk must be a positive integer")
        self.configure_fastv(enabled=False)
        self.clse_configured = True
        self.clse_enabled = bool(enabled) and keep_ratio < 1
        self.clse_keep_ratio = keep_ratio
        self.clse_cutoff, self.clse_epsilon, self.clse_temperature = cutoff_ratio, epsilon, temperature
        self.clse_channel_chunk = channel_chunk
        self.clse_capture = bool(capture_visualization or capture_attention)
        self.clse_capture_attention = bool(capture_attention)
        if self.clse_enabled and len({p.device for layer in self.layers for p in layer.parameters()}) != 1:
            raise ValueError("CLSE requires decoder weights on one device")

    def configure_priortr(self, **kwargs):
        self.clse_enabled = self.clse_configured = False
        return super().configure_priortr(**kwargs)

    def configure_pyramiddrop(self, **kwargs):
        self.clse_enabled = self.clse_configured = False
        return super().configure_pyramiddrop(**kwargs)

    def configure_vico(self, **kwargs):
        self.clse_enabled = self.clse_configured = False
        return super().configure_vico(**kwargs)

    def set_clse_image_layout(self, spans, length, layout):
        if self.clse_enabled:
            if not layout or len(layout.get("grids", [])) != 2:
                raise ValueError("CLSE requires separate complete Base and AnyRes grids")
            spatial = [i for grid in layout["grids"] for i in grid["indices"]]
            if (len(spatial) != len(set(spatial)) or not spatial or min(spatial) < 0
                    or max(spatial) >= layout["packed_tokens"]
                    or any(len(grid["indices"]) != grid["rows"] * grid["cols"] for grid in layout["grids"])):
                raise ValueError("Invalid or overlapping CLSE spatial grids")
        self._clse_spans, self._clse_prompt_length, self._clse_layout = spans, length, layout

    def validate_clse_generation(self, kwargs, generation_config):
        if not self.clse_enabled:
            return
        config = kwargs.get("generation_config", generation_config)
        def option(name, default=None):
            return kwargs.get(name, getattr(config, name, default))
        if (option("do_sample", False) or option("num_beams", 1) != 1
                or option("num_return_sequences", 1) != 1 or option("num_beam_groups", 1) != 1
                or kwargs.get("assistant_model") is not None or option("prompt_lookup_num_tokens")
                or option("penalty_alpha") is not None or option("constraints") is not None
                or option("force_words_ids") is not None or option("dola_layers") is not None):
            raise ValueError("CLSE supports only batch-1 greedy autoregressive generation")
        if not option("use_cache", True) or option("cache_implementation") not in (None, "dynamic"):
            raise ValueError("CLSE requires use_cache=True and DynamicCache")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("CLSE generate requires a fresh image prompt")

    def _clse_energy(self, hidden):
        values, indices = [], []
        for grid in self._clse_layout["grids"]:
            indices.extend(grid["indices"])
            values.append(spatial_spectral_energy(hidden, grid["indices"], grid["rows"], grid["cols"],
                                                  cutoff_ratio=self.clse_cutoff, channel_chunk=self.clse_channel_chunk))
        return torch.cat(values), indices

    def get_clse_stats(self):
        if self.clse_enabled:
            return dict(self._clse_stats)
        spans = self._clse_spans[0] if self._clse_spans else []
        count = sum(b - a for a, b in spans)
        length = self._clse_prompt_length
        return dict(mode="disabled_baseline", original_image_tokens=count, final_llm_visual_tokens=count,
                    keep_ratio=1., layers=[self._priortr_layer_record(i + 1, count, count, length, length, count)
                                           for i in range(len(self.layers))])

    def get_clse_stages(self):
        if self._clse_stage is None:
            raise RuntimeError("No CLSE visualization captured")
        item = self._clse_stage
        start, end = self._clse_spans[0][0]
        keep = torch.zeros(end - start, dtype=torch.bool)
        keep[item["indices"].detach().cpu()] = True
        stage = dict(item["stats"], mask={"span": [start, end], "keep": keep.tolist()})
        if "scores" in item:
            stage["attention"] = {"span": [start, end], "scores": item["scores"].detach().cpu().tolist(),
                                  "score_kind": self._clse_score_kind()}
        return [stage]

    @torch.no_grad()
    def clse_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                     past_key_values=None, inputs_embeds=None, use_cache=None,
                     output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None):
        if self.training or output_attentions or (output_attentions is None and self.config.output_attentions):
            raise ValueError("CLSE is eval-only and never requests decoder attention outputs")
        if getattr(self.config, "use_sliding_window", False):
            raise ValueError("CLSE does not support sliding-window attention")
        if self.config._attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported CLSE attention backend")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if use_cache is False or (use_cache is None and not self.config.use_cache):
            raise ValueError("CLSE requires use_cache=True")
        if past_key_values is not None and type(past_key_values) is not DynamicCache:
            raise ValueError("CLSE requires DynamicCache")
        cache = past_key_values if past_key_values is not None else DynamicCache()
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.shape[0] != 1:
            raise ValueError("CLSE supports batch size one")
        if attention_mask is not None and (attention_mask.ndim != 2 or attention_mask.shape[0] != 1
                                           or not attention_mask.bool().all()):
            raise ValueError("CLSE requires an unpadded prompt")
        prefill = cache.get_seq_length() == 0
        reference = None
        if prefill:
            if len(self._clse_spans) != 1 or len(self._clse_spans[0]) != 1 or self._clse_layout is None:
                raise ValueError("CLSE requires exactly one packed image with actual spatial layout")
            start, end = self._clse_spans[0][0]
            if not 0 <= start < end < hidden.shape[1]:
                raise ValueError("CLSE requires an image followed by prompt text")
            if end - start != self._clse_layout["packed_tokens"]:
                raise ValueError("CLSE cannot FFT a truncated image grid")
            ids = torch.arange(hidden.shape[1], device=hidden.device)[None]
            if position_ids is not None and not torch.equal(position_ids.to(hidden.device), ids):
                raise ValueError("CLSE requires contiguous zero-based initial positions")
            if attention_mask is not None and attention_mask.shape[-1] != hidden.shape[1]:
                raise ValueError("CLSE initial mask must match packed prompt length")
            original = current = end - start
            self._clse_next_position = hidden.shape[1]
            self._clse_stats = dict(mode="physical_drop", original_image_tokens=original,
                                    original_prompt_tokens=hidden.shape[1], layers=[], pruning_layer=self._clse_boundary(),
                                    position_policy="preserve_original", attention_source="none_pure_spectral",
                                    score="clse", budget_scope="joint Base + AnyRes; structural newlines protected",
                                    rounding="floor packed target, reserve newlines", min_spatial_tokens=1)
            self._clse_stats.update(self._clse_extra_stats())
            self._clse_stage = None
            with measure_pruning(self, hidden.device):
                reference, spatial_indices = self._clse_reference(hidden, start, end)
                # Check feasibility BEFORE running the decoder, not by silently clamping.
                target = max(1, math.floor(original * self.clse_keep_ratio + 1e-10))
                protected = original - len(spatial_indices)
                if target <= protected:
                    raise ValueError(f"CLSE packed budget keeps {target}/{original} tokens but {protected} structural "
                                     "newlines plus one spatial token are required; lower the pruning target")
        else:
            if hidden.shape[1] != 1 or self._clse_next_position is None:
                raise ValueError("CLSE decode needs one token and matching prefill state")
            ids = torch.tensor([[self._clse_next_position]], device=hidden.device)
            self._clse_next_position += 1
        all_hidden = () if (self.config.output_hidden_states if output_hidden_states is None else output_hidden_states) else None
        for index, layer in enumerate(self.layers):
            if all_hidden is not None:
                all_hidden += (hidden,)
            physical_past = cache.get_seq_length(index)
            physical_positions = torch.arange(physical_past, physical_past + hidden.shape[1], device=hidden.device)
            rotary = self.rotary_emb(hidden, ids)
            causal = self._pyramiddrop_causal_mask(hidden, physical_past, self.config._attn_implementation)
            hidden = layer(hidden, attention_mask=causal, position_ids=ids, past_key_value=cache,
                           output_attentions=False, use_cache=True, cache_position=physical_positions,
                           position_embeddings=rotary)[0]
            if not prefill:
                continue
            before, seq_before = current, hidden.shape[1]
            if index + 1 == self._clse_boundary():
                with measure_pruning(self, hidden.device):
                    ranking = self._clse_rank(hidden, start, end, ids, reference)
                    reference = None
                    selected, retained, protected = self._clse_select(ranking, spatial_indices, original)
                    selection = torch.cat((torch.arange(start, device=hidden.device), selected + start,
                                           torch.arange(end, hidden.shape[1], device=hidden.device)))
                    hidden = hidden.index_select(1, selection).contiguous()
                    ids = ids.index_select(1, selection)
                    for cached_layer in range(len(cache.key_cache)):
                        cache.key_cache[cached_layer] = cache.key_cache[cached_layer].index_select(2, selection).contiguous()
                        cache.value_cache[cached_layer] = cache.value_cache[cached_layer].index_select(2, selection).contiguous()
                    cache._seen_tokens = hidden.shape[1]
                    current = retained
                    self._clse_stats.update(protected_newline_tokens=protected, spatial_tokens=original - protected,
                                            kept_spatial_tokens=retained - protected)
                    stage = dict(after_layer=index + 1, scoring_layer=index + 1, target_keep_ratio=self.clse_keep_ratio,
                                 image_tokens_before=before, image_tokens_after=retained,
                                 removed_this_stage=before - retained, cumulative_prune_rate=100 * (1 - retained / original))
                    if self.clse_capture:
                        self._clse_stage = {"stats": stage, "indices": selected.detach()}
                        if self.clse_capture_attention:
                            full_scores = torch.zeros(original, device=ranking.device)
                            full_scores[torch.tensor(spatial_indices, device=ranking.device)] = ranking
                            self._clse_stage["scores"] = full_scores
            self._clse_stats["layers"].append(
                self._priortr_layer_record(index + 1, before, current, seq_before, hidden.shape[1], original))
        hidden = self.norm(hidden)
        if prefill:
            self._clse_stats.update(final_llm_visual_tokens=current, final_sequence_tokens=hidden.shape[1],
                                    actual_prune_rate=100 * (1 - current / original))
        if all_hidden is not None:
            all_hidden += (hidden,)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not return_dict:
            return tuple(v for v in (hidden, cache, all_hidden) if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache, hidden_states=all_hidden)
