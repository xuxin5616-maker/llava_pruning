"""PyramidDrop inference for LLaVA-OneVision / Qwen2, Transformers 4.46.1.

Adapted from Cooperx521/PyramidDrop, commit
6444f304aee4b7edcb022839fb9e8cd07859704a, ``pdrop_rank_drop``.
The upstream repository has an MIT license; see third_party/pyramiddrop/LICENSE.
Its inherited modeling source retains its original Apache-2.0 header.
Ranking uses the NEXT block's
input norm and Q/K projections, including RoPE, after each configured boundary.
This adapter adds Qwen2 grouped-query attention and per-layer DynamicCache
positions; it does not change the loaded model's dtype or attention backend.
"""

import math

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .pruning_timing import measure_pruning


def _rotate_half(values):
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@torch.no_grad()
def last_prompt_attention(decoder_layer, hidden_states, position_embeddings):
    """Official ranking row [1, length], with Qwen2 GQA support.

    QK multiplication uses the model dtype. Like upstream, softmax runs in
    FP32, is cast back to the query dtype, and is then averaged over heads.
    The last token is guaranteed to be an instruction token by the caller.
    """
    batch, length, _ = hidden_states.shape
    attention = decoder_layer.self_attn
    normalized = decoder_layer.input_layernorm(hidden_states)
    query = attention.q_proj(normalized[:, -1:, :])
    key = attention.k_proj(normalized)
    query = query.view(batch, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    cos, sin = cos.to(query.device), sin.to(query.device)
    query = query * cos[:, None, -1:, :] + _rotate_half(query) * sin[:, None, -1:, :]
    key = key * cos[:, None, :, :] + _rotate_half(key) * sin[:, None, :, :]
    key = key.repeat_interleave(attention.num_key_value_groups, dim=1)
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(attention.head_dim)
    if not torch.isfinite(logits).all():
        raise FloatingPointError(
            "PyramidDrop Q/K scores are non-finite in the model dtype; "
            "check the checkpoint and baseline precision."
        )
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    return weights.mean(dim=1).squeeze(1)


class PyramidDropMixin:
    def init_pyramiddrop(self):
        self.pyramiddrop_enabled = False
        self.pyramiddrop_configured = False
        self.pyramiddrop_layers = ()
        self.pyramiddrop_keep_ratios = ()
        self.pyramiddrop_capture_stages = False
        self.reset_pyramiddrop_state()

    def reset_pyramiddrop_state(self):
        self._pyramiddrop_image_spans = []
        self._pyramiddrop_prompt_length = None
        self._pyramiddrop_stats = {}
        self._pyramiddrop_stages = []

    def configure_pyramiddrop(self, *, enabled, layers=(8, 16, 24),
                              keep_ratios=(0.5, 0.25, 0.125), capture_stages=False):
        """Set cumulative ORIGINAL-image fractions after 1-based boundaries.

        ``layers=[8,16,24]`` ranks after those blocks, using blocks 9/17/25.
        Every packed image token, including image_newline, is a candidate.
        Targets use upstream's ``floor(N_original * ratio)``. A minimum of one
        visual token is an explicit safety extension for very small prompts.
        All-one ratios or ``enabled=False`` dispatch to the original Qwen2.
        """
        layers, ratios = tuple(layers), tuple(float(r) for r in keep_ratios)
        if (not layers or any(type(n) is not int or not 1 <= n < len(self.layers) for n in layers)
                or tuple(sorted(set(layers))) != layers):
            raise ValueError(
                "PyramidDrop layers must be increasing 1-based boundaries below the final layer"
            )
        if (len(ratios) != len(layers)
                or any(not math.isfinite(r) or not 0.0 < r <= 1.0 for r in ratios)
                or any(a < b for a, b in zip(ratios, ratios[1:]))):
            raise ValueError(
                "PyramidDrop keep_ratios must be positive non-increasing cumulative fractions"
            )
        active = bool(enabled) and ratios[-1] < 1.0
        if active and len({layer.self_attn.q_proj.weight.device for layer in self.layers}) != 1:
            raise ValueError("PyramidDrop requires all decoder layers on one device")
        self.fastv_enabled = False
        self.vico_enabled = False
        self.vico_configured = False
        if hasattr(self, "reset_fastv_state"):
            self.reset_fastv_state()
        else:
            self.reset_pyramiddrop_state()
        self.pyramiddrop_configured = True
        self.pyramiddrop_enabled = active
        self.pyramiddrop_layers = layers
        self.pyramiddrop_keep_ratios = ratios
        self.pyramiddrop_capture_stages = bool(capture_stages)

    def configure_vico(self, **kwargs):
        # Keep the pre-existing pruning modes mutually exclusive on mode changes.
        self.pyramiddrop_enabled = False
        self.pyramiddrop_configured = False
        return super().configure_vico(**kwargs)

    def set_pyramiddrop_image_spans(self, spans, prompt_length=None):
        """Receive ``batch -> images -> (start, exclusive_end)`` after packing."""
        self._pyramiddrop_image_spans = spans
        self._pyramiddrop_prompt_length = prompt_length

    @staticmethod
    def _pyramiddrop_layer_record(layer, before, after, seq_before, seq_after, original):
        return {
            "layer": layer,
            "image_tokens_in": before,
            "image_tokens_out": after,
            "sequence_tokens_in": seq_before,
            "sequence_tokens_out": seq_after,
            "removed_after_layer": before - after,
            "cumulative_prune_rate": 100.0 * (1.0 - after / original) if original else 0.0,
        }

    def get_pyramiddrop_stats(self):
        if self.pyramiddrop_enabled:
            return dict(self._pyramiddrop_stats)
        spans = self._pyramiddrop_image_spans[0] if self._pyramiddrop_image_spans else []
        count = sum(end - start for start, end in spans)
        length = self._pyramiddrop_prompt_length
        return {
            "mode": "disabled_baseline",
            "keep_ratio": 1.0,
            "rate_semantics": "final_cumulative",
            "original_image_tokens": count,
            "final_llm_visual_tokens": count,
            "actual_keep_ratio": 1.0,
            "actual_prune_rate": 0.0,
            "original_prompt_tokens": length,
            "final_sequence_tokens": length,
            "stage_ratios": list(self.pyramiddrop_keep_ratios),
            "stages": [],
            "layers": [self._pyramiddrop_layer_record(i + 1, count, count, length, length, count)
                       for i in range(len(self.layers))],
        }

    def get_pyramiddrop_stages(self):
        """Materialize original-packed-span masks and ranking rows on demand.

        ``mask``/``attention`` match the existing single-image ViCo viewer.
        ``masks``/``attentions`` also expose every span for multi-image prompts.
        Scores for tokens already removed by earlier stages are invalid, not
        zero-attention observations; ``valid`` explicitly marks this distinction.
        """
        if not self._pyramiddrop_stages:
            raise RuntimeError("No PyramidDrop stages captured; enable capture_stages and generate first")
        spans = self._pyramiddrop_image_spans[0]
        length = self._pyramiddrop_prompt_length
        result = []
        for item in self._pyramiddrop_stages:
            keep = torch.zeros(length, dtype=torch.bool)
            keep[item["kept_original_positions"].detach().cpu()] = True
            valid = torch.zeros(length, dtype=torch.bool)
            active = item["active_original_positions"].detach().cpu()
            valid[active] = True
            scores = torch.zeros(length, dtype=torch.float32)
            scores[active] = item["scores"].detach().float().cpu()
            stage = dict(item["stats"])
            stage["masks"] = [
                {"span": [int(start), int(end)], "keep": keep[start:end].tolist()}
                for start, end in spans
            ]
            stage["attentions"] = [
                {"span": [int(start), int(end)], "scores": scores[start:end].tolist(),
                 "valid": valid[start:end].tolist()}
                for start, end in spans
            ]
            if len(spans) == 1:
                stage["mask"] = stage["masks"][0]
                stage["attention"] = stage["attentions"][0]
            result.append(stage)
        return result

    def validate_pyramiddrop_generation(self, kwargs, generation_config):
        """Reject unsupported generation modes before vision encoding/prefill."""
        if not self.pyramiddrop_enabled:
            return
        config = kwargs.get("generation_config", generation_config)

        def option(name, default=None):
            return kwargs.get(name, getattr(config, name, default))

        if (option("do_sample", False) or option("num_beams", 1) != 1
                or option("num_beam_groups", 1) != 1
                or option("num_return_sequences", 1) != 1
                or option("penalty_alpha", None) is not None
                or option("constraints", None) is not None
                or option("force_words_ids", None) is not None
                or kwargs.get("assistant_model") is not None
                or option("prompt_lookup_num_tokens", None) is not None):
            raise ValueError("PyramidDrop supports batch=1 greedy decoding only")
        if not option("use_cache", True):
            raise ValueError("PyramidDrop greedy decoding requires use_cache=True with DynamicCache")
        if option("cache_implementation", None) not in (None, "dynamic"):
            raise ValueError("PyramidDrop requires DynamicCache; static/sliding/quantized caches are unsupported")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("PyramidDrop generate requires a fresh prompt; external cache continuation is unsupported")

    @staticmethod
    def _pyramiddrop_causal_mask(hidden, past_length, implementation):
        # Unpadded prefill is causal; single-token decode can see every cached
        # key. Qwen2's SDPA/FA2 implementations handle both cases natively.
        if implementation in ("flash_attention_2", "sdpa"):
            return None
        q_length = hidden.shape[1]
        keys = torch.arange(past_length + q_length, device=hidden.device)
        queries = torch.arange(past_length, past_length + q_length, device=hidden.device)
        visible = keys[None, :] <= queries[:, None]
        return torch.zeros(
            (1, 1, q_length, past_length + q_length), dtype=hidden.dtype, device=hidden.device,
        ).masked_fill(~visible[None, None], torch.finfo(hidden.dtype).min)

    @torch.no_grad()
    def pyramiddrop_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                            past_key_values=None, inputs_embeds=None, use_cache=None,
                            output_attentions=None, output_hidden_states=None,
                            return_dict=None, cache_position=None):
        if self.training:
            raise ValueError("PyramidDrop is inference-only; call eval()")
        if output_attentions or (output_attentions is None and self.config.output_attentions):
            raise ValueError("PyramidDrop uses independent Q/K ranking; output_attentions must be False")
        if getattr(self.config, "use_sliding_window", False):
            raise ValueError("PyramidDrop does not support sliding-window attention")
        if self.config._attn_implementation not in ("eager", "sdpa", "flash_attention_2"):
            raise ValueError("PyramidDrop supports eager, sdpa, and flash_attention_2 attention only")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if not use_cache:
            raise ValueError("PyramidDrop requires use_cache=True with DynamicCache")
        if past_key_values is not None and type(past_key_values) is not DynamicCache:
            raise ValueError("PyramidDrop requires DynamicCache; legacy/static/sliding/quantized caches are unsupported")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError("PyramidDrop supports batch size 1 (no beam expansion)")
        if attention_mask is not None and (
                attention_mask.ndim != 2 or attention_mask.shape[0] != 1
                or not attention_mask.bool().all()):
            raise ValueError("PyramidDrop requires an unpadded batch=1 prompt")
        if past_key_values is None:
            past_key_values = DynamicCache()
        prefill = past_key_values.get_seq_length(0) == 0
        if not prefill and inputs_embeds.shape[1] != 1:
            raise ValueError("PyramidDrop cache continuation supports one greedy token per step")
        output_hidden_states = (self.config.output_hidden_states if output_hidden_states is None
                                else output_hidden_states)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        hidden = inputs_embeds
        if prefill:
            if len(self._pyramiddrop_image_spans) != 1 or not self._pyramiddrop_image_spans[0]:
                raise ValueError("PyramidDrop requires packed image spans for one prompt")
            spans = self._pyramiddrop_image_spans[0]
            previous_end = 0
            for start, end in spans:
                if not 0 <= previous_end <= start < end < hidden.shape[1]:
                    raise ValueError(
                        "PyramidDrop image spans must be ordered, nonoverlapping, and followed by instruction text"
                    )
                previous_end = end
            expected_positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
            if position_ids is not None and not torch.equal(position_ids.to(hidden.device), expected_positions):
                raise ValueError("PyramidDrop requires contiguous zero-based prefill position IDs")
            if attention_mask is not None and attention_mask.shape[1] != hidden.shape[1]:
                raise ValueError("PyramidDrop prefill attention mask must match the packed prompt length")
            self._pyramiddrop_prompt_length = hidden.shape[1]
            original = sum(end - start for start, end in spans)
            visual = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
            for start, end in spans:
                visual[start:end] = True
            original_positions = (torch.arange(hidden.shape[1], device=hidden.device)
                                  if self.pyramiddrop_capture_stages else None)
            visual_count = original
            self._pyramiddrop_stats = {
                "mode": "physical_drop", "attention_source": "independent_next_layer_qk_rope",
                "score_dtype": str(hidden.dtype).replace("torch.", ""),
                "softmax_dtype": "float32", "softmax_cast_before_head_mean": True,
                "rate_semantics": "final_cumulative", "rounding": "floor",
                "min_tokens_safety_extension": 1,
                "original_image_tokens": original, "original_prompt_tokens": hidden.shape[1],
                "keep_ratio": self.pyramiddrop_keep_ratios[-1],
                "stage_ratios": list(self.pyramiddrop_keep_ratios),
                "position_policy": "reindex_after_each_drop",
                "image_newline_policy": "rank_with_all_image_tokens",
                "multi_image_policy": "global_visual_ranking",
                "stages": [], "layers": [],
            }
            self._pyramiddrop_stages = []
        elif not self._pyramiddrop_stats:
            raise ValueError("PyramidDrop cache continuation requires its matching prefill state")
        else:
            original = self._pyramiddrop_stats["original_image_tokens"]

        all_hidden = () if output_hidden_states else None
        rotary_key = None
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden += (hidden,)
            past_length = past_key_values.get_seq_length(index)
            # Layer 0 retains the original prompt cache. Deeper stages have
            # shorter caches, and use their own compacted decode position IDs.
            key = (past_length, hidden.shape[1])
            if key != rotary_key:
                positions = torch.arange(past_length, past_length + hidden.shape[1], device=hidden.device)
                ids = positions[None]
                rotary = self.rotary_emb(hidden, ids)
                causal = self._pyramiddrop_causal_mask(hidden, past_length, self.config._attn_implementation)
                rotary_key = key
            layer_output = layer(
                hidden, attention_mask=causal, position_ids=ids, past_key_value=past_key_values,
                output_attentions=False, use_cache=True, cache_position=positions,
                position_embeddings=rotary,
            )
            hidden = layer_output[0]
            if not prefill:
                continue
            before, sequence_before = visual_count, hidden.shape[1]
            if index + 1 in self.pyramiddrop_layers:
                with measure_pruning(self, hidden.device):
                    stage_index = self.pyramiddrop_layers.index(index + 1)
                    ratio = self.pyramiddrop_keep_ratios[stage_index]
                    official_target = math.floor(original * ratio)
                    keep_count = min(before, max(1, official_target))
                    scores = last_prompt_attention(self.layers[index + 1], hidden, rotary)
                    visual_positions = visual.nonzero(as_tuple=False).flatten()
                    visual_scores = scores[0].index_select(0, visual_positions)
                    selected = visual_scores.topk(keep_count).indices.sort().values
                    selected_positions = visual_positions.index_select(0, selected)
                    keep = ~visual
                    keep[selected_positions] = True
                    selection = keep.nonzero(as_tuple=False).flatten()
                    stage_stats = {
                        "after_layer": index + 1, "scoring_layer": index + 2,
                        "target_keep_ratio": ratio, "official_floor_target": official_target,
                        "minimum_token_guard_applied": official_target == 0,
                        "image_tokens_before": before, "image_tokens_after": keep_count,
                        "removed_this_stage": before - keep_count,
                        "sequence_tokens_before": sequence_before,
                        "sequence_tokens_after": sequence_before - before + keep_count,
                        "actual_keep_ratio": keep_count / original,
                        "cumulative_prune_rate": 100.0 * (1.0 - keep_count / original),
                    }
                    self._pyramiddrop_stats["stages"].append(stage_stats)
                    if self.pyramiddrop_capture_stages:
                        self._pyramiddrop_stages.append({
                            "stats": stage_stats,
                            "kept_original_positions": original_positions.index_select(0, selected_positions),
                            "active_original_positions": original_positions.index_select(0, visual_positions),
                            "scores": visual_scores.detach(),
                        })
                        original_positions = original_positions.index_select(0, selection)
                    hidden = hidden.index_select(1, selection)
                    visual = visual.index_select(0, selection)
                    visual_count = keep_count
                rotary_key = None
            self._pyramiddrop_stats["layers"].append(self._pyramiddrop_layer_record(
                index + 1, before, visual_count, sequence_before, hidden.shape[1], original,
            ))
        hidden = self.norm(hidden)
        if prefill:
            self._pyramiddrop_stats.update(
                final_llm_visual_tokens=visual_count,
                final_sequence_tokens=hidden.shape[1],
                actual_keep_ratio=visual_count / original,
                actual_prune_rate=100.0 * (1.0 - visual_count / original),
            )
        if output_hidden_states:
            all_hidden += (hidden,)
        if not return_dict:
            return tuple(value for value in (hidden, past_key_values, all_hidden) if value is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=past_key_values,
            hidden_states=all_hidden, attentions=None,
        )
