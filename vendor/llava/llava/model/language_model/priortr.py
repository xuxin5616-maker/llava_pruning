"""PriorTR OneVision/Qwen2 adaptation; native decoder precision/backend retained.

Algorithm: CodeChildCZJ/PriorTR, e442a4c14af1b333067a6b8310c9111cad922033.
Modified for Qwen2 GQA, independent bounded Q/K rows, Transformers 4.46.1
DynamicCache and actual packed image spans. Reference snapshots and Apache-2.0
license: third_party/priortr. Inference only, single unpadded image, greedy.
"""

import math

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .pruning_timing import measure_pruning


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


@torch.no_grad()
def priortr_scores(layer, hidden, rotary, span, *, query_aggregation="last",
                   head_aggregation="mean", epsilon=1e-10, chunk_size=32):
    """Recompute only required causal rows; never request decoder attentions.

    Prior: first text token immediately AFTER the packed image (not image_newline).
    Posterior: last prompt token, or all tokens from that separator to prompt end.
    Native QK/softmax-cast/head aggregation; FP32 normalization/log as pinned code.
    Question aggregation uses bounded query chunks, never a T x T allocation.
    """
    start, end = span
    if hidden.shape[0] != 1 or not 0 <= start < end < hidden.shape[1] - 1:
        raise ValueError("PriorTR needs one image followed by a separator and instruction tokens")
    if query_aggregation not in {"last", "question"} or head_aggregation not in {"mean", "max"}:
        raise ValueError("Unsupported PriorTR attention aggregation")
    attn = layer.self_attn
    normalized = layer.input_layernorm(hidden)
    length = hidden.shape[1]
    key = attn.k_proj(normalized).view(1, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    cos, sin = (v.to(hidden.device) for v in rotary)
    key = key * cos[:, None] + _rotate_half(key) * sin[:, None]
    key = key.repeat_interleave(attn.num_key_value_groups, dim=1)
    key_positions = torch.arange(length, device=hidden.device)

    def rows(indices):
        query = attn.q_proj(normalized.index_select(1, indices))
        query = query.view(1, len(indices), attn.num_heads, attn.head_dim).transpose(1, 2)
        query = (query * cos[:, None, indices] + _rotate_half(query) * sin[:, None, indices])
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        visible = key_positions[None, :] <= indices[:, None]
        if not torch.isfinite(logits.masked_select(visible[None, None])).all():
            raise FloatingPointError("Non-finite native-dtype PriorTR Q/K scores; model precision is not changed")
        logits = logits.masked_fill(~visible[None, None], float("-inf"))
        return torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)[0, :, :, start:end]

    if query_aggregation == "last":
        attention = rows(torch.tensor([end, length - 1], device=hidden.device))
        prior, posterior = attention[:, 0], attention[:, 1]
    else:
        accumulated = None
        prior = None
        for first in range(end, length, chunk_size):
            indices = torch.arange(first, min(first + chunk_size, length), device=hidden.device)
            attention = rows(indices)
            if prior is None:
                prior = attention[:, 0]
            contribution = attention.float().sum(dim=1)
            accumulated = contribution if accumulated is None else accumulated + contribution
        posterior = (accumulated / (length - end)).to(prior.dtype)
    if head_aggregation == "mean":
        prior, posterior = prior.mean(0), posterior.mean(0)
    else:
        prior, posterior = prior.max(0).values, posterior.max(0).values
    original_dtype = posterior.dtype
    p, q = posterior.float(), prior.float()
    p, q = p / (p.sum() + epsilon), q / (q.sum() + epsilon)
    score = p * torch.log((p + epsilon) / (q + epsilon))
    if not torch.isfinite(score).all():
        raise FloatingPointError("Non-finite PriorTR corrected scores")
    return score.to(original_dtype)


class PriorTRMixin:
    def init_priortr(self):
        self.priortr_enabled = False
        self.priortr_configured = False
        self.reset_priortr_state()

    def reset_priortr_state(self):
        self._priortr_spans = []
        self._priortr_prompt_length = None
        self._priortr_stats = {}
        self._priortr_stage = None
        self._priortr_next_position = None

    def configure_priortr(self, *, enabled, layer=2, keep_ratio=.3,
                          query_aggregation="last", head_aggregation="mean", epsilon=1e-10,
                          capture_visualization=False, capture_attention=False):
        if type(layer) is not int or not 1 <= layer < len(self.layers):
            raise ValueError("PriorTR layer is a 1-based completed block below the final block")
        if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
            raise ValueError("PriorTR keep_ratio must be in (0,1]")
        if query_aggregation not in {"last", "question"} or head_aggregation not in {"mean", "max"}:
            raise ValueError("Unsupported PriorTR aggregation")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("PriorTR epsilon must be positive and finite")
        self.configure_fastv(enabled=False)
        self.priortr_enabled = bool(enabled) and keep_ratio < 1
        self.priortr_configured = True
        self.priortr_layer, self.priortr_keep_ratio = layer, keep_ratio
        self.priortr_query, self.priortr_head = query_aggregation, head_aggregation
        self.priortr_epsilon = epsilon
        self.priortr_capture = bool(capture_visualization or capture_attention)
        self.priortr_capture_attention = bool(capture_attention)
        if self.priortr_enabled and len({p.device for layer in self.layers for p in layer.parameters()}) != 1:
            raise ValueError("PriorTR requires decoder weights on one device")

    def configure_vico(self, **kwargs):
        self.priortr_enabled = self.priortr_configured = False
        return super().configure_vico(**kwargs)

    def configure_pyramiddrop(self, **kwargs):
        self.priortr_enabled = self.priortr_configured = False
        return super().configure_pyramiddrop(**kwargs)

    def set_priortr_image_spans(self, spans, length):
        self._priortr_spans, self._priortr_prompt_length = spans, length

    def validate_priortr_generation(self, kwargs, generation_config):
        if not self.priortr_enabled:
            return
        config = kwargs.get("generation_config", generation_config)
        def option(name, default=None):
            return kwargs.get(name, getattr(config, name, default))
        if (option("do_sample", False) or option("num_beams", 1) != 1
                or option("num_return_sequences", 1) != 1 or option("num_beam_groups", 1) != 1
                or kwargs.get("assistant_model") is not None or option("prompt_lookup_num_tokens")
                or option("penalty_alpha") is not None or option("constraints") is not None
                or option("force_words_ids") is not None or option("dola_layers") is not None):
            raise ValueError("PriorTR supports only batch-1 greedy autoregressive generation")
        if not option("use_cache", True) or option("cache_implementation") not in (None, "dynamic"):
            raise ValueError("PriorTR requires use_cache=True and DynamicCache")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("PriorTR generate requires a fresh image prompt")

    @staticmethod
    def _priortr_layer_record(layer, before, after, seq_before, seq_after, original):
        return dict(layer=layer, image_tokens_in=before, image_tokens_out=after,
                    sequence_tokens_in=seq_before, sequence_tokens_out=seq_after,
                    removed_after_layer=before - after,
                    cumulative_prune_rate=100 * (1 - after / original) if original else 0)

    def get_priortr_stats(self):
        if self.priortr_enabled:
            return dict(self._priortr_stats)
        spans = self._priortr_spans[0] if self._priortr_spans else []
        count = sum(b - a for a, b in spans)
        length = self._priortr_prompt_length
        return dict(mode="disabled_baseline", original_image_tokens=count, final_llm_visual_tokens=count,
                    keep_ratio=1., layers=[self._priortr_layer_record(i + 1, count, count, length, length, count)
                                           for i in range(len(self.layers))])

    def get_priortr_stages(self):
        if self._priortr_stage is None:
            raise RuntimeError("No PriorTR visualization captured")
        item = self._priortr_stage
        start, end = self._priortr_spans[0][0]
        keep = torch.zeros(end - start, dtype=torch.bool)
        keep[item["indices"].detach().cpu()] = True
        stage = dict(item["stats"], mask={"span": [start, end], "keep": keep.tolist()})
        if "scores" in item:
            stage["attention"] = {"span": [start, end], "scores": item["scores"].detach().float().cpu().tolist(),
                                  "score_kind": "priortr"}
        return [stage]

    @torch.no_grad()
    def priortr_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None,
                        return_dict=None, cache_position=None):
        if self.training or output_attentions or (output_attentions is None and self.config.output_attentions):
            raise ValueError("PriorTR is eval-only and never requests decoder attention outputs")
        if getattr(self.config, "use_sliding_window", False):
            raise ValueError("PriorTR does not support sliding-window attention")
        if self.config._attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported PriorTR attention backend")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if use_cache is False or (use_cache is None and not self.config.use_cache):
            raise ValueError("PriorTR requires use_cache=True")
        if past_key_values is not None and type(past_key_values) is not DynamicCache:
            raise ValueError("PriorTR requires DynamicCache")
        cache = past_key_values if past_key_values is not None else DynamicCache()
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.shape[0] != 1:
            raise ValueError("PriorTR supports batch size one")
        if attention_mask is not None and (attention_mask.ndim != 2 or attention_mask.shape[0] != 1
                                           or not attention_mask.bool().all()):
            raise ValueError("PriorTR requires an unpadded prompt")
        prefill = cache.get_seq_length() == 0
        if prefill:
            if len(self._priortr_spans) != 1 or len(self._priortr_spans[0]) != 1:
                raise ValueError("PriorTR requires exactly one packed image span")
            start, end = self._priortr_spans[0][0]
            if not 0 <= start < end < hidden.shape[1] - 1:
                raise ValueError("PriorTR image must be followed by separator and instruction text")
            ids = torch.arange(hidden.shape[1], device=hidden.device)[None]
            if position_ids is not None and not torch.equal(position_ids.to(hidden.device), ids):
                raise ValueError("PriorTR requires contiguous zero-based initial positions")
            if attention_mask is not None and attention_mask.shape[-1] != hidden.shape[1]:
                raise ValueError("PriorTR initial mask must match packed prompt length")
            original = current = end - start
            original_length = hidden.shape[1]
            self._priortr_next_position = original_length
            self._priortr_stats = dict(mode="physical_drop", original_image_tokens=original,
                                      original_prompt_tokens=original_length, layers=[],
                                      pruning_layer=self.priortr_layer, position_policy="preserve_original",
                                      attention_source="independent_causal_qk", score="P*log((P+eps)/(Q+eps))",
                                      query_aggregation=self.priortr_query, head_aggregation=self.priortr_head,
                                      epsilon=self.priortr_epsilon, budget_scope="all packed visual tokens including newlines",
                                      rounding="floor", min_tokens=1)
            self._priortr_stage = None
        else:
            if hidden.shape[1] != 1 or self._priortr_next_position is None:
                raise ValueError("PriorTR decode needs one token and matching prefill state")
            ids = torch.tensor([[self._priortr_next_position]], device=hidden.device)
            self._priortr_next_position += 1
        all_hidden = () if (self.config.output_hidden_states if output_hidden_states is None else output_hidden_states) else None
        for index, layer in enumerate(self.layers):
            if all_hidden is not None:
                all_hidden += (hidden,)
            physical_past = cache.get_seq_length(index)
            physical_positions = torch.arange(physical_past, physical_past + hidden.shape[1], device=hidden.device)
            rotary = self.rotary_emb(hidden, ids)
            causal = self._pyramiddrop_causal_mask(hidden, physical_past, self.config._attn_implementation)
            ranking = None
            if prefill and index + 1 == self.priortr_layer:
                with measure_pruning(self, hidden.device):
                    ranking = priortr_scores(layer, hidden, rotary, (start, end),
                                             query_aggregation=self.priortr_query, head_aggregation=self.priortr_head,
                                             epsilon=self.priortr_epsilon)
            hidden = layer(hidden, attention_mask=causal, position_ids=ids, past_key_value=cache,
                           output_attentions=False, use_cache=True, cache_position=physical_positions,
                           position_embeddings=rotary)[0]
            if not prefill:
                continue
            before, seq_before = current, hidden.shape[1]
            if ranking is not None:
                with measure_pruning(self, hidden.device):
                    target = math.floor(original * self.priortr_keep_ratio)
                    retained = min(original, max(1, target))
                    self._priortr_stats.update(official_floor_target=target,
                                               minimum_token_guard_applied=target == 0)
                    selected = ranking.topk(retained).indices.sort().values
                    selection = torch.cat((torch.arange(start, device=hidden.device), selected + start,
                                           torch.arange(end, hidden.shape[1], device=hidden.device)))
                    hidden = hidden.index_select(1, selection).contiguous()
                    ids = ids.index_select(1, selection)
                    # Compact ALL caches already written, including shallow layers.
                    for cached_layer in range(len(cache.key_cache)):
                        cache.key_cache[cached_layer] = cache.key_cache[cached_layer].index_select(2, selection).contiguous()
                        cache.value_cache[cached_layer] = cache.value_cache[cached_layer].index_select(2, selection).contiguous()
                    cache._seen_tokens = hidden.shape[1]  # physical length; logical positions tracked separately
                    current = retained
                    stage_stats = dict(after_layer=index + 1, scoring_layer=index + 1,
                                       target_keep_ratio=self.priortr_keep_ratio, image_tokens_before=before,
                                       image_tokens_after=retained, removed_this_stage=before - retained,
                                       cumulative_prune_rate=100 * (1 - retained / original))
                    if self.priortr_capture:
                        self._priortr_stage = {"stats": stage_stats, "indices": selected.detach()}
                        if self.priortr_capture_attention:
                            self._priortr_stage["scores"] = ranking.detach()
            self._priortr_stats["layers"].append(
                self._priortr_layer_record(index + 1, before, current, seq_before, hidden.shape[1], original))
        hidden = self.norm(hidden)
        if prefill:
            self._priortr_stats.update(final_llm_visual_tokens=current, final_sequence_tokens=hidden.shape[1],
                                      actual_prune_rate=100 * (1 - current / original))
        if all_hidden is not None:
            all_hidden += (hidden,)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not return_dict:
            return tuple(v for v in (hidden, cache, all_hidden) if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache, hidden_states=all_hidden)
