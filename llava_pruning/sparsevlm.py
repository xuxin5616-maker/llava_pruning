"""SparseVLM paper rules on OneVision/Qwen2, scoped to one generation."""

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import rotate_half


PAPER = "https://arxiv.org/html/2410.04417#S3"
REFERENCE = "https://github.com/Gumpest/SparseVLMs/tree/v1.5/llava/model/language_model"
# v1.5 prunes AFTER the blocks indexed (2, 6, 15) in its zero-based loop.
AFTER_BLOCKS = (3, 7, 16)


def text_raters(hidden, start, end):
    """Paper Eqs. 5–6: normalize over text, then average over the image."""
    visual = hidden[0, start:end].float()
    text = hidden[0, end:].float()
    relevance = (visual @ text.T).softmax(dim=1).mean(dim=0)
    if not torch.isfinite(relevance).all():
        raise FloatingPointError("Non-finite SparseVLM text relevance")
    return (relevance >= relevance.mean()).nonzero(as_tuple=True)[0]


def rater_queries(layer, hidden, rotary, indices):
    """Native Q projection and RoPE for the selected rows only."""
    attn = layer.self_attn
    x = layer.input_layernorm(hidden.index_select(1, indices))
    query = attn.q_proj(x).view(1, len(indices), attn.num_heads, attn.head_dim).transpose(1, 2)
    cos, sin = (tensor.index_select(1, indices).unsqueeze(1) for tensor in rotary)
    return query * cos + rotate_half(query) * sin


def rater_attention(query, key, indices, start, end, chunk_size=16):
    """Mean-head causal probabilities, retaining only the visual key columns."""
    kv_heads, length, width = key.shape[1:]
    key = key[0].float()
    queries = query[0].float().reshape(kv_heads, -1, len(indices), width)
    positions = torch.arange(length, device=key.device)
    chunks = []
    for offset in range(0, len(indices), chunk_size):
        rows = indices[offset:offset + chunk_size]
        logits = torch.einsum("hgrd,hkd->hgrk", queries[:, :, offset:offset + len(rows)], key)
        logits.div_(math.sqrt(width))
        logits.masked_fill_(positions[None, :] > rows[:, None], float("-inf"))
        probs = logits.softmax(dim=-1)
        chunks.append(probs[..., start:end].mean(dim=(0, 1)))
    return torch.cat(chunks, dim=0)


def recycle_tokens(tokens, centers_count):
    """Density-peak centers, cosine assignment, and sum as in paper Eq. 10."""
    x = tokens.float()
    n, width = x.shape
    if centers_count == 1:
        return x.sum(dim=0, keepdim=True).to(tokens.dtype)

    # Matrix multiplication avoids an [N, N, hidden_size] broadcast.
    distance = torch.cdist(x, x, compute_mode="use_mm_for_euclid_dist") / math.sqrt(width)
    distance.fill_diagonal_(0)
    density = (-distance.topk(centers_count, largest=False).values.square().mean(dim=1)).exp()
    index = torch.arange(n, device=x.device)
    denser = ((density[None, :] > density[:, None])
              | ((density[None, :] == density[:, None]) & (index[None, :] < index[:, None])))
    separation = distance.masked_fill(~denser, float("inf")).amin(dim=1)
    separation = torch.where(torch.isfinite(separation), separation, distance.max())
    centers = (density * separation).argsort(descending=True, stable=True)[:centers_count]
    unit = F.normalize(x, dim=-1)
    assignment = (unit @ unit[centers].T).argmax(dim=1)
    assignment[centers] = torch.arange(centers_count, device=x.device)
    merged = x.new_zeros((centers_count, width))
    return merged.index_add_(0, assignment, x).to(tokens.dtype)


def sparsify(hidden, attention, start, end, scale):
    """Apply Eq. 7 and put recycled tokens after retained visual tokens."""
    visual_count = end - start
    rank = int(torch.linalg.matrix_rank(attention.T.contiguous()).item())
    dropped = min(visual_count - 1, math.floor(scale * (visual_count - rank)))
    if dropped <= 0:
        return hidden, dict(rank=rank, dropped=0, recycled=0, recycle_candidates=0)
    keep = visual_count - dropped
    ranking = attention.mean(dim=0).argsort(descending=True, stable=True)
    candidates = min(dropped, math.floor(.3 * dropped) + 1)
    centers = min(candidates, candidates // 10 + 1)
    visual = hidden[0, start:end]
    retained = visual.index_select(0, ranking[:keep].sort().values)
    merged = recycle_tokens(visual.index_select(0, ranking[keep:keep + candidates]), centers)
    result = torch.cat((hidden[:, :start], retained[None], merged[None], hidden[:, end:]), dim=1)
    return result, dict(rank=rank, dropped=dropped, recycled=centers, recycle_candidates=candidates)


class SparseVLMMethod:
    def __init__(self, core, scale, after_blocks=AFTER_BLOCKS):
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("SparseVLM scale must be in (0, 1]")
        if (not after_blocks or tuple(sorted(set(after_blocks))) != tuple(after_blocks)
                or not all(type(block) is int and 0 < block < len(core.layers) for block in after_blocks)):
            raise ValueError("SparseVLM pruning stages must be distinct completed non-final decoder blocks")
        self.core = core
        self.scale = scale
        self.after_blocks = tuple(after_blocks)
        self.details = None

    def configure(self, core, rate, **capture):
        if core is not self.core or any(capture.values()):
            raise ValueError("SparseVLM benchmark expects one model and disables visualization")
        if (core.config.image_aspect_ratio != "anyres_max_9"
                or core.config.mm_patch_merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}
                or getattr(core.config, "image_token_order", "base_first") != "base_first"):
            raise ValueError("SparseVLM expects native Base-first AnyRes max9 packing")
        core.configure_fastv(enabled=False)
        self.details = None

    def prepare_inputs(self, image_size, tokens):
        from llava.constants import IMAGE_TOKEN_INDEX
        from .visa_comparison_methods import native_counts

        markers = (tokens[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if tokens.shape[0] != 1 or len(markers) != 1 or int(markers[0]) >= tokens.shape[1] - 1:
            raise ValueError("SparseVLM requires one image followed by text")
        self.start = int(markers[0])
        self.text_count = tokens.shape[1] - 1
        _, counts = native_counts(self.core, image_size)
        self.original = sum(counts)

    @contextmanager
    def inference_context(self, model):
        """Use the adapter for this model call and restore the native forward."""
        core = self.core
        if model.get_model() is not core:
            raise ValueError("SparseVLM context received a different model")
        local = "forward" in core.__dict__
        previous = core.forward
        core.forward = self.decoder_forward
        try:
            yield
        finally:
            if local:
                core.forward = previous
            else:
                del core.forward

    @torch.no_grad()
    def decoder_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None, return_dict=None,
                        cache_position=None):
        core = self.core
        implementation = core.config._attn_implementation
        if (core.training or getattr(core.config, "use_sliding_window", False)
                or implementation not in {"eager", "sdpa", "flash_attention_2"}
                or (core.config.output_attentions if output_attentions is None else output_attentions)
                or (core.config.output_hidden_states if output_hidden_states is None else output_hidden_states)):
            raise ValueError("SparseVLM supports eval without sliding windows or hidden/attention dumps")
        if not (core.config.use_cache if use_cache is None else use_cache):
            raise ValueError("SparseVLM requires cached generation")
        if past_key_values is not None and type(past_key_values) is not DynamicCache:
            raise ValueError("SparseVLM requires Transformers DynamicCache")
        cache = DynamicCache() if past_key_values is None else past_key_values
        hidden = core.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if (hidden.shape[0] != 1 or (attention_mask is not None
                and (attention_mask.ndim != 2 or attention_mask.shape[0] != 1 or not attention_mask.bool().all()))):
            raise ValueError("SparseVLM needs a single unpadded prompt")

        prefill = cache.get_seq_length(0) == 0
        if prefill:
            if hidden.shape[1] != self.text_count + self.original:
                raise ValueError("SparseVLM packed image count disagrees with native AnyRes geometry")
            start, end = self.start, self.start + self.original
            raters = text_raters(hidden, start, end)
            layer_counts, stages = [], []
        elif self.details is None or hidden.shape[1] != 1:
            raise ValueError("SparseVLM decode requires its matching prefill and one new token")

        rotary_shape = None
        for index, layer in enumerate(core.layers):
            past = cache.get_seq_length(index)
            shape = (past, hidden.shape[1])
            if shape != rotary_shape:
                positions = torch.arange(past, past + hidden.shape[1], device=hidden.device)
                ids = positions[None]
                rotary = core.rotary_emb(hidden, ids)
                causal = core._pyramiddrop_causal_mask(hidden, past, implementation)
                rotary_shape = shape
            should_prune = prefill and index + 1 in self.after_blocks
            if prefill:
                layer_counts.append(end - start)
            if should_prune:
                indices = end + raters
                query = rater_queries(layer, hidden, rotary, indices)
            hidden = layer(hidden, attention_mask=causal, position_ids=ids,
                           past_key_value=cache, use_cache=True, output_attentions=False,
                           cache_position=positions, position_embeddings=rotary)[0]
            if should_prune:
                # The block's K cache is already rotated. Earlier layers retain
                # the tokens they saw; later layers cache their compact inputs.
                scores = rater_attention(query, cache.key_cache[index], indices, start, end)
                before = end - start
                hidden, stage = sparsify(hidden, scores, start, end, self.scale)
                end = hidden.shape[1] - (self.text_count - start)
                stages.append(dict(after_block=index + 1, visual_in=before, visual_out=end - start, **stage))
                del query, scores

        hidden = core.norm(hidden)
        if prefill:
            self.details = dict(
                original_image_tokens=self.original,
                final_layer_visual_tokens=layer_counts[-1],
                observed_layer_token_sum=sum(layer_counts),
                actual_average_prune_rate=100 * (1 - sum(layer_counts) / (len(layer_counts) * self.original)),
                layer_visual_tokens=layer_counts,
                stages=stages,
                raters=len(raters),
            )
        if not (core.config.use_return_dict if return_dict is None else return_dict):
            return hidden, cache
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache)

    def stats(self, core, rate):
        if self.details is None:
            raise RuntimeError("SparseVLM prefill did not execute")
        return dict(self.details)
