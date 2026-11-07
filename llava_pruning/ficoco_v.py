"""FiCoCo-V paper-based, non-CLS SigLIP adaptation; no LLM modifications.

Reference: arXiv:2411.17686v4, Algorithm1 and Appendix10.2/9.
Attention axes are [batch, query, key]; vision redundancy is attention-IN
(mean over queries), as clarified in Appendix10.3. Scores are computed from
captured Q/K independently of the native attention output. See docs/ficoco_v.md
for differences from the released CLIP-only implementation and sparse packing.
"""

from contextlib import contextmanager
from decimal import Decimal, ROUND_FLOOR
from itertools import groupby
import math
import types


PAPER = "https://arxiv.org/html/2411.17686v4"
UPSTREAM = "https://github.com/kawhiiiileo/FiCoCo"
VERSION = "1.0-paper-siglip"


def progressive_keeps(original, target, layers):
    if any(type(v) is not int for v in (original,target,layers)) or not 1<=target<=original or layers<1:
        raise ValueError("Invalid progressive FiCoCo-V budget")
    quotient,remainder=divmod(original-target,layers)
    return [original-(index+1)*quotient-min(index+1,remainder) for index in range(layers)]


def packed_plan(original, views, side, grid_height, rate, final_newline=False):
    if (type(original) is not int or original<1 or type(views) is not int or views<2 or
            type(side) is not int or side<1 or type(grid_height) is not int or grid_height<1 or
            type(rate) not in (int,float) or not math.isfinite(rate) or not 0<=rate<100):
        raise ValueError("Invalid FiCoCo-V packed budget")
    if rate==0: return original,grid_height*side+int(final_newline),[side*side]*views
    target=int((Decimal(original)*(1-Decimal(str(rate))/100)).to_integral_value(rounding=ROUND_FLOOR))
    protected=grid_height*side+int(final_newline)
    content=target-protected
    quotient,remainder=divmod(content,views)
    keeps=[quotient+int(i<remainder) for i in range(views)]
    if min(keeps)<1 or max(keeps)>side*side:
        raise ValueError(f"FiCoCo-V {rate}% packed target={target} cannot preserve {protected} row newlines and one token/view")
    return target,protected,keeps


def independent_attention(query, key, heads, scale, head_chunk=4):
    """Return FP32 mean-head [B,Q,K] probabilities and mean-head keys."""
    import torch
    if query.ndim!=3 or key.shape!=query.shape or query.shape[-1]%heads or head_chunk<1:
        raise ValueError("Invalid FiCoCo-V Q/K dimensions")
    with torch.autocast(device_type=query.device.type,enabled=False):
        batch,count,channels=query.shape
        q=query.float().view(batch,count,heads,channels//heads).transpose(1,2)
        k=key.float().view(batch,count,heads,channels//heads).transpose(1,2)
        probability=q.new_zeros(batch,count,count)
        for start in range(0,heads,head_chunk):
            logits=q[:,start:start+head_chunk]@k[:,start:start+head_chunk].transpose(-1,-2)*scale
            probability+=logits.softmax(-1).sum(1)
        return probability/heads,k.mean(1)


def local_penalty(scores, anchors, side, coefficient=2.):
    """Boost ONE maximum/view/2x2 window; retain original sparse coordinates.

    The odd last row/column of SigLIP27x27 is a partial window, not omitted.
    Deterministic ties select the first surviving token in spatial order.
    """
    import torch
    if scores.shape!=anchors.shape or scores.ndim!=2 or side<1:
        raise ValueError("Invalid FiCoCo-V sparse score grid")
    width=(side+1)//2
    windows=(anchors//side//2)*width+(anchors%side//2)
    maxima=scores.new_full((scores.shape[0],width*width),-torch.inf)
    maxima.scatter_reduce_(1,windows,scores,reduce="amax",include_self=True)
    positions=torch.arange(scores.shape[1],device=scores.device).expand_as(anchors)
    first=anchors.new_full(maxima.shape,scores.shape[1])
    first.scatter_reduce_(1,windows,torch.where(scores==maxima.gather(1,windows),positions,scores.shape[1]),
                          reduce="amin",include_self=True)
    selected=positions==first.gather(1,windows)
    raised=torch.where(scores>=0,scores*coefficient,scores/coefficient)
    return torch.where(selected,raised,scores)


def compress(features, probability, keys, anchors, target, side, *, balance=.35, quantile=.998):
    """Filter -> many-to-many correlate -> paper Eq2 self-preserving merge.

    L2-normalize the two score terms as in the released Filter, then use the
    paper's lambda*vision - (1-lambda)*semantic coefficient/sign convention.
    All SigLIP tokens are patches, including index0. Fusion keeps feature dtype.
    """
    import torch
    from torch.nn import functional as F
    batch,count,channels=features.shape
    if (probability.shape!=(batch,count,count) or keys.shape[:2]!=(batch,count) or
            anchors.shape!=(batch,count) or type(target) is not int or not 1<=target<=count):
        raise ValueError("Invalid FiCoCo-V compression shapes/budget")
    if target==count: return features,anchors
    with torch.autocast(device_type=features.device.type,enabled=False):
        received=probability.float().mean(-2)
        equivalent=-F.cosine_similarity(keys.float(),keys.float().mean(1,keepdim=True),dim=-1,eps=1e-6)
        redundancy=balance*F.normalize(received,dim=-1)-(1-balance)*F.normalize(equivalent,dim=-1)
        redundancy=local_penalty(redundancy,anchors,side)
        if not torch.isfinite(redundancy).all(): raise FloatingPointError("Non-finite FiCoCo-V redundancy")
        order=torch.argsort(redundancy,dim=-1,descending=True,stable=True)
        discarded=order[:,:count-target]
        retained=order[:,count-target:].sort(-1).values
        correlation=probability.gather(1,discarded[:,:,None].expand(-1,-1,count))
        correlation=correlation.gather(2,retained[:,None,:].expand(-1,count-target,-1)).float()
        threshold=torch.quantile(correlation,quantile,dim=-1,keepdim=True)
        weights=torch.where(correlation>=threshold,correlation.clamp_min(torch.finfo(torch.float32).tiny),0.)
        weights=weights/weights.sum(-1,keepdim=True)
    selected=features.gather(1,retained[:,:,None].expand(-1,-1,channels))
    sources=features.gather(1,discarded[:,:,None].expand(-1,-1,channels))
    # Native model/feature arithmetic stays FP16 on the server. The additional
    # decision computation above is FP32; it never changes model weights/dtype.
    weights=weights.to(features.dtype)
    with torch.autocast(device_type=features.device.type,enabled=False):
        result=(selected+weights.transpose(1,2)@sources)/(1+weights.sum(1)[:,:,None])
    return result,anchors.gather(1,retained)


def pack_sparse(features, indices, grid, side, newline, final_newline=False):
    """Base then original AnyRes raster anchors and one marker per raw row."""
    import torch
    width,height=grid
    if len(features)!=1+width*height or len(indices)!=len(features):
        raise ValueError("FiCoCo-V packing view count mismatch")
    positions=[];values=[]
    for tile,(hidden,anchor) in enumerate(zip(features[1:],indices[1:])):
        y,x=divmod(tile,width)
        positions.append((y*side+anchor//side)*(width*side+1)+x*side+anchor%side)
        values.append(hidden)
    positions.append(torch.arange(height*side,device=newline.device)*(width*side+1)+width*side)
    values.append(newline[None].expand(height*side,-1))
    raster=torch.cat(values)[torch.argsort(torch.cat(positions))]
    pieces=[features[0],raster]
    if final_newline: pieces.append(newline[None])
    return torch.cat(pieces)


class FiCoCoVAdapter:
    def __init__(self, core, *, start_layer=12, balance=.35, quantile=.998):
        self.core,self.tower=core,core.get_vision_tower()
        self.start_layer,self.balance,self.quantile=start_layer,balance,quantile
        self.stats={}

    def prepare_inputs(self, size, tokens):
        from .visa_comparison_methods import native_counts
        self.size=tuple(size)
        self.grid,counts=native_counts(self.core,size)
        self.original=sum(counts)
        self.final_newline=self.core.config.mm_patch_merge_type.endswith("_add_newl")
        self.side=self.tower.num_patches_per_side
        self.target,self.protected,self.keeps=packed_plan(self.original,1+self.grid[0]*self.grid[1],
            self.side,self.grid[1],self.rate,self.final_newline)

    @contextmanager
    def compression_context(self, keeps, batch):
        import torch
        blocks=self.tower.vision_tower.vision_model.encoder.layers
        state=dict(anchors=torch.arange(self.side*self.side,device=self.core.image_newline.device).expand(batch,-1))
        plan=progressive_keeps(self.side*self.side,keeps,len(blocks)-self.start_layer+1)
        saved=[];hooks=[];captured={}
        def capture(index,kind):
            def record(module,args,output): captured[index,kind]=output
            return record
        def replacement(index):
            def forward(layer,hidden_states,attention_mask=None,output_attentions=False):
                if attention_mask is not None: raise ValueError("FiCoCo-V expects unmasked SigLIP views")
                residual=hidden_states
                attended,attn_weights=layer.self_attn(layer.layer_norm1(hidden_states),
                    attention_mask=attention_mask,output_attentions=output_attentions)
                hidden_states=residual+attended
                query=captured.pop((index,"q"));key=captured.pop((index,"k"))
                if not output_attentions: attn_weights=None
                target=plan[index-(self.start_layer-1)]
                if target<hidden_states.shape[1]:
                    probability,mean_key=independent_attention(query,key,layer.self_attn.num_heads,layer.self_attn.scale)
                    hidden_states,state["anchors"]=compress(hidden_states,probability,mean_key,state["anchors"],
                        target,self.side,balance=self.balance,quantile=self.quantile)
                hidden_states=hidden_states+layer.mlp(layer.layer_norm2(hidden_states))
                return (hidden_states,attn_weights) if output_attentions else (hidden_states,)
            return forward
        try:
            for index,block in enumerate(blocks):
                if index<self.start_layer-1: continue
                saved.append((block,"forward" in block.__dict__,block.__dict__.get("forward")))
                block.forward=types.MethodType(replacement(index),block)
                hooks.extend((block.self_attn.q_proj.register_forward_hook(capture(index,"q")),
                              block.self_attn.k_proj.register_forward_hook(capture(index,"k"))))
            yield state
        finally:
            for hook in hooks: hook.remove()
            for block,present,value in saved:
                if present: block.forward=value
                else: delattr(block,"forward")
            captured.clear()

    def encode_images(self, images):
        import torch
        from runVisionZip import _match_feature_memory_order
        if images.shape[0]!=len(self.keeps): raise ValueError("FiCoCo-V view count differs from AnyRes geometry")
        self.indices=[]
        if self.rate==0:
            features=self.tower(images)
            self.indices=list(torch.arange(self.side*self.side,device=images.device).expand(images.shape[0],-1).unbind())
            self.stats=dict(full_grid_identity=True)
            return self.core.mm_projector(features)
        outputs=[]
        vision=self.tower.vision_tower
        # At most two adjacent quota groups. Native vision forward is retained,
        # including its post-norm/head and hidden-state lifetime; only the
        # active block forwards insert compression between attention and MLP.
        for keep,group in groupby(enumerate(self.keeps),key=lambda item:item[1]):
            positions=[i for i,_ in group];start,end=positions[0],positions[-1]+1
            with self.compression_context(keep,end-start) as state:
                result=vision(images[start:end].to(device=self.tower.device,dtype=self.tower.dtype),output_hidden_states=True)
                features=result.hidden_states[-1].to(images.dtype)
            if features.shape[1]!=keep: raise RuntimeError("FiCoCo-V progressive encoder missed its exact quota")
            # Mirror the encoder's channel-major feature layout when storing
            # compact projector inputs, as in the current VisionZip adapter.
            reference=result.hidden_states[0]
            features=_match_feature_memory_order(features,reference)
            projected=self.core.mm_projector(features)
            outputs.extend(projected.unbind());self.indices.extend(state["anchors"].unbind())
        self.stats=dict(full_grid_identity=False)
        return outputs

    def pack(self, features, image_sizes, modalities):
        if len(features)!=1 or tuple(image_sizes[0])!=self.size or list(modalities)!=["image"]:
            raise ValueError("FiCoCo-V expects one unchanged source image")
        if self.rate==0:
            import torch
            from runVisionZip import TokenDecisions,pack_full_views
            count=self.side*self.side
            decisions=TokenDecisions(count,count,torch.stack(self.indices))
            packed,self.protected=pack_full_views(features[0],decisions,self.grid,self.size,self.core.image_newline,
                                    self.core.config.mm_patch_merge_type)
        else:
            packed=pack_sparse(features[0],self.indices,self.grid,self.side,self.core.image_newline,self.final_newline)
        if len(packed)!=self.target: raise RuntimeError("FiCoCo-V sparse packing changed the exact LLM budget")
        self.stats.update(original_image_tokens=self.original,final_llm_visual_tokens=len(packed),
            actual_average_prune_rate=100*(1-len(packed)/self.original),
            raw_view_keeps=self.keeps,protected_newlines=self.protected,
            vision_start_layer=self.start_layer,vision_depth=len(self.tower.vision_tower.vision_model.encoder.layers),
            packing="native full-grid at zero; sparse Base-first spatial anchors + raw-row newlines otherwise")
        return [packed]


class FiCoCoVMethod:
    name,visualize_rates="ficoco_v",frozenset()

    def __init__(self, core, rate, *, start_layer=12, balance=.35, quantile=.998):
        blocks=core.get_vision_tower().vision_tower.vision_model.encoder.layers
        if (type(start_layer) is not int or not 1<=start_layer<=len(blocks) or
                not 0<=balance<=1 or not 0<=quantile<=1):
            raise ValueError("Invalid FiCoCo-V vision layers or score hyperparameters")
        self.adapter=FiCoCoVAdapter(core,start_layer=start_layer,balance=balance,quantile=quantile)
        self.rate=rate

    def configure(self, core, rate, **capture):
        if rate!=self.rate or any(capture.values()): raise ValueError("FiCoCo-V benchmark expects its configured rate and no PNG")
        if (core.training or core.config.image_aspect_ratio!="anyres_max_9" or
                core.config.mm_patch_merge_type not in {"spatial_unpad","spatial_unpad_add_newl"} or
                getattr(core.config,"image_token_order","base_first")!="base_first"):
            raise ValueError("FiCoCo-V requires native AnyRes spatial_unpad, Base first and model.eval()")
        core.configure_fastv(enabled=False)
        self.adapter.rate=rate;self.adapter.stats={}

    def prepare_inputs(self, size, tokens): self.adapter.prepare_inputs(size,tokens)

    def stats(self, core, rate): return dict(self.adapter.stats)

    @contextmanager
    def inference_context(self, model):
        from runVisionZip import visionzip_generation
        with visionzip_generation(model,self.adapter): yield
