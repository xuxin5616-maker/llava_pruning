"""Budget-only adapters for the matched VisA method comparison.

The old standalone entry points/configurations are untouched. FastV retains
its mask implementation; VisionZip retains its scores, merging and sparse
dominant/contextual packing, with an exact native-packed input budget.
"""

from contextlib import contextmanager
from decimal import Decimal, ROUND_FLOOR
from itertools import groupby


def single_drop_keep(original, rate, depth=28, boundary=0):
    if not 0 <= boundary < depth or not 0 <= rate < 100:
        raise ValueError("Invalid layer-average budget")
    n,d,k,r = Decimal(original),Decimal(depth),Decimal(boundary),Decimal(str(rate))/100
    target = int(((d*n*(1-r)-k*n)/(d-k)).to_integral_value(rounding=ROUND_FLOOR))
    if not 1 <= target <= original:
        raise ValueError("Layer-average target is unreachable")
    return target


def native_counts(core, original_size):
    from llava.mm_utils import get_anyres_image_grid_shape
    from llava.model.language_model.attention_predictor_selection import native_anyres_counts
    tower = core.get_vision_tower()
    grid = get_anyres_image_grid_shape(original_size,core.config.image_grid_pinpoints,tower.image_size)
    counts = native_anyres_counts(tower.num_patches_per_side,grid,original_size,
        final_newline=core.config.mm_patch_merge_type.endswith("_add_newl"))
    return grid,counts


class AverageFastV:
    name,layer,visualize_rates = "fastv",2,frozenset()

    def __init__(self, rate, depth=28):
        self.rate,self.depth = rate,depth

    def configure(self, core, rate, **capture):
        if rate != self.rate or any(capture.values()): raise ValueError("Wrong FastV benchmark options")
        self.core = core
        core.configure_fastv(enabled=True,layer=self.layer,keep_ratio=1-rate*self.depth/(self.depth-self.layer)/100,
                             capture_attention=False,preserve_image_newline=True,budget_mode="joint")

    def prepare_inputs(self, original_size, tokens):
        _,counts = native_counts(self.core,original_size)
        self.original = sum(counts)
        self.target = single_drop_keep(self.original,self.rate,self.depth,self.layer)
        if self.target < 2: raise ValueError("FastV needs one spatial token and its protected final marker")
        # Native FastV ceils the candidate quota and protects its last marker.
        # An interior point reaches the exact packed target without FP edges.
        self.core.fastv_keep_ratio = (self.target-1-.25)/(self.original-1)

    def stats(self, core, rate):
        result = dict(core.get_fastv_stats())
        original = sum(row["image_tokens"] for row in result["batch"])
        retained = sum(row["kept_image_tokens"] for row in result["batch"])
        if (original,retained) != (self.original,self.target):
            raise RuntimeError("FastV mask count differs from the exact layer-average budget")
        result.update(original_image_tokens=original,retained_visible_image_tokens=retained,
            actual_average_prune_rate=100*(self.depth-self.layer)/self.depth*(1-retained/original),
            physical_average_prune_rate=0.,budget_kind="effective_attention_mask",pruning_layer=self.layer)
        return result


def visionzip_view_keeps(original, raw_views, raw_rows, rate, final_newline=False):
    target = single_drop_keep(original,rate)
    protected = raw_rows+int(final_newline)
    content = target-protected
    quotient,remainder = divmod(content,raw_views)
    keeps = [quotient+int(i<remainder) for i in range(raw_views)]
    if min(keeps)<1 or max(keeps)>729:
        raise ValueError("VisionZip target cannot preserve raw row newlines and a valid per-view budget")
    return target,protected,keeps


def pack_variable_visionzip(projected, dominant_indices, dominant_counts, grid, newline, final_newline=False):
    """Official sparse Base / raster dominants / tile contextual order; raw newlines."""
    import torch
    width,height = grid
    if len(projected) != 1+width*height: raise ValueError("VisionZip crop count changed")
    positions,dominants,contexts = [],[],[]
    for tile,(features,indices,dominant) in enumerate(zip(projected[1:],dominant_indices[1:],dominant_counts[1:])):
        row,col = divmod(tile,width)
        location = (row*27+indices//27)*(width*27)+col*27+indices%27
        positions.append((location//(width*27))*(width*27+1)+location%(width*27))
        dominants.append(features[:dominant]);contexts.append(features[dominant:])
    rows = height*27
    positions.append(torch.arange(rows,device=newline.device)*(width*27+1)+width*27)
    dominants.append(newline[None].expand(rows,-1))
    order = torch.argsort(torch.cat(positions))
    pieces = [projected[0],torch.cat(dominants)[order],torch.cat(contexts)]
    if final_newline: pieces.append(newline[None])
    return torch.cat(pieces)


class AverageVisionZip:
    name,visualize_rates = "visionzip",frozenset()

    def __init__(self, core, rate):
        self.core,self.rate = core,rate
        self.tower = core.get_vision_tower()
        self.stats = {} # Native VisionZip preparation reads this adapter field.

    def configure(self, core, rate, **capture):
        if rate!=self.rate or any(capture.values()): raise ValueError("Wrong VisionZip benchmark options")
        core.configure_fastv(enabled=False)
        blocks = self.tower.vision_tower.vision_model.encoder.layers
        if (len(blocks)!=26 or self.tower.num_patches!=729 or
                core.config.image_aspect_ratio!="anyres_max_9" or
                core.config.mm_patch_merge_type not in {"spatial_unpad","spatial_unpad_add_newl"} or
                getattr(core.config,"image_token_order","base_first")!="base_first"):
            raise ValueError("VisionZip comparison requires the current native SigLIP/AnyRes spatial_unpad layout")
        self.stats = {}

    def prepare_inputs(self, original_size, tokens):
        self.original_size = tuple(original_size)
        self.grid,counts = native_counts(self.core,original_size)
        self.original = sum(counts)
        self.final_newline = self.core.config.mm_patch_merge_type.endswith("_add_newl")
        self.target,self.protected,self.keeps = visionzip_view_keeps(self.original,1+self.grid[0]*self.grid[1],
            self.grid[1]*27,self.rate,self.final_newline)

    def encode_images(self, images):
        import torch
        import runVisionZip as visionzip
        attention = self.tower.vision_tower.vision_model.encoder.layers[-1].self_attn
        captured = {}
        def key(module,args,output):
            b,n,_ = output.shape
            captured["metric"] = output.view(b,n,attention.num_heads,attention.head_dim).permute(0,2,1,3).mean(1)
        def score(module,args,output):
            if output[1] is None: raise RuntimeError("Missing native SigLIP received attention")
            captured["scores"] = output[1].mean(1).mean(1)
        handles = [attention.k_proj.register_forward_hook(key),attention.register_forward_hook(score)]
        try: features = self.tower(images)
        finally:
            for handle in handles: handle.remove()
        if features.shape[:2] != (len(self.keeps),729):
            raise ValueError("VisionZip requires the native 729-token SigLIP views")
        projected,indices,dominants = [],[],[]
        # At most two consecutive quota groups, differing by one token. Keep
        # the official encoder-axis storage through compression/projector.
        for keep,items in groupby(enumerate(self.keeps),key=lambda pair:pair[1]):
            views = [index for index,_ in items];start,end = views[0],views[-1]+1
            dominant = keep if keep==729 else round(keep*visionzip.DOMINANT_RATIO)
            zipped,decisions = visionzip.compress_tokens(features[start:end],captured["scores"][start:end],
                captured["metric"][start:end],dominant,keep-dominant,capture=False)
            encoded = self.core.mm_projector(zipped)
            projected.extend(encoded.unbind(0));indices.extend(decisions.dominant_indices.unbind(0))
            dominants.extend([dominant]*len(views))
        self.indices,self.dominants = indices,dominants
        self.stats = dict(full_grid_identity=False,baseline_packed_tokens=self.original,
            original_image_tokens=self.original,raw_view_keeps=list(self.keeps),protected_newlines=self.protected,
            budget_kind="native_packed_layer_average",selection_layer=26,
            packing="original VisionZip sparse Base/raster dominants/contextual, with raw row newlines")
        return projected

    def pack(self, image_features, image_sizes, modalities):
        if len(image_features)!=1 or tuple(image_sizes[0])!=self.original_size or list(modalities)!=["image"]:
            raise ValueError("VisionZip benchmark expects one unchanged image")
        packed = pack_variable_visionzip(image_features[0],self.indices,self.dominants,self.grid,
            self.core.image_newline,self.final_newline)
        if len(packed)!=self.target: raise RuntimeError("VisionZip packing changed the exact layer-average target")
        self.stats.update(final_llm_visual_tokens=len(packed),actual_average_prune_rate=100*(1-len(packed)/self.original))
        return [packed]


class VisionZipMethod:
    name,visualize_rates = "visionzip",frozenset()
    def __init__(self, core, rate): self.adapter,self.rate = AverageVisionZip(core,rate),rate
    def configure(self,core,rate,**capture): self.adapter.configure(core,rate,**capture)
    def prepare_inputs(self,size,tokens): self.adapter.prepare_inputs(size,tokens)
    def stats(self,core,rate): return dict(self.adapter.stats)
    @contextmanager
    def inference_context(self,model):
        from runVisionZip import visionzip_generation
        with visionzip_generation(model,self.adapter): yield


def make_method(name, core, rate, checkpoint):
    depth = len(core.layers)
    if name=="llava_unpruned":
        from .methods import FastVMethod
        return FastVMethod(2,(0,),frozenset(),1,True)
    if name=="fastv": return AverageFastV(rate,depth)
    if name=="visionzip": return VisionZipMethod(core,rate)
    if name=="anchorprune":
        from .anchorprune import AnchorPruneMethod
        return AnchorPruneMethod((rate,))
    if name=="clse":
        from .clse import CLSEMethod,source_details
        source_details()
        return CLSEMethod((rate,),frozenset())
    if name=="priortr":
        from .priortr import PriorTRMethod,source_details
        source_details()
        return PriorTRMethod(2,(rate,),frozenset(),"last","mean",1e-10,rate_semantics="layer_average")
    if name=="pyramiddrop":
        from runPyramidDrop import build_parser,resolve_plan,PyramidDropMethod,check_source
        check_source()
        args = build_parser().parse_args(["--average-prune-rates",str(rate),"--layers","7","14","21"])
        return PyramidDropMethod(resolve_plan(args,depth)[0])
    if name=="hybrid_baseline":
        from .pool_distill import PoolDistillMethod
        from runMyPruningPoolSpatial import MethodSettings
        return PoolDistillMethod(core,checkpoint,rate=rate,share=.7,mode="base_full",
            settings=MethodSettings(prune_rate=0),pruning_audit="lean",tail_drop_layers=9,
            predictor_input="base_text",base_mapping="fast",base_policy="anyres_first")
    raise ValueError(f"Unknown comparison method: {name}")


@contextmanager
def observe_prefill(core):
    lengths = {}
    handles = []
    def hook(index):
        def record(module,args,kwargs):
            if index not in lengths:
                hidden = args[0] if args else kwargs["hidden_states"]
                lengths[index] = int(hidden.shape[1])
        return record
    try:
        handles = [layer.register_forward_pre_hook(hook(i),with_kwargs=True) for i,layer in enumerate(core.layers)]
        yield lengths
    finally:
        for handle in handles: handle.remove()


def observed_budget(name, rate, stats, lengths, original, depth=28):
    if len(lengths)!=depth: raise RuntimeError("Incomplete observed LLM-layer lengths")
    first = (stats["pre_llm_tokens"] if name=="hybrid_baseline" else
             stats["final_llm_visual_tokens"] if name in {"visionzip","anchorprune"} else original)
    text = lengths[0]-first
    counts = [lengths[i]-text for i in range(depth)]
    if text<1 or min(counts)<0: raise RuntimeError("Invalid observed visual/text decomposition")
    physical = 100*(1-sum(counts)/(depth*original))
    actual = stats["actual_average_prune_rate"] if name=="fastv" else physical
    if name=="fastv" and any(n!=original for n in counts):
        raise RuntimeError("FastV mask unexpectedly changed physical sequence lengths")
    if "observed_layer_token_sum" in stats and stats["observed_layer_token_sum"]!=sum(counts):
        raise RuntimeError("Method accounting differs from observed physical token lengths")
    return dict(original_image_tokens=original,physical_average_prune_rate=physical,
        effective_average_prune_rate=actual,actual_average_prune_rate=actual,
        budget_kind="effective_attention_mask" if name=="fastv" else "physical_tokens",
        observed_layer_token_sum=sum(counts),final_layer_visual_tokens=counts[-1],
        two_stage_average_prune_rate=stats.get("two_stage_average_prune_rate"),
        actual_tail_contribution=stats.get("actual_tail_contribution",0.),
        first_stage_visual_tokens=first)
