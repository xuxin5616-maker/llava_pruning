"""White per-encoder-view displays for native-packed AnchorPrune/PyramidDrop.

Packed AnyRes tokens can be unpadded/downsampled. Their binary spatial mask is
projected back to the raw crop grid with nearest interpolation. Projected raw
cells are explicitly NOT counted as packed tokens after max9 downsampling.
"""

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image


@contextmanager
def capture_inputs():
    from llava import mm_utils
    original = mm_utils.process_images
    trace = {}
    def process(*args,**kwargs):
        result = original(*args,**kwargs)
        pixels,metadata = result
        if trace or not kwargs.get("return_pro_data"): raise ValueError("Expected one preprocessing call")
        if isinstance(pixels,list):
            if len(pixels)!=1: raise ValueError("Expected one image")
            pixels = pixels[0]
        elif pixels.ndim==5 and pixels.shape[0]==1: pixels=pixels[0]
        if pixels.ndim!=4 or pixels.device.type!="cpu": raise ValueError("Expected CPU encoder views")
        trace.update(pixels=pixels,metadata=metadata[0])
        return result
    try:
        mm_utils.process_images = process
        yield trace
    finally: mm_utils.process_images = original


class TracedPyramid:
    """Enable only the existing diagnostic stage snapshots; native scores unchanged."""
    name = "pyramiddrop"
    def __init__(self,method): self.method=method
    def configure(self,core,rate,**kwargs): self.method.configure(core,rate,capture_visualization=True)
    def stats(self,core,rate): return self.method.stats(core,rate)


def view_masks(keep, metadata, side=27):
    """Exact no-resize mapping; spatial projection when native max9 resizes."""
    from .visualization import build_anyres_layout,_spatial_indices
    if metadata.get("image_token_order","base_first")!="base_first": raise ValueError("Expected Base first")
    layouts=build_anyres_layout(metadata,side,14)
    base,anyres=layouts
    h,w=anyres['grid_shape'];gw,gh=metadata['grid_patches']
    expected=side**2+h*(w+1)+int(metadata.get('final_newline',False))
    keep=np.asarray(keep,dtype=bool)
    if keep.shape!=(expected,): raise ValueError("Packed mask does not match native unpad/max9 geometry")
    raw_h,raw_w=gh*side,gw*side
    iw,ih=metadata['original_size'];top,left=0,0;up_h,up_w=raw_h,raw_w
    if iw/ih>raw_w/raw_h:
        top=(raw_h-int(ih*raw_w/iw))//2;up_h=raw_h-2*top
    else:
        left=(raw_w-int(iw*raw_h/ih))//2;up_w=raw_w-2*left
    native=keep[_spatial_indices(anyres)].reshape(h,w)
    projected=np.asarray(Image.fromarray(native.astype(np.uint8)).resize((up_w,up_h),Image.Resampling.NEAREST),dtype=bool)
    raw=np.zeros((raw_h,raw_w),bool);valid=np.zeros_like(raw)
    raw[top:top+up_h,left:left+up_w]=projected;valid[top:top+up_h,left:left+up_w]=True
    masks=[keep[_spatial_indices(base)]];valids=[np.ones(side**2,bool)]
    for y in range(gh):
        for x in range(gw):
            masks.append(raw[y*side:(y+1)*side,x*side:(x+1)*side].ravel())
            valids.append(valid[y*side:(y+1)*side,x*side:(x+1)*side].ravel())
    spatial=np.concatenate([_spatial_indices(v) for v in layouts])
    return np.stack(masks),np.stack(valids),dict(packed_tokens=expected,packed_kept=int(keep.sum()),
        packed_spatial_kept=int(keep[spatial].sum()),packed_newline_kept=int(keep.sum()-keep[spatial].sum()),
        native_anyres_grid=[h,w],raw_anyres_grid=[raw_h,raw_w],max9_resized=(h,w)!=(up_h,up_w),
        mapping="nearest projection of native packed mask; raw displayed cells are not packed-token counts")


def stages(method,core,rate):
    if method.name=='anchorprune':
        stats=method.stats(core,rate);n=stats['original_image_tokens']
        keep=np.zeros(n,bool);keep[method.selected.cpu().tolist()]=True
        return [dict(after_layer=0,image_tokens_after=int(keep.sum()),mask={'keep':keep})]
    return core.get_pyramiddrop_stages()


def save_views(trace,processor,result,sample,stage_data,output):
    from .pool_distill_visualization import input_rgb,white_positions,save_individual_views
    from .visualization import prediction_caption
    views=input_rgb(trace['pixels'],processor);metadata=trace['metadata']
    grid=metadata['grid_patches'];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    caption=f"ID: {sample.sample_id} | {prediction_caption(sample.gt,result['answer'])}"
    files=save_individual_views(views,grid,output,stage='input',caption=caption,counts=[729]*len(views))
    rows=[];summaries=[]
    previous=None
    for stage in stage_data:
        keep=np.asarray(stage['mask']['keep'],bool)
        if previous is not None and (keep&~previous).any(): raise ValueError("Pruned tokens reappeared")
        previous=keep
        masks,valids,summary=view_masks(keep,metadata)
        if int(keep.sum())!=stage['image_tokens_after']: raise ValueError("Stage count differs from actual decision")
        panels=[]
        for rgb,mask,valid in zip(views,masks,valids):
            panel=white_positions(rgb,mask|~valid)
            excluded=(~valid.reshape(27,27)).repeat(14,0).repeat(14,1)
            panel[:378,:378][excluded]=160
            panels.append(panel)
        layer=stage['after_layer'];prefix='pre_llm' if layer==0 else f'after_layer_{layer:02d}'
        note="White=pruned; gray=native unpad. Display counts are projected raw cells."
        files += save_individual_views(panels,grid,output,stage=prefix,
            title='AnchorPrune (before LLM)' if layer==0 else f'PyramidDrop after layer {layer}',
            caption=f"{caption} | {note}",counts=[int(mask.sum()) for mask in masks],count_label='projected cells')
        summaries.append(dict(after_layer=layer,**summary))
        for view,(mask,valid) in enumerate(zip(masks,valids)):
            rows.append(dict(after_layer=layer,view='base' if view==0 else f'anyres_{view:02d}',
                valid_raw_cells=int(valid.sum()),projected_kept_raw_cells=int(mask.sum()),
                projected_pruned_raw_cells=int((valid&~mask).sum()),packed_stage_kept=int(keep.sum())))
    with (output/'view_counts.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]));writer.writeheader();writer.writerows(rows)
    return dict(outputs=files,native_grid=list(grid),encoder_views=len(views),stages=summaries,
        mask_semantics="white=removed native spatial mask projected to raw views; gray=excluded by native unpadding; unencoded 6px edges unchanged")


def render(backend,method,samples,spec,job,output,measured,identity):
    import torch
    import benchmark_visa_methods as common
    from .prompts import GENERIC_PROMPT_V0
    from .visa_comparison_methods import observe_prefill,observed_budget,native_counts
    output.mkdir(parents=True,exist_ok=True)
    for index,(sample,old) in enumerate(zip(samples,measured)):
        folder=output/f'sample_{index:09d}';marker=folder/'run.json'
        if marker.is_file():
            saved=json.loads(marker.read_text(encoding='utf-8'))
            if saved.get('identity')!=identity: raise ValueError("Drawing configuration changed; use a new visualization folder")
            if all((folder/name).is_file() for name in (*saved['outputs'],'view_counts.csv')): continue
        with torch.inference_mode(),capture_inputs() as trace,observe_prefill(backend.model.get_model()) as lengths:
            result=backend.generate(sample,GENERIC_PROMPT_V0,method,job['prune_rate'],'anyres_max_9',
                capture_visualization=False,capture_attention=False,random_seed=(spec['seed']+index)%2**32,
                do_sample=False,include_pruning_time=True,benchmark_metrics=False)
        core=backend.model.get_model()
        trace['metadata']=dict(trace['metadata'],final_newline=core.config.mm_patch_merge_type.endswith('_add_newl'))
        original=sum(native_counts(core,sample.source['image_size'])[1])
        budget=observed_budget(job['method'],job['prune_rate'],result['stats'],lengths,original,len(core.layers))
        for key in ('original_image_tokens','observed_layer_token_sum'):
            if budget[key]!=old['budget'][key]: raise ValueError("Visualization budget differs from measured inference")
        details=save_views(trace,backend.processor,result,sample,stages(method,core,job['prune_rate']),folder)
        common.save_json(marker,dict(identity=identity,question_id=sample.sample_id,gt=sample.gt,
            measured_answer=old['answer'],visualization_answer=result['answer'],
            answer_matches_measured=old['answer']==result['answer'],**details))
        print(f"VIS {job['job_id']} {index+1}/{len(samples)}",flush=True)
        del trace,result
