"""One original/Base/AnyRes PNG from the real post-distillation decisions."""

from PIL import Image, ImageDraw, ImageFont

from .pool_distill_visualization import anchor_masks, input_rgb, white_positions


def font(size):
    try: return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        try: return ImageFont.load_default(size=size)
        except TypeError: return ImageFont.load_default()


def anchor_decisions(decisions):
    """Export ONLY actual retained anchors; lean self mode omits merge maps.

    Merge members never change packed anchor order and are intentionally
    white in this visualization. An empty membership list is used solely by
    the shared anchor-mask packer; it does not replace model provenance.
    """
    def indices(value):
        return value.detach().cpu().tolist() if hasattr(value,'detach') else list(value)
    return [dict(dominant_indices=indices(d['dominant_indices']),
                 contextual_indices=indices(d['contextual_indices']),cluster_assignments=[])
            for d in decisions]


def render_compact(method,trace,processor,result,sample,rate,panel_height=768,*,pre_llm_label=None):
    from .visualization import prediction_caption
    if type(panel_height) is not int or panel_height<1: raise ValueError('panel-height must be positive')
    if 'pixels' not in trace or (pre_llm_label is None and 'selected' not in trace):
        raise RuntimeError('Missing actual encoder input or pruning selection')
    adapter=method.adapter;grid=tuple(adapter.geometry.grid);width,height=grid
    views=input_rgb(trace['pixels'],processor)
    decisions=anchor_decisions(adapter.device_decisions)
    plan=adapter.plan
    selected=trace['selected'].cpu().tolist() if pre_llm_label is None else range(plan['pre_llm_tokens'])
    before,after,newlines=anchor_masks(decisions,grid,selected,
        final_newline=adapter.core.config.mm_patch_merge_type.endswith('_add_newl'))
    if len(views)!=1+width*height or after.shape!=(len(views),729):
        raise ValueError('Input views differ from actual AnyRes geometry')
    expected = plan['final_llm_tokens'] if pre_llm_label is None else plan['pre_llm_tokens']
    if int(before.sum())+newlines!=plan['pre_llm_tokens'] or int(after.sum())+newlines!=expected:
        raise RuntimeError('Rendered anchors disagree with the actual two-stage token budget')
    patch=int(adapter.geometry.patch_size)
    base=Image.fromarray(white_positions(views[0],after[0],patch))
    tile_h,tile_w=views.shape[1:3]
    mosaic=Image.new('RGB',(width*tile_w,height*tile_h),'white')
    for tile in range(width*height):
        row,col=divmod(tile,width)
        mosaic.paste(Image.fromarray(white_positions(views[tile+1],after[tile+1],patch)),(col*tile_w,row*tile_h))
    with Image.open(sample.image) as image: original=image.convert('RGB')
    pictures=[original,base,mosaic];scaled=[]
    try:
        for picture in pictures:
            scaled.append(picture.resize((max(1,round(picture.width*panel_height/picture.height)),panel_height),Image.Resampling.NEAREST))
        margin,gap,header,footer=20,24,100,36
        canvas=Image.new('RGB',(2*margin+sum(p.width for p in scaled)+2*gap,header+panel_height+footer),'white')
        draw=ImageDraw.Draw(canvas)
        title='Joint self-attention Top-k' if getattr(method,'base_policy','anyres_first')=='joint' else 'Baseline'
        share = getattr(method,'share',.7)
        heading = (f'{title} | Avg pruning {rate:g}% | lambda={share:g}' if pre_llm_label is None
                   else f'{pre_llm_label} | Target pruning {rate:g}%')
        stage = 'After MyPool + distillation, BEFORE tail9' if pre_llm_label is None else 'After MyPool, before LLM'
        draw.text((margin,10),f'{heading} | ID: {sample.sample_id}',fill='black',font=font(20))
        draw.text((margin,40),prediction_caption(sample.gt,result['answer'])+' | '+stage,fill='black',font=font(16))
        labels=('Original',f'Base (pruned; {int(after[0].sum())}/729)',
                f'AnyRes (pruned; {width} x {height}; {int(after[1:].sum())}/{729*(len(views)-1)})')
        left=margin;panels=[]
        for label,picture in zip(labels,scaled):
            draw.text((left,72),label,fill='black',font=font(17))
            canvas.paste(picture,(left,header));panels.append([left,header,left+picture.width,header+panel_height])
            left+=picture.width+gap
        draw.text((margin,header+panel_height+8),'White = removed independent token position (merged sources included). Unencoded crop edges unchanged.',fill='black',font=font(13))
    finally:
        for picture in pictures+scaled: picture.close()
    details = dict(native_grid=list(grid),encoder_views=len(views),
        base_kept=int(after[0].sum()),anyres_kept=int(after[1:].sum()),protected_newlines=newlines,
        baseline_packed_tokens=plan['baseline_packed_tokens'],pre_llm_tokens=plan['pre_llm_tokens'],
        panels=panels,post_snapshot=stage,
        mask_semantics='RGB255 at removed independent anchors, including merged-away source positions')
    if pre_llm_label is None:
        details.update(post_distill_tokens=len(selected), post_snapshot='after MyPool and block2 distillation, before tail9')
    else:
        details['anyres_kept_per_crop'] = after[1:].sum(axis=1).tolist()
    return canvas, details
