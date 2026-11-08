"""One original/Base/AnyRes PNG from the real post-distillation decisions."""

from PIL import Image, ImageDraw, ImageFont

from .pool_distill_visualization import anchor_masks, input_rgb, white_positions


def font(size):
    try: return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        try: return ImageFont.load_default(size=size)
        except TypeError: return ImageFont.load_default()


def render_compact(method,trace,processor,result,sample,rate,panel_height=768):
    from .pool_lean import export_source_decisions
    from .visualization import prediction_caption
    if type(panel_height) is not int or panel_height<1: raise ValueError('panel-height must be positive')
    if 'pixels' not in trace or 'selected' not in trace:
        raise RuntimeError('Missing actual encoder input or pre-tail distillation selection')
    adapter=method.adapter;grid=tuple(adapter.geometry.grid);width,height=grid
    views=input_rgb(trace['pixels'],processor)
    decisions=export_source_decisions(adapter.device_decisions)
    selected=trace['selected'].cpu().tolist()
    before,after,newlines=anchor_masks(decisions,grid,selected,
        final_newline=adapter.core.config.mm_patch_merge_type.endswith('_add_newl'))
    if len(views)!=1+width*height or after.shape!=(len(views),729):
        raise ValueError('Input views differ from actual AnyRes geometry')
    plan=adapter.plan
    if int(before.sum())+newlines!=plan['pre_llm_tokens'] or int(after.sum())+newlines!=plan['final_llm_tokens']:
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
        draw.text((margin,10),f'Baseline | Avg pruning {rate:g}% | lambda=0.7 | ID: {sample.sample_id}',fill='black',font=font(20))
        draw.text((margin,40),prediction_caption(sample.gt,result['answer'])+' | After MyPool + distillation, BEFORE tail9',fill='black',font=font(16))
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
    return canvas,dict(native_grid=list(grid),encoder_views=len(views),
        base_kept=int(after[0].sum()),anyres_kept=int(after[1:].sum()),protected_newlines=newlines,
        baseline_packed_tokens=plan['baseline_packed_tokens'],pre_llm_tokens=plan['pre_llm_tokens'],post_distill_tokens=len(selected),
        panels=panels,post_snapshot='after MyPool and block2 distillation, before tail9',
        mask_semantics='RGB255 at removed independent anchors, including merged-away source positions')
