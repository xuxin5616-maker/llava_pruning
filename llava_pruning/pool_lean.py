"""Inference-only MyPool auxiliary path; no pruning compute is subtracted.

Native SigLIP/merge/projector arithmetic stays the same. Static shape data is
prepared before generate; unused diagnostics are not produced. Essential scores,
CPU quotas, native top-k/Key merging and sparse packing remain on the Lat path.
"""

import torch
import runMyPruningPoolSpatial as pool


def prepare_geometry(geometry, device):
    width, height = geometry.grid
    side = geometry.token_side
    bins = []
    for crop in range(width*height):
        row, col = divmod(crop, width)
        y0, x0 = row*(side//height), col*(side//width)
        y1 = side if row == height-1 else y0+side//height
        x1 = side if col == width-1 else x0+side//width
        bins.append([y*side+x for y in range(y0,y1) for x in range(x0,x1)])
    return dict(grid=tuple(geometry.grid), side=side, base_token_indices=bins,
                spatial_indices=torch.arange(side*side,device=device),
                row_ends=torch.arange(height*side,device=device)*(width*side+1)+width*side,
                base_index_tensors=[torch.tensor(indices,device=device,dtype=torch.long) for indices in bins],
                base_token_counts=torch.tensor([len(indices) for indices in bins],device=device,dtype=torch.long))


def pack_device_rows(projected_views, decisions, grid, newline, merge_type, *, prepared=None):
    """Same anchor-row ordering as pack_anchor_rows, without CPU index round trips."""
    width, height = grid
    side = 27
    indices = [torch.cat((row["dominant_indices"],row["contextual_indices"])) for row in decisions]
    base = projected_views[0][torch.argsort(indices[0])]
    positions, rows = [], []
    raw_width, raw_rows = width*side, height*side
    for tile, local in enumerate(indices[1:]):
        tile_row, tile_col = divmod(tile,width)
        positions.append((tile_row*side+local//side)*raw_width+tile_col*side+local%side)
        rows.append(projected_views[tile+1])
    positions, rows = torch.cat(positions), torch.cat(rows)
    positions = (positions//raw_width)*(raw_width+1)+positions%raw_width
    row_ends = prepared["row_ends"] if prepared is not None else torch.arange(raw_rows,device=rows.device)*(raw_width+1)+raw_width
    positions = torch.cat((positions,row_ends))
    roles = torch.cat((torch.ones(rows.shape[0],device=rows.device,dtype=torch.long),
                       torch.full((raw_rows,),2,device=rows.device,dtype=torch.long)))
    rows = torch.cat((rows,newline.to(device=rows.device,dtype=rows.dtype)[None].expand(raw_rows,-1)))
    order = torch.argsort(positions)
    packed = torch.cat((base,rows[order]))
    roles = torch.cat((torch.zeros(len(base),device=rows.device,dtype=torch.long),roles[order]))
    if merge_type == "spatial_unpad_add_newl":
        packed = torch.cat((packed,newline[None].to(device=packed.device,dtype=packed.dtype)))
        roles = torch.cat((roles,roles.new_tensor([2])))
    return packed, roles


def compress_views(features, scores, metric, keeps, grid, projector, newline, merge_type, *, source_provenance=True, prepared=None):
    if features.shape[:2] != scores.shape or metric.shape[:2] != scores.shape:
        raise ValueError("Lean compression feature/score/Key shapes disagree")
    if features.shape[0] != len(keeps) or features.shape[1] != 729:
        raise ValueError("Lean MyPool requires Base + raw729-token views")
    if all(keep==729 for keep in keeps):
        raise ValueError("Full retention must use the existing native full-grid endpoint")
    # One batch validation instead of repeated full feature/score/Key scans per view.
    pool._validate_inputs(features,scores,metric,0,0)
    spatial_indices=prepared["spatial_indices"] if prepared is not None else torch.arange(729,device=features.device)
    compressed, decisions = [], []
    for view,keep in enumerate(keeps):
        if type(keep) is not int or not 4 <= keep <= 729:
            raise ValueError("Invalid lean view quota")
        dominant = keep if keep == 729 else round(keep*pool.DOMINANT_RATIO)
        output, decision = pool.compress_tokens(features[view:view+1],scores[view:view+1],metric[view:view+1],
            dominant,keep-dominant,audit=False,validate=False,source_provenance=source_provenance,spatial_indices=spatial_indices)
        compressed.append(output[0])
        decisions.append(decision[0])
    projected = projector(torch.cat(compressed))
    if projected.ndim != 2 or projected.shape[0] != sum(keeps) or not bool(torch.isfinite(projected).all()):
        raise ValueError("Lean projector produced invalid features")
    packed,roles = pack_device_rows(list(torch.split(projected,keeps)),decisions,grid,newline,merge_type,prepared=prepared)
    expected = sum(keeps)+grid[1]*27+int(merge_type.endswith("_add_newl"))
    if packed.shape[0] != expected:
        raise RuntimeError("Lean packing violated the exact token budget")
    return packed,decisions,roles


def export_source_decisions(decisions):
    """Only indices/assignments REQUIRED by Base mapping; one integer transfer."""
    joined = torch.cat([torch.cat((d["dominant_indices"],d["contextual_indices"],d["cluster_assignments"]))
                        for d in decisions]).cpu().tolist()
    cursor, result = 0, []
    for decision in decisions:
        dominant, contextual = decision["dominant_count"],decision["contextual_count"]
        result.append(dict(dominant_indices=joined[cursor:cursor+dominant],
            contextual_indices=joined[cursor+dominant:cursor+dominant+contextual],
            cluster_assignments=joined[cursor+dominant+contextual:cursor+dominant+contextual+729]))
        cursor += dominant+contextual+729
    return result


def encode(adapter, images):
    if adapter.core.training or adapter.geometry is None:
        raise ValueError("Lean MyPool requires eval mode and prepared image geometry")
    expected = 1+adapter.geometry.tile_count
    if images.ndim != 4 or images.shape[0] != expected:
        raise ValueError("Lean encoder input disagrees with Base/AnyRes geometry")
    blocks = adapter.tower.vision_tower.vision_model.encoder.layers
    attention = blocks[-1].self_attn
    keys, scores, redundancies, handles = {}, {}, {}, []
    def key_hook(module, args, output):
        batch,tokens,_ = output.shape
        keys["metric"] = output.view(batch,tokens,attention.num_heads,attention.head_dim).permute(0,2,1,3).mean(1)
    def score_hook(layer):
        def capture(module,args,kwargs,output):
            if layer in scores or output[1] is None:
                raise RuntimeError("Missing or repeated native SigLIP attention capture")
            scores[layer] = output[1].mean(dim=1).mean(dim=1)
        return capture
    def feature_hook(layer):
        def capture(module,args,output):
            if layer in redundancies:
                raise RuntimeError("Repeated redundancy layer")
            features = output[0] if isinstance(output,(tuple,list)) else output
            redundancies[layer] = pool.pooled_crop_redundancy(features,adapter.geometry.grid,
                reduction=adapter.settings.redundancy_reduction,audit=False,prepared=adapter.prepared)
        return capture
    try:
        handles.append(attention.k_proj.register_forward_hook(key_hook))
        for layer in adapter.settings.visionzip_layers:
            handles.append(blocks[layer-1].self_attn.register_forward_hook(score_hook(layer),with_kwargs=True))
        for layer in adapter.settings.redundancy_layers:
            handles.append(blocks[layer-1].register_forward_hook(feature_hook(layer)))
        features = adapter.tower(images)
    finally:
        for handle in handles: handle.remove()
    if (set(scores)!=set(adapter.settings.visionzip_layers)
            or set(redundancies)!=set(adapter.settings.redundancy_layers)
            or "metric" not in keys or features.shape[:2]!=(expected,729)):
        raise RuntimeError("Incomplete lean SigLIP capture")
    selected_scores = (scores[adapter.settings.visionzip_layers[0]] if len(scores)==1 else
        torch.stack([scores[layer].float() for layer in adapter.settings.visionzip_layers]).mean(0))
    first = redundancies[adapter.settings.redundancy_layers[0]]
    redundancy = (first["redundancy"] if len(redundancies)==1 else
        torch.stack([redundancies[layer]["redundancy"] for layer in adapter.settings.redundancy_layers]).mean(0))
    # CPU transfer is necessary: actual crop quota allocation still runs in NumPy.
    raw = redundancy.detach().to(dtype=torch.float64).cpu().numpy()
    spatial = dict(spatial_prior=adapter.prepared["spatial_prior"])
    allocation,_,base_d,base_c = adapter.allocate_view_budgets(raw,first,spatial)
    keeps = [base_d+base_c]+[crop["keep"] for crop in allocation["crops"]]
    packed,decisions,roles = compress_views(features,selected_scores,keys["metric"],keeps,
        adapter.geometry.grid,adapter.core.mm_projector,adapter.core.image_newline,adapter.core.config.mm_patch_merge_type,
        source_provenance=adapter.mode=="base_full",prepared=adapter.prepared)
    adapter.allocation,adapter.device_decisions,adapter.roles = allocation,decisions,roles
    # Minimal fields consumed by prepare/pack, no reporting lists or heatmaps.
    adapter.stats = dict(final_llm_visual_tokens=len(packed),full_grid_identity=False)
    return packed
