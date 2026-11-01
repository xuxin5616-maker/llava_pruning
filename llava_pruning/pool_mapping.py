"""Optional Base mapping fast path; same sources, weights and FP32 equations.

Replace per-token Python dictionaries with vectorized NumPy provenance, and
sample all AnyRes crop grids in one call. Dynamic source transfers still occur
inside generate/Lat. No changes to model/side precision or pruning budgets.
"""

import numpy as np
import torch
from torch.nn import functional as F


def prepare_mapping_geometry(geometry, device):
    side,patch,tile=geometry.token_side,geometry.patch_size,geometry.tile_size
    if side!=27:
        raise ValueError("Base mapping requires the 27x27 SigLIP grid")
    local=(torch.arange(side,device=device,dtype=torch.float32)+.5)*patch
    yy,xx=torch.meshgrid(local,local,indexing="ij")
    grids,masks=[],[]
    for crop in range(geometry.tile_count):
        row,col=divmod(crop,geometry.grid[0])
        x=(xx+col*tile-geometry.paste_xy[0])/geometry.resized_size[0]
        y=(yy+row*tile-geometry.paste_xy[1])/geometry.resized_size[1]
        grids.append(torch.stack((2*x*tile/(side*patch)-1,2*y*tile/(side*patch)-1),dim=-1))
        masks.append((x<0)|(x>=1)|(y<0)|(y>=1))
    return dict(coordinates=torch.stack(grids),padding=torch.stack(masks))


def prepare_mapping_sources(decisions, grid):
    """One integer D2H copy, bounded view loops, two compact H2D arrays.

    Outputs exactly match weighted_sources(source_records(...)): owners are
    spatially sorted, each owner emits its anchor first then ascending sources.
    Contextual normalization is anchor=1/2, other=1/(2*m) when m>0, else 1.
    These FP32 values match the legacy double-Python normalization for m<=728.
    """
    if len(decisions)!=1+grid[0]*grid[1]:
        raise ValueError("Source views disagree with the AnyRes grid")
    device=decisions[0]["dominant_indices"].device
    joined=torch.cat([torch.cat((d["dominant_indices"],d["contextual_indices"],d["cluster_assignments"]))
                      for d in decisions]).detach().cpu().numpy()
    parsed,positions=[],[]
    cursor=0
    for view,d in enumerate(decisions):
        dominant,contextual=d["dominant_count"],d["contextual_count"]
        anchors=joined[cursor:cursor+dominant+contextual]
        assignments=joined[cursor+dominant+contextual:cursor+dominant+contextual+729]
        cursor+=dominant+contextual+729
        parsed.append((dominant,contextual,anchors,assignments))
        if view:
            row,col=divmod(view-1,grid[0])
            positions.append((row*27+anchors//27)*(grid[0]*27)+col*27+anchors%27)
    inverse=[]
    for raw in (parsed[0][2],np.concatenate(positions)):
        order=np.argsort(raw,kind="stable")
        inv=np.empty_like(order);inv[order]=np.arange(len(order))
        inverse.append(inv)
    groups=[[],[]]
    offset=0
    for view,(dominant,contextual,anchors,assignments) in enumerate(parsed):
        owner=np.full(729,-1,dtype=np.int64)
        owner[anchors[:dominant]]=np.arange(dominant)
        mask=assignments>=0
        if np.any(assignments[mask]>=contextual):
            raise ValueError("Invalid merged-source cluster")
        owner[mask]=dominant+assignments[mask]
        source=np.flatnonzero(owner>=0)
        local_owner=owner[source]
        is_anchor=source==anchors[local_owner]
        weights=np.ones(len(source),dtype=np.float64)
        if contextual:
            counts=np.bincount(assignments[mask],minlength=contextual)-1
            context=local_owner>=dominant
            m=counts[np.maximum(local_owner-dominant,0)]
            weights[context&(m>0)]=.5
            other=context&~is_anchor
            weights[other]=.5/m[other]
        mapped_owner=inverse[0][local_owner] if view==0 else inverse[1][offset+local_owner]
        if view: offset+=len(anchors)
        # The legacy per-record weighted sum sees anchor, then sorted others.
        order=np.lexsort((np.where(is_anchor,0,source+1),mapped_owner))
        groups[int(view>0)].append((mapped_owner[order],source[order]+(view-1)*729 if view else source[order],
                                    weights[order].astype(np.float32),np.where(is_anchor,0,source+1)[order]))
    vectors=[]
    for group in groups:
        values=[np.concatenate([row[i] for row in group]) for i in range(4)]
        # AnyRes owners interleave rows of different views in the packed grid.
        order=np.lexsort((values[3],values[0]))
        vectors.append(tuple(value[order] for value in values[:3]))
    split=len(vectors[0][0])
    integer=np.stack((np.concatenate([v[0] for v in vectors]),np.concatenate([v[1] for v in vectors])))
    weights=np.concatenate([v[2] for v in vectors])
    integer=torch.from_numpy(integer).to(device=device)
    weights=torch.from_numpy(weights).to(device=device)
    return dict(base=(integer[0,:split],integer[1,:split],weights[:split]),
                anyres=(integer[0,split:],integer[1,split:],weights[split:]),
                base_count=len(parsed[0][2]),anyres_count=len(inverse[1]))


def mapped_scores_fast(base_log, sources, geometry):
    with torch.autocast(device_type=base_log.device.type,enabled=False):
        probability=F.softmax(base_log.float(),dim=0)
        owner,source,weight=sources["base"]
        dense=probability.new_zeros(729).index_add(0,source,probability[owner]*weight)
        if not torch.isclose(dense.sum(),probability.new_tensor(1.),atol=1e-5):
            raise RuntimeError("Base source backprojection lost probability mass")
        count=geometry["coordinates"].shape[0]
        maps=F.grid_sample(dense.reshape(1,1,27,27).expand(count,-1,-1,-1),geometry["coordinates"],
                           mode="bilinear",padding_mode="border",align_corners=False)[:,0]
        maps=maps.masked_fill(geometry["padding"],0).flatten()
        owner,source,weight=sources["anyres"]
        scores=probability.new_zeros(sources["anyres_count"]).index_add(0,owner,maps[source]*weight)
        joined=torch.cat((probability,scores)).clamp_min(torch.finfo(torch.float32).tiny).log()
        return joined-torch.logsumexp(joined,dim=0)
