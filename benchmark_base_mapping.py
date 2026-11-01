"""Check/time Base source mapping only; no model/checkpoint/dataset is loaded.

Uses real MyPool selector/merger decisions at representative 27x27 view shapes.
Includes dynamic D2H/provenance/H2D and FP32 mapping. This is NOT full-model Lat.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import time


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu",type=int,default=5)
    p.add_argument("--cpu",action="store_true",help="CPU diagnostic only; not GPU speed evidence")
    p.add_argument("--prune-rates",type=float,nargs="+",default=[30,50,70])
    p.add_argument("--lambdas",type=float,nargs="+",default=[.7,.8,.9])
    p.add_argument("--grid",type=int,nargs=2,default=[3,3])
    p.add_argument("--image-size",type=int,nargs=2,default=[1024,1024])
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--output-dir",type=Path,default=Path("output/base_mapping_benchmark_01"))
    return p


def run(args):
    if args.gpu<0 or args.warmup<0 or args.repeats<1 or min(*args.grid,*args.image_size)<1:
        raise ValueError("Invalid benchmark device/geometry/repeat count")
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import torch
    import runMyPruningPoolSpatial as pool
    from llava_pruning.pool_lean import compress_views,export_source_decisions
    from llava_pruning.pool_distill import source_records,weighted_sources,mapped_scores
    from llava_pruning.pool_distill_budget import hybrid_budget,pre_view_budget
    from llava_pruning.pool_mapping import prepare_mapping_sources,prepare_mapping_geometry,mapped_scores_fast
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --cpu only for CPU diagnostics")
    device=torch.device("cpu" if args.cpu else "cuda:0")
    sync=(lambda:None) if args.cpu else lambda:torch.cuda.synchronize(0)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    grid=tuple(args.grid)
    geo=pool.ScoreGeometry.create(tuple(args.image_size),grid,384,14)
    static=prepare_mapping_geometry(geo,device)
    native=pool.baseline_packed_count(tuple(args.image_size),grid,27,"spatial_unpad")
    rows=[]
    with torch.inference_mode():
        for rate in args.prune_rates:
            for share in args.lambdas:
                plan=hybrid_budget(native,rate,share);plan["raw_grid_height"]=grid[1]
                budget=pre_view_budget(plan,geo.tile_count)
                k,q=divmod(budget["anyres_keep"],geo.tile_count)
                keeps=[budget["base_keep"]]+[k+int(i<q) for i in range(geo.tile_count)]
                dtype=torch.float32 if args.cpu else torch.float16
                features=torch.randn(1+geo.tile_count,729,8,device=device,dtype=dtype)
                scores=torch.rand(features.shape[:2],device=device,dtype=dtype)
                keys=torch.randn(1+geo.tile_count,729,4,device=device,dtype=dtype)
                _,decisions,_=compress_views(features,scores,keys,keeps,grid,torch.nn.Identity(),
                    features.new_zeros(8),"spatial_unpad")
                logits=torch.randn(keeps[0],device=device,dtype=torch.float32)
                def reference():
                    records=source_records(export_source_decisions(decisions),grid,fast=True)
                    base=[r for r in records if r is not None and r["view"]==0]
                    anyres=[r for r in records if r is not None and r["view"]!=0]
                    return mapped_scores(logits,base,anyres,geo)
                def fast():
                    return mapped_scores_fast(logits,prepare_mapping_sources(decisions,grid),static)
                records=source_records(export_source_decisions(decisions),grid,fast=True)
                base=[r for r in records if r is not None and r["view"]==0]
                anyres=[r for r in records if r is not None and r["view"]!=0]
                sources=prepare_mapping_sources(decisions,grid)
                for name,record,offset in (("base",base,False),("anyres",anyres,True)):
                    expected=weighted_sources(record,view_offset=offset,device=device)
                    if not all(torch.equal(a,b) for a,b in zip(expected,sources[name])):
                        raise RuntimeError(f"Source/weight mismatch for {name}; do not adopt fast mapping")
                old,new=reference(),fast()
                torch.testing.assert_close(old,new,rtol=1e-6,atol=2e-6)
                keep=plan["final_llm_tokens"]-keeps[0]-grid[1]*27
                old_indices=old[keeps[0]:].argsort(descending=True,stable=True)[:keep].sort().values
                new_indices=new[keeps[0]:].argsort(descending=True,stable=True)[:keep].sort().values
                selected_equal=torch.equal(old_indices,new_indices)
                if not selected_equal:
                    raise RuntimeError("Selected tokens differ; keep reference mapping and investigate")
                for _ in range(args.warmup): reference();fast()
                times={"reference":[],"fast":[]}
                for repeat in range(args.repeats):
                    for name,fn in (("reference",reference),("fast",fast))[::1 if repeat%2==0 else -1]:
                        sync();started=time.perf_counter();fn();sync()
                        times[name].append(1000*(time.perf_counter()-started))
                a,b=statistics.mean(times["reference"]),statistics.mean(times["fast"])
                row=dict(prune_rate=rate,distill_share=share,reference_mapping_ms=a,fast_mapping_ms=b,
                    mapping_speedup=a/b,selected_equal=selected_equal,score_bitwise_equal=torch.equal(old,new),
                    max_abs_score_error=float((old-new).abs().max()),base_tokens=keeps[0],anyres_tokens=sum(keeps[1:]))
                rows.append(row)
                print(f"rate={rate:g} lambda={share:g}: reference={a:.3f}ms fast={b:.3f}ms "
                      f"mapping-only speedup={a/b:.2f}x; selected_equal={selected_equal}",flush=True)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/"benchmark.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    metadata=dict(device=str(device),gpu=None if args.cpu else torch.cuda.get_device_name(0),
        torch=torch.__version__,cuda=torch.version.cuda,grid=grid,image_size=args.image_size,
        warmup=args.warmup,repeats=args.repeats,
        scope="dynamic provenance transfers/assembly + Base mapping only; NOT whole-model Lat",
        inputs="synthetic real MyPool decisions; verify real-data predictions separately before comparing experiments",
        static_geometry="prepared once as in the lean input-preparation path",
        score_dtype="float32",baseline_precision_unchanged=True)
    (output/"benchmark.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    return rows


if __name__=="__main__": run(parser().parse_args())
