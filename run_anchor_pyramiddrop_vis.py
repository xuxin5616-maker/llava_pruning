"""Full official VisA test: AnchorPrune/PyramidDrop metrics, then white-mask PNGs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import benchmark_visa_methods as common


PROJECT=Path(__file__).resolve().parent
KIND='full-visa-anchor-pyramiddrop'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='5')
    p.add_argument('--model-path',default='/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/')
    p.add_argument('--data-root',default='/home/yz/xxy/data/datasets/VisA')
    p.add_argument('--visa-split-csv')
    p.add_argument('--prune-rates',nargs='+',type=int,choices=(50,70,90),default=[50,70,90])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--output-dir',type=Path,default=Path('output/visa_anchor_pyramiddrop_full_vis_01'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--visualize-only',action='store_true',help='Replay a completed run without changing its measured CSV/Lat/Mem')
    p.add_argument('--vis-output-dir',type=Path)
    p.add_argument('--visualization-worker',help=argparse.SUPPRESS)
    p.add_argument('--manifest',type=Path,help=argparse.SUPPRESS)
    return p


def load_run(output):
    output=Path(output).expanduser().resolve()
    spec=json.loads((output/'run.json').read_text(encoding='utf-8'))
    if spec.get('experiment_kind')!=KIND: raise ValueError('Not a full VisA AnchorPrune/PyramidDrop run')
    common.verify_inputs(spec)
    return output,spec


def setup(args):
    options=common.parser().parse_args([])
    for key in ('gpu','model_path','data_root','visa_split_csv','prune_rates','seed','warmup','output_dir','resume'):
        setattr(options,key,getattr(args,key))
    output,spec=common.setup(options)
    if args.resume:
        if spec.get('experiment_kind')!=KIND: raise ValueError('Cannot resume another experiment type')
    else:
        spec['jobs']=[j for j in spec['jobs'] if j['method'] in {'llava_unpruned','anchorprune','pyramiddrop'}]
        spec['experiment_kind']=KIND
        common.save_json(output/'run.json',spec)
    options.resume=True
    return output,spec,options


def identity(output,spec):
    jobs=[j for j in spec['jobs'] if j['method'] in {'anchorprune','pyramiddrop'} and j['status']=='pending']
    return dict(input_sha256=spec['input_sha256'],comparison_manifest_sha256=common.digest(output/'run.json'),
        replay_source_identity=common.source_identity(),gpu=spec['gpu'],
        measurements={j['job_id']:{name:common.digest(output/j['job_id']/name)
            for name in ('metrics.json','predictions.jsonl')} for j in jobs})


def records(output,spec,job,samples):
    from llava_pruning.pool_distill_resume import read_prefix,validate_prediction
    folder=output/job['job_id']
    report=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    if (report.get('status')!='success' or not report.get('complete') or
            report.get('job_id')!=job['job_id'] or report.get('input_sha256')!=spec['input_sha256'] or report.get('evaluated_samples')!=len(samples)):
        raise ValueError(f"Finish measurement before drawing {job['job_id']}")
    rows,_=read_prefix(folder/'predictions.jsonl')
    if len(rows)!=len(samples): raise ValueError('Incomplete measured sample list')
    for i,(row,sample) in enumerate(zip(rows,samples)): validate_prediction(row,sample.sample_id,sample.gt,i)
    return rows,report


def worker(manifest,job_id,vis_root):
    import torch
    from llava_pruning.backend import LlavaBackend
    from llava_pruning.data import load_samples
    from llava_pruning.visa_comparison_methods import make_method
    from llava_pruning.native_pruning_visualization import TracedPyramid,render
    output,spec=load_run(Path(manifest).parent)
    job=next(j for j in spec['jobs'] if j['job_id']==job_id and j['status']=='pending')
    samples=load_samples(spec['input_json'],spec['data_root']);rows,report=records(output,spec,job,samples)
    vis_root=Path(vis_root).resolve();expected=identity(output,spec)
    if json.loads((vis_root/'run.json').read_text(encoding='utf-8'))['identity']!=expected:
        raise ValueError('Drawing identity changed')
    backend=LlavaBackend(spec['model_path'],'anyres_max_9',image_token_order='base_first')
    common.validate_single_gpu(backend.model,torch)
    if backend.inference_config!=report.get('inference'): raise ValueError('Replay precision/backend/library settings changed')
    method=make_method(job['method'],backend.model.get_model(),job['prune_rate'],None)
    if method.name=='pyramiddrop': method=TracedPyramid(method)
    folder=vis_root/job['method']/f"prune_{job['prune_rate']:02d}"
    render(backend,method,samples,spec,job,folder,rows,expected)
    common.save_json(folder/'visualization_status.json',dict(complete=True,samples=len(samples),identity=expected))
    return 0


def visualize(output,vis_root=None):
    from llava_pruning.data import load_samples
    from llava_pruning.distilled_visa100_visualization import completed_job
    output,spec=load_run(output)
    samples=load_samples(spec['input_json'],spec['data_root'])
    jobs=[j for j in spec['jobs'] if j['method'] in {'anchorprune','pyramiddrop'} and j['status']=='pending']
    for job in jobs: records(output,spec,job,samples)
    vis_root=Path(vis_root).expanduser().resolve() if vis_root else output/'visualizations'
    expected=identity(output,spec);marker=vis_root/'run.json'
    if marker.is_file():
        if json.loads(marker.read_text(encoding='utf-8'))['identity']!=expected:
            raise ValueError('Existing drawings use another configuration; choose --vis-output-dir')
    elif vis_root.exists() and any(vis_root.iterdir()): raise FileExistsError('Visualization directory is not empty')
    vis_root.mkdir(parents=True,exist_ok=True)
    common.save_json(marker,dict(identity=expected,samples=len(samples),jobs=[j['job_id'] for j in jobs],
        measurement_source_identity=spec['source_identity'],lat_mem_recomputed=False))
    common.save_csv(vis_root/'sample_index.csv',[dict(folder=f'sample_{i:09d}',question_id=s.sample_id,gt=s.gt,image=s.origin_path)
        for i,s in enumerate(samples)],('folder','question_id','gt','image'))
    active=None;failures=[]
    try:
        for job in jobs:
            folder=vis_root/job['method']/f"prune_{job['prune_rate']:02d}"
            if completed_job(folder,samples,expected):
                print(f"VIS reuse {job['job_id']}",flush=True);continue
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--visualization-worker',job['job_id'],
                '--manifest',str(output/'run.json'),'--vis-output-dir',str(vis_root)]
            log_path=vis_root/(job['job_id']+'.log')
            print(f"VIS start {job['job_id']}: {len(samples)} samples; log={log_path}",flush=True)
            with log_path.open('a',encoding='utf-8') as log:
                active=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=spec['gpu'],PYTHONUNBUFFERED='1'),
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!='nt')
                while True:
                    try: code=active.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        done=sum((p/'run.json').is_file() for p in folder.glob('sample_*'))
                        print(f"VIS {job['job_id']}: {done}/{len(samples)} saved",flush=True)
            active=None
            if code: failures.append(job['job_id'])
            print(f"VIS finished {job['job_id']}: exit={code}",flush=True)
    except KeyboardInterrupt:
        if active is not None:
            active.terminate()
            try: active.wait(timeout=15)
            except subprocess.TimeoutExpired: active.kill();active.wait()
        return 130
    print(f'Pictures: {vis_root}',flush=True)
    if failures: print('Failed drawings: '+', '.join(failures),flush=True)
    return int(bool(failures))


def run(args):
    if args.visualize_only: return visualize(args.output_dir,args.vis_output_dir)
    output,spec,options=setup(args)
    print(f"FULL official VisA test: {spec['sample_count']} images; no 100-image limit. Metrics first, pictures second.",flush=True)
    code=common.run(options)
    return code if code else visualize(output,args.vis_output_dir)


def main():
    try:
        args=parser().parse_args()
        if args.visualization_worker: return worker(args.manifest,args.visualization_worker,args.vis_output_dir)
        return run(args)
    except (ValueError,OSError) as error:
        print(f'Error: {error}',file=sys.stderr);return 1


if __name__=='__main__': sys.exit(main())
