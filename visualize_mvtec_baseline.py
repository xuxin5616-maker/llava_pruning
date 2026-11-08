"""Full MVTec baseline: one original/pruned-Base/pruned-AnyRes PNG per sample.

Default GPU5=50%, GPU6=70%. Snapshot after MyPool+distillation, BEFORE tail9.
Diagnostic generation only: no clean Lat/Mem benchmark, no additional PNGs.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import benchmark_visa_methods as common
from compare_distilled_visa100 import student_identity
from llava_pruning.data import load_samples
from visualize_pool_distill import BASELINE


PROJECT=Path(__file__).resolve().parent
VERSION='1.0-mvtec-baseline-compact'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[5,6])
    p.add_argument('--prune-rates',type=int,nargs='+',choices=(50,70),default=[50,70])
    p.add_argument('--model-path',default='/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/')
    p.add_argument('--checkpoint',default='output/attention_predictor_twig_train_01/best.pt')
    p.add_argument('--input-json',default='/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl')
    p.add_argument('--data-root',default='/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/')
    p.add_argument('--prompt-version',choices=('v0','v1','v2','v3'),default='v0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--panel-height',type=int,default=768)
    p.add_argument('--output-dir',type=Path,default=Path('output/mvtec_baseline_compact_50_70_gpu56_01'))
    p.add_argument('--resume',action='store_true',help='Restore saved settings, skip completed PNGs, continue missing samples')
    p.add_argument('--check-data',action='store_true',help='Validate/count the FULL input; no model, GPU or output files')
    p.add_argument('--find-existing',action='store_true',help='Read-only search of output/outputs for MVTec visualization records')
    p.add_argument('--worker',help=argparse.SUPPRESS)
    p.add_argument('--manifest',type=Path,help=argparse.SUPPRESS)
    return p


def find_existing(roots=None):
    """Candidates from actual metadata AND PNGs; absence is not proof of no run."""
    roots=roots or [PROJECT/'output',PROJECT/'outputs'];found=[]
    for root in roots:
        if not root.is_dir(): continue
        for directory,children,_ in os.walk(root):
            folder=Path(directory);depth=len(folder.relative_to(root).parts)
            children[:]=[n for n in children if not n.startswith(('.', 'sample_', 'image_')) and n not in {'visualizations','_runtime','logs'}]
            if depth>=4: children[:]=[]
            for filename in ('run.json','comparison.json'):
                marker=folder/filename
                if not marker.is_file(): continue
                try: saved=json.loads(marker.read_text(encoding='utf-8'))
                except (OSError,ValueError): continue
                def input_values(value,keys):
                    if isinstance(value,dict):
                        for k,v in value.items():
                            if k in keys and isinstance(v,str): yield v
                            elif isinstance(v,(dict,list)): yield from input_values(v,keys)
                    elif isinstance(value,list):
                        for v in value: yield from input_values(v,keys)
                configuration=saved.get('configuration',{})
                if not isinstance(configuration,dict): configuration={}
                data_root=saved.get('data_root',configuration.get('data_root'))
                paths=[data_root] if isinstance(data_root,str) else list(input_values(saved,{'data_root'}))
                if not paths: paths=list(input_values(saved,{'input_json'}))
                if not any('mvtec' in v.lower() for v in paths): continue
                picture=None
                for base,dirs,names in os.walk(folder):
                    if len(Path(base).relative_to(folder).parts)>6: dirs[:]=[];continue
                    picture=next((str(Path(base)/n) for n in names if n.lower().endswith('.png')),None)
                    if picture: break
                if picture:
                    record=dict(directory=str(folder),samples=saved.get('sample_count',saved.get('samples',1 if 'question_id' in saved else 'unknown')),example_png=picture)
                    found.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
                    break
    if not found: print('No matching saved MVTec PNG records found under output/outputs; this does not prove an experiment was never run.',flush=True)
    return found


def freeze(args):
    from PIL import Image
    from llava_pruning.prompts import resolve_prompt
    samples=load_samples(args.input_json,args.data_root)
    if not samples: raise ValueError('Empty MVTec input')
    if not 0<=args.seed<=2**32-len(samples): raise ValueError('seed + sample index must fit NumPy seed range')
    rows=[]
    for sample in samples:
        with Image.open(sample.image) as image: size=list(image.size)
        stat=sample.image.stat()
        rows.append(dict(question_id=sample.sample_id,image=sample.source['image'],origin_path=sample.origin_path,
            category=sample.category,gt=sample.gt,text=sample.text,prompt=resolve_prompt(args.prompt_version,sample.category,sample.text)[0],
            resolved_image=str(sample.image),image_size=size,image_bytes=stat.st_size,image_mtime_ns=stat.st_mtime_ns))
    print(f'FULL input: {len(rows)} samples; every normal/abnormal image, no ID subset or 100-image limit',flush=True)
    return rows


def verify(spec):
    if spec.get('version')!=VERSION or spec['configuration']!=BASELINE:
        raise ValueError('Saved visualization version or baseline settings changed')
    for path,expected in ((spec['input_json'],spec['input_sha256']),
            (spec['original_input_json'],spec['original_input_sha256']),
            (Path(spec['model_path'])/'config.json',spec['model_config_sha256'])):
        if common.digest(path)!=expected: raise ValueError(f'Saved input/model config changed: {path}')


def setup(args):
    output=args.output_dir.expanduser().resolve()
    if args.resume:
        spec=json.loads((output/'run.json').read_text(encoding='utf-8'));verify(spec)
        if student_identity(spec['checkpoint']['path'],'visa')!=spec['checkpoint']:
            raise ValueError('Saved student or training manifest changed')
        return output,spec
    if (not args.gpus or len(set(args.gpus))!=len(args.gpus) or any(g<0 for g in args.gpus) or
            not args.prune_rates or len(set(args.prune_rates))!=len(args.prune_rates) or args.panel_height<1):
        raise ValueError('Use distinct nonnegative GPUs, distinct rates and positive panel-height')
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Use a new/empty directory or --resume: {output}')
    rows=freeze(args);model=Path(args.model_path).expanduser().resolve()
    if json.loads((model/'config.json').read_text(encoding='utf-8')).get('num_hidden_layers')!=28:
        raise ValueError('Expected the existing 28-layer OneVision/Qwen2 model')
    checkpoint=student_identity(args.checkpoint,'visa')
    output.mkdir(parents=True,exist_ok=True);path=output/'input_frozen.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    jobs=[dict(job_id=f'prune_{rate:02d}',prune_rate=rate,gpu=str(args.gpus[i%len(args.gpus)])) for i,rate in enumerate(args.prune_rates)]
    spec=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),configuration=dict(BASELINE),
        model_path=str(model),model_config_sha256=common.digest(model/'config.json'),checkpoint=checkpoint,
        original_input_json=str(Path(args.input_json).expanduser().resolve()),original_input_sha256=common.digest(args.input_json),
        input_json=str(path),input_sha256=common.digest(path),data_root=str(Path(args.data_root).expanduser().resolve()),
        sample_count=len(rows),prompt_version=args.prompt_version,seed=args.seed,panel_height=args.panel_height,jobs=jobs,
        source_identity=common.source_identity(),
        definitions=dict(stage='After MyPool and block2 distillation; BEFORE tail9. Model generation still executes tail9.',
            mask='RGB255 at removed independent token anchors, including merged-away source positions. Native unencoded edges retained.',
            images='Exactly one PNG per sample/rate: original disk image, pruned Base, native AnyRes mosaic. No intermediate PNGs.',
            measurement='Diagnostic generation/rendering only; no clean Lat/Mem measurements.',
            source='Use current project code; record source fingerprints per job. Resume checks inputs/model/student/runtime. No whole-project source-change blocking.'))
    common.save_json(output/'run.json',spec)
    return output,spec


def samples_for(spec):
    samples=load_samples(spec['input_json'],spec['data_root'])
    if len(samples)!=spec['sample_count']: raise ValueError('Frozen sample count changed')
    for sample in samples:
        stat=sample.image.stat();row=sample.source
        if str(sample.image)!=row['resolved_image'] or (stat.st_size,stat.st_mtime_ns)!=(row['image_bytes'],row['image_mtime_ns']):
            raise ValueError(f'Image metadata changed: {sample.image}')
    return samples


def restored(folder,samples):
    path=folder/'predictions.jsonl';rows,valid=common.read_prefix(path)
    if len(rows)>len(samples): raise ValueError('Too many saved visualization rows')
    for i,(row,sample) in enumerate(zip(rows,samples)):
        if row.get('index')!=i or row.get('question_id')!=sample.sample_id or row.get('gt')!=sample.gt:
            raise ValueError('Saved image ID/order/GT changed')
        if row.get('output')!=f'visualizations/sample_{i:09d}.png' or not isinstance(row.get('answer'),str):
            raise ValueError('Invalid saved image/output record')
        if not (folder/row['output']).is_file(): raise FileNotFoundError(folder/row['output'])
    if path.is_file() and path.stat().st_size!=valid:
        with path.open('r+b') as stream: stream.truncate(valid)
    if rows and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as stream: stream.write(b'\n')
    return rows


def render_samples(backend,method,samples,spec,job,folder):
    import torch
    from llava_pruning.pool_distill_visualization import capture_baseline_views
    from llava_pruning.pool_compact_visualization import render_compact
    folder.mkdir(parents=True,exist_ok=True);pictures=folder/'visualizations';pictures.mkdir(exist_ok=True)
    rows=restored(folder,samples);resumed=len(rows);started=time.perf_counter();last=started
    with torch.inference_mode(),method.generation_context(backend.model),(folder/'predictions.jsonl').open('a',encoding='utf-8') as stream:
        for index in range(resumed,len(samples)):
            sample=samples[index]
            with capture_baseline_views(method) as trace:
                result=backend.generate(sample,sample.source['prompt'],method,job['prune_rate'],'anyres_max_9',
                    capture_visualization=False,capture_attention=False,random_seed=spec['seed']+index,
                    do_sample=False,include_pruning_time=True,benchmark_metrics=False)
            picture,details=render_compact(method,trace,backend.processor,result,sample,job['prune_rate'],spec['panel_height'])
            name=f'visualizations/sample_{index:09d}.png';target=folder/name;temporary=target.with_suffix('.png.tmp')
            try: picture.save(temporary,format='PNG',compress_level=2);temporary.replace(target)
            finally: picture.close()
            row=dict(index=index,question_id=sample.sample_id,gt=sample.gt,answer=result['answer'],output=name,
                sample_seed=spec['seed']+index,**details)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();rows.append(row)
            del trace,result,picture
            elapsed=time.perf_counter()-started
            if index==resumed or (index+1)%25==0 or index+1==len(samples) or time.perf_counter()-last>=15:
                eta=elapsed*(len(samples)-index-1)/(index+1-resumed)
                print(f"VIS GPU{job['gpu']} prune={job['prune_rate']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s",flush=True)
                last=time.perf_counter()
    common.save_csv(folder/'image_index.csv',rows,('index','question_id','gt','answer','output','base_kept','anyres_kept',
        'baseline_packed_tokens','pre_llm_tokens','post_distill_tokens','protected_newlines'))
    return dict(complete=True,samples=len(rows),resumed_samples=resumed,fresh_samples=len(rows)-resumed,pngs=len(rows),lat_mem_benchmark=False)


def worker(manifest,job_id):
    spec=json.loads(manifest.read_text(encoding='utf-8'));verify(spec)
    job=next(j for j in spec['jobs'] if j['job_id']==job_id);folder=manifest.parent/job_id;folder.mkdir(exist_ok=True)
    report=dict(job_id=job_id,status='failed',complete=False,input_sha256=spec['input_sha256'],gpu=job['gpu'])
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=job['gpu']: raise ValueError('Wrong visible GPU')
        samples=samples_for(spec)
        if student_identity(spec['checkpoint']['path'],'visa')!=spec['checkpoint']: raise ValueError('Student changed')
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.pool_distill import PoolDistillMethod
        from runMyPruningPoolSpatial import MethodSettings
        torch.manual_seed(spec['seed'])
        backend=LlavaBackend(spec['model_path'],'anyres_max_9',image_token_order='base_first')
        common.validate_single_gpu(backend.model,torch);core=backend.model.get_model()
        if len(core.layers)!=28: raise ValueError('Expected 28 LLM layers')
        device=torch.cuda.get_device_properties(0)
        common.verify_runtime(folder,backend.inference_config,device.name,device.total_memory)
        method=PoolDistillMethod(core,spec['checkpoint']['path'],rate=job['prune_rate'],share=.7,mode='base_full',
            settings=MethodSettings(prune_rate=0,redundancy_layers=(26,),visionzip_layers=(26,),allocation_temperature=.05,spatial_strength=0.),
            pruning_audit='lean',tail_drop_layers=9,predictor_input='base_text',base_mapping='fast',base_policy='anyres_first')
        report.update(inference=backend.inference_config,source_identity=common.source_identity(),
            **render_samples(backend,method,samples,spec,job,folder));report['status']='success'
    except Exception as error: report['error']=f'{type(error).__name__}: {error}';traceback.print_exc()
    finally: common.save_json(folder/'status.json',report)
    return int(report['status']!='success')


def run(args):
    if args.find_existing: find_existing();return 0
    if args.check_data: freeze(args);return 0
    output,spec=setup(args);queues={j['gpu']:deque() for j in spec['jobs']};active={};failures=[]
    for job in spec['jobs']: queues[job['gpu']].append(job)
    print(f"Full MVTec: {spec['sample_count']} samples/rate; {spec['jobs']}; one PNG each; stage=before tail9",flush=True)
    last=time.perf_counter()
    try:
        while active or any(queues.values()):
            for gpu,queue in queues.items():
                while gpu not in active and queue:
                    job=queue.popleft();folder=output/job['job_id'];marker=folder/'status.json'
                    old=json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else {}
                    if old.get('complete') and old.get('status')=='success':
                        if old.get('input_sha256')!=spec['input_sha256'] or old.get('job_id')!=job['job_id']:
                            raise ValueError('Saved completed images belong to another input/job')
                        if len(restored(folder,samples_for(spec)))!=spec['sample_count']: raise ValueError('Incomplete completed PNG set')
                        print(f"Reuse complete {job['job_id']}",flush=True);continue
                    logpath=output/(job['job_id']+'.log');log=logpath.open('a',encoding='utf-8')
                    command=[sys.executable,'-u',str(PROJECT/'visualize_mvtec_baseline.py'),'--worker',job['job_id'],'--manifest',str(output/'run.json')]
                    try:
                        process=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                            start_new_session=os.name!='nt',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1'))
                    except BaseException: log.close();raise
                    active[gpu]=(process,log,job,time.perf_counter());print(f"Start GPU{gpu}: {job['job_id']}; log={logpath}",flush=True)
            for gpu,(process,log,job,started) in list(active.items()):
                code=process.poll()
                if code is None: continue
                process.wait();log.close();del active[gpu]
                marker=output/job['job_id']/'status.json';report=json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else {}
                if code or report.get('status')!='success' or not report.get('complete'): failures.append(job['job_id'])
                print(f"Finish GPU{gpu}: {job['job_id']}; exit={code}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            if time.perf_counter()-last>=30:
                for gpu,(_,_,job,started) in active.items():
                    count=len(common.read_prefix(output/job['job_id']/'predictions.jsonl')[0])
                    print(f"GPU{gpu} {job['job_id']}: {count}/{spec['sample_count']} saved; elapsed={time.perf_counter()-started:.0f}s; ETA in log",flush=True)
                last=time.perf_counter()
            if active: time.sleep(.25)
    except KeyboardInterrupt:
        print(f'Interrupted; resume with --resume --output-dir {output}',flush=True);return 130
    finally:
        for process,_,_,_ in active.values():
            if process.poll() is None:
                if os.name=='posix': process.send_signal(signal.SIGINT)
                else: process.terminate()
        for process,log,_,_ in active.values():
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            finally: log.close()
    print(f'Pictures: {output}; failures={failures}',flush=True)
    return int(bool(failures))


def main():
    args=parser().parse_args()
    try: return worker(args.manifest,args.worker) if args.worker else run(args)
    except (OSError,ValueError) as error: print(f'Error: {error}',file=sys.stderr);return 1


if __name__=='__main__': sys.exit(main())
