"""FiCoCo-V on MVTec, MMAD and MPDD: one server command, no PNG.

Default GPU5/MVTec, GPU6/MMAD, GPU7/MPDD; each dataset has a native unpruned
reference and layer-average50/70/90%. No distilled student or training needed.
Use --include-visa to include MMAD's VisA source (excluded by default).
"""

import argparse
from collections import defaultdict, deque
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
from benchmark_mmad_binary import FIELDS, SOURCE_FIELDS, metric_row
from llava_pruning.data import Sample, load_samples
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.mmad_data import load_binary, canonical_answer
from llava_pruning.metrics import OPTION
from llava_pruning.mpdd_data import load_mpdd_test
from llava_pruning.prompts import resolve_prompt
from llava_pruning.ficoco_v import PAPER, UPSTREAM, VERSION as METHOD_VERSION


PROJECT=Path(__file__).resolve().parent
VERSION="1.1-ficoco-optional-mmad-visa"
DATASETS=("mvtec","mmad","mpdd")
BUDGET_FIELDS=(*common.BUDGET_FIELDS,"vision_start_layer","vision_depth","raw_retained_tokens","protected_newlines","packing")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus",nargs="+",type=int,default=[5,6,7])
    p.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    p.add_argument("--model-path",default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--mvtec-input-json",default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl")
    p.add_argument("--mvtec-data-root",default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/")
    p.add_argument("--mmad-root",default="/home/yz/xxy/data/datasets/MMAD")
    p.add_argument("--mmad-annotations")
    p.add_argument("--mmad-image-root")
    p.add_argument("--mmad-mvtec-root",default="/home/yz/xxy/data/datasets/MVTec-AD")
    p.add_argument("--include-visa",action="store_true",help="Include the VisA-source Anomaly Detection questions inside MMAD, for both FiCoCo-V and the unpruned reference")
    p.add_argument("--mpdd-root",default="/home/yz/xxy/data/datasets/MPDD")
    p.add_argument("--prune-rates",nargs="+",type=int,choices=(50,70,90),default=[50,70,90])
    p.add_argument("--vision-start-layer",type=int,default=12,help="One-based SigLIP layer; compress after attention, before MLP")
    p.add_argument("--balance",type=float,default=.35,help="FiCoCo-V score balance, NOT the hybrid distillation lambda")
    p.add_argument("--quantile",type=float,default=.998)
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output-dir",type=Path,default=Path("output/ficoco_v_mvtec_mmad_mpdd_01"))
    p.add_argument("--check-data",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--worker",help=argparse.SUPPRESS)
    p.add_argument("--manifest",type=Path,help=argparse.SUPPRESS)
    return p


def prepare_data(args):
    """Only CPU/file work, all data checked before output or GPU processes."""
    from PIL import Image
    datasets={}
    for name in args.datasets:
        if name=="mmad":
            rows,info=load_binary(args.mmad_root,args.mmad_annotations,args.mmad_image_root,args.seed,
                                 exclude_sources=() if args.include_visa else ("VisA",),mvtec_root=args.mmad_mvtec_root)
        else:
            if name=="mvtec":
                samples=load_samples(args.mvtec_input_json,args.mvtec_data_root)
                info=dict(layout="original_question_json",input_json=str(Path(args.mvtec_input_json).resolve()),
                    input_sha256=common.digest(args.mvtec_input_json))
            else: samples,info=load_mpdd_test(args.mpdd_root)
            rows=[]
            for sample in samples:
                if sample.gt not in (0,1): raise ValueError(f"Missing binary label: {sample.sample_id}")
                text=resolve_prompt("v0",sample.category,sample.text)[0] if name=="mvtec" else sample.text
                rows.append(dict(question_id=sample.sample_id,image=sample.source["image"],
                    resolved_image=str(sample.image.resolve()),origin_path=sample.origin_path,category=sample.category,
                    gt=sample.gt,text=text,dataset="MVTec-AD" if name=="mvtec" else "MPDD"))
        if not rows: raise ValueError(f"Empty test selection: {name}")
        seen=set()
        for row in rows:
            if row['question_id'] in seen: raise ValueError(f"Duplicate {name} question ID")
            seen.add(row['question_id']);row['benchmark_dataset']=name
            path=Path(row['resolved_image'])
            with Image.open(path) as image: row['image_size']=list(image.size)
            stat=path.stat();row.update(image_bytes=stat.st_size,image_mtime_ns=stat.st_mtime_ns)
        datasets[name]=dict(rows=rows,info=info)
        counts=defaultdict(int)
        for row in rows: counts[row['dataset']]+=1
        print(f"{name}: {len(rows)} test questions; sources={dict(counts)}",flush=True)
    return datasets


def schedule(names,rates,gpus,start,*,balance=.35,quantile=.998,include_visa=False):
    if not gpus or len(set(gpus))!=len(gpus) or any(type(g) is not int or g<0 for g in gpus):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    jobs=[]
    for index,name in enumerate(names):
        gpu=str(gpus[index%len(gpus)])
        for rate in (0,*rates):
            method="ficoco_v" if rate else "llava_unpruned"
            config=f"dataset={name};method={method};prune={rate}%;rate=layer_average;shot=0;gpu={gpu}"
            if rate: config+=f";vision_start={start};balance={balance:g};quantile={quantile:g}"
            if name=='mmad': config+=';exclude=none' if include_visa else ';exclude=VisA'
            jobs.append(dict(job_id=f"{name}_{method}_{rate:02d}",dataset=name,method=method,prune_rate=rate,gpu=gpu,config=config))
    return jobs


def verify(spec):
    if spec.get('version')!=VERSION or spec['source_identity']!=common.source_identity():
        raise ValueError("FiCoCo benchmark source/version changed; use a new directory")
    if common.digest(Path(spec['model_path'])/'config.json')!=spec['model_config_sha256']:
        raise ValueError("Model config changed")
    for name,ds in spec['datasets'].items():
        if common.digest(ds['input_json'])!=ds['input_sha256']: raise ValueError(f"Frozen {name} input changed")
        info=ds['info']
        if name=='mvtec' and common.digest(info['input_json'])!=info['input_sha256']:
            raise ValueError("Original MVTec question JSON changed")
        excluded=[] if spec.get('include_visa',False) else ['VisA']
        if name=='mmad' and (info['excluded_sources']!=excluded or set(info['source_counts'])&set(excluded) or
                common.digest(info['annotations'])!=info['annotations_sha256']):
            raise ValueError("MMAD annotation/exclusion changed")


def setup(args):
    output=args.output_dir.expanduser().resolve()
    if args.resume:
        spec=json.loads((output/'run.json').read_text(encoding='utf-8'));verify(spec)
        return output,spec
    jobs=schedule(args.datasets,args.prune_rates,args.gpus,args.vision_start_layer,
                  balance=args.balance,quantile=args.quantile,include_visa=args.include_visa)
    if (len(set(args.datasets))!=len(args.datasets) or len(set(args.prune_rates))!=len(args.prune_rates) or
            args.warmup<0 or not 0<=args.seed<2**32 or not 1<=args.vision_start_layer<=26 or
            not 0<=args.balance<=1 or not 0<=args.quantile<=1):
        raise ValueError("Invalid rates, vision settings, dataset list, seed or warmup")
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Use a new/empty directory or --resume: {output}")
    data=prepare_data(args)
    model=Path(args.model_path).expanduser().resolve()
    if json.loads((model/'config.json').read_text(encoding='utf-8')).get('num_hidden_layers')!=28:
        raise ValueError("Expected the existing 28-layer OneVision/Qwen2 model")
    output.mkdir(parents=True,exist_ok=True);(output/'inputs').mkdir(exist_ok=True)
    datasets={}
    for name,value in data.items():
        path=output/'inputs'/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in value['rows']),encoding='utf-8')
        datasets[name]=dict(info=value['info'],input_json=str(path),input_sha256=common.digest(path),samples=len(value['rows']))
    spec=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),gpus=[str(g) for g in args.gpus],
        model_path=str(model),model_config_sha256=common.digest(model/'config.json'),datasets=datasets,jobs=jobs,
        seed=args.seed,warmup=args.warmup,vision_start_layer=args.vision_start_layer,balance=args.balance,quantile=args.quantile,
        include_visa=args.include_visa,
        source_identity=common.source_identity(),method_version=METHOD_VERSION,paper=PAPER,upstream=UPSTREAM,
        definitions=dict(common.DEFINITIONS,
            rate="Physical mean removed packed image-token fraction over ALL LLM blocks. FiCoCo-V compresses in vision; every LLM block sees the same reduced sequence.",
            attention="Independent FP32 QK/key-softmax from native SigLIP projection outputs; no LLM attention return or precision/backend changes.",
            vision="Paper-based non-CLS Key-mean negative-cosine; attention-IN; L2 terms as in released filter; paper coefficient signs; 2x2 local maximum penalty; quantile many-to-many Eq2 merge. Constant per-layer reduction with integer remainder. See docs/ficoco_v.md for source discrepancies.",
            packing="At nonzero rates, sparse spatial anchors with RAW row newlines, no dense unpad/max9 feature interpolation; budgets target native unpruned packed denominator. At zero, full native packing.",
            accuracy="MVTec/MPDD retain strict original A/B parsing. MMAD uses its frozen option mapping; invalid/Maybe/Unknown answers count as ACC errors; parsed binary answers only for confusion/PRE/Recall/TNR/F1/BACC.",
            relative_accuracy="Each dataset's own actually measured unpruned ACC; not a fixed91.18% reference.",
            timing="All online score/selection/merge/packing/vision/projector/prefill included in resident-input TTFT; no PNG, no timing exclusions.",
            units_csv="acc/pre/recall/tnr/bacc/f1/mmad_score/relative_accuracy_pct=percent; Lat=mean TTFT milliseconds; mem=peak allocated GiB.",
            training="No student checkpoint or training; FiCoCo-V and the unpruned reference do not use the VisA-trained student.",
            source_selection="MMAD includes all four sources including VisA." if args.include_visa else
                "MMAD excludes VisA; evaluate MVTec-AD, MVTec-LOCO and GoodsAD."))
    common.save_json(output/'run.json',spec)
    return output,spec


def frozen_samples(ds,name):
    rows=[json.loads(line) for line in Path(ds['input_json']).read_text(encoding='utf-8').splitlines() if line.strip()]
    samples=[]
    for row in rows:
        if row['benchmark_dataset']!=name or (name=='mmad' and row['dataset'] in ds['info']['excluded_sources']):
            raise ValueError("Frozen dataset/source mismatch")
        path=Path(row['resolved_image']);stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(row['image_bytes'],row['image_mtime_ns']):
            raise ValueError(f"Image metadata changed: {path}")
        samples.append(Sample(row['question_id'],path,row['category'],row['origin_path'],row['text'],row['gt'],None,None,row))
    if len(samples)!=ds['samples']: raise ValueError("Frozen test count changed")
    return samples


def parse(row,answer):
    if row['benchmark_dataset']=='mmad': return canonical_answer(row,answer)
    match=OPTION.match(answer)
    choice=match.group(1).upper() if match else None
    return choice or '',choice


def summarize(records,samples,job,total_memory):
    def group(pairs,expected):
        canonical=[];unparsed=nonbinary=0;normal=abnormal=normal_ok=abnormal_ok=0
        for row,sample in pairs:
            answer,key=parse(sample.source,row['answer'])
            canonical.append(dict(row,answer=answer))
            unparsed+=key is None;nonbinary+=key is not None and not answer
            normal+=sample.gt==0;abnormal+=sample.gt==1
            correct=bool(answer) and (answer=='A')==bool(sample.gt)
            normal_ok+=sample.gt==0 and correct;abnormal_ok+=sample.gt==1 and correct
        result=common.summarize_records(canonical,expected,job,total_memory)
        extra=extended_metrics(result,1.)
        result.update({key:extra[key] for key in ('f1','bacc')})
        result.update(unparsed=unparsed,nonbinary=nonbinary,parsed_samples=len(pairs)-unparsed-nonbinary,
            normal=normal,abnormal=abnormal,
            mmad_score=(normal_ok/normal+abnormal_ok/abnormal)/2 if normal and abnormal and job['dataset']=='mmad' else None,
            rule="ACC: invalid/nonbinary answers are errors; confusion/PRE/Recall/TNR/F1/BACC use parsed binary only. Original strict A/B for MVTec/MPDD; actual frozen choices for MMAD.")
        return result
    pairs=list(zip(records,samples));result=group(pairs,len(samples));sources=defaultdict(list)
    for row,sample in pairs: sources[sample.source['dataset']].append((row,sample))
    result['sources']={name:group(values,len(values)) for name,values in sorted(sources.items())}
    if job['dataset']=='mmad':
        scores=[s['mmad_score'] for s in result['sources'].values()]
        result['mmad_score']=sum(scores)/len(scores) if scores and all(s is not None for s in scores) else None
    return result


def evaluate(backend,method,samples,spec,job,output,total_memory):
    from llava_pruning.visa_comparison_methods import native_counts, observe_prefill
    core=backend.model.get_model();path=output/'predictions.jsonl'
    records=common.restore_records(path,samples)
    for row,sample in zip(records,samples):
        if (row.get('canonical_answer'),row.get('selected_option'))!=parse(sample.source,row['answer']):
            raise ValueError("Saved raw answer/option mapping changed")
    if records and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as stream: stream.write(b'\n')
    resumed=len(records)
    def generate(index):
        sample=samples[index]
        with observe_prefill(core) as lengths:
            result=backend.generate(sample,sample.text,method,job['prune_rate'],'anyres_max_9',
                capture_visualization=False,capture_attention=False,random_seed=(spec['seed']+index)%2**32,
                do_sample=False,include_pruning_time=True,benchmark_metrics=True)
        _,counts=native_counts(core,sample.source['image_size']);original=sum(counts)
        stats=result['stats'];first=stats.get('final_llm_visual_tokens',original)
        text=lengths[0]-first
        if len(lengths)!=len(core.layers) or text<1 or any(n-text!=first for n in lengths.values()):
            raise RuntimeError("FiCoCo-V LLM sequence count differs across layers")
        actual=100*(1-first/original)
        if job['prune_rate'] and abs(actual-stats['actual_average_prune_rate'])>1e-7:
            raise RuntimeError("FiCoCo-V stats disagree with observed LLM inputs")
        budget=dict(original_image_tokens=original,first_stage_visual_tokens=first,final_layer_visual_tokens=first,
            actual_average_prune_rate=actual,physical_average_prune_rate=actual,effective_average_prune_rate=actual,
            budget_kind='physical_tokens',observed_layer_token_sum=first*len(core.layers),actual_tail_contribution=0.,
            protected_newlines=stats.get('protected_newlines',counts[2]),vision_start_layer=stats.get('vision_start_layer'),
            vision_depth=stats.get('vision_depth'),raw_retained_tokens=sum(stats['raw_view_keeps']) if 'raw_view_keeps' in stats else None,
            packing=stats.get('packing','native unpruned'))
        canonical,key=parse(sample.source,result['answer'])
        return dict(question_id=sample.sample_id,gt=sample.gt,answer=result['answer'],canonical_answer=canonical,
            selected_option=key,**{key:result[key] for key in common.TIMING},budget=budget)
    indices=common.warmup_indices(len(samples),spec['warmup']) if resumed<len(samples) else []
    for position,index in enumerate(indices,1):
        generate(index);print(f"{job['job_id']} warmup {position}/{len(indices)}",flush=True)
    started=time.perf_counter()
    with path.open('a',encoding='utf-8') as stream:
        for index in range(resumed,len(samples)):
            row=generate(index);records.append(row)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            if index==resumed or (index+1)%25==0 or index+1==len(samples):
                elapsed=time.perf_counter()-started;eta=elapsed*(len(samples)-index-1)/(index+1-resumed)
                print(f"{job['job_id']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s Lat={1000*row['lat_seconds']:.2f}ms",flush=True)
    return dict(summarize(records,samples,job,total_memory),resumed_samples=resumed,fresh_samples=len(records)-resumed,
                warmup_samples=len(indices))


def worker(manifest,job_id):
    spec=json.loads(manifest.read_text(encoding='utf-8'));job=next(j for j in spec['jobs'] if j['job_id']==job_id)
    ds=spec['datasets'][job['dataset']];output=manifest.parent/job_id;output.mkdir(exist_ok=True)
    report=dict(job_id=job_id,dataset=job['dataset'],input_sha256=ds['input_sha256'],gpu_id=job['gpu'],status='failed',complete=False)
    try:
        verify(spec)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=job['gpu']: raise ValueError("Worker exposed a different GPU")
        samples=frozen_samples(ds,job['dataset'])
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.ficoco_v import FiCoCoVMethod
        from llava_pruning.visa_comparison_methods import make_method
        torch.manual_seed(spec['seed'])
        backend=LlavaBackend(spec['model_path'],'anyres_max_9',image_token_order='base_first')
        common.validate_single_gpu(backend.model,torch);core=backend.model.get_model()
        method=(FiCoCoVMethod(core,job['prune_rate'],start_layer=spec['vision_start_layer'],balance=spec['balance'],quantile=spec['quantile'])
                if job['method']=='ficoco_v' else make_method('llava_unpruned',core,0,None))
        # Static per-geometry budget validation before warmup and measurements.
        if job['method']=='ficoco_v':
            from llava_pruning.visa_comparison_methods import native_counts
            from llava_pruning.ficoco_v import packed_plan
            tower=core.get_vision_tower()
            for size in {tuple(s.source['image_size']) for s in samples}:
                grid,counts=native_counts(core,size)
                packed_plan(sum(counts),1+grid[0]*grid[1],tower.num_patches_per_side,grid[1],job['prune_rate'],
                            core.config.mm_patch_merge_type.endswith('_add_newl'))
        device=torch.cuda.get_device_properties(0)
        common.verify_runtime(output,backend.inference_config,device.name,device.total_memory)
        report.update(inference=backend.inference_config,**evaluate(backend,method,samples,spec,job,output,device.total_memory))
        report['status']='success'
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc()
    finally: common.save_json(output/'metrics.json',report)
    return 0 if report['status']=='success' else 1


def write_reports(output,spec):
    results={}
    for job in spec['jobs']:
        path=output/job['job_id']/'metrics.json'
        result=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        if result and (result.get('job_id')!=job['job_id'] or result.get('input_sha256')!=spec['datasets'][job['dataset']]['input_sha256']):
            raise ValueError("Saved result belongs to another dataset/trial")
        results[job['job_id']]=result
    reference={name:results[f'{name}_llava_unpruned_00'] for name in spec['datasets']}
    rows=[];sources=[];budgets=[]
    for job in spec['jobs']:
        result=results[job['job_id']];status=result.get('status','pending')
        complete=status=='success' and result.get('complete')
        config=job['config'] if complete else job['config']+f';status={status}'
        base=reference[job['dataset']]
        base=base if base.get('status')=='success' and base.get('complete') else {}
        rows.append(metric_row(result,config,base.get('accuracy')) if complete else dict(config=config))
        if complete:
            for source,value in result['sources'].items():
                denominator=base.get('sources',{}).get(source,{}).get('accuracy')
                sources.append(dict(metric_row(value,config,denominator),source=source,
                    questions=value['evaluated_samples'],normal=value['normal'],abnormal=value['abnormal']))
        budgets.append(dict(result.get('budget',{}),config=config,status=status,reason=result.get('error',''),
                            target_average_prune_rate=job['prune_rate']))
    common.save_csv(output/'summary.csv',rows,FIELDS)
    common.save_csv(output/'summary_by_source.csv',sources,SOURCE_FIELDS)
    common.save_csv(output/'budget_summary.csv',budgets,BUDGET_FIELDS)


def run(args):
    if args.check_data: prepare_data(args);return 0
    output,spec=setup(args);write_reports(output,spec)
    queues={gpu:deque() for gpu in spec['gpus']}
    for job in spec['jobs']: queues[job['gpu']].append(job)
    active={};failures=[];last_progress=time.perf_counter()
    print(f"FiCoCo-V: {len(spec['jobs'])} trials, GPUs={spec['gpus']}; no student, PNG or retraining",flush=True)
    try:
        while active or any(queues.values()):
            for gpu,queue in queues.items():
                while gpu not in active and queue:
                    job=queue.popleft();path=output/job['job_id']/'metrics.json'
                    old=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
                    if old.get('status')=='success' and old.get('complete'):
                        print(f"Reuse {job['job_id']}",flush=True);continue
                    logpath=output/(job['job_id']+'.log');log=logpath.open('a',encoding='utf-8')
                    command=[sys.executable,'-u',str(PROJECT/'benchmark_ficoco_v.py'),'--worker',job['job_id'],'--manifest',str(output/'run.json')]
                    try:
                        process=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                            start_new_session=os.name!='nt',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1'))
                    except BaseException: log.close();raise
                    active[gpu]=(process,log,job,time.perf_counter())
                    print(f"Start GPU {gpu}: {job['job_id']}; log={logpath}",flush=True)
            for gpu,(process,log,job,started) in list(active.items()):
                code=process.poll()
                if code is None: continue
                process.wait();log.close();del active[gpu]
                path=output/job['job_id']/'metrics.json'
                result=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
                if code or result.get('status')!='success' or not result.get('complete'):
                    failures.append(job['job_id']);path.parent.mkdir(exist_ok=True)
                    common.save_json(path,dict(result,job_id=job['job_id'],status='failed',complete=False,
                        input_sha256=spec['datasets'][job['dataset']]['input_sha256'],error=result.get('error') or f'Worker exit={code}; see log'))
                print(f"Finish GPU {gpu}: {job['job_id']}; exit={code}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
                write_reports(output,spec)
            if time.perf_counter()-last_progress>=30:
                for gpu,(_,_,job,started) in active.items():
                    count=len(common.read_prefix(output/job['job_id']/'predictions.jsonl')[0])
                    expected=spec['datasets'][job['dataset']]['samples']
                    print(f"GPU {gpu}: {job['job_id']} {count}/{expected}; elapsed={time.perf_counter()-started:.0f}s; sample ETA in log",flush=True)
                last_progress=time.perf_counter()
            if active: time.sleep(.25)
    except KeyboardInterrupt:
        print(f"Interrupted; resume: python benchmark_ficoco_v.py --resume --output-dir {output}",flush=True);return 130
    finally:
        for process,_,_,_ in active.values():
            if process.poll() is None:
                if os.name=='posix': process.send_signal(signal.SIGINT)
                else: process.terminate()
        for process,log,_,_ in active.values():
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            finally: log.close()
        write_reports(output,spec)
    print(f"Summary: {output/'summary.csv'}",flush=True)
    if failures: print('Failed trials: '+', '.join(failures),flush=True)
    return int(bool(failures))


def main():
    args=parser().parse_args()
    try: return worker(args.manifest,args.worker) if args.worker else run(args)
    except (ValueError,OSError) as error:
        print(f'Error: {error}',file=sys.stderr);return 1


if __name__=='__main__': sys.exit(main())
