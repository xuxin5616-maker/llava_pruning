"""Three datasets share frozen prompts, clean timings, GPU queues and reports."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import benchmark_ficoco_v as entry
import benchmark_visa_methods as common
from llava_pruning.mpdd_data import CATEGORIES
from llava_pruning.prompts import resolve_prompt
from test_mmad_binary import fixture as source_fixture


def fixture(root):
    legacy=source_fixture(root)
    mvtec=root/'mvtec';mvtec.mkdir();rows=[]
    for gt in (0,1):
        path=mvtec/f'{gt}.png';Image.new('RGB',(32,32)).save(path)
        rows.append(dict(question_id=f'{gt:09d}',image=path.name,origin_path=f'bottle/test/{"broken" if gt else "good"}/{gt}.png',
            category='bottle',gt=gt,text='RAW TEXT MUST NOT REPLACE V0'))
    questions=root/'questions.json';questions.write_text(json.dumps(rows),encoding='utf-8')
    mpdd=root/'MPDD'
    for category in CATEGORIES:
        for label in ('good','defect'):
            path=mpdd/category/'test'/label/'000.png';path.parent.mkdir(parents=True)
            Image.new('RGB',(32,32)).save(path)
    args=entry.parser().parse_args(['--model-path',legacy.model_path,'--mvtec-input-json',str(questions),
        '--mvtec-data-root',str(mvtec),'--mmad-root',legacy.data_root,'--mpdd-root',str(mpdd),
        '--output-dir',str(root/'output')])
    args.mmad_mvtec_root=None
    return args


def records_for(samples):
    rows=[]
    for sample in samples:
        answer=sample.source['correct_option'] if sample.source['benchmark_dataset']=='mmad' else ('A' if sample.gt else 'B')
        canonical,key=entry.parse(sample.source,answer)
        rows.append(dict(question_id=sample.sample_id,gt=sample.gt,answer=answer,canonical_answer=canonical,
            selected_option=key,lat_seconds=.1,generation_seconds=.2,generated_tokens=2,
            peak_allocated_bytes=100,peak_reserved_bytes=200,budget=dict(physical_average_prune_rate=50.)))
    return rows


class BenchmarkTests(unittest.TestCase):
    def test_data_selection_keeps_original_prompts_source_exclusion_and_test_split(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));data=entry.prepare_data(args)
            self.assertEqual({name:len(ds['rows']) for name,ds in data.items()},{'mvtec':2,'mmad':8,'mpdd':12})
            self.assertTrue(all(row['text']==resolve_prompt('v0','bottle',None)[0] for row in data['mvtec']['rows']))
            self.assertTrue(all(row['dataset']!='VisA' for row in data['mmad']['rows']))
            self.assertEqual(data['mmad']['info']['excluded_sources'],['VisA'])
            self.assertTrue(all('/test/' in row['origin_path'] for row in data['mpdd']['rows']))
            self.assertTrue(all('mask' not in row for ds in data.values() for row in ds['rows']))
            args.check_data=True;args.model_path='MISSING_MODEL'
            self.assertEqual(entry.run(args),0);self.assertFalse(args.output_dir.exists())

    def test_schedule_keeps_each_dataset_curve_on_one_gpu(self):
        jobs=entry.schedule(entry.DATASETS,[50,70,90],[5,6,7],12)
        self.assertEqual(len(jobs),12)
        for name,gpu in zip(entry.DATASETS,('5','6','7')):
            group=[j for j in jobs if j['dataset']==name]
            self.assertEqual({j['gpu'] for j in group},{gpu})
            self.assertEqual([j['prune_rate'] for j in group],[0,50,70,90])
        with self.assertRaises(ValueError): entry.schedule(entry.DATASETS,[50],[5,5],12)

    def test_include_visa_reaches_preflight_frozen_inputs_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.include_visa=True
            args.check_data=True
            self.assertEqual(entry.run(args),0);self.assertFalse(args.output_dir.exists())
            args.check_data=False;_,spec=entry.setup(args);entry.verify(spec)
            self.assertTrue(spec['include_visa'])
            ds=spec['datasets']['mmad'];samples=entry.frozen_samples(ds,'mmad')
            self.assertEqual(len(samples),10)
            self.assertEqual(ds['info']['excluded_sources'],[])
            self.assertEqual(ds['info']['source_counts']['VisA'],2)
            for sample in samples:
                if sample.source['dataset']=='VisA':
                    self.assertEqual(len(sample.source['options']),4)
                    self.assertIn(sample.source['question'],sample.text)
            self.assertTrue(all('exclude=none' in j['config'] for j in spec['jobs'] if j['dataset']=='mmad'))
            self.assertTrue(all('exclude=' not in j['config'] for j in spec['jobs'] if j['dataset']!='mmad'))
            # Resume restores the saved inclusion policy; a missing CLI flag
            # cannot silently drop VisA halfway through this experiment.
            args.resume=True;args.include_visa=False
            self.assertEqual(entry.setup(args)[1],spec)
            spec['include_visa']=False
            with self.assertRaisesRegex(ValueError,'MMAD annotation/exclusion'): entry.verify(spec)
            ds['info']['excluded_sources']=['VisA']
            with self.assertRaisesRegex(ValueError,'Frozen dataset/source'): entry.frozen_samples(ds,'mmad')

    def test_include_visa_exports_four_sources_and_full_mmad_reference(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.include_visa=True;args.datasets=['mmad'];calls=[]
            def process(command,**kwargs):
                manifest=Path(command[command.index('--manifest')+1]);spec=json.loads(manifest.read_text(encoding='utf-8'))
                job=next(j for j in spec['jobs'] if j['job_id']==command[command.index('--worker')+1])
                ds=spec['datasets']['mmad'];samples=entry.frozen_samples(ds,'mmad');rows=records_for(samples)
                for sample,row in zip(samples,rows):
                    if sample.source['dataset']=='VisA': row['answer']='I cannot decide.'
                result=entry.summarize(rows,samples,job,2**30)
                folder=manifest.parent/job['job_id'];folder.mkdir()
                common.save_json(folder/'metrics.json',dict(result,job_id=job['job_id'],status='success',input_sha256=ds['input_sha256']))
                calls.append(job);return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0)
            with patch.object(entry.subprocess,'Popen',side_effect=process):
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),4)
                args.resume=True;self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),4)
            with (args.output_dir/'summary.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(all(float(r['acc'])==80. and float(r['mmad_score'])==75. and
                float(r['relative_accuracy_pct'])==100. for r in rows))
            with (args.output_dir/'summary_by_source.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),16)
            self.assertEqual({r['source'] for r in rows},{'MVTec-AD','MVTec-LOCO','GoodsAD','VisA'})
            self.assertTrue(all(float(r['acc'])==0. for r in rows if r['source']=='VisA'))

    def test_parallel_reports_use_per_dataset_references_and_resume_same_inputs(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));busy=set();maximum=[0];calls=[]
            class Process:
                def __init__(self,job,spec,manifest,gpu):
                    self.job,self.spec,self.manifest,self.gpu=job,spec,manifest,gpu;self.done=False;self.polls=0
                def poll(self):
                    if self.done: return 0
                    self.polls+=1
                    if self.polls==1: return None
                    self.done=True;busy.remove(self.gpu)
                    ds=self.spec['datasets'][self.job['dataset']]
                    samples=entry.frozen_samples(ds,self.job['dataset'])
                    expected={'mvtec':2,'mmad':8,'mpdd':12}[self.job['dataset']]
                    assert len(samples)==expected
                    result=entry.summarize(records_for(samples),samples,self.job,2**30)
                    folder=self.manifest.parent/self.job['job_id'];folder.mkdir()
                    common.save_json(folder/'metrics.json',dict(result,job_id=self.job['job_id'],status='success',
                        input_sha256=ds['input_sha256']))
                    return 0
                def wait(self,**kwargs): return 0
            def process(command,**kwargs):
                calls.append(command);manifest=Path(command[command.index('--manifest')+1])
                spec=json.loads(manifest.read_text(encoding='utf-8'))
                job=next(j for j in spec['jobs'] if j['job_id']==command[command.index('--worker')+1])
                gpu=kwargs['env']['CUDA_VISIBLE_DEVICES'];self.assertEqual(gpu,job['gpu'])
                self.assertNotIn(gpu,busy);busy.add(gpu);maximum[0]=max(maximum[0],len(busy))
                return Process(job,spec,manifest,gpu)
            with patch.object(entry.subprocess,'Popen',side_effect=process),patch.object(entry.time,'sleep'):
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),12);self.assertEqual(maximum[0],3)
                args.resume=True;self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),12)
            with (args.output_dir/'summary.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),12)
            self.assertTrue(all(float(row['acc'])==100 and float(row['relative_accuracy_pct'])==100 for row in rows))
            with (args.output_dir/'summary_by_source.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),20);self.assertNotIn('VisA',{r['source'] for r in rows})

    def test_invalid_answers_and_source_options_remain_separate_from_raw_letters(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));output,spec=entry.setup(args)
            ds=spec['datasets']['mmad'];samples=entry.frozen_samples(ds,'mmad');rows=records_for(samples)
            self.assertEqual(entry.summarize(rows,samples,{'dataset':'mmad','prune_rate':50},2**30)['accuracy'],1.)
            rows[0]['answer']='I cannot decide.'
            result=entry.summarize(rows,samples,{'dataset':'mmad','prune_rate':50},2**30)
            self.assertEqual(result['unparsed'],1);self.assertEqual(result['correct'],7)
            self.assertEqual(entry.parse({'benchmark_dataset':'mvtec'},'Yes.'),('',None))
            # Original MVTec A/B parser, not a new forced fuzzy Yes/No parser.
            self.assertEqual(entry.parse({'benchmark_dataset':'mvtec'},'A.'),('A','A'))
            Path(ds['input_json']).write_text('modified',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Frozen.*input changed'): entry.verify(spec)

    def test_interrupt_stops_all_dataset_workers(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));processes=[];raised=[False]
            class Process:
                def __init__(self,log): self.log=log;self.stopped=False;self.waited=False
                def poll(self):
                    if not raised[0]: raised[0]=True;raise KeyboardInterrupt
                    return -15 if self.stopped else None
                def terminate(self): self.stopped=True
                def send_signal(self,sig): self.stopped=True
                def wait(self,**kwargs): self.waited=True;return -15
            def process(command,**kwargs):
                value=Process(kwargs['stdout']);processes.append(value);return value
            with patch.object(entry.subprocess,'Popen',side_effect=process): self.assertEqual(entry.run(args),130)
            self.assertEqual(len(processes),3)
            self.assertTrue(all(p.stopped and p.waited and p.log.closed for p in processes))

    def test_real_measurement_loop_excludes_warmups_and_keeps_observed_budget(self):
        import torch
        from test_ficoco_v import mini_model
        from llava_pruning.ficoco_v import FiCoCoVMethod
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            model,options=mini_model();core=model.get_model()
            samples=[SimpleNamespace(sample_id=str(gt),gt=gt,text='Fixed prompt',
                source=dict(benchmark_dataset='mvtec',dataset='MVTec-AD',image_size=[384,384])) for gt in (0,1)]
            class Backend:
                def __init__(self): self.model=model;self.calls=0
                def generate(self,sample,text,method,rate,mode,**kwargs):
                    assert text==sample.text and mode=='anyres_max_9'
                    assert kwargs['benchmark_metrics'] and kwargs['include_pruning_time']
                    assert not any(kwargs[k] for k in ('do_sample','capture_attention','capture_visualization'))
                    method.configure(core,rate);method.prepare_inputs((384,384),options['inputs'])
                    with method.inference_context(model),torch.inference_mode(): model.generate(**options)
                    self.calls+=1
                    return dict(answer='A' if sample.gt else 'B',stats=method.stats(core,rate),
                        lat_seconds=.01*self.calls,generation_seconds=.02*self.calls,
                        generated_tokens=3,peak_allocated_bytes=100,peak_reserved_bytes=200)
            backend=Backend();method=FiCoCoVMethod(core,50,start_layer=2)
            job=dict(job_id='mvtec_ficoco_v_50',dataset='mvtec',method='ficoco_v',prune_rate=50)
            with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
                output=Path(temporary);spec=dict(seed=42,warmup=2)
                result=entry.evaluate(backend,method,samples,spec,job,output,2**30)
                self.assertEqual(backend.calls,4);self.assertEqual(result['accuracy'],1.)
                self.assertAlmostEqual(result['lat_mean_ms'],35.)
                self.assertAlmostEqual(result['budget']['physical_average_prune_rate'],50,delta=.6)
                resumed=entry.evaluate(backend,method,samples,spec,job,output,2**30)
                self.assertEqual(backend.calls,4);self.assertEqual(resumed['fresh_samples'],0)
                self.assertEqual(resumed['lat_mean_ms'],result['lat_mean_ms'])
        finally: torch.set_num_threads(threads)


if __name__=='__main__': unittest.main()
