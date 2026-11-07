"""MPDD-only data, both Base-guided tails, shared measurements and two-GPU queues."""

from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import benchmark_mpdd as entry
import benchmark_visa_methods as common
from llava_pruning.data import load_samples
from llava_pruning.mpdd_data import CATEGORIES, load_mpdd_test
from llava_pruning.prompts import GENERIC_PROMPT_V0


def fixture(root):
    data=root/'MPDD';data.mkdir()
    for category in CATEGORIES:
        for label in ('good','hole'):
            path=data/category/'test'/label/'000.png';path.parent.mkdir(parents=True)
            Image.new('RGB',(32,32),(130,70,20)).save(path)
        # Must not be opened or evaluated, even when invalid as image files.
        for relative in ('train/good/not_test.png','ground_truth/hole/000.png'):
            path=data/category/relative;path.parent.mkdir(parents=True);path.write_bytes(b'NOT A TEST IMAGE')
    model=root/'model';model.mkdir()
    common.save_json(model/'config.json',dict(num_hidden_layers=28))
    student=root/'visa_student';student.mkdir();(student/'best.pt').write_bytes(b'visa-student')
    common.save_json(student/'train_run.json',dict(configuration={'dataset':'visa'},
        original_visa_test_is_training_source=True))
    return entry.parser().parse_args(['--data-root',str(data),'--model-path',str(model),
        '--checkpoint',str(student/'best.pt'),'--output-dir',str(root/'output')])


def predictions(samples, wrong=0):
    return [dict(question_id=s.sample_id,gt=s.gt,answer='A' if (s.gt==1) != (i<wrong) else 'B',
        lat_seconds=.1,generation_seconds=.2,generated_tokens=2,peak_allocated_bytes=100,
        peak_reserved_bytes=200,budget={}) for i,s in enumerate(samples)]


class MPDDDataTests(unittest.TestCase):
    def test_six_test_categories_only_generic_v0_and_native_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=fixture(Path(temporary));data=Path(args.data_root)
            large=data/'bracket_black/test/good/000.png'
            Image.new('RGB',(1024,1024)).save(large)
            before={p:p.read_bytes() for p in data.rglob('*.png')}
            samples,info=load_mpdd_test(data)
            self.assertEqual((len(samples),info['normal'],info['abnormal']),(12,6,6))
            self.assertEqual(info['image_sizes'],{'1024x1024':1,'32x32':11})
            self.assertEqual(set(info['categories']),set(CATEGORIES))
            self.assertEqual([s.sample_id for s in samples],sorted(s.sample_id for s in samples))
            for s in samples:
                self.assertIn('test',s.image.parts)
                self.assertEqual(s.gt,int(s.image.parent.name!='good'))
                self.assertEqual(s.text,GENERIC_PROMPT_V0)
                self.assertIsNone(s.mask);self.assertIsNone(s.bbox)
                self.assertEqual(s.source['image_size'],[1024,1024] if s.image==large.resolve() else [32,32])
            self.assertEqual({p:p.read_bytes() for p in data.rglob('*.png')},before)

    def test_missing_class_unlabeled_images_and_reserved_folders_fail(self):
        for change,message in (('missing','missing category/test'),('unlabeled','Unlabeled'),
                ('reserved','Reserved/unlabeled'),('empty','both normal and abnormal')):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                args=fixture(Path(temporary));root=Path(args.data_root)
                if change=='missing': (root/'tubes/test').rename(root/'tubes/test_missing')
                if change=='unlabeled': Image.new('RGB',(2,2)).save(root/'tubes/test/unknown.png')
                if change=='reserved': (root/'tubes/test/masks').mkdir()
                if change=='empty': (root/'tubes/test/hole/000.png').unlink()
                with self.assertRaisesRegex(ValueError,message): load_mpdd_test(root)

    def test_check_data_writes_nothing_and_needs_no_model_or_student(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()) as output:
            args=fixture(Path(temporary));args.check_data=True
            args.model_path='MISSING_MODEL';args.checkpoint='MISSING_STUDENT'
            self.assertEqual(entry.run(args),0)
            self.assertIn('MPDD test: 12 images',output.getvalue())
            self.assertIn('32x32',output.getvalue())
            self.assertFalse(args.output_dir.exists())


class MPDDSetupTests(unittest.TestCase):
    def test_full_schedule_and_tail_labels_use_fixed_method_gpu_assignments(self):
        jobs=entry.schedule([50,70,90],[3,4])
        self.assertEqual(len(jobs),25)
        self.assertEqual(sum(j['status']=='pending' for j in jobs),24)
        self.assertEqual([j['job_id'] for j in jobs if j['status']=='unreachable'],['pyramiddrop_90'])
        self.assertEqual(sum(j['method']=='llava_unpruned' for j in jobs),1)
        for method in entry.METHODS:
            selected=[j for j in jobs if j['method']==method]
            self.assertEqual(len(selected),3)
            self.assertEqual(len({j['gpu'] for j in selected}),1)
        for tail in (9,0):
            selected=[j for j in jobs if j.get('tail_drop_layers')==tail]
            self.assertEqual(len(selected),3)
            self.assertTrue(all(f'tail={tail}'+('(extra)' if tail else '') in j['config'] for j in selected))
            self.assertTrue(all('lambda=0.7;base_full;input=base_text;mapping=fast;base_policy=anyres_first' in j['config']
                for j in selected))
        self.assertTrue(all('dataset=MPDD;resize=native' in j['config'] for j in jobs))
        for gpus in ([],[3,3],[-1],['3']):
            with self.assertRaises(ValueError): entry.schedule([50],gpus)

    def test_setup_resume_keeps_same_images_student_prompt_and_current_source_policy(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));model=Path(args.model_path)/'config.json'
            original=model.read_bytes()
            output,spec=entry.setup(args)
            self.assertEqual(spec['gpus'],['3','4'])
            self.assertEqual(spec['source_policy'],'current_directory')
            self.assertTrue(spec['checkpoint']['visa_test_used_as_training_source'])
            self.assertFalse((output/'_runtime').exists())
            self.assertFalse(list(output.rglob('*.png')))
            samples=load_samples(spec['input_json'],spec['data_root'])
            self.assertEqual(len(samples),12)
            self.assertTrue(all(s.text==GENERIC_PROMPT_V0 and '/test/' in s.source['image'] for s in samples))
            self.assertEqual(model.read_bytes(),original)
            with patch.object(common,'source_identity',return_value={'sha256':'later-edits','files':1}):
                entry.verify(spec)
                self.assertEqual(common.verify_worker_source(spec)['sha256'],'later-edits')
            args.resume=True;args.gpus=[7];args.prune_rates=[90]
            self.assertEqual(entry.setup(args),(output,spec))
            model.write_text('{"num_hidden_layers":28,"changed":true}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Model config changed'): entry.setup(args)

    def test_resume_rejects_changed_images_and_student(self):
        for change,message in (('image','MPDD test selection'),('student','student checkpoint')):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
                args=fixture(Path(temporary));_,spec=entry.setup(args);args.resume=True
                if change=='image':
                    Image.new('RGB',(34,32)).save(Path(args.data_root)/'tubes/test/good/000.png')
                else: Path(args.checkpoint).write_bytes(b'different-student')
                with self.assertRaisesRegex(ValueError,message): entry.setup(args)

    def test_two_gpu_launcher_metrics_failure_and_resume_do_not_repeat_successes(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()) as output:
            args=fixture(Path(temporary));busy=set();attempts={};maximum=[0]
            class Process:
                def __init__(self,command,**kwargs):
                    self.manifest=Path(command[command.index('--manifest')+1])
                    self.spec=json.loads(self.manifest.read_text(encoding='utf-8'))
                    self.job=next(j for j in self.spec['jobs'] if j['job_id']==command[command.index('--worker')+1])
                    self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']
                    if Path(command[2]).name!='benchmark_mpdd.py' or self.gpu in busy or self.gpu!=self.job['gpu']:
                        raise AssertionError('Wrong worker or overlapping GPU')
                    busy.add(self.gpu);maximum[0]=max(maximum[0],len(busy));self.done=False
                    attempts[self.job['job_id']]=attempts.get(self.job['job_id'],0)+1
                def poll(self):
                    if self.done: return self.code
                    self.done=True;busy.remove(self.gpu)
                    folder=self.manifest.parent/self.job['job_id'];folder.mkdir(exist_ok=True)
                    fail=self.job['job_id']=='fastv_50' and attempts['fastv_50']==1
                    samples=load_samples(self.spec['input_json'],self.spec['data_root'])
                    result=common.summarize_records(predictions(samples,1 if self.job['prune_rate']==0 else 2),
                        len(samples),self.job,2**30)
                    common.save_json(folder/'metrics.json',dict(result,job_id=self.job['job_id'],
                        status='failed' if fail else 'success',complete=not fail,input_sha256=self.spec['input_sha256']))
                    self.code=int(fail);return self.code
            with patch.object(entry.parallel.time,'sleep'):
                self.assertEqual(entry.run(args,process_factory=Process),1)
                self.assertEqual(len(attempts),24);self.assertEqual(maximum[0],2);self.assertFalse(busy)
                args.resume=True
                self.assertEqual(entry.run(args,process_factory=Process),0)
                self.assertEqual(sum(attempts.values()),25);self.assertEqual(attempts['fastv_50'],2)
            self.assertIn('MPDD test images; native AnyRes max9',output.getvalue())
            self.assertNotIn('official VisA test images',output.getvalue())
            with (args.output_dir/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);self.assertEqual(tuple(reader.fieldnames),common.FIELDS);rows=list(reader)
            self.assertEqual(len(rows),25)
            self.assertEqual(float(rows[0]['relative_accuracy_pct']),100)
            completed=[r for r in rows[1:] if 'status=unreachable' not in r['config']]
            self.assertTrue(all(abs(float(r['relative_accuracy_pct'])-100*10/11)<1e-8 for r in completed))
            self.assertTrue(all(float(r['Lat'])==100 for r in completed))
            self.assertTrue(all(r['bacc'] and r['f1'] and r['mem'] for r in completed))
            saved=json.loads((args.output_dir/'run.json').read_text(encoding='utf-8'))
            self.assertTrue(saved['complete']);self.assertEqual(saved['completed_trials'],24)


class MPDDGenerationTests(unittest.TestCase):
    def test_real_worker_uses_both_tail_configs_and_shared_timing_loop(self):
        import torch
        from test_pool_ablations import model_and_options
        from test_attention_predictor import make_checkpoint
        from llava_pruning import backend as backend_module
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for tail in (9,0):
                with self.subTest(tail=tail),tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
                    torch.manual_seed(117)
                    args=fixture(Path(temporary));args.warmup=1;args.prune_rates=[50]
                    model,options=model_and_options(28);core=model.get_model()
                    make_checkpoint(Path(args.checkpoint),core)
                    output,spec=entry.setup(args)
                    job=next(j for j in spec['jobs'] if j.get('tail_drop_layers')==tail)
                    calls=[]
                    class Backend:
                        def __init__(self,path,mode,**kw):
                            self.model=model;self.inference_config={'test_backend':'CPU tiny model'}
                            assert mode=='anyres_max_9' and kw['image_token_order']=='base_first'
                        def generate(self,sample,prompt,method,rate,mode,**kwargs):
                            assert prompt==GENERIC_PROMPT_V0 and mode=='anyres_max_9'
                            assert method.tail_drop_layers==tail and method.mode=='base_full'
                            assert method.predictor_input=='base_text' and method.adapter.base_mapping=='fast'
                            assert method.base_policy=='anyres_first'
                            assert kwargs['benchmark_metrics'] and kwargs['include_pruning_time']
                            assert not any(kwargs[k] for k in ('do_sample','capture_attention','capture_visualization'))
                            method.configure(core,rate,capture_visualization=False,capture_attention=False)
                            method.prepare_inputs(tuple(sample.source['image_size']),options['inputs'])
                            with torch.inference_mode(): model.generate(**options)
                            calls.append(sample.sample_id)
                            return dict(answer='A' if sample.gt else 'B',stats=method.stats(core,rate),
                                lat_seconds=.1,generation_seconds=.2,generated_tokens=3,
                                peak_allocated_bytes=100,peak_reserved_bytes=200)
                    device=SimpleNamespace(name='CPU test double',total_memory=2**30)
                    with patch.object(backend_module,'LlavaBackend',Backend),\
                         patch.object(common,'validate_single_gpu'),\
                         patch.object(torch.cuda,'get_device_properties',return_value=device),\
                         patch.dict(os.environ,CUDA_VISIBLE_DEVICES=job['gpu']):
                        self.assertEqual(entry.worker(output/'run.json',job['job_id']),0)
                    result=json.loads((output/job['job_id']/'metrics.json').read_text(encoding='utf-8'))
                    self.assertEqual(result['status'],'success');self.assertTrue(result['complete'])
                    self.assertEqual((len(calls),result['evaluated_samples'],result['warmup_samples']),(13,12,1))
                    self.assertEqual(result['accuracy'],1.)
                    self.assertAlmostEqual(result['lat_mean_ms'],100)
                    budget=result['budget']
                    self.assertAlmostEqual(budget['two_stage_average_prune_rate'],50,delta=.15)
                    self.assertEqual(budget['final_layer_visual_tokens']==0,tail==9)
                    self.assertEqual(budget['actual_tail_contribution']>0,tail==9)
                    self.assertFalse(list(output.rglob('*.png')))
                    self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))
        finally: torch.set_num_threads(threads)


if __name__=='__main__':
    unittest.main()
