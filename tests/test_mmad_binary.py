"""MMAD task filtering, dynamic choice semantics, matched trials and clean timing."""

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

import benchmark_mmad_binary as entry
import benchmark_visa_methods as common
from llava_pruning.mmad_data import (load_binary, load_frozen, canonical_answer,
    parse_option, prompt, find_annotations, resolve_image, relative_image)


def fixture(root):
    data=root/'MMAD';data.mkdir()
    model=root/'model';model.mkdir();(model/'config.json').write_text('{"num_hidden_layers":28}',encoding='utf-8')
    student=root/'visa_student';student.mkdir();(student/'best.pt').write_bytes(b'visa-test-student')
    common.save_json(student/'train_run.json',dict(configuration={'dataset':'visa'},
        original_visa_test_is_training_source=True))
    records={}
    for source in ('DS-MVTec','MVTec-AD','MVTec-LOCO','VisA','GoodsAD'):
        for label in ('good','broken'):
            relative=f'{source}/object/image/{label}/000.png'
            path=data/relative;path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(384,384)).save(path)
            records[relative]=dict(image_path=relative,mask_path='NOT_USED.png',similar_templates=['MISSING_REF.png'],
                conversation=[dict(type='Anomaly Detection',Question='Is there any defect in the object?',
                    Options={'A':'No.','B':'Yes.','C':'Maybe.','D':'Unknown.'} if source=='VisA' else {'A':'No.','B':'Yes.'},
                    Answer='A' if label=='good' else 'B'),
                    dict(type='Defect Location',Question='Where?',Options={'A':'top','B':'left'},Answer='A')],
                conversation_alternate=[dict(type='Anomaly Detection',Question='DUPLICATE',Options={'A':'No.','B':'Yes.'},Answer='A')])
    # An annotated image without the binary task must not be measured.
    records['GoodsAD/object/image/broken/only_other.png']=dict(conversation=[
        dict(type='Defect Type',Question='What?',Options={'A':'hole','B':'scratch'},Answer='A')])
    common.save_json(data/'mmad.json',records)
    return entry.parser().parse_args(['--data-root',str(data),'--model-path',str(model),
        '--checkpoint',str(student/'best.pt'),'--output-dir',str(root/'output')])


def prediction(sample, choice=None, answer=None):
    answer=choice if answer is None else answer
    canonical,key=canonical_answer(sample.source,answer)
    return dict(question_id=sample.sample_id,gt=sample.gt,answer=answer,selected_option=key,
        canonical_answer=canonical,lat_seconds=.1,generation_seconds=.2,generated_tokens=2,
        peak_allocated_bytes=100,peak_reserved_bytes=200,budget={'physical_average_prune_rate':0.})


class MMADDataTests(unittest.TestCase):
    def test_binary_filter_freeze_and_no_reference_or_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=fixture(Path(temporary))
            rows,info=load_binary(args.data_root)
            self.assertEqual(len(rows),10);self.assertEqual(info['query_images'],11)
            self.assertEqual(info['source_counts'],{'MVTec-AD':4,'MVTec-LOCO':2,'VisA':2,'GoodsAD':2})
            self.assertEqual(info['binary_questions_with_distractors'],2)
            self.assertEqual(rows,load_binary(args.data_root)[0])
            self.assertNotEqual([r['options'] for r in rows],[r['options'] for r in load_binary(args.data_root,seed=7)[0]])
            for row in rows:
                self.assertEqual(row['text'],prompt(row))
                self.assertIn('Is there any defect in the object?',row['text'])
                self.assertNotIn('correct_option',row['text'])
                self.assertNotIn('DUPLICATE',row['text'])
                self.assertNotIn('MISSING_REF',row['text'])
                original='A' if row['gt']==0 else 'B'
                self.assertEqual(row['option_key_mapping'][row['correct_option']],original)

    def test_parser_does_not_assign_a_fixed_binary_class_or_force_uncertainty(self):
        row={'gt':0,'options':{'A':'Yes.','B':'Unknown.','C':'No.','D':'Maybe.'},'correct_option':'C'}
        self.assertEqual(canonical_answer(row,'C.'),('B','C'))
        self.assertEqual(canonical_answer(row,'Answer: A.'),('A','A'))
        self.assertEqual(canonical_answer(row,'Unknown.'),('','B'))
        self.assertEqual(canonical_answer(row,'Maybe.'),('','D'))
        self.assertEqual(canonical_answer(row,'I cannot decide.'),('',None))
        self.assertEqual(parse_option('(C)',row['options']),'C')
        # A normality question reverses Yes/No semantics; annotation resolves it.
        reverse=dict(row,gt=1,correct_option='C')
        self.assertEqual(canonical_answer(reverse,'No.'),('A','C'))
        self.assertEqual(canonical_answer(reverse,'Yes.'),('B','A'))
        self.assertIsNone(parse_option('Aardvark',row['options']))
        self.assertIsNone(parse_option('E',row['options']))

    def test_source_exclusion_happens_before_image_loading_and_normalizes_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=fixture(Path(temporary));root=Path(args.data_root)
            full,_=load_binary(root)
            # Excluded-source images need not even be installed on the server.
            for path in (root/'VisA').rglob('*.png'): path.unlink()
            rows,info=load_binary(root,exclude_sources=('VisA',))
            self.assertEqual(rows,[r for r in full if r['dataset']!='VisA'])
            self.assertEqual(len(rows),8)
            self.assertEqual(info['excluded_sources'],['VisA'])
            self.assertEqual(info['excluded_binary_questions'],{'VisA':2})
            self.assertEqual(info['selected_query_images'],8)
            self.assertEqual(info['source_counts'],{'MVTec-AD':4,'MVTec-LOCO':2,'GoodsAD':2})
            rows,_=load_binary(root,exclude_sources=('MVTec-AD','VisA'))
            self.assertTrue(all(r['raw_source'] not in {'DS-MVTec','MVTec-AD','VisA'} for r in rows))
            with self.assertRaisesRegex(ValueError,'Unknown MMAD'): load_binary(root,exclude_sources=('typo',))

    def test_frozen_source_validation_rejects_visa_and_mislabeled_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=fixture(Path(temporary));rows,_=load_binary(args.data_root)
            path=Path(temporary)/'frozen.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Excluded MMAD source'):
                load_frozen(path,exclude_sources=entry.EXCLUDED_SOURCES)
            row=rows[0];row['dataset']='GoodsAD'
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'differs from image path'): load_frozen(path)

    def test_missing_paths_ambiguity_and_traversal_are_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=fixture(Path(temporary));root=Path(args.data_root)
            self.assertEqual(find_annotations(root),(root/'mmad.json').resolve())
            (root/'annotations').mkdir()
            common.save_json(root/'annotations/mmad.json',{})
            with self.assertRaisesRegex(ValueError,'exactly one'): find_annotations(root)
            self.assertEqual(find_annotations(root,root/'mmad.json'),(root/'mmad.json').resolve())
            relative='DS-MVTec/object/image/good/000.png'
            duplicate=root/'images'/relative;duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes((root/relative).read_bytes())
            with self.assertRaisesRegex(ValueError,'2 candidate'): resolve_image(root,relative)
            self.assertEqual(resolve_image(root,relative,root),(root/relative).resolve())
            for value in ('../VisA/c/image/good/a.png','/VisA/c/image/good/a.png','C:/VisA/c/image/good/a.png','Unknown/c/image/good/a.png'):
                with self.assertRaises(ValueError): relative_image(value)
            with self.assertRaisesRegex(ValueError,'0 candidate'): resolve_image(root,'VisA/object/image/good/missing.png')

    def test_native_test_paths_prefix_aliases_and_annotation_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            cases=[('DS-MVTec','MVTec-AD','bottle'),('MVTec-AD','DS-MVTec','pill'),
                   ('MVTec-LOCO','MVTec-LOCO','breakfast_box'),('GoodsAD','GoodsAD','food_box')]
            for annotation_source,installed_source,category in cases:
                with self.subTest(source=annotation_source):
                    relative=f'{annotation_source}/{category}/image/broken/000.png'
                    native=root/installed_source/category/'test/broken/000.png'
                    native.parent.mkdir(parents=True,exist_ok=True);native.write_bytes(b'native')
                    self.assertEqual(resolve_image(root,relative),native.resolve())
                    converted=root/relative;converted.parent.mkdir(parents=True,exist_ok=True)
                    converted.write_bytes(b'converted MMAD')
                    self.assertEqual(resolve_image(root,relative),converted.resolve())

    def test_explicit_mvtec_root_is_authoritative_and_never_uses_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'MMAD';root.mkdir()
            external=Path(temporary)/'original_MVTec';external.mkdir()
            installed=root/'DS-MVTec/bottle/image/good/000.png'
            installed.parent.mkdir(parents=True);installed.write_bytes(b'converted')
            train=external/'bottle/train/good/000.png'
            train.parent.mkdir(parents=True);train.write_bytes(b'NOT A TEST IMAGE')
            relative='DS-MVTec/bottle/image/good/000.png'
            with self.assertRaisesRegex(ValueError,'0 candidate files') as error:
                resolve_image(root,relative,mvtec_root=external)
            self.assertIn(str((external/'bottle/test/good/000.png').resolve()),str(error.exception))
            native=external/'bottle/test/good/000.png'
            native.parent.mkdir(parents=True);native.write_bytes(b'external native')
            for prefix in ('DS-MVTec','MVTec-AD'):
                self.assertEqual(resolve_image(root,f'{prefix}/bottle/image/good/000.png',
                    mvtec_root=external),native.resolve())
            with self.assertRaisesRegex(NotADirectoryError,'mvtec-root not found'):
                resolve_image(root,relative,mvtec_root=external/'missing')
            duplicate=root/'images/MVTec-LOCO/c/test/good/000.png'
            original=root/'MVTec-LOCO/c/test/good/000.png'
            for path in (original,duplicate): path.parent.mkdir(parents=True);path.write_bytes(b'c')
            with self.assertRaisesRegex(ValueError,'2 candidate files'):
                resolve_image(root,'MVTec-LOCO/c/image/good/000.png',mvtec_root=external)

    def test_native_install_and_external_mvtec_freeze_resume(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()) as text:
            root=Path(temporary);args=fixture(root);data=Path(args.data_root)
            external=root/'original_MVTec';external.mkdir();args.mvtec_root=str(external)
            for source in ('DS-MVTec','MVTec-AD','MVTec-LOCO','GoodsAD'):
                for label in ('good','broken'):
                    original=data/source/'object/image'/label/'000.png'
                    target=(external if source in {'DS-MVTec','MVTec-AD'} else data/source)/'object/test'/label/'000.png'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(original.read_bytes());original.unlink()
            args.check_data=True
            self.assertEqual(entry.run(args),0);self.assertFalse(args.output_dir.exists())
            self.assertIn('MVTec image source:',text.getvalue())
            args.check_data=False
            _,spec=entry.setup(args);samples=load_frozen(spec['input_json'],exclude_sources=entry.EXCLUDED_SOURCES)
            self.assertEqual(len(samples),8)
            self.assertEqual(spec['dataset_info']['mvtec_root'],str(external.resolve()))
            for sample in samples:
                self.assertIn('/image/',sample.source['image'])
                self.assertIn('test',sample.image.parts)
                if sample.source['dataset']=='MVTec-AD': self.assertTrue(sample.image.is_relative_to(external.resolve()))
                self.assertEqual(sample.text,prompt(sample.source))
            args.resume=True
            self.assertEqual(entry.setup(args)[1],spec)

    def test_missing_image_fails_before_checkpoint_fingerprinting(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=fixture(Path(temporary));args.image_root=str(Path(temporary)/'missing_images')
            with patch.object(entry,'student_identity') as fingerprint:
                with self.assertRaisesRegex(ValueError,'0 candidate files'): entry.setup(args)
                fingerprint.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_check_data_needs_no_model_or_checkpoint_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()) as text:
            root=Path(temporary);args=fixture(root);args.check_data=True
            args.model_path='MISSING_MODEL';args.checkpoint='MISSING_STUDENT'
            self.assertEqual(entry.run(args),0)
            self.assertFalse(args.output_dir.exists());self.assertIn('8 Anomaly Detection',text.getvalue())
            self.assertIn('Excluded source question counts: {"VisA": 2}',text.getvalue())

    def test_manifest_uses_declared_visa_student_and_frozen_original_qa(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));output,spec=entry.setup(args)
            self.assertTrue(spec['checkpoint']['visa_test_used_as_training_source'])
            samples=load_frozen(spec['input_json'])
            self.assertEqual(len(samples),8)
            self.assertTrue(all(s.source['dataset']!='VisA' for s in samples))
            self.assertEqual(spec['dataset_info']['excluded_sources'],['VisA'])
            self.assertNotIn('VisA',spec['dataset_info']['source_counts'])
            self.assertTrue(all('exclude=VisA' in j['config'] for j in spec['jobs']))
            self.assertTrue(all(s.mask is None and s.bbox is None and s.text==prompt(s.source) for s in samples))
            self.assertEqual(len(spec['jobs']),25)
            self.assertEqual(sum(j['status']=='pending' for j in spec['jobs']),24)
            original=[j for j in spec['jobs'] if j.get('variant')=='base_guided']
            added=[j for j in spec['jobs'] if j.get('variant')=='joint_self']
            self.assertEqual(len(original),3);self.assertEqual(len(added),3)
            self.assertTrue(all('tail=9(extra)' in j['config'] and 'input=base_text' in j['config'] for j in original))
            self.assertTrue(all('tail=0' in j['config'] and 'input=full' in j['config'] and 'base_policy=joint' in j['config'] for j in added))
            self.assertTrue(all('student=VisA' in j['config'] for j in original+added))
            with self.assertRaisesRegex(ValueError,'VisA test'): common.checkpoint_identity(args.checkpoint)
            args.resume=True
            self.assertEqual(entry.setup(args)[1],spec)
            Path(spec['input_json']).write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'changed'): entry.setup(args)

    def test_strict_metrics_source_macro_and_distractors_remain_errors(self):
        # Source1: one normal right, three abnormal wrong (incl. uncertainty).
        # Source2: both classes right. Pooled ACC=3/6, BACC(parsed)=.75,
        # MMAD source-macro=(.5+1)/2=.75, pooled all-denominator BACC=.75.
        samples=[];rows=[]
        for source,labels,answers in (('MVTec-AD',[0,1,1,1],['B','B','C','???']),('VisA',[0,1],['B','A'])):
            for gt,answer in zip(labels,answers):
                sample=SimpleNamespace(sample_id=str(len(samples)),gt=gt,source=dict(dataset=source,gt=gt,
                    options={'A':'Yes.','B':'No.','C':'Maybe.'},correct_option='A' if gt else 'B'))
                samples.append(sample);rows.append(prediction(sample,answer=answer))
        result=entry.summarize_records(rows,samples,{'prune_rate':50},2**30)
        self.assertEqual((result['tp'],result['fp'],result['tn'],result['fn']),(1,0,2,1))
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['unparsed'],1);self.assertEqual(result['nonbinary'],1)
        self.assertEqual(result['parsed_samples'],4)
        self.assertEqual(result['bacc'],.75)
        self.assertEqual(result['mmad_score'],.75)
        self.assertAlmostEqual(result['lat_mean_ms'],100)
        self.assertAlmostEqual(result['sources']['MVTec-AD']['mmad_score'],.5)
        self.assertNotIn('baseline_accuracy',result)

    def test_source_macro_is_not_pooled_bacc(self):
        samples=[];rows=[]
        for source,labels in (('s1',[0,1]),('s2',[0]*5+[1]*5)):
            for gt in labels:
                sample=SimpleNamespace(sample_id=str(len(samples)),gt=gt,source=dict(dataset=source,gt=gt,
                    options={'A':'Yes.','B':'No.'},correct_option='A' if gt else 'B'))
                samples.append(sample)
                rows.append(prediction(sample,choice=sample.source['correct_option'] if source=='s1' else ('B' if gt else 'A')))
        result=entry.summarize_records(rows,samples,{'prune_rate':50},2**30)
        self.assertAlmostEqual(result['mmad_score'],.5)
        self.assertAlmostEqual(result['bacc'],1/6)

    def test_restore_preserves_raw_answers_and_rejects_wrong_canonical_choice(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));output,spec=entry.setup(args)
            samples=load_frozen(spec['input_json']);sample=samples[0]
            row=prediction(sample,sample.source['correct_option']);path=output/'predictions.jsonl'
            prefix=json.dumps(row)+'\n';path.write_text(prefix+'{"torn"',encoding='utf-8')
            self.assertEqual(entry.restore_records(path,samples),[row]);self.assertEqual(path.read_text(),prefix)
            path.write_text(json.dumps(row),encoding='utf-8')
            self.assertEqual(entry.restore_records(path,samples),[row]);self.assertEqual(path.read_text(),prefix)
            row['canonical_answer']='WRONG';path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'canonical'): entry.restore_records(path,samples)

    def test_sequential_launcher_report_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));calls=[]
            def process(command,**kwargs):
                calls.append(command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'5')
                self.assertIn('benchmark_mmad_binary.py',command[2])
                manifest=Path(command[command.index('--manifest')+1])
                spec=json.loads(manifest.read_text(encoding='utf-8'))
                job_id=command[command.index('--worker')+1]
                job=next(j for j in spec['jobs'] if j['job_id']==job_id)
                samples=load_frozen(spec['input_json'])
                report=entry.summarize_records([prediction(s,s.source['correct_option']) for s in samples],samples,job,2**30)
                folder=manifest.parent/job_id;folder.mkdir()
                common.save_json(folder/'metrics.json',dict(report,job_id=job_id,status='success',input_sha256=spec['input_sha256']))
                return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0)
            with patch.object(entry.subprocess,'Popen',side_effect=process):
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),24)
                args.resume=True
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),24)
            with (args.output_dir/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);self.assertEqual(tuple(reader.fieldnames),entry.FIELDS);rows=list(reader)
            self.assertEqual(len(rows),25)
            self.assertEqual(sum('status=unreachable' in r['config'] for r in rows),1)
            for row in rows:
                if 'status=unreachable' not in row['config']:
                    self.assertEqual(float(row['acc']),100)
                    self.assertEqual(float(row['relative_accuracy_pct']),100)
            with (args.output_dir/'summary_by_source.csv').open(encoding='utf-8-sig') as stream:
                source_rows=list(csv.DictReader(stream))
                self.assertEqual(len(source_rows),25*3)
                self.assertEqual({row['source'] for row in source_rows},{'MVTec-AD','MVTec-LOCO','GoodsAD'})

    def test_three_gpu_queues_failures_and_resume_keep_fixed_assignments(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.gpus=[5,6,7]
            busy=set();maximum=[0];attempts={};assignments={}
            class Process:
                def __init__(self,job,spec,manifest,gpu):
                    self.job,self.spec,self.manifest,self.gpu=job,spec,manifest,gpu
                    self.calls=0;self.done=False;self.code=None
                def poll(self):
                    if self.done: return self.code
                    self.calls+=1
                    if self.calls==1: return None
                    self.done=True;busy.remove(self.gpu)
                    folder=self.manifest.parent/self.job['job_id'];folder.mkdir(exist_ok=True)
                    fail=self.job['job_id']=='fastv_50' and attempts[self.job['job_id']]==1
                    if fail:
                        common.save_json(folder/'metrics.json',dict(job_id=self.job['job_id'],status='failed',
                            complete=False,input_sha256=self.spec['input_sha256'],error='simulated worker failure'))
                    else:
                        samples=load_frozen(self.spec['input_json'])
                        result=entry.summarize_records([prediction(s,s.source['correct_option']) for s in samples],
                            samples,self.job,2**30)
                        common.save_json(folder/'metrics.json',dict(result,job_id=self.job['job_id'],status='success',
                            input_sha256=self.spec['input_sha256']))
                    self.code=int(fail);return self.code
                def wait(self,**kwargs): return self.code
            def process(command,**kwargs):
                manifest=Path(command[command.index('--manifest')+1])
                spec=json.loads(manifest.read_text(encoding='utf-8'))
                job_id=command[command.index('--worker')+1]
                job=next(j for j in spec['jobs'] if j['job_id']==job_id)
                gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']
                self.assertEqual(gpu,job['gpu']);self.assertNotIn(gpu,busy)
                self.assertIn(gpu,{'5','6','7'})
                assignments.setdefault(job_id,gpu);self.assertEqual(assignments[job_id],gpu)
                attempts[job_id]=attempts.get(job_id,0)+1
                busy.add(gpu);maximum[0]=max(maximum[0],len(busy))
                return Process(job,spec,manifest,gpu)
            with patch.object(entry.subprocess,'Popen',side_effect=process),patch.object(entry.time,'sleep'):
                self.assertEqual(entry.run(args),1)
                self.assertEqual(maximum[0],3);self.assertFalse(busy)
                self.assertEqual(len(attempts),24)
                saved=json.loads((args.output_dir/'run.json').read_text(encoding='utf-8'))
                self.assertFalse(saved['complete']);self.assertEqual(saved['failed_trials'],['fastv_50'])
                self.assertEqual([sum(j.get('gpu')==g for j in saved['jobs']) for g in ('5','6','7')],[8,8,8])
                args.resume=True;args.gpus=[2]  # Resume restores the ORIGINAL GPU allocation.
                self.assertEqual(entry.run(args),0)
                self.assertEqual(sum(attempts.values()),25)
                self.assertEqual(attempts['fastv_50'],2)
                final=json.loads((args.output_dir/'run.json').read_text(encoding='utf-8'))
                self.assertTrue(final['complete']);self.assertEqual(final['completed_trials'],24)
                self.assertEqual(final['gpus'],['5','6','7'])
            with (args.output_dir/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),25)
            self.assertTrue(all('gpu=' in r['config'] for r in rows if 'status=unreachable' not in r['config']))

    def test_interrupt_stops_every_gpu_worker_and_closes_logs(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.gpus=[5,6,7];processes=[];raised=[False]
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
            with patch.object(entry.subprocess,'Popen',side_effect=process):
                self.assertEqual(entry.run(args),130)
            self.assertEqual(len(processes),3)
            self.assertTrue(all(p.stopped and p.waited and p.log.closed for p in processes))

    def test_gpu_list_validation_and_single_gpu_compatibility(self):
        args=entry.parser().parse_args(['--gpus','5','6','7'])
        self.assertEqual(args.gpus,[5,6,7])
        self.assertEqual(entry.parser().parse_args(['--gpu','6']).gpu,'6')
        for gpus in ([],[5,5],[-1],['5,6']):
            with self.assertRaisesRegex(ValueError,'GPU IDs'): entry.schedule([50],gpus)


class MMADGenerationTests(unittest.TestCase):
    def test_joint_self_factory_real_generation_at_all_three_rates(self):
        import torch
        from test_pool_ablations import model_and_options
        from test_attention_predictor import make_checkpoint
        from llava_pruning.visa_comparison_methods import observe_prefill, observed_budget
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for rate in (50,70,90):
                with self.subTest(rate=rate),tempfile.TemporaryDirectory() as temporary:
                    torch.manual_seed(1801)
                    model,options=model_and_options(28);core=model.get_model()
                    checkpoint=Path(temporary)/'best.pt';make_checkpoint(checkpoint,core)
                    job=next(j for j in entry.schedule([rate]) if j.get('variant')=='joint_self')
                    method=entry.make_trial_method(job,core,checkpoint)
                    self.assertEqual((method.mode,method.predictor_input,method.tail_drop_layers,method.share),
                        ('anyres_self','full',0,.7))
                    self.assertEqual(method.base_policy,'joint')
                    method.configure(core,rate)
                    method.prepare_inputs((384,384),options['inputs'])
                    with patch.object(core.attention_predictor,'forward',wraps=core.attention_predictor.forward) as predict,\
                         method.generation_context(model),observe_prefill(core) as lengths,torch.inference_mode():
                        generated=model.generate(**options)
                    stats=method.stats(core,rate)
                    budget=observed_budget(job['method'],rate,stats,lengths,1485)
                    self.assertEqual(generated.sequences.shape[-1],3)
                    self.assertEqual(predict.call_count,1)
                    self.assertEqual(predict.call_args.args[0].shape[1],stats['pre_llm_tokens']+4)
                    self.assertEqual(len(predict.call_args.args[2]),len(method.adapter.spatial))
                    self.assertEqual(stats['stage2_base_tokens']+stats['stage2_anyres_tokens']+stats['protected_newlines'],
                        stats['final_llm_tokens'])
                    self.assertEqual(stats['stage2_base_policy'],'joint')
                    self.assertEqual(budget['actual_tail_contribution'],0)
                    self.assertGreater(budget['final_layer_visual_tokens'],0)
                    self.assertAlmostEqual(budget['actual_average_prune_rate'],rate,delta=.15)
                    self.assertEqual([lengths[i] for i in range(28)],
                        [stats['pre_llm_tokens']+4]*2+[stats['final_llm_tokens']+4]*26)
                    self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))
                    original=next(j for j in entry.schedule([rate]) if j.get('variant')=='base_guided')
                    original_method=entry.make_trial_method(original,core,checkpoint)
                    self.assertEqual((original_method.mode,original_method.predictor_input,original_method.tail_drop_layers),
                        ('base_full','base_text',9))
                    self.assertEqual(original_method.base_policy,'anyres_first')
        finally: torch.set_num_threads(threads)

    def test_measurement_loop_passes_frozen_qa_excludes_warmups_and_resumes(self):
        import torch
        from test_pool_ablations import model_and_options
        from llava_pruning.visa_comparison_methods import make_method
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            model,options=model_and_options(28);core=model.get_model()
            samples=[]
            for gt in (0,1):
                source=dict(dataset='VisA',gt=gt,question='Is there any defect?',image_size=[384,384],
                    options={'A':'No.','B':'Yes.'},correct_option='B' if gt else 'A')
                samples.append(SimpleNamespace(sample_id=str(gt),gt=gt,source=source,text=prompt(source)))
            class Backend:
                def __init__(self): self.model=model;self.calls=0
                def generate(self,sample,text,method,rate,mode,**kwargs):
                    assert text==sample.text and mode=='anyres_max_9'
                    assert kwargs['benchmark_metrics'] and kwargs['include_pruning_time']
                    assert not any(kwargs[k] for k in ('do_sample','capture_visualization','capture_attention'))
                    method.configure(core,rate,capture_visualization=False,capture_attention=False)
                    with torch.inference_mode(): model.generate(**options)
                    self.calls+=1
                    return dict(answer='A' if sample.gt==0 else 'B',stats=method.stats(core,rate),
                        lat_seconds=.02*self.calls,generation_seconds=.03*self.calls,generated_tokens=3,
                        peak_allocated_bytes=100,peak_reserved_bytes=200)
            backend=Backend();method=make_method('llava_unpruned',core,0,None)
            job=common.schedule([50])[0]
            with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
                output=Path(temporary);spec={'seed':42,'warmup':2}
                result=entry.evaluate_samples(backend,method,samples,spec,job,output,2**30)
                self.assertEqual(backend.calls,4);self.assertEqual(result['accuracy'],1.)
                self.assertEqual((result['tp'],result['tn']),(1,1))
                self.assertAlmostEqual(result['lat_mean_ms'],70.)
                resumed=entry.evaluate_samples(backend,method,samples,spec,job,output,2**30)
                self.assertEqual(backend.calls,4);self.assertEqual(resumed['fresh_samples'],0)
                self.assertEqual(resumed['warmup_samples'],0)
                self.assertEqual(resumed['lat_mean_ms'],result['lat_mean_ms'])
                self.assertTrue(all(not l._forward_pre_hooks for l in core.layers))
        finally: torch.set_num_threads(threads)


if __name__=='__main__': unittest.main()
