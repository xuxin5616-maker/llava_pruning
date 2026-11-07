"""Reusing historical MMAD answers/timings must never regenerate or relabel them."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_mmad_binary as entry
import benchmark_visa_methods as common
from llava_pruning.mmad_data import load_frozen
from llava_pruning.mmad_extension import prepare_reuse, verify_imported_prefix
from test_mmad_binary import fixture, prediction


def historical(root, *, complete=True, legacy=False):
    args=fixture(root);args.gpus=[5,6,7];output,spec=entry.setup(args)
    samples=load_frozen(spec['input_json'])
    if legacy:
        spec['version']='1.4-mmad-native-test-paths'
        for name in ('include_visa','experiment_group','base_guided_tail'): spec.pop(name,None)
        for job in spec['jobs']: job.pop('tail_drop_layers',None)
        common.save_json(output/'run.json',spec)
    for job in spec['jobs']:
        if job['status']!='pending': continue
        folder=output/job['job_id'];folder.mkdir()
        rows=[prediction(s,s.source['correct_option']) for s in samples]
        if not complete and job['job_id']=='fastv_50': rows=rows[:3]
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        result=entry.summarize_records(rows,samples,job,2**30)
        common.save_json(folder/'metrics.json',dict(result,job_id=job['job_id'],input_sha256=spec['input_sha256'],
            status='success' if result['complete'] else 'failed',inference={'precision':'float16','attention':'flash_attention_2'}))
        common.save_json(folder/'inference.json',dict(inference={'precision':'float16','attention':'flash_attention_2'},
            device_name='TestGPU',device_total_bytes=2**30))
    new=entry.parser().parse_args(['--include-visa','--reuse-dir',str(output),'--gpus','5','6','7',
        '--output-dir',str(root/'with_visa')])
    return args,output,spec,new


def snapshot(root):
    return {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}


class ExtensionTests(unittest.TestCase):
    def test_reuse_is_read_only_preserves_every_frozen_choice_and_valid_prefix(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);_,old,saved,args=historical(root,complete=False,legacy=True)
            # Valid final JSON without newline and a torn final JSON are both
            # copied safely. Neither historical source is repaired in-place.
            path=old/'fastv_50/predictions.jsonl';path.write_bytes(path.read_bytes()+b'{"torn"')
            path=old/'llava_unpruned_00/predictions.jsonl';path.write_bytes(path.read_bytes().rstrip(b'\n'))
            before=snapshot(old)
            output,spec=entry.setup(args)
            self.assertEqual(snapshot(old),before)
            samples=load_frozen(spec['input_json']);previous=load_frozen(saved['input_json'])
            self.assertEqual([s.source for s in samples[:8]],[s.source for s in previous])
            self.assertEqual([s.source['dataset'] for s in samples[8:]],['VisA']*2)
            self.assertEqual((spec['sample_count'],spec['dataset_info']['source_counts']['VisA']),(10,2))
            self.assertEqual(spec['seed'],saved['seed']);self.assertEqual(spec['checkpoint'],saved['checkpoint'])
            self.assertEqual(spec['model_path'],saved['model_path'])
            self.assertEqual(spec['dataset_info']['excluded_sources'],[])
            self.assertIn('NOT independent',spec['training_overlap']['VisA'])
            self.assertTrue(all('exclude=none' in j['config'] for j in spec['jobs']))
            self.assertTrue(all('training_overlap=VisA' in j['config'] for j in spec['jobs'] if j['method']=='hybrid_baseline'))
            self.assertTrue(all('training_overlap' not in j['config'] for j in spec['jobs'] if j['method']!='hybrid_baseline'))
            self.assertEqual(spec['reuse_history']['trials']['fastv_50']['records'],3)
            args.gpus=None;args.gpu='5'
            self.assertEqual({j.get('gpu') for j in prepare_reuse(args,entry,fingerprint_student=False)[3]
                              if j['status']=='pending'},{'5'})
            self.assertEqual(len(entry.restore_records(output/'fastv_50/predictions.jsonl',samples)),3)
            for job in spec['jobs']:
                if job['status']=='pending': verify_imported_prefix(output/job['job_id'],spec,job)
            args.resume=True
            # New output is self-contained: original directory is not needed on resume.
            old.rename(root/'archived_old')
            self.assertEqual(entry.setup(args)[1],spec)

    def test_full_include_visa_and_default_exclusion_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.include_visa=True
            _,spec=entry.setup(args)
            self.assertEqual(spec['sample_count'],10)
            self.assertEqual(spec['dataset_info']['excluded_sources'],[])
            entry.verify_inputs(spec)
            self.assertIn('including VisA',spec['definitions']['relative_accuracy'])

    def test_check_only_does_not_need_student_or_create_output(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);_,_,saved,args=historical(root)
            Path(saved['checkpoint']['path']).unlink();args.check_data=True
            with patch.object(entry,'student_identity') as fingerprint:
                self.assertEqual(entry.run(args),0);fingerprint.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_corrupt_choices_runtime_or_completed_rows_are_rejected_without_writing(self):
        for failure in ('choice','runtime','missing_records','configuration','input','student','image'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
                root=Path(temporary);_,old,spec,args=historical(root)
                folder=old/'fastv_50';path=folder/'predictions.jsonl'
                if failure=='choice':
                    rows=common.read_prefix(path)[0];rows[0]['canonical_answer']='WRONG'
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                elif failure=='runtime': (folder/'inference.json').unlink()
                elif failure=='missing_records': path.unlink()
                elif failure=='configuration':
                    spec['jobs'][1]['config']+=';layer=20';common.save_json(old/'run.json',spec)
                elif failure=='input': Path(spec['input_json']).write_text('{}',encoding='utf-8')
                elif failure=='student': Path(spec['checkpoint']['path']).write_bytes(b'changed')
                elif failure=='image': load_frozen(spec['input_json'])[0].image.write_bytes(b'changed')
                with self.assertRaises((ValueError,OSError)): entry.setup(args)
                self.assertFalse(args.output_dir.exists())

    def test_output_separation_and_import_checksum(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);_,old,_,args=historical(root)
            for target in (old,old/'child',old.parent):
                args.output_dir=target
                with self.assertRaises(ValueError): prepare_reuse(args,entry,fingerprint_student=False)
            args.output_dir=root/'new';args.include_visa=False
            with self.assertRaisesRegex(ValueError,'requires --include-visa'): entry.setup(args)
            args.include_visa=True;output,spec=entry.setup(args);job=spec['jobs'][0]
            path=output/job['job_id']/'predictions.jsonl'
            path.write_bytes(path.read_bytes().replace(b'"lat_seconds": 0.1',b'"lat_seconds": 0.2',1))
            with self.assertRaisesRegex(ValueError,'Imported historical'): verify_imported_prefix(path.parent,spec,job)

    def test_parallel_completion_exports_pooled_and_four_source_metrics_with_resume(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);_,old,_,args=historical(root);before=snapshot(old);calls=[]
            def process(command,**kwargs):
                manifest=Path(command[command.index('--manifest')+1]);spec=json.loads(manifest.read_text(encoding='utf-8'))
                job=next(j for j in spec['jobs'] if j['job_id']==command[command.index('--worker')+1])
                folder=manifest.parent/job['job_id'];samples=load_frozen(spec['input_json'])
                rows=entry.restore_records(folder/'predictions.jsonl',samples)
                self.assertEqual(len(rows),8);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],job['gpu'])
                self.assertTrue(all(s.source['dataset']=='VisA' for s in samples[len(rows):]))
                calls.append(job['job_id'])
                for sample in samples[len(rows):]:
                    row=prediction(sample,'???');row.update(lat_seconds=.4,generation_seconds=.8,peak_allocated_bytes=300)
                    rows.append(row)
                result=entry.summarize_records(rows,samples,job,2**30)
                (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                common.save_json(folder/'metrics.json',dict(result,job_id=job['job_id'],status='success',input_sha256=spec['input_sha256']))
                return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0)
            with patch.object(entry.subprocess,'Popen',side_effect=process):
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),24)
                args.resume=True;self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),24)
            self.assertEqual(snapshot(old),before)
            with (args.output_dir/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),25)
            for row in rows:
                if 'status=unreachable' in row['config']: self.assertEqual(row['acc'],'');continue
                self.assertEqual(float(row['acc']),80.)
                self.assertAlmostEqual(float(row['Lat']),160.)  # 8*100ms + 2*400ms, divided by10
                self.assertAlmostEqual(float(row['mem']),300/2**30)  # maximum, never mean/sum
                self.assertEqual(float(row['relative_accuracy_pct']),100.)
            with (args.output_dir/'summary_by_source.csv').open(encoding='utf-8-sig',newline='') as stream:
                sources=list(csv.DictReader(stream))
            self.assertEqual(len(sources),25*4);self.assertEqual({r['source'] for r in sources},{'VisA','MVTec-AD','MVTec-LOCO','GoodsAD'})
            self.assertTrue(all(float(r['acc'])==100. for r in sources if r['acc'] and r['source']!='VisA'))

    def test_timing_loop_only_generates_remaining_visa_with_real_tiny_llm(self):
        import torch
        from test_pool_ablations import model_and_options
        from llava_pruning.visa_comparison_methods import make_method
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
                root=Path(temporary);_,_,_,args=historical(root);output,spec=entry.setup(args)
                spec['warmup']=2;model,options=model_and_options(28);core=model.get_model();calls=[]
                class Backend:
                    def __init__(self): self.model=model
                    def generate(self,sample,text,method,rate,mode,**kwargs):
                        assert sample.source['dataset']=='VisA'
                        assert text==sample.text and kwargs['do_sample'] is False
                        calls.append(sample.sample_id)
                        method.configure(core,rate,capture_visualization=False,capture_attention=False)
                        with torch.inference_mode(): model.generate(**options)
                        return dict(answer=sample.source['correct_option'],stats=method.stats(core,rate),
                            lat_seconds=.4,generation_seconds=.8,generated_tokens=2,peak_allocated_bytes=300,peak_reserved_bytes=400)
                method=make_method('llava_unpruned',core,0,None);samples=load_frozen(spec['input_json']);job=spec['jobs'][0]
                report=entry.evaluate_samples(Backend(),method,samples,spec,job,output/job['job_id'],2**30)
                self.assertEqual((len(calls),report['fresh_samples'],report['reused_from_previous'],report['resumed_samples']),(4,2,8,8))
                self.assertEqual(report['accuracy'],1.)
                self.assertAlmostEqual(report['lat_mean_ms'],160.)
                report2=entry.evaluate_samples(Backend(),method,samples,spec,job,output/job['job_id'],2**30)
                self.assertEqual(len(calls),4);self.assertEqual(report2['fresh_samples'],0)
                self.assertEqual(report2['lat_mean_ms'],report['lat_mean_ms'])
        finally: torch.set_num_threads(threads)


if __name__=='__main__': unittest.main()
