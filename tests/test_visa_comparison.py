"""Matched VisA budgets, real tiny-model generation, reporting and resume."""

from contextlib import nullcontext, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import benchmark_visa_methods as entry
from llava_pruning.visa_comparison_methods import (AverageFastV, make_method, native_counts,
    observed_budget, observe_prefill, pack_variable_visionzip, single_drop_keep, visionzip_view_keeps)
from llava_pruning.prompts import GENERIC_PROMPT_V0
from test_pool_ablations import model_and_options
from test_attention_predictor import make_checkpoint
import runVisionZip as zip_entry


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_schedule_and_unsupported_are_explicit(self):
        jobs = entry.schedule([50,70,90])
        self.assertEqual(len(jobs),22)
        self.assertEqual(sum(j['status']=='pending' for j in jobs),21)
        unavailable = [j for j in jobs if j['status']=='unreachable']
        self.assertEqual([j['job_id'] for j in unavailable],['pyramiddrop_90'])
        self.assertIn('7/28',unavailable[0]['reason'])
        self.assertIn('tail=9(extra)',next(j['config'] for j in jobs if j['method']=='hybrid_baseline'))

    def test_integer_budgets_reach_average_instead_of_final_rate(self):
        for original in (1485,7290+81,7126,9999):
            for rate in (50,70,90):
                keep = single_drop_keep(original,rate,28,2)
                actual = 100*(1-(2*original+26*keep)/(28*original))
                self.assertGreaterEqual(actual+1e-12,rate)
                self.assertLess(actual-rate,100/original)
                target,protected,keeps = visionzip_view_keeps(original,10,81,rate)
                self.assertEqual(sum(keeps)+protected,target)
                self.assertEqual(target,original*(100-rate)//100)
                self.assertLessEqual(max(keeps)-min(keeps),1)
        with self.assertRaises(ValueError): single_drop_keep(100,95,28,2)

    def test_variable_packing_matches_existing_visionzip_and_raw_newlines(self):
        torch.manual_seed(1)
        for grid in ((1,1),(3,2)):
            views = 1+grid[0]*grid[1]
            features = torch.randn(views,32,8)
            indices = torch.stack([torch.randperm(729)[:27].sort().values for _ in range(views)])
            decisions = zip_entry.TokenDecisions(729,32,indices)
            newline = torch.full((8,),-99.)
            for final in (False,True):
                merge = 'spatial_unpad_add_newl' if final else 'spatial_unpad'
                expected = zip_entry.pack_sparse_views(features,decisions,grid,27,image_newline=newline,merge_type=merge)
                actual = pack_variable_visionzip(list(features),list(indices),[27]*views,grid,newline,final)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                self.assertEqual(int((actual[:,0]==-99).sum()),grid[1]*27+int(final))

    def test_real_generation_every_method_all_three_rates_and_budget_observers(self):
        # Real autoregressive forward/KV updates with native local method adapters;
        # no mocked pruning counts. CUDA/FP16/Flash numerical parity needs server.
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'best.pt'
            for name in ('llava_unpruned',*entry.METHODS):
                for rate in ([0] if name=='llava_unpruned' else [50,70,90]):
                    if name=='pyramiddrop' and rate==90: continue
                    with self.subTest(method=name,rate=rate):
                        torch.manual_seed(2)
                        model,options = model_and_options(28)
                        core = model.get_model()
                        if name=='anchorprune':
                            from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention, SigLipVisionConfig
                            core.vision_tower.blocks[-1].self_attn = SigLipAttention(
                                SigLipVisionConfig(hidden_size=8,num_attention_heads=2))
                        if name=='hybrid_baseline': make_checkpoint(checkpoint,core)
                        method = make_method(name,core,rate,checkpoint)
                        method.configure(core,rate,capture_visualization=False,capture_attention=False)
                        if hasattr(method,'prepare_inputs'): method.prepare_inputs((384,384),options['inputs'])
                        if name=='anchorprune': method.query_ids = torch.tensor([[1,2]])
                        outer = method.generation_context(model) if hasattr(method,'generation_context') else nullcontext()
                        inner = method.inference_context(model) if hasattr(method,'inference_context') else nullcontext()
                        with outer,observe_prefill(core) as lengths,inner,torch.inference_mode():
                            generated = model.generate(**options)
                        stats = method.stats(core,rate)
                        budget = observed_budget(name,rate,stats,lengths,1485)
                        self.assertEqual(generated.sequences.shape[1],3)
                        self.assertEqual(len(lengths),28)
                        if name=='fastv':
                            self.assertEqual(budget['physical_average_prune_rate'],0)
                            self.assertGreaterEqual(budget['effective_average_prune_rate']+1e-9,rate)
                        elif name=='hybrid_baseline':
                            self.assertAlmostEqual(budget['actual_average_prune_rate'],
                                budget['two_stage_average_prune_rate']+budget['actual_tail_contribution'])
                            self.assertEqual(budget['final_layer_visual_tokens'],0)
                        elif rate:
                            self.assertAlmostEqual(budget['physical_average_prune_rate'],rate,delta=.2)
                        else: self.assertEqual(budget['actual_average_prune_rate'],0)
                        self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))

    def test_frozen_test_only_input_and_resume_identity(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model';model.mkdir()
            (model/'config.json').write_text(json.dumps({'num_hidden_layers':28}))
            checkpoint = root/'best.pt';checkpoint.write_bytes(b'test weights')
            data = root/'VisA';(data/'split_csv').mkdir(parents=True)
            selected = []
            for label,split,index in (('Normal','train',0),('Normal','test',1),('Anomaly','test',2)):
                relative = f'candle/Data/Images/{label}/{index}.png'
                path = data/relative;path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(16+index,16)).save(path)
                selected.append(['candle',split,label.lower(),relative])
            with (data/'split_csv/1cls.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['object','split','label','image']);writer.writerows(selected)
            for provided in (False,True):
                with self.subTest(checkpoint_provided=provided):
                    options=['--model-path',str(model),'--data-root',str(data),
                        '--output-dir',str(root/f'output_{provided}')]
                    if provided: options+=['--checkpoint',str(checkpoint)]
                    args=entry.parser().parse_args(options)
                    output,spec=entry.setup(args)
                    self.assertEqual(spec['sample_count'],2)
                    self.assertEqual(spec['checkpoint'] is not None,provided)
                    self.assertEqual(sum(j['status']=='skipped' for j in spec['jobs']),0 if provided else 3)
                    frozen=[json.loads(line) for line in Path(spec['input_json']).read_text().splitlines()]
                    self.assertEqual({r['gt'] for r in frozen},{0,1})
                    self.assertTrue(all(r['text']==GENERIC_PROMPT_V0 and 'image_size' in r and 'mask' not in r for r in frozen))
                    self.assertNotIn('/0.png',json.dumps(frozen))
                    args.resume=True
                    self.assertEqual(entry.setup(args)[1],spec)
                    if not provided:
                        args.checkpoint=str(checkpoint)
                        with self.assertRaisesRegex(ValueError,'skipped the hybrid'): entry.setup(args)
                        args.checkpoint=None
                    Path(spec['input_json']).write_text('modified')
                    with self.assertRaisesRegex(ValueError,'Frozen VisA'): entry.setup(args)

    def test_old_visa_training_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoint=root/'best.pt';checkpoint.write_bytes(b'weights')
            (root/'train_run.json').write_text(json.dumps({'original_visa_test_is_training_source':True}))
            with self.assertRaisesRegex(ValueError,'VisA test'): entry.checkpoint_identity(checkpoint)

    def test_summary_uses_measured_visa_baseline_not_mvtec_and_empty_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            spec={'jobs':entry.schedule([50,70,90])}
            for job_id,acc in (('llava_unpruned_00',.8),('fastv_50',.9)):
                folder=root/job_id;folder.mkdir()
                entry.save_json(folder/'metrics.json',dict(job_id=job_id,status='success',complete=True,
                    accuracy=acc,tp=8,fp=1,tn=1,fn=0,lat_mean_ms=20,mem_peak_allocated_gib=12,
                    budget={'physical_average_prune_rate':0}))
            entry.write_reports(root,spec)
            with (root/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);self.assertEqual(tuple(reader.fieldnames),entry.FIELDS);rows=list(reader)
            self.assertEqual(float(rows[1]['relative_accuracy_pct']),112.5)
            unsupported=next(r for r in rows if 'status=unreachable' in r['config'])
            self.assertTrue(all(not unsupported[key] for key in entry.FIELDS[1:]))
            self.assertEqual(float(rows[1]['bacc']),75)

    def test_resume_only_truncates_torn_final_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'predictions.jsonl'
            row=dict(question_id='1',gt=0,answer='B',lat_seconds=.1,generation_seconds=.2,
                generated_tokens=2,peak_allocated_bytes=100,peak_reserved_bytes=200,budget={})
            prefix=json.dumps(row)+'\n';path.write_text(prefix+'{"torn"')
            sample=SimpleNamespace(sample_id='1',gt=0)
            self.assertEqual(entry.restore_records(path,[sample]),[row])
            self.assertEqual(path.read_text(),prefix)
            path.write_text('{"corrupt"\n')
            with self.assertRaises(ValueError): entry.restore_records(path,[sample])

    def test_shared_loop_excludes_warmups_and_resumes_saved_samples(self):
        model,options=model_and_options(28)
        core=model.get_model()
        class Backend:
            def __init__(self): self.model=model;self.calls=0
            def generate(self,sample,prompt,method,rate,mode,**kwargs):
                assert prompt==GENERIC_PROMPT_V0 and mode=='anyres_max_9'
                assert kwargs['benchmark_metrics'] and kwargs['include_pruning_time']
                assert not any(kwargs[k] for k in ('do_sample','capture_visualization','capture_attention'))
                method.configure(core,rate,capture_visualization=False,capture_attention=False)
                with torch.inference_mode(): model.generate(**options)
                self.calls+=1
                return dict(answer='B',lat_seconds=.02*self.calls,generation_seconds=.03*self.calls,
                    generated_tokens=3,peak_allocated_bytes=100,peak_reserved_bytes=200,stats=method.stats(core,rate))
        backend=Backend()
        method=make_method('llava_unpruned',core,0,None)
        samples=[SimpleNamespace(sample_id=str(i),gt=0,source={'image_size':[384,384]}) for i in range(2)]
        job=entry.schedule([50])[0]
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            output=Path(temporary)
            first=entry.evaluate_samples(backend,method,samples,{'seed':42,'warmup':2},job,output,2**30)
            self.assertEqual(backend.calls,4)
            self.assertEqual(first['evaluated_samples'],2)
            self.assertAlmostEqual(first['lat_mean_ms'],70)
            second=entry.evaluate_samples(backend,method,samples,{'seed':42,'warmup':2},job,output,2**30)
            self.assertEqual(backend.calls,4)
            self.assertEqual(second['resumed_samples'],2)
            self.assertEqual(second['fresh_samples'],0)
            self.assertEqual(second['warmup_samples'],0)
            self.assertEqual(second['lat_mean_ms'],first['lat_mean_ms'])

    def test_launcher_runs_fresh_processes_on_same_gpu_and_reuses_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            spec={'gpu':'5','sample_count':2,'checkpoint':{'path':'new-mvtec.pt'},
                  'input_sha256':'input-hash','jobs':entry.schedule([50,70,90])}
            entry.save_json(output/'run.json',spec)
            calls=[]
            def process(command,**kwargs):
                calls.append((command,kwargs))
                job_id=command[command.index('--worker')+1]
                folder=output/job_id;folder.mkdir()
                entry.save_json(folder/'metrics.json',dict(job_id=job_id,status='success',complete=True,
                    input_sha256='input-hash',accuracy=.8,tp=1,fp=0,tn=1,fn=0,
                    lat_mean_ms=10,mem_peak_allocated_gib=12))
                return SimpleNamespace(wait=lambda **kw:0)
            with patch.object(entry,'setup',return_value=(output,spec)),patch.object(entry.subprocess,'Popen',side_effect=process),redirect_stdout(io.StringIO()):
                self.assertEqual(entry.run(None),0)
                self.assertEqual(len(calls),21)
                self.assertTrue(all(k['env']['CUDA_VISIBLE_DEVICES']=='5' for _,k in calls))
                self.assertTrue(all('pyramiddrop_90' not in c for c,_ in calls))
                self.assertEqual(entry.run(None),0)
                self.assertEqual(len(calls),21)

    def test_without_checkpoint_only_hybrid_is_skipped_in_launcher_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            jobs=entry.schedule([50,70,90],include_hybrid=False)
            spec={'gpu':'5','sample_count':2,'checkpoint':None,'input_sha256':'input-hash','jobs':jobs}
            self.assertEqual(sum(j['status']=='pending' for j in jobs),18)
            skipped=[j for j in jobs if j['status']=='skipped']
            self.assertEqual({j['method'] for j in skipped},{'hybrid_baseline'})
            calls=[]
            def process(command,**kwargs):
                job_id=command[command.index('--worker')+1];calls.append(job_id)
                self.assertFalse(job_id.startswith('hybrid_baseline'))
                return SimpleNamespace(wait=lambda **kw:0)
            with patch.object(entry,'setup',return_value=(output,spec)),patch.object(entry.subprocess,'Popen',side_effect=process),redirect_stdout(io.StringIO()):
                self.assertEqual(entry.run(None),0)
            self.assertEqual(len(calls),18)
            self.assertTrue(any(j.startswith('llava_unpruned') for j in calls))
            self.assertTrue(all(any(j.startswith(m+'_') for j in calls) for m in entry.METHODS[:-1]))
            with (output/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            skipped_rows=[r for r in rows if 'status=skipped' in r['config']]
            self.assertEqual(len(skipped_rows),3)
            self.assertTrue(all('No MVTec-trained' in r['config'] for r in skipped_rows))
            self.assertTrue(all(not r[key] for r in skipped_rows for key in entry.FIELDS[1:]))

    def test_resume_rejects_changed_dtype_or_gpu_timing_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            inference={'model_dtype':'torch.float16','torch':'2.1.2'}
            entry.verify_runtime(root,inference,'A100',40*2**30)
            entry.verify_runtime(root,inference,'A100',40*2**30)
            with self.assertRaisesRegex(ValueError,'Resume inference'):
                entry.verify_runtime(root,dict(inference,model_dtype='torch.bfloat16'),'A100',40*2**30)
            with self.assertRaisesRegex(ValueError,'Resume inference'):
                entry.verify_runtime(root,inference,'H100',80*2**30)


if __name__=='__main__': unittest.main()
