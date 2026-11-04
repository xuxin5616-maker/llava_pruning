"""Two-process startup, immutable result reuse, and hybrid dependency checks."""

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

import benchmark_visa_methods as benchmark
import run_mvtec_visa_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def args(self,*options):
        return pipeline.parser().parse_args(['--train-gpu','4','--eval-gpu','5',*options])

    def test_both_processes_start_before_waiting_and_have_separate_devices(self):
        events=[];calls=[]
        class Process:
            def __init__(self,index): self.index=index;self.polls=0
            def poll(self):
                events.append('poll');self.polls+=1
                return None if self.polls==1 else 0
            def wait(self,**kwargs): return 0
        def spawn(command,**kwargs):
            events.append('spawn');calls.append((command,kwargs));return Process(len(calls))
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()),\
                patch.object(pipeline.subprocess,'Popen',side_effect=spawn),patch.object(pipeline.time,'sleep'):
            args=self.args()
            self.assertEqual(pipeline.launch_parallel(args,Path(temporary)),[0,0])
        self.assertEqual(events[:2],['spawn','spawn'])
        self.assertEqual([kw['env']['CUDA_VISIBLE_DEVICES'] for _,kw in calls],['4','5'])
        train,eval_=[c for c,_ in calls]
        self.assertEqual(train[train.index('--validation-fraction')+1],'0')
        self.assertTrue(train[2].endswith('train_attention_predictor_mvtec.py'))
        self.assertNotIn('--checkpoint',eval_)
        self.assertEqual(eval_[eval_.index('--gpu')+1],'5')

    def fixture(self,root):
        model=root/'model';model.mkdir();(model/'config.json').write_text('{"num_hidden_layers":28}',encoding='utf-8')
        visa=root/'VisA';(visa/'split_csv').mkdir(parents=True)
        rows=[]
        for label in ('Normal','Anomaly'):
            relative=f'candle/Data/Images/{label}/000.png'
            path=visa/relative;path.parent.mkdir(parents=True)
            Image.new('RGB',(16,16)).save(path)
            rows.append(['candle','test',label.lower(),relative])
        with (visa/'split_csv/1cls.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['object','split','label','image']);writer.writerows(rows)
        args=self.args('--model-path',str(model),'--visa-root',str(visa))
        student=root/'student';student.mkdir()
        (student/'last.pt').write_bytes(b'student weights')
        benchmark.save_json(student/'training_status.json',{'complete':True})
        benchmark.save_json(student/'train_run.json',{'original_visa_test_is_training_source':False})
        prior_dir,prior=benchmark.setup(benchmark.parser().parse_args(pipeline.eval_options(args,root/'visa_independent')))
        for job in prior['jobs']:
            if job['status']!='pending': continue
            folder=prior_dir/job['job_id'];folder.mkdir()
            report=dict(job_id=job['job_id'],status='success',complete=True,evaluated_samples=2,
                input_sha256=prior['input_sha256'],accuracy=1.,tp=1,tn=1,fp=0,fn=0,lat_mean_ms=123.,mem_peak_allocated_gib=12.)
            benchmark.save_json(folder/'metrics.json',report)
            (folder/'predictions.jsonl').write_text('{"preserved":"answer, timing, budget"}\n',encoding='utf-8')
            benchmark.save_json(folder/'inference.json',{'model_dtype':'float16'})
        return args,prior

    def test_reuses_all_18_trials_and_runs_only_three_hybrid_jobs(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args,prior=self.fixture(root)
            options=pipeline.prepare_final(args,root)
            final=root/'visa_all_methods'
            spec=json.loads((final/'run.json').read_text(encoding='utf-8'))
            self.assertEqual(len(spec['reused_trials']['jobs']),18)
            self.assertEqual(spec['checkpoint']['path'],str((root/'student/last.pt').resolve()))
            self.assertIsNone(prior['checkpoint'])
            original=root/'visa_independent/fastv_50'
            self.assertEqual((original/'predictions.jsonl').read_bytes(),(final/'fastv_50/predictions.jsonl').read_bytes())
            self.assertEqual((original/'metrics.json').read_bytes(),(final/'fastv_50/metrics.json').read_bytes())
            calls=[]
            def spawn(command,**kwargs):
                job_id=command[command.index('--worker')+1];calls.append(job_id)
                self.assertTrue(job_id.startswith('hybrid_baseline_'))
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'5')
                folder=final/job_id;folder.mkdir()
                benchmark.save_json(folder/'metrics.json',dict(job_id=job_id,status='success',complete=True,
                    input_sha256=spec['input_sha256'],accuracy=.5,tp=1,fp=1,tn=0,fn=0,lat_mean_ms=222,mem_peak_allocated_gib=20))
                return SimpleNamespace(wait=lambda **kw:0)
            with patch.object(benchmark.subprocess,'Popen',side_effect=spawn):
                self.assertEqual(benchmark.run(options),0)
            self.assertEqual(calls,['hybrid_baseline_50','hybrid_baseline_70','hybrid_baseline_90'])
            with (final/'summary.csv').open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),22)
            self.assertEqual(sum('status=unreachable' in r['config'] for r in rows),1)
            reused=next(r for r in rows if 'method=fastv;prune=50%' in r['config'])
            self.assertEqual(float(reused['Lat']),123.)
            self.assertEqual(float(reused['mem']),12.)

    def test_incomplete_training_or_changed_evaluation_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);args,_=self.fixture(root)
            benchmark.save_json(root/'student/training_status.json',{'complete':False})
            with self.assertRaisesRegex(ValueError,'Training has not completed'): pipeline.prepare_final(args,root)
            self.assertFalse((root/'visa_all_methods').exists())
            benchmark.save_json(root/'student/training_status.json',{'complete':True})
            args.seed=0
            with self.assertRaisesRegex(ValueError,'seed differs'): pipeline.prepare_final(args,root)

    def test_same_device_is_rejected_before_startup(self):
        args=self.args('--eval-gpu','4')
        with patch.object(pipeline,'launch_parallel') as launch:
            with self.assertRaisesRegex(ValueError,'DIFFERENT'): pipeline.run(args)
            launch.assert_not_called()


if __name__=='__main__': unittest.main()
