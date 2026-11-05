"""Separate diagnostic replay, existing-result preservation and native snapshots."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from llava import mm_utils
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava_pruning.backend import LlavaBackend
from llava_pruning.data import load_samples
from llava_pruning.prompts import GENERIC_PROMPT_V0
from llava_pruning.visa_comparison_methods import make_method,observe_prefill,observed_budget
from llava_pruning.distilled_visa100_visualization import render_job,completed_job
import llava_pruning.distilled_visa100_visualization as vis
import compare_distilled_visa100 as entry
import benchmark_visa_methods as common
from test_attention_predictor import make_checkpoint
from test_pool_ablations import model_and_options


class StudentVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_opt_in_is_after_measurements_and_visualize_only_never_runs_measurements(self):
        args=entry.parser().parse_args(['--visualize-only','--output-dir','existing'])
        with patch.object(entry,'setup',side_effect=AssertionError('Do not run measurement setup')),\
                patch.object(vis,'run',return_value=0) as draw:
            self.assertEqual(entry.run(args),0)
            draw.assert_called_once_with(Path('existing'),None)
        args=entry.parser().parse_args(['--save-prune-vis'])
        output=Path('existing')
        spec={'sample_count':100,'gpu':'5','jobs':[]}
        events=[]
        with patch.object(entry,'setup',return_value=(output,spec)),\
                patch.object(entry,'write_reports',side_effect=lambda *a:events.append('metrics')),\
                patch.object(vis,'run',side_effect=lambda *a:events.append('draw') or 0),redirect_stdout(io.StringIO()):
            self.assertEqual(entry.run(args),0)
        self.assertEqual(events,['metrics','metrics','draw'])

    def measured_fixture(self,root):
        from test_distilled_visa100 import StudentComparisonTests
        args=StudentComparisonTests().fixture(root)
        output,spec=entry.setup(args)
        samples=load_samples(spec['input_json'],spec['data_root'])
        for job in spec['jobs']:
            folder=output/job['group']/job['job_id'];folder.mkdir()
            report=dict(job_id=job['job_id'],input_sha256=spec['input_sha256'],status='success',complete=True,
                evaluated_samples=100,accuracy=1.,tp=50,tn=50,fp=0,fn=0,lat_mean_ms=100,mem_peak_allocated_gib=20,
                inference={'fixture':'native'})
            common.save_json(folder/'metrics.json',report)
            rows=[dict(question_id=s.sample_id,gt=s.gt,answer='A' if s.gt else 'B',
                lat_seconds=.1,generation_seconds=.2,generated_tokens=2,peak_allocated_bytes=10,
                peak_reserved_bytes=20,budget={}) for s in samples]
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        # Simulate a completed run from BEFORE this drawing feature was added.
        spec['source_identity']={'files':0,'sha256':'older-measurement-code'}
        common.save_json(output/'run.json',spec)
        entry.write_reports(output,spec)
        return output,spec

    def test_old_result_visualize_only_launches_six_jobs_and_preserves_all_measurements(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);output,spec=self.measured_fixture(root)
            files=[output/'run.json',output/'summary.csv',output/'budget_summary.csv',Path(spec['input_json'])]
            files += list(output.glob('*/*/metrics.json'))+list(output.glob('*/*/predictions.jsonl'))
            before={p:p.read_bytes() for p in files}
            calls=[]
            def process(command,**kwargs):
                calls.append(command)
                self.assertIn('--visualization-worker',command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'5')
                self.assertNotIn('--worker',command)
                return SimpleNamespace(wait=lambda **kw:0)
            with patch.object(vis.subprocess,'Popen',side_effect=process):
                self.assertEqual(vis.run(output),0)
            self.assertEqual(len(calls),6)
            self.assertEqual({c[c.index('--visualization-worker')+1] for c in calls},
                {f'{source}_{rate}' for source in ('mvtec','visa') for rate in (50,70,90)})
            self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
            manifest=json.loads((output/'visualizations/run.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['samples'],100)
            self.assertNotEqual(manifest['identity']['replay_source_identity'],spec['source_identity'])
            self.assertEqual(manifest['measurement_source_identity'],spec['source_identity'])

    def test_native_replay_keeps_tokens_and_exports_all_three_stages_without_benchmark(self):
        torch.manual_seed(882)
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);image=root/'image.png';Image.new('RGB',(384,384),(10,70,170)).save(image)
            source=root/'input.json'
            source.write_text(json.dumps([dict(question_id=f'candle/{i}.png',image='image.png',gt=i,text=GENERIC_PROMPT_V0)
                for i in (0,1)]),encoding='utf-8')
            samples=load_samples(source,root)
            model,options=model_and_options(28);core=model.get_model();core.config.max_position_embeddings=4096
            checkpoint=root/'best.pt';make_checkpoint(checkpoint,core)
            backend=LlavaBackend.__new__(LlavaBackend)
            backend.model,backend.vision_tower=model,core.get_vision_tower()
            backend.processor=SigLipImageProcessor()
            backend.tokenizer=SimpleNamespace(batch_decode=lambda *a,**kw:['B'])
            original=model.generate;sequences=[]
            def generate(**kwargs):
                kwargs['max_new_tokens']=3
                value=original(**kwargs);sequences.append(value.detach().clone());return value
            job=dict(job_id='mvtec_90',method='hybrid_baseline',prune_rate=90)
            spec={'seed':42};rows=[]
            with patch.object(model,'generate',side_effect=generate),patch.object(torch.cuda,'synchronize'),\
                    patch.object(mm_utils,'tokenizer_image_token',return_value=options['inputs'][0]):
                clean=make_method('hybrid_baseline',core,90,checkpoint)
                with torch.inference_mode(),clean.generation_context(model):
                    for i,sample in enumerate(samples):
                        with observe_prefill(core) as lengths:
                            result=backend.generate(sample,GENERIC_PROMPT_V0,clean,90,'anyres_max_9',
                                capture_visualization=False,capture_attention=False,random_seed=42+i,benchmark_metrics=False)
                        rows.append(dict(answer=result['answer'],budget=observed_budget('hybrid_baseline',90,result['stats'],lengths,1485)))
                method=make_method('hybrid_baseline',core,90,checkpoint)
                output=root/'pictures';identity={'fixture':'same-model-input-and-checkpoint'}
                with patch('llava_pruning.benchmark_timing.generate_with_benchmark',side_effect=AssertionError('No Lat/Mem measurement')):
                    status=render_job(backend,method,samples,spec,job,output,rows,identity)
                    common.save_json(output/'visualization_status.json',status)
                    self.assertTrue(completed_job(output,samples,identity))
                    self.assertEqual(len(sequences),4)
                    self.assertTrue(all(torch.equal(sequences[i],sequences[i+2]) for i in (0,1)))
                    # Repeating the drawing resumes the complete sample folders.
                    render_job(backend,method,samples,spec,job,output,rows,identity)
                    self.assertEqual(len(sequences),4)
                for i in (0,1):
                    folder=output/f'sample_{i:09d}'
                    self.assertEqual({p.name for p in folder.glob('*.png')},
                        {f'{stage}_{view}.png' for stage in ('input','pre_llm','post_distill') for view in ('base','anyres_01')})
                    meta=json.loads((folder/'run.json').read_text(encoding='utf-8'))
                    self.assertTrue(meta['answer_matches_measured'])
                    self.assertEqual(meta['gt'],i)
                    self.assertIn('before tail=9',meta['post_snapshot'])
                    self.assertEqual(meta['encoder_views'],2)  # actual native count, not forced9
                    self.assertGreater(meta['post_distill_tokens'],0)
                (output/'sample_000000000/pre_llm_base.png').unlink()
                self.assertFalse(completed_job(output,samples,identity))


if __name__=='__main__': unittest.main()
