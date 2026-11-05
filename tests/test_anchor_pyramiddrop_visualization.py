"""Full-test scope, native geometry projection and real clean/capture parity."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from llava import mm_utils
from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention,SigLipVisionConfig,SigLipImageProcessor
from llava_pruning.backend import LlavaBackend
from llava_pruning.data import load_samples
from llava_pruning.prompts import GENERIC_PROMPT_V0
from llava_pruning.visualization import build_anyres_layout
from llava_pruning.visa_comparison_methods import make_method,observe_prefill,observed_budget
from llava_pruning.native_pruning_visualization import TracedPyramid,view_masks,render,save_views
import run_anchor_pyramiddrop_vis as entry
import benchmark_visa_methods as common
from test_pool_ablations import model_and_options


class NativeVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def fixture(self,root):
        from test_distilled_visa100 import StudentComparisonTests
        old=StudentComparisonTests().fixture(root)
        return entry.parser().parse_args(['--model-path',old.model_path,'--data-root',old.data_root,
            '--output-dir',str(root/'full_output')])

    def test_full_test_scope_over_100_no_student_and_unreachable_90(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=self.fixture(root)
            with patch.object(common,'checkpoint_identity',side_effect=AssertionError('No student required')):
                output,spec,options=entry.setup(args)
            self.assertEqual(spec['sample_count'],120)
            self.assertEqual(len(load_samples(spec['input_json'],spec['data_root'])),120)
            self.assertIsNone(spec['checkpoint']);self.assertTrue(options.resume)
            self.assertEqual(len(spec['jobs']),7)
            self.assertEqual(sum(j['status']=='pending' for j in spec['jobs']),6)
            self.assertEqual([j['job_id'] for j in spec['jobs'] if j['status']=='unreachable'],['pyramiddrop_90'])
            self.assertEqual({j['method'] for j in spec['jobs']},{'llava_unpruned','anchorprune','pyramiddrop'})

    def test_native_unpad_max9_projection_and_exact_token_counts(self):
        for grid,size in (((1,1),(384,384)),((3,2),(1024,300)),((6,6),(3000,2700))):
            with self.subTest(grid=grid,size=size):
                meta=dict(mode='anyres_max_9',original_size=size,grid_patches=grid)
                layouts=build_anyres_layout(meta,27,14);h,w=layouts[1]['grid_shape'];n=729+h*(w+1)
                keep=np.ones(n,bool);keep[729:]=np.arange(n-729)%3!=0
                masks,valids,details=view_masks(keep,meta)
                self.assertEqual(masks.shape,(1+grid[0]*grid[1],729))
                self.assertFalse((masks&~valids).any())
                self.assertEqual(details['packed_kept'],int(keep.sum()))
                if not details['max9_resized']:
                    self.assertEqual(int(masks.sum())+details['packed_newline_kept'],int(keep.sum()))
                else:
                    self.assertGreater(int(valids.sum()),details['packed_tokens'])
                all_kept,valid,_=view_masks(np.ones(n,bool),meta)
                self.assertTrue(np.array_equal(all_kept,valid))
                with self.assertRaises(ValueError): view_masks(keep[:-1],meta)

    def test_real_anchor_and_pyramiddrop_tokens_unchanged_and_stage_images(self):
        for name,rates in (('anchorprune',(50,70,90)),('pyramiddrop',(50,70))):
            for rate in rates:
                with self.subTest(method=name,rate=rate),tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
                    root=Path(temporary);Image.new('RGB',(384,384),(10,70,170)).save(root/'image.png')
                    source=root/'input.json';source.write_text(json.dumps([dict(question_id='candle/test/image.png',
                        image='image.png',gt=0,text=GENERIC_PROMPT_V0,image_size=[384,384])]),encoding='utf-8')
                    samples=load_samples(source,root)
                    torch.manual_seed(43)
                    model,options=model_and_options(28);core=model.get_model();core.config.max_position_embeddings=4096
                    if name=='anchorprune':
                        core.vision_tower.blocks[-1].self_attn=SigLipAttention(SigLipVisionConfig(hidden_size=8,num_attention_heads=2))
                    class Tokenizer:
                        def __call__(self,*a,**kw): return SimpleNamespace(input_ids=torch.tensor([[2,3]]))
                        def batch_decode(self,*a,**kw): return ['B']
                    backend=LlavaBackend.__new__(LlavaBackend)
                    backend.model,backend.vision_tower=model,core.get_vision_tower()
                    backend.processor,backend.tokenizer=SigLipImageProcessor(),Tokenizer()
                    original=model.generate;sequences=[]
                    def generate(**kwargs):
                        kwargs['max_new_tokens']=3
                        value=original(**kwargs);sequences.append(value.detach().clone());return value
                    job=dict(method=name,job_id=f'{name}_{rate}',prune_rate=rate);spec={'seed':42}
                    with patch.object(model,'generate',side_effect=generate),patch.object(torch.cuda,'synchronize'),\
                            patch.object(mm_utils,'tokenizer_image_token',return_value=options['inputs'][0]):
                        clean=make_method(name,core,rate,None)
                        with torch.inference_mode(),observe_prefill(core) as lengths:
                            result=backend.generate(samples[0],GENERIC_PROMPT_V0,clean,rate,'anyres_max_9',
                                capture_visualization=False,capture_attention=False,random_seed=42,benchmark_metrics=False)
                        rows=[dict(answer=result['answer'],budget=observed_budget(name,rate,result['stats'],lengths,1485))]
                        method=make_method(name,core,rate,None)
                        if name=='pyramiddrop': method=TracedPyramid(method)
                        render(backend,method,samples,spec,job,root/'pictures',rows,{'same':'identity'})
                    self.assertTrue(torch.equal(sequences[0],sequences[1]))
                    folder=root/'pictures/sample_000000000'
                    prefixes=['input','pre_llm'] if name=='anchorprune' else ['input','after_layer_07','after_layer_14','after_layer_21']
                    self.assertEqual({p.name for p in folder.glob('*.png')},
                        {f'{stage}_{view}.png' for stage in prefixes for view in ('base','anyres_01')})
                    saved=json.loads((folder/'run.json').read_text(encoding='utf-8'))
                    self.assertTrue(saved['answer_matches_measured'])
                    self.assertEqual([s['after_layer'] for s in saved['stages']],[0] if name=='anchorprune' else [7,14,21])
                    self.assertEqual(core.config._attn_implementation,'sdpa')

    def test_white_pruning_and_gray_unpadding_are_distinct_in_actual_pngs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            rgb=np.empty((3,384,384,3),np.uint8);rgb[:]=[10,70,170]
            pixels=torch.from_numpy(rgb.transpose(0,3,1,2).copy()).float()/255*2-1
            metadata=dict(mode='anyres_max_9',original_size=[384,128],grid_patches=[2,1])
            h,w=build_anyres_layout(metadata,27,14)[1]['grid_shape']
            keep=np.ones(729+h*(w+1),bool);keep[:729]=False
            save_views(dict(pixels=pixels,metadata=metadata),SigLipImageProcessor(),{'answer':'B'},
                SimpleNamespace(sample_id='test',gt=0),[dict(after_layer=0,image_tokens_after=int(keep.sum()),mask={'keep':keep})],root)
            def body(name):
                with Image.open(root/name) as image: return np.asarray(image)[-384:]
            self.assertTrue((body('pre_llm_base.png')[7,7]==255).all())
            self.assertTrue((body('pre_llm_anyres_01.png')[7,7]==160).all())
            self.assertTrue(np.array_equal(body('pre_llm_anyres_01.png')[4*14+7,7],[10,70,170]))
            self.assertTrue(np.array_equal(body('input_anyres_01.png')[7,7],[10,70,170]))

    def test_full_visualization_runs_only_five_reachable_jobs_preserving_metrics(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=self.fixture(root);output,spec,_=entry.setup(args)
            samples=load_samples(spec['input_json'],spec['data_root'])
            for job in spec['jobs']:
                if job['status']!='pending': continue
                folder=output/job['job_id'];folder.mkdir()
                common.save_json(folder/'metrics.json',dict(job_id=job['job_id'],status='success',complete=True,
                    input_sha256=spec['input_sha256'],evaluated_samples=120,accuracy=1.,tp=60,tn=60,fp=0,fn=0,
                    lat_mean_ms=100,mem_peak_allocated_gib=20,inference={'fixture':'native'}))
                rows=[dict(question_id=s.sample_id,gt=s.gt,answer='A' if s.gt else 'B',lat_seconds=.1,generation_seconds=.2,
                    generated_tokens=2,peak_allocated_bytes=10,peak_reserved_bytes=20,budget={}) for s in samples]
                (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            common.write_reports(output,spec)
            paths=[output/'summary.csv',output/'budget_summary.csv',output/'run.json']+list(output.glob('*/metrics.json'))+list(output.glob('*/predictions.jsonl'))
            before={p:p.read_bytes() for p in paths};calls=[]
            def process(command,**kwargs):
                calls.append(command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'5')
                return SimpleNamespace(wait=lambda **kw:0)
            with patch.object(entry.subprocess,'Popen',side_effect=process): self.assertEqual(entry.visualize(output),0)
            self.assertEqual(len(calls),5)
            self.assertNotIn('pyramiddrop_90',[c[c.index('--visualization-worker')+1] for c in calls])
            self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
            drawing=json.loads((output/'visualizations/run.json').read_text(encoding='utf-8'))
            self.assertEqual(drawing['samples'],120)


if __name__=='__main__': unittest.main()
