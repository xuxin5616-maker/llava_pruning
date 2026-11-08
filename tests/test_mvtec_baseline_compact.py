"""Full-input baseline PNGs, exact white anchors, two GPUs and safe continuation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import visualize_mvtec_baseline as entry
import benchmark_visa_methods as common
from llava_pruning.pool_compact_visualization import render_compact
from llava_pruning.pool_distill import PoolDistillMethod,source_records
from llava_pruning.backend import LlavaBackend
from llava import mm_utils
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from test_pool_ablations import model_and_options
from test_attention_predictor import make_checkpoint
from runMyPruningPoolSpatial import MethodSettings


def fixture(root):
    model=root/'model';model.mkdir();(model/'config.json').write_text('{"num_hidden_layers":28}',encoding='utf-8')
    data=root/'mvtec';data.mkdir();rows=[]
    for index in range(2):
        Image.new('RGB',(384,384),(10+index*20,70,170)).save(data/f'{index}.png')
        rows.append(dict(question_id=f'{index:09d}',image=f'{index}.png',gt=index,
            origin_path=f'bottle/test/{"broken" if index else "good"}/{index}.png',text='RAW DATA TEXT'))
    source=root/'questions.json';source.write_text(json.dumps(rows),encoding='utf-8')
    checkpoint=root/'best.pt';checkpoint.write_bytes(b'fake-test-student')
    return entry.parser().parse_args(['--model-path',str(model),'--data-root',str(data),
        '--input-json',str(source),'--checkpoint',str(checkpoint),'--output-dir',str(root/'pictures'),'--panel-height','384'])


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_native_grids_three_panels_use_actual_selected_anchors_and_original_disk_photo(self):
        for grid in ((1,1),(2,1),(3,3)):
            with self.subTest(grid=grid),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);original=root/'original.png';Image.new('RGB',(384,384),(91,40,130)).save(original)
                decisions=[dict(dominant_indices=[7,1],contextual_indices=[3],cluster_assignments=[-1]*729)
                           for _ in range(1+grid[0]*grid[1])]
                records=source_records(decisions,grid,fast=True)
                selected=[i for i,r in enumerate(records) if r is None or r['anchor']==(3 if r['view']==0 else 7)]
                colors=[(10+i*7,70,170) for i in range(len(decisions))]
                rgb=np.stack([np.full((384,384,3),color,dtype=np.uint8) for color in colors])
                pixels=torch.tensor(rgb.transpose(0,3,1,2).copy(),dtype=torch.float32)/255*2-1
                newlines=sum(r is None for r in records)
                adapter=SimpleNamespace(geometry=SimpleNamespace(grid=grid,patch_size=14),device_decisions=decisions,
                    core=SimpleNamespace(config=SimpleNamespace(mm_patch_merge_type='spatial_unpad')),
                    plan=dict(pre_llm_tokens=len(records),final_llm_tokens=len(selected),baseline_packed_tokens=729*len(decisions)+newlines))
                trace=dict(pixels=pixels,selected=torch.tensor(selected));height=grid[1]*384
                with patch('llava_pruning.pool_lean.export_source_decisions',side_effect=lambda x:x):
                    image,details=render_compact(SimpleNamespace(adapter=adapter),trace,SigLipImageProcessor(),
                        {'answer':'B'},SimpleNamespace(image=original,sample_id='000000048',gt=0),50,height)
                self.assertEqual(details['native_grid'],list(grid))
                self.assertEqual(details['base_kept'],1);self.assertEqual(details['anyres_kept'],len(decisions)-1)
                panels=[image.crop(box) for box in details['panels']]
                self.assertTrue((np.asarray(panels[0])==[91,40,130]).all())
                base=np.asarray(panels[1]);scale=height//384
                self.assertTrue((base[:14*scale,:14*scale]==255).all())
                self.assertTrue((base[:14*scale,42*scale:56*scale]==colors[0]).all())
                anyres=np.asarray(panels[2])
                self.assertEqual(anyres.shape[:2],(384*grid[1],384*grid[0]))
                for tile in range(len(decisions)-1):
                    row,col=divmod(tile,grid[0]);view=anyres[row*384:(row+1)*384,col*384:(col+1)*384]
                    self.assertTrue((view[:14,:14]==255).all())
                    self.assertTrue((view[:14,98:112]==colors[tile+1]).all())
                    self.assertTrue((view[378:]==colors[tile+1]).all())
                self.assertEqual(len(list(root.glob('*.png'))),1)  # renderer writes nothing on its own
                for panel in panels: panel.close()
                image.close()

    def test_real_full_input_generation_saves_one_png_each_and_resume_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=fixture(root)
            torch.manual_seed(733);model,options=model_and_options(28);core=model.get_model()
            core.config.max_position_embeddings=4096
            make_checkpoint(args.checkpoint,core)
            output,spec=entry.setup(args);samples=entry.samples_for(spec)
            backend=LlavaBackend.__new__(LlavaBackend);backend.model=model;backend.vision_tower=core.get_vision_tower()
            backend.processor=SigLipImageProcessor();backend.tokenizer=type('Tokenizer',(),{'batch_decode':lambda *a,**k:['B']})()
            native=model.generate;sequences=[]
            def short_generate(**kwargs):
                kwargs['max_new_tokens']=3;result=native(**kwargs);sequences.append(result.detach().clone());return result
            with patch.object(model,'generate',side_effect=short_generate),patch.object(torch.cuda,'synchronize'),\
                    patch.object(mm_utils,'tokenizer_image_token',return_value=options['inputs'][0]):
                for job in spec['jobs']:
                    method=PoolDistillMethod(core,args.checkpoint,rate=job['prune_rate'],share=.7,mode='base_full',
                        settings=MethodSettings(prune_rate=0,redundancy_layers=(26,),visionzip_layers=(26,),allocation_temperature=.05,spatial_strength=0.),
                        pruning_audit='lean',tail_drop_layers=9,predictor_input='base_text',base_mapping='fast',base_policy='anyres_first')
                    first_call=len(sequences)
                    report=entry.render_samples(backend,method,samples,spec,job,output/job['job_id'])
                    self.assertEqual(len(sequences)-first_call,2)
                    self.assertEqual((report['samples'],report['fresh_samples'],report['pngs']),(2,2,2))
                    rows=entry.restored(output/job['job_id'],samples)
                    self.assertEqual({r['gt'] for r in rows},{0,1})
                    self.assertEqual(len(list((output/job['job_id']).rglob('*.png'))),2)
                    self.assertTrue(all(r['post_distill_tokens']>0 and 'before tail9' in r['post_snapshot'] for r in rows))
                    self.assertTrue(all(r['post_distill_tokens']==r['base_kept']+r['anyres_kept']+r['protected_newlines'] for r in rows))
                    again=entry.render_samples(backend,method,samples,spec,job,output/job['job_id'])
                    self.assertEqual(again['fresh_samples'],0);self.assertEqual(len(sequences)-first_call,2)
                    with torch.inference_mode(),method.generation_context(model):
                        reference=backend.generate(samples[0],samples[0].source['prompt'],method,job['prune_rate'],'anyres_max_9',
                            capture_visualization=False,capture_attention=False,random_seed=spec['seed'],
                            do_sample=False,include_pruning_time=True,benchmark_metrics=False)
                    self.assertTrue(torch.equal(sequences[first_call],sequences[-1]))
                    self.assertEqual(rows[0]['answer'],reference['answer'])
                    self.assertEqual(reference['stats']['final_llm_visual_tokens'],0)  # actual model still tail9
            self.assertEqual(len(list(output.rglob('*.png'))),4)
            self.assertNotIn('select',method.adapter.__dict__)

    def test_two_gpu_launcher_full_input_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));busy=set();peak=[0];calls=[]
            class Process:
                def __init__(self,command,kwargs):
                    manifest=Path(command[command.index('--manifest')+1]);self.spec=json.loads(manifest.read_text(encoding='utf-8'))
                    self.job=next(j for j in self.spec['jobs'] if j['job_id']==command[command.index('--worker')+1])
                    self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES'];self.folder=manifest.parent/self.job['job_id'];self.polls=0
                    self.done=False;self.assertion=self.gpu==self.job['gpu'];assert self.assertion and self.gpu not in busy
                    busy.add(self.gpu);peak[0]=max(peak[0],len(busy));calls.append(self.job)
                def poll(self):
                    if self.done: return 0
                    self.polls+=1
                    if self.polls==1: return None
                    self.done=True;busy.remove(self.gpu);self.folder.mkdir();(self.folder/'visualizations').mkdir()
                    samples=entry.samples_for(self.spec);rows=[]
                    for i,sample in enumerate(samples):
                        name=f'visualizations/sample_{i:09d}.png';Image.new('RGB',(10,10),'white').save(self.folder/name)
                        rows.append(dict(index=i,question_id=sample.sample_id,gt=sample.gt,answer='B',output=name))
                    (self.folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                    common.save_json(self.folder/'status.json',dict(job_id=self.job['job_id'],complete=True,status='success',input_sha256=self.spec['input_sha256']))
                    return 0
                def wait(self,**kwargs): return 0
            with patch.object(entry.subprocess,'Popen',side_effect=lambda command,**kwargs:Process(command,kwargs)),patch.object(entry.time,'sleep'):
                self.assertEqual(entry.run(args),0);self.assertEqual(peak[0],2)
                self.assertEqual([(j['prune_rate'],j['gpu']) for j in calls],[(50,'5'),(70,'6')])
                args.resume=True;args.gpus=[0]
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),2)

    def test_preflight_and_existing_query_are_read_only_and_distinguish_training_dataset_names(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=fixture(root);args.check_data=True
            args.model_path='MISSING';args.checkpoint='MISSING'
            self.assertEqual(entry.run(args),0);self.assertFalse(args.output_dir.exists())
            out=root/'output';real=out/'mvtec_real';real.mkdir(parents=True)
            common.save_json(real/'run.json',dict(data_root=str(root/'mvtec'),sample_count=1725))
            Image.new('RGB',(1,1)).save(real/'sample.png')
            other=out/'mvtec_student_on_visa';other.mkdir()
            common.save_json(other/'run.json',dict(data_root='/datasets/VisA',input_json=str(other/'input_frozen.jsonl')))
            Image.new('RGB',(1,1)).save(other/'sample.png')
            found=entry.find_existing([out]);self.assertEqual([r['directory'] for r in found],[str(real)])
            code="import sys; import visualize_mvtec_baseline; assert 'torch' not in sys.modules; print('CPU-only CLI')"
            result=subprocess.run([sys.executable,'-B','-c',code],cwd=entry.PROJECT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_torn_last_row_and_missing_newline_are_repaired_only_in_new_output(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));output,spec=entry.setup(args);samples=entry.samples_for(spec)
            folder=output/'prune_50';(folder/'visualizations').mkdir(parents=True)
            name='visualizations/sample_000000000.png';Image.new('RGB',(1,1)).save(folder/name)
            row=dict(index=0,question_id=samples[0].sample_id,gt=samples[0].gt,answer='B',output=name)
            path=folder/'predictions.jsonl';path.write_text(json.dumps(row)+'\n{"torn"',encoding='utf-8')
            self.assertEqual(entry.restored(folder,samples),[row])
            self.assertEqual(path.read_text(encoding='utf-8'),json.dumps(row)+'\n')
            path.write_text(json.dumps(row),encoding='utf-8');entry.restored(folder,samples)
            self.assertTrue(path.read_bytes().endswith(b'\n'))

    def test_interrupt_stops_both_workers_and_closes_logs(self):
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
            self.assertEqual(len(processes),2)
            self.assertTrue(all(p.stopped and p.waited and p.log.closed for p in processes))


if __name__=='__main__': unittest.main()
