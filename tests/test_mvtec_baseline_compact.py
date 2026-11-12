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
from llava_pruning.pool_distill import source_records
from llava_pruning.backend import LlavaBackend
from llava import mm_utils
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from test_pool_ablations import model_and_options
from test_attention_predictor import make_checkpoint


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
            args.selection_modes=['baseline','joint_self']
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
                    mode=job['selection_mode'];config=entry.mode_configuration(mode)
                    method=entry.make_method(core,args.checkpoint,job['prune_rate'],mode)
                    self.assertEqual((method.mode,method.predictor_input,method.base_policy,method.tail_drop_layers),
                        (config['selection_mode'],config['predictor_input'],config['base_policy'],9))
                    method.configure(core,job['prune_rate'],capture_visualization=False,capture_attention=False)
                    first_call=len(sequences)
                    folder=entry.job_folder(output,job)
                    with patch.object(core.attention_predictor,'forward',wraps=core.attention_predictor.forward) as predict:
                        report=entry.render_samples(backend,method,samples,spec,job,folder)
                    self.assertEqual(predict.call_count,2)
                    if mode=='joint_self':
                        self.assertTrue(all(d['cluster_assignments'] is None for d in method.adapter.device_decisions))
                    expected_input=method.adapter.plan['pre_llm_tokens'] if mode=='joint_self' else len(method.adapter.base)
                    self.assertEqual(predict.call_args.args[0].shape[1],expected_input+4)
                    self.assertEqual(len(sequences)-first_call,2)
                    self.assertEqual((report['samples'],report['fresh_samples'],report['pngs']),(2,2,2))
                    rows=entry.restored(folder,samples,mode)
                    self.assertTrue(all(r['selection_mode']==mode for r in rows))
                    self.assertEqual({r['gt'] for r in rows},{0,1})
                    self.assertEqual(len(list(folder.rglob('*.png'))),2)
                    self.assertTrue(all(r['post_distill_tokens']>0 and 'before tail9' in r['post_snapshot'] for r in rows))
                    self.assertTrue(all(r['post_distill_tokens']==r['base_kept']+r['anyres_kept']+r['protected_newlines'] for r in rows))
                    again=entry.render_samples(backend,method,samples,spec,job,folder)
                    self.assertEqual(again['fresh_samples'],0);self.assertEqual(len(sequences)-first_call,2)
                    with torch.inference_mode(),method.generation_context(model):
                        reference=backend.generate(samples[0],samples[0].source['prompt'],method,job['prune_rate'],'anyres_max_9',
                            capture_visualization=False,capture_attention=False,random_seed=spec['seed'],
                            do_sample=False,include_pruning_time=True,benchmark_metrics=False)
                    self.assertTrue(torch.equal(sequences[first_call],sequences[-1]))
                    self.assertEqual(rows[0]['answer'],reference['answer'])
                    self.assertEqual(reference['stats']['final_llm_visual_tokens'],0)  # actual model still tail9
            self.assertEqual(len(list(output.rglob('*.png'))),8)
            self.assertNotIn('select',method.adapter.__dict__)

    def test_lambda_subset_uses_original_filenames_and_runs_the_new_budget(self):
        self.check_lambda_subset('baseline')

    def test_joint_lambda_subset_uses_full_input_and_joint_topk(self):
        self.check_lambda_subset('joint_self')

    def check_lambda_subset(self, mode):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = fixture(Path(temporary))
            args.selection_modes = ['baseline', 'joint_self']
            model, options = model_and_options(28)
            core = model.get_model()
            model.config.max_position_embeddings = 4096
            make_checkpoint(args.checkpoint, core)
            original, old = entry.setup(args)
            old_manifest = (original / 'run.json').read_bytes()
            cli = ['--source-run', str(original), '--sample-indices', '1',
                   '--distill-share', '0.3', '--selection-modes', mode, '--gpus', '0', '1']
            subset, spec = entry.setup(entry.parser().parse_args(cli))
            entry.verify(spec)
            expected_folder = 'lambda_0.3' if mode == 'baseline' else 'lambda_0.3_joint_self'
            self.assertEqual(subset, original / expected_folder)
            config = old['mode_configurations'][mode]
            self.assertEqual(spec['configuration'], dict(config, distill_share=.3))
            self.assertEqual(spec['selection_modes'], [mode])
            self.assertEqual([job['gpu'] for job in spec['jobs']], ['0', '1'])
            self.assertEqual((original / 'run.json').read_bytes(), old_manifest)
            samples = entry.samples_for(spec)
            backend = LlavaBackend.__new__(LlavaBackend)
            backend.model, backend.vision_tower = model, core.get_vision_tower()
            backend.processor = SigLipImageProcessor()
            backend.tokenizer = type('Tokenizer', (), {'batch_decode': lambda *a, **k: ['B']})()
            native = model.generate
            saved_pictures = []
            with patch.object(model, 'generate', side_effect=lambda **kw: native(**dict(kw, max_new_tokens=2))), \
                    patch.object(torch.cuda, 'synchronize'), \
                    patch.object(mm_utils, 'tokenizer_image_token', return_value=options['inputs'][0]):
                for job in spec['jobs']:
                    rate = job['prune_rate']
                    method = entry.make_method(core, spec['checkpoint']['path'], rate, mode,
                                               share=spec['distill_share'])
                    self.assertEqual((method.predictor_input, method.base_policy),
                                     (config['predictor_input'], config['base_policy']))
                    folder = entry.job_folder(subset, job)
                    report = entry.render_samples(backend, method, samples, spec, job, folder)
                    self.assertEqual(report['pngs'], 3)
                    rows = entry.restored(folder, samples, mode, sample_indices=[1])
                    self.assertEqual(rows[0]['question_id'], '000000001')
                    self.assertEqual(rows[0]['output'], 'visualizations/sample_000000001.png')
                    self.assertEqual(rows[0]['sample_seed'], spec['seed'] + 1)
                    self.assertEqual(method.adapter.plan['distill_share'], .3)
                    self.assertAlmostEqual(method.adapter.plan['target_pre_contribution'], rate * .7)
                    name = 'sample_000000001.png'
                    composite = folder / rows[0]['output']
                    original_png = folder / 'originals' / name
                    pair = folder / 'base_anyres' / name
                    with Image.open(original_png) as actual, Image.open(samples[0].image) as expected:
                        self.assertEqual(actual.tobytes(), expected.tobytes())
                    _, base, anyres = rows[0]['panels']
                    with Image.open(pair) as actual, Image.open(composite) as three:
                        with three.crop((base[0], base[1], anyres[2], anyres[3])) as expected:
                            self.assertEqual(actual.tobytes(), expected.tobytes())
                    saved_pictures.append((composite, composite.read_bytes(), original_png, pair))
                    entry.common.save_json(folder / 'status.json', dict(
                        report, status='success', job_id=job['job_id'], input_sha256=spec['input_sha256']))
            self.assertEqual(len(list(subset.rglob('*.png'))), 6)
            for _, _, original_png, pair in saved_pictures:
                original_png.unlink()
                pair.unlink()
            with patch.object(entry.subprocess, 'Popen', side_effect=AssertionError('completed PNGs should be reused')):
                self.assertEqual(entry.run(entry.parser().parse_args([*cli, '--resume'])), 0)
            for composite, pixels, original_png, pair in saved_pictures:
                self.assertEqual(composite.read_bytes(), pixels)
                self.assertTrue(original_png.is_file() and pair.is_file())
            self.assertEqual(entry.setup(entry.parser().parse_args([*cli, '--resume'])), (subset, spec))
            another = original / 'lambda_0.3_samples_0'
            other_cli = [*cli, '--sample-indices', '0', '--output-dir', str(another)]
            new_output, new_spec = entry.setup(entry.parser().parse_args(other_cli))
            self.assertEqual(new_output, another)
            self.assertEqual(new_spec['sample_indices'], [0])
            self.assertEqual(entry.setup(entry.parser().parse_args([*other_cli, '--resume'])),
                             (another, new_spec))
            self.assertEqual(entry.setup(entry.parser().parse_args([*cli, '--resume'])), (subset, spec))

    def test_two_gpu_launcher_full_input_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));busy=set();peak=[0];calls=[]
            class Process:
                def __init__(self,command,kwargs):
                    manifest=Path(command[command.index('--manifest')+1]);self.spec=json.loads(manifest.read_text(encoding='utf-8'))
                    self.job=next(j for j in self.spec['jobs'] if j['job_id']==command[command.index('--worker')+1])
                    self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES'];self.folder=entry.job_folder(manifest.parent,self.job);self.polls=0
                    self.done=False;self.assertion=self.gpu==self.job['gpu'];assert self.assertion and self.gpu not in busy
                    busy.add(self.gpu);peak[0]=max(peak[0],len(busy));calls.append(self.job)
                def poll(self):
                    if self.done: return 0
                    self.polls+=1
                    if self.polls==1: return None
                    self.done=True;busy.remove(self.gpu);self.folder.mkdir(parents=True);(self.folder/'visualizations').mkdir()
                    samples=entry.samples_for(self.spec);rows=[]
                    for i,sample in enumerate(samples):
                        name=f'visualizations/sample_{i:09d}.png';Image.new('RGB',(10,10),'white').save(self.folder/name)
                        rows.append(dict(index=i,question_id=sample.sample_id,gt=sample.gt,answer='B',output=name,selection_mode=self.job['selection_mode']))
                    (self.folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                    common.save_json(self.folder/'status.json',dict(job_id=self.job['job_id'],complete=True,status='success',input_sha256=self.spec['input_sha256']))
                    return 0
                def wait(self,**kwargs): return 0
            with patch.object(entry.subprocess,'Popen',side_effect=lambda command,**kwargs:Process(command,kwargs)),patch.object(entry.time,'sleep'):
                self.assertEqual(entry.run(args),0);self.assertEqual(peak[0],2)
                self.assertEqual([(j['prune_rate'],j['gpu']) for j in calls],[(50,'5'),(70,'6')])
                args.resume=True;args.gpus=[0]
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),2)
                preserved={p:p.read_bytes() for p in args.output_dir.rglob('*.png')}
                args.add_joint_self=True
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),4)
                self.assertEqual([(j['selection_mode'],j['prune_rate'],j['gpu']) for j in calls[2:]],
                    [('joint_self',50,'5'),('joint_self',70,'6')])
                self.assertEqual({p:p.read_bytes() for p in preserved},preserved)
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),4)

    def test_four_jobs_two_mode_configs_and_legacy_baseline_manifest_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.selection_modes=['baseline','joint_self']
            output,spec=entry.setup(args);entry.verify(spec)
            self.assertEqual(len(spec['jobs']),4)
            self.assertEqual([(j['selection_mode'],j['prune_rate'],j['gpu']) for j in spec['jobs']],
                [('baseline',50,'5'),('baseline',70,'6'),('joint_self',50,'5'),('joint_self',70,'6')])
            config=spec['mode_configurations']['joint_self']
            self.assertEqual((config['selection_mode'],config['predictor_input'],config['base_policy']),('anyres_self','full','joint'))
            self.assertEqual((config['distill_share'],config['tail_drop_layers']),(.7,9))
            spec['jobs']=spec['jobs'][:2]
            for job in spec['jobs']:
                for key in ('folder','selection_mode'): job.pop(key)
            for key in ('selection_modes','mode_configurations'): spec.pop(key)
            spec['version']=entry.LEGACY_VERSION;common.save_json(output/'run.json',spec)
            entry.verify(spec);args.resume=True;args.add_joint_self=True
            _,updated=entry.setup(args)
            self.assertEqual(len(updated['jobs']),4)
            self.assertEqual(updated['jobs'][:2],spec['jobs'])

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
            with self.assertRaisesRegex(ValueError,'selection mode changed'):
                entry.restored(folder,samples,'joint_self')

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
