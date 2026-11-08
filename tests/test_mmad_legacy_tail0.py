"""Regression for the actual 67136cf full-MMAD, 25-row experiment."""

import ast
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
import llava_pruning.mmad_source as source
from llava_pruning.mmad_data import load_frozen
from test_mmad_binary import fixture, prediction


class ReleasedTail0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(entry.__file__).resolve().parent
        try:
            cls.old=source.revision_files(cls.root,'67136cf',{})
        except ValueError as error:
            raise unittest.SkipTest('Released-source regression requires local 67136cf Git history: '+str(error))
        cls.files=source.working_files(cls.root)
        cls.current=source.scope_identity(cls.files)

    def spec(self):
        return dict(source_identity=source.whole_identity(self.old),
            jobs=entry.schedule([50,70,90],[5,6,7],include_visa=True))

    def test_exact_released_entry_and_extension_are_compatible_for_saved_tail0(self):
        spec=self.spec();result=source.verify_source(spec,self.root)
        recovery=result['legacy_recovery']
        self.assertTrue(recovery['reference'].startswith('67136cf'))
        self.assertEqual(recovery['compatibility']['transition'],'67136cf_to_6aea384_default_tail0')
        self.assertEqual(spec,self.spec())
        self.assertEqual(result['identity'],self.current)

    def test_nonzero_saved_tail_is_not_compatible_with_literal_tail0(self):
        spec=self.spec()
        next(j for j in spec['jobs'] if j.get('variant')=='joint_self')['tail_drop_layers']=1
        with self.assertRaisesRegex(ValueError,'inference dependencies changed'):
            source.verify_source(spec,self.root)
        spec=self.spec();spec['joint_self_tails']=[0,1]
        self.assertIsNone(source.tail0_compatibility(self.old,self.files,spec,self.current))

    def test_other_method_backend_metrics_or_extension_changes_still_fail(self):
        for filename in ('llava_pruning/backend.py',source.EXTENSION,source.ENTRY):
            with self.subTest(file=filename):
                files=dict(self.files)
                if filename==source.ENTRY:
                    text=files[filename].decode('utf-8')
                    self.assertIn('share=.7,mode="anyres_self"',text)
                    files[filename]=text.replace('share=.7,mode="anyres_self"','share=.6,mode="anyres_self"',1).encode('utf-8')
                else: files[filename]+=b'\nCHANGED=1\n'
                current=source.scope_identity(files)
                self.assertIsNone(source.tail0_compatibility(self.old,files,self.spec(),current))

    def test_21_successes_are_preserved_and_only_three_failed_jobs_are_resumed(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.include_visa=True;args.gpus=[5,6,7]
            output,spec=entry.setup(args)
            spec.pop('mmad_source_identity');spec.pop('source_policy');spec.pop('joint_self_tails')
            spec['source_identity']=source.whole_identity(self.old)
            common.save_json(output/'run.json',spec)
            original_manifest=(output/'run.json').read_bytes();samples=load_frozen(spec['input_json'])
            failed={'hybrid_baseline_50','hybrid_joint_self_50','hybrid_joint_self_70'};preserved={}
            for job in spec['jobs']:
                if job['status']!='pending': continue
                folder=output/job['job_id'];folder.mkdir()
                rows=[prediction(s,s.source['correct_option']) for s in samples]
                if job['job_id'] in failed:
                    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:2]),encoding='utf-8')
                    common.save_json(folder/'metrics.json',dict(job_id=job['job_id'],input_sha256=spec['input_sha256'],
                        status='failed',complete=False,error='Inference source changed; use a new output directory'))
                else:
                    result=entry.summarize_records(rows,samples,job,2**30)
                    common.save_json(folder/'metrics.json',dict(result,job_id=job['job_id'],status='success',input_sha256=spec['input_sha256']))
                    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                    for name in ('metrics.json','predictions.jsonl'):
                        preserved[folder/name]=(folder/name).read_bytes()
            calls=[];real_popen=entry.subprocess.Popen
            def process(command,**kwargs):
                if command[0]=='git': return real_popen(command,**kwargs)
                manifest=Path(command[command.index('--manifest')+1]);saved=json.loads(manifest.read_text(encoding='utf-8'))
                entry.verify_inputs(saved)
                job=next(j for j in saved['jobs'] if j['job_id']==command[command.index('--worker')+1])
                self.assertIn(job['job_id'],failed);self.assertEqual(job['gpu'],kwargs['env']['CUDA_VISIBLE_DEVICES'])
                calls.append(job['job_id']);folder=output/job['job_id']
                rows=entry.restore_records(folder/'predictions.jsonl',samples);self.assertEqual(len(rows),2)
                rows.extend(prediction(s,s.source['correct_option']) for s in samples[2:])
                result=entry.summarize_records(rows,samples,job,2**30)
                common.save_json(folder/'metrics.json',dict(result,job_id=job['job_id'],status='success',input_sha256=saved['input_sha256']))
                return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0)
            args.resume=True
            with patch.object(entry.subprocess,'Popen',side_effect=process):
                self.assertEqual(entry.run(args),0);self.assertEqual(set(calls),failed)
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),3)
            self.assertEqual({p:p.read_bytes() for p in preserved},preserved)
            self.assertEqual((output/'run.before_scoped_source.json').read_bytes(),original_manifest)
            saved=json.loads((output/'run.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['gpus'],['5','6','7'])
            self.assertEqual(saved['source_recovery']['compatibility']['joint_self_tails'],[0])
            with (output/'summary.csv').open(encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),25);self.assertEqual(sum(bool(r['acc']) for r in rows),24)
            self.assertEqual(sum('status=unreachable' in r['config'] for r in rows),1)

    def test_real_old_and_new_joint_tail0_generation_match_at_all_three_rates(self):
        import torch
        from test_pool_ablations import model_and_options
        from test_attention_predictor import make_checkpoint
        from llava_pruning.visa_comparison_methods import observe_prefill
        # Execute only the already hash-verified historical constructor for
        # this numerical regression; production recovery executes no old code.
        node=next(n for n in ast.parse(self.old[source.ENTRY]).body if isinstance(n,ast.FunctionDef) and n.name=='make_trial_method')
        self.assertEqual(source.digest(ast.dump(node,include_attributes=False).encode()),source.TAIL0_FUNCTION_PAIRS['make_trial_method'][0])
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'reviewed_67136cf_constructor','exec'),namespace)
        old_constructor=namespace['make_trial_method']
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for rate in (50,70,90):
                with self.subTest(rate=rate),tempfile.TemporaryDirectory() as temporary:
                    observations=[]
                    job=next(j for j in entry.schedule([rate]) if j.get('variant')=='joint_self')
                    for constructor in (old_constructor,entry.make_trial_method):
                        torch.manual_seed(138)
                        model,options=model_and_options(28);core=model.get_model()
                        checkpoint=Path(temporary)/'best.pt';make_checkpoint(checkpoint,core)
                        method=constructor(job,core,checkpoint)
                        method.configure(core,rate);method.prepare_inputs((384,384),options['inputs'])
                        with method.generation_context(model),observe_prefill(core) as lengths,torch.inference_mode():
                            result=model.generate(**options)
                        stats=method.stats(core,rate)
                        observations.append((result.sequences.clone(),lengths,tuple(stats[k] for k in
                            ('pre_llm_tokens','final_llm_tokens','stage2_base_tokens','stage2_anyres_tokens'))))
                    self.assertTrue(torch.equal(observations[0][0],observations[1][0]))
                    self.assertEqual(observations[0][1:],observations[1][1:])
        finally: torch.set_num_threads(threads)


if __name__=='__main__': unittest.main()
