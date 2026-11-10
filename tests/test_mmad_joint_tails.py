"""Full-source MMAD joint attention tail sweep, budget invariance and reports."""

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
from test_mmad_binary import fixture, prediction


class JointTailScheduleTests(unittest.TestCase):
    def test_33_joint_trials_share_one_reference_and_keep_real_tail_labels(self):
        args=entry.parser().parse_args(['--gpus','3','4','--include-visa','--experiment-group','joint_self',
            '--joint-self-tails',*map(str,range(11))])
        jobs=entry.schedule(args.prune_rates,args.gpus,experiment_group=args.experiment_group,
            joint_self_tails=args.joint_self_tails,include_visa=args.include_visa)
        self.assertEqual(len(jobs),34)
        self.assertEqual(len({j['job_id'] for j in jobs}),34)
        self.assertEqual(sum(j['method']=='llava_unpruned' for j in jobs),1)
        self.assertTrue(all(j['status']=='pending' for j in jobs))
        self.assertEqual({j['gpu'] for j in jobs},{'3','4'})
        for rate in (50,70,90):
            selected=[j for j in jobs if j['prune_rate']==rate]
            self.assertEqual([j['tail_drop_layers'] for j in selected],list(range(11)))
            for job in selected:
                tail=job['tail_drop_layers']
                self.assertEqual(job['variant'],'joint_self')
                self.assertIn(f'tail={tail}'+('(extra)' if tail else '')+';',job['config'])
                self.assertIn('lambda=0.7;anyres_self;input=full;base_policy=joint;',job['config'])
                self.assertIn('exclude=none',job['config']);self.assertIn('training_overlap=VisA',job['config'])
            self.assertEqual(selected[0]['job_id'],f'hybrid_joint_self_{rate:02d}')
        for tails in ([],[0,0],[-1],[27],[.5],[True]):
            with self.assertRaisesRegex(ValueError,'Joint-self tails'):
                entry.schedule([50],joint_self_tails=tails)

    def test_full_source_manifest_launcher_and_resume_preserve_the_whole_sweep(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            args=fixture(Path(temporary));args.gpus=[3,4];args.include_visa=True
            args.experiment_group='joint_self';args.joint_self_tails=list(range(11))
            calls=[]
            def process(command,**kwargs):
                manifest=Path(command[command.index('--manifest')+1])
                spec=json.loads(manifest.read_text(encoding='utf-8'))
                self.assertEqual(spec['joint_self_tails'],list(range(11)))
                self.assertEqual(spec['dataset_info']['excluded_sources'],[])
                self.assertEqual(set(spec['dataset_info']['source_counts']),{'GoodsAD','MVTec-AD','MVTec-LOCO','VisA'})
                self.assertIn('NOT independent',spec['training_overlap']['VisA'])
                job_id=command[command.index('--worker')+1]
                job=next(j for j in spec['jobs'] if j['job_id']==job_id)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],job['gpu'])
                calls.append(job)
                samples=load_frozen(spec['input_json'])
                result=entry.summarize_records([prediction(s,s.source['correct_option']) for s in samples],samples,job,2**30)
                folder=manifest.parent/job_id;folder.mkdir()
                common.save_json(folder/'metrics.json',dict(result,job_id=job_id,status='success',input_sha256=spec['input_sha256']))
                return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0)
            with patch.object(entry.subprocess,'Popen',side_effect=process):
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),34)
                args.resume=True;args.include_visa=False;args.joint_self_tails=[0];args.gpus=[7]
                self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),34)
            with (args.output_dir/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),34)
            self.assertTrue(all(float(row['relative_accuracy_pct'])==100 for row in rows))
            self.assertTrue(all('exclude=none' in row['config'] for row in rows))
            with (args.output_dir/'summary_by_source.csv').open(encoding='utf-8-sig',newline='') as stream:
                source_rows=list(csv.DictReader(stream))
            self.assertEqual(len(source_rows),136)
            self.assertEqual({r['source'] for r in source_rows},{'GoodsAD','MVTec-AD','MVTec-LOCO','VisA'})

    def test_append_stretched_tail8_rates_reuses_inputs_and_preserves_original_results(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = entry.parser().parse_args([
                "--experiment-group", "baselines_joint90", "--resize-mode", "stretch",
                "--joint-self-tails", "8", "9", "--gpus", "0", "1", "2", "3", "4", "5",
            ], namespace=fixture(root))
            common.save_json(Path(args.model_path) / "config.json",
                             {"num_hidden_layers": 28, "image_grid_pinpoints": [[1152, 1152]]})
            calls = []

            def process(command, **kwargs):
                manifest = Path(command[command.index("--manifest") + 1])
                spec = json.loads(manifest.read_text(encoding="utf-8"))
                job_id = command[command.index("--worker") + 1]
                job = next(j for j in spec["jobs"] if j["job_id"] == job_id)
                calls.append(job_id)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], job["gpu"])
                samples = load_frozen(spec["input_json"], exclude_sources=("VisA",))
                records = [prediction(s, s.source["correct_option"]) for s in samples]
                report = entry.summarize_records(records, samples, job, 2**30)
                folder = manifest.parent / job_id
                folder.mkdir()
                common.save_json(folder / "metrics.json", dict(report, job_id=job_id, status="success",
                                                               input_sha256=spec["input_sha256"]))
                return SimpleNamespace(poll=lambda: 0, wait=lambda **kw: 0)

            with patch.object(entry.subprocess, "Popen", side_effect=process):
                self.assertEqual(entry.run(args), 0)
                self.assertEqual(len(calls), 20)
                output = args.output_dir
                old = json.loads((output / "run.json").read_text(encoding="utf-8"))
                old_summary = (output / "summary.csv").read_bytes()
                old_assets = {path: path.stat().st_mtime_ns for path in (output / "prepared_images").glob("*.png")}
                extra = entry.parser().parse_args([
                    "--resume", "--append-joint-self-rates", "50", "70", "--joint-self-tails", "8",
                    "--output-dir", str(output),
                ])
                self.assertEqual(entry.run(extra), 0)
                self.assertEqual(calls[20:], ["hybrid_joint_self_tail08_50", "hybrid_joint_self_tail08_70"])
                self.assertEqual(entry.run(extra), 0)
                self.assertEqual(len(calls), 22)
                new = json.loads((output / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(new["jobs"][:len(old["jobs"])], old["jobs"])
                self.assertEqual(new["input_sha256"], old["input_sha256"])
                self.assertEqual(new["gpus"], old["gpus"])
                self.assertEqual(new["completed_trials"], 22)
                self.assertTrue(new["complete"])
                self.assertTrue((output / "summary.csv").read_bytes().startswith(old_summary))
                self.assertTrue(all(path.stat().st_mtime_ns == stamp for path, stamp in old_assets.items()))

    def test_custom_tail_list_survives_historical_visa_extension(self):
        from test_mmad_extension import historical
        def custom_fixture(root):
            args=fixture(root);args.experiment_group='joint_self';args.joint_self_tails=[0,2,10]
            return args
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()),\
             patch('test_mmad_extension.fixture',side_effect=custom_fixture):
            _,old,saved,args=historical(Path(temporary))
            output,spec=entry.setup(args)
            self.assertEqual(spec['joint_self_tails'],[0,2,10])
            self.assertEqual(len(spec['jobs']),10)
            self.assertEqual({j.get('tail_drop_layers') for j in spec['jobs'][1:]},{0,2,10})
            self.assertEqual(set(spec['reuse_history']['trials']),{j['job_id'] for j in spec['jobs']})
            self.assertEqual(spec['include_visa'],True)
            self.assertNotEqual(output,old)


class JointTailGenerationTests(unittest.TestCase):
    def test_all_33_real_prefills_keep_budget_and_drop_images_only_at_the_selected_tail(self):
        import torch
        from test_pool_ablations import model_and_options
        from test_attention_predictor import make_checkpoint
        from llava_pruning.visa_comparison_methods import observe_prefill, observed_budget
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            jobs=entry.schedule([50,70,90],experiment_group='joint_self',joint_self_tails=list(range(11)))
            reference={}
            for job in jobs[1:]:
                rate,tail=job['prune_rate'],job['tail_drop_layers']
                with self.subTest(rate=rate,tail=tail),tempfile.TemporaryDirectory() as temporary:
                    torch.manual_seed(138)
                    model,options=model_and_options(28);core=model.get_model()
                    checkpoint=Path(temporary)/'best.pt';make_checkpoint(checkpoint,core)
                    method=entry.make_trial_method(job,core,checkpoint)
                    self.assertEqual((method.mode,method.predictor_input,method.base_policy,method.share),
                        ('anyres_self','full','joint',.7))
                    self.assertEqual(method.tail_drop_layers,tail)
                    method.configure(core,rate);method.prepare_inputs((384,384),options['inputs'])
                    with method.generation_context(model),observe_prefill(core) as lengths,torch.inference_mode():
                        result=model.generate(**options)
                    stats=method.stats(core,rate)
                    budget=observed_budget(job['method'],rate,stats,lengths,1485)
                    self.assertEqual(result.sequences.shape[-1],3)
                    signature=tuple(stats[k] for k in ('pre_llm_tokens','final_llm_tokens','stage2_base_tokens','stage2_anyres_tokens'))
                    self.assertEqual(reference.setdefault(rate,signature),signature)
                    self.assertEqual([lengths[i] for i in range(28)],
                        [stats['pre_llm_tokens']+4]*2+[stats['final_llm_tokens']+4]*(26-tail)+[4]*tail)
                    self.assertAlmostEqual(budget['two_stage_average_prune_rate'],rate,delta=.15)
                    self.assertAlmostEqual(budget['actual_average_prune_rate'],
                        budget['two_stage_average_prune_rate']+budget['actual_tail_contribution'])
                    self.assertEqual(budget['final_layer_visual_tokens']==0,tail>0)
                    self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))
        finally:
            torch.set_num_threads(threads)


if __name__=='__main__':
    unittest.main()
