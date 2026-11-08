"""Unrelated experiments must not abort a sweep; real inference edits still do."""

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
from llava_pruning import mmad_source as source
from llava_pruning.mmad_data import load_frozen
from test_mmad_binary import fixture, prediction


def source_repo(root):
    root.mkdir()
    text='import json\nFIELDS=("config",)\n'
    for name in sorted(source.FUNCTIONS):
        text+=f'def {name}():\n'
        if name=='worker': text+='    from llava_pruning.backend import generate\n'
        text+='    return 1\n'
    text+='def verify_inputs():\n    return "old whole-source guard"\n'
    files={source.ENTRY:text,'benchmark_ficoco_v.py':'VERSION=1\n',
        'llava_pruning/__init__.py':'',
        'llava_pruning/backend.py':'def generate():\n    from .scores import rank\n    return rank()\n',
        'llava_pruning/scores.py':'def rank(): return 1\n',
        'vendor/llava/llava/model.py':'NATIVE=1\n','third_party/pinned.py':'PINNED=1\n',
        'run_fastv_layer_sweep.py':'GPU=5\n'}
    for name,text in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    source.git(root,'init','-q')
    commit(root,'original')
    return root


def commit(root,message,paths=None):
    source.git(root,'add',*(paths or ['.']))
    source.git(root,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm',message)
    return source.git(root,'rev-parse','HEAD').decode().strip()


class SourceTests(unittest.TestCase):
    def test_unrelated_root_edits_and_guard_changes_do_not_change_inference_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=source_repo(Path(temporary)/'src');initial=source.scope_identity(source.working_files(root))
            (root/'benchmark_ficoco_v.py').write_text('VERSION=2\n',encoding='utf-8')
            (root/'new_experiment.py').write_text('NEW=1\n',encoding='utf-8')
            path=root/source.ENTRY;path.write_text(path.read_text(encoding='utf-8').replace('old whole-source guard','scoped guard'),encoding='utf-8')
            self.assertEqual(source.verify_source({'mmad_source_identity':initial},root)['identity'],initial)
            self.assertIn('llava_pruning/scores.py',initial['files'])
            self.assertNotIn('benchmark_ficoco_v.py',initial['files'])

    def test_exact_legacy_git_hash_recovers_without_overwriting_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=source_repo(Path(temporary)/'src');old=source.git(root,'rev-parse','HEAD').decode().strip()
            for windows in (False,True):
                legacy=source.whole_identity(source.working_files(root),windows=windows)
                (root/'benchmark_ficoco_v.py').write_text('VERSION=2\n',encoding='utf-8');commit(root,'only FiCoCo')
                spec={'source_identity':legacy};result=source.verify_source(spec,root)
                self.assertEqual(result['legacy_recovery']['reference'],old)
                self.assertEqual(spec,{'source_identity':legacy})
                # Reset just this fixture for the second platform ordering.
                source.git(root,'reset','--hard',old)

    def test_backend_scores_entry_and_vendor_changes_are_rejected(self):
        for filename in ('llava_pruning/backend.py','llava_pruning/scores.py',source.ENTRY,'vendor/llava/llava/model.py'):
            with self.subTest(file=filename),tempfile.TemporaryDirectory() as temporary:
                root=source_repo(Path(temporary)/'src');files=source.working_files(root)
                legacy=source.whole_identity(files);scoped=source.scope_identity(files)
                path=root/filename;text=path.read_text(encoding='utf-8')
                path.write_text(text.replace('return 1','return 2') if filename==source.ENTRY else text+'\nCHANGED=1\n',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'inference dependencies changed'):
                    source.verify_source({'mmad_source_identity':scoped},root)
                with self.assertRaisesRegex(ValueError,'inference dependencies changed'):
                    source.verify_source({'source_identity':legacy},root)

    def test_unknown_legacy_identity_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=source_repo(Path(temporary)/'src')
            with self.assertRaisesRegex(ValueError,'Cannot identify the saved legacy source hash'):
                source.verify_source({'source_identity':{'files':999,'sha256':'unknown-old-code'}},root)

    def test_server_seven_extra_scripts_and_local_sweep_edit_recover_exact_old_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=source_repo(Path(temporary)/'src')
            scripts=['1.py','runMyPruningPoolRedundancy.py','runPoolRedundancyLayerSweep.py',
                'runPoolSpatialAblationSweep.py','runPoolSpatialStrengthSweep.py',
                'runPoolTemperatureSweep.py','runVisionZipAnchorOrder.py']
            for index,name in enumerate(scripts): (root/name).write_text(f'LOCAL={index}\n',encoding='utf-8')
            (root/'run_fastv_layer_sweep.py').write_text('GPU=3\n',encoding='utf-8')
            (root/'runMyPruningPoolSpatial.py.local-backup').write_text('not Python identity',encoding='utf-8')
            (root/'runPoolBaseFullSix.sh').write_text('# shell\n',encoding='utf-8')
            legacy=source.whole_identity(source.working_files(root),windows=True)
            old=source.git(root,'rev-parse','HEAD').decode().strip()
            (root/'benchmark_ficoco_v.py').write_text('VERSION=2\n',encoding='utf-8')
            helper=root/'llava_pruning/mmad_source.py';helper.write_text('POLICY="new guard"\n',encoding='utf-8')
            commit(root,'new tracked guard and unrelated update',paths=['benchmark_ficoco_v.py','llava_pruning/mmad_source.py'])
            before=source.working_files(root)
            result=source.verify_source({'source_identity':legacy},root)
            recovery=result['legacy_recovery']
            self.assertEqual(recovery['reference'],old)
            self.assertEqual(recovery['local_overlay']['mode'],'git_plus_local_changes')
            self.assertEqual(recovery['local_overlay']['added'],sorted(scripts))
            self.assertEqual(recovery['local_overlay']['modified'],['run_fastv_layer_sweep.py'])
            self.assertEqual(len(before),legacy['files']+1)
            self.assertEqual(source.working_files(root),before)

    def test_local_inference_edit_that_already_existed_is_proved_not_discarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=source_repo(Path(temporary)/'src');path=root/'llava_pruning/scores.py'
            path.write_text('def rank(): return 3\n',encoding='utf-8')
            legacy=source.whole_identity(source.working_files(root))
            (root/'benchmark_ficoco_v.py').write_text('VERSION=2\n',encoding='utf-8')
            commit(root,'unrelated',paths=['benchmark_ficoco_v.py'])
            result=source.verify_source({'source_identity':legacy},root)
            self.assertEqual(result['legacy_recovery']['local_overlay']['modified'],['llava_pruning/scores.py'])
            self.assertEqual(path.read_text(encoding='utf-8'),'def rank(): return 3\n')

    def test_changed_local_script_or_inference_after_old_run_is_not_assumed_unchanged(self):
        for path in ('1.py','llava_pruning/scores.py'):
            with self.subTest(file=path),tempfile.TemporaryDirectory() as temporary:
                root=source_repo(Path(temporary)/'src');(root/'1.py').write_text('LOCAL=1\n',encoding='utf-8')
                legacy=source.whole_identity(source.working_files(root))
                target=root/path;target.write_text(target.read_text(encoding='utf-8')+'\nCHANGED=2\n',encoding='utf-8')
                (root/'benchmark_ficoco_v.py').write_text('VERSION=2\n',encoding='utf-8')
                commit(root,'unrelated',paths=['benchmark_ficoco_v.py'])
                with self.assertRaisesRegex(ValueError,'Cannot identify|inference dependencies changed'):
                    source.verify_source({'source_identity':legacy},root)

    def test_recovery_preserves_two_completed_trials_and_runs_only_the_remaining_32(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);code=source_repo(root/'src');args=fixture(root)
            (code/'1.py').write_text('LOCAL=1\n',encoding='utf-8')
            (code/'run_fastv_layer_sweep.py').write_text('GPU=3\n',encoding='utf-8')
            args.gpus=[3,4];args.include_visa=True;args.experiment_group='joint_self';args.joint_self_tails=list(range(11))
            with patch.object(entry,'PROJECT',code):
                output,spec=entry.setup(args);spec.pop('mmad_source_identity');spec.pop('source_policy')
                spec['source_identity']=source.whole_identity(source.working_files(code),windows=True)
                common.save_json(output/'run.json',spec);old_manifest=(output/'run.json').read_bytes()
                samples=load_frozen(spec['input_json']);saved={}
                for job in spec['jobs'][:2]:
                    folder=output/job['job_id'];folder.mkdir()
                    result=entry.summarize_records([prediction(s,s.source['correct_option']) for s in samples],samples,job,2**30)
                    common.save_json(folder/'metrics.json',dict(result,job_id=job['job_id'],status='success',input_sha256=spec['input_sha256']))
                    saved[folder/'metrics.json']=(folder/'metrics.json').read_bytes()
                (code/'benchmark_ficoco_v.py').write_text('VERSION=2\n',encoding='utf-8')
                commit(code,'unrelated new benchmark',paths=['benchmark_ficoco_v.py'])
                calls=[]
                real_popen=entry.subprocess.Popen
                def process(command,**kwargs):
                    if command[0]=='git': return real_popen(command,**kwargs)
                    manifest=Path(command[command.index('--manifest')+1]);current=json.loads(manifest.read_text(encoding='utf-8'))
                    entry.verify_inputs(current)
                    job=next(j for j in current['jobs'] if j['job_id']==command[command.index('--worker')+1])
                    self.assertNotIn(job['job_id'],[j['job_id'] for j in spec['jobs'][:2]])
                    calls.append(job['job_id']);folder=output/job['job_id'];folder.mkdir()
                    result=entry.summarize_records([prediction(s,s.source['correct_option']) for s in samples],samples,job,2**30)
                    common.save_json(folder/'metrics.json',dict(result,job_id=job['job_id'],status='success',input_sha256=current['input_sha256']))
                    return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0)
                args.resume=True
                with patch.object(entry.subprocess,'Popen',side_effect=process):
                    self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),32)
                    self.assertEqual(entry.run(args),0);self.assertEqual(len(calls),32)
                self.assertEqual((output/'run.before_scoped_source.json').read_bytes(),old_manifest)
                self.assertEqual({path:path.read_bytes() for path in saved},saved)
                final=json.loads((output/'run.json').read_text(encoding='utf-8'))
                self.assertEqual(final['source_identity'],spec['source_identity'])
                self.assertIn('source_recovery',final);self.assertEqual(final['gpus'],['3','4'])
                self.assertEqual(final['joint_self_tails'],list(range(11)))
                with (output/'summary.csv').open(encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
                self.assertEqual(len(rows),34);self.assertTrue(all(row['acc'] for row in rows))


if __name__=='__main__': unittest.main()
