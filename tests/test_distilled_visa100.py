"""Fixed 100-image pairing, explicit provenance, shared worker jobs and resume."""

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

import compare_distilled_visa100 as entry
import benchmark_visa_methods as common
from llava_pruning.data import load_samples
from llava_pruning.prompts import GENERIC_PROMPT_V0


class StudentComparisonTests(unittest.TestCase):
    def test_selection_is_exact_reproducible_and_covers_category_labels(self):
        samples=[SimpleNamespace(sample_id=f'{cat}/{gt}/{i:03d}',category=cat,gt=gt)
                 for cat in ('a','b','c') for gt in (0,1) for i in range(30+gt*10)]
        first=entry.select_samples(samples)
        self.assertEqual(len(first),100)
        self.assertEqual(len({s.sample_id for s in first}),100)
        self.assertEqual(first,entry.select_samples(list(reversed(samples))))
        self.assertNotEqual(first,entry.select_samples(samples,seed=43))
        self.assertEqual({(s.category,s.gt) for s in first},{(s.category,s.gt) for s in samples})
        self.assertEqual(entry.select_samples(samples,len(samples)),sorted(samples,key=lambda s:s.sample_id))
        with self.assertRaises(ValueError): entry.select_samples(samples,1000)
        with self.assertRaises(ValueError): entry.select_samples(samples,3)

    def fixture(self,root):
        model=root/'model';model.mkdir()
        (model/'config.json').write_text('{"num_hidden_layers":28}',encoding='utf-8')
        data=root/'VisA';(data/'split_csv').mkdir(parents=True)
        rows=[]
        for category in ('candle','cashew'):
            for label in ('Normal','Anomaly'):
                for i in range(30):
                    relative=f'{category}/Data/Images/{label}/{i:03d}.png'
                    path=data/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(16,16)).save(path)
                    rows.append([category,'test',label.lower(),relative])
        rows.append(['candle','train','normal','candle/Data/Images/Normal/train_only.png'])
        with (data/'split_csv/1cls.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['object','split','label','image']);writer.writerows(rows)
        for source in ('mvtec','visa'):
            folder=root/(source+'_student');folder.mkdir()
            path=folder/('last.pt' if source=='mvtec' else 'best.pt');path.write_bytes(source.encode())
            common.save_json(folder/'train_run.json',dict(original_visa_test_is_training_source=source=='visa',
                configuration={'dataset':source}))
        visa_ids=[dict(image=row[3]) for row in rows if row[1]=='test']
        for split,records in (('train',visa_ids[:15]),('validation',visa_ids[15:])):
            (root/'visa_student'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
        return entry.parser().parse_args(['--model-path',str(model),'--data-root',str(data),
            '--mvtec-checkpoint',str(root/'mvtec_student/last.pt'),'--visa-checkpoint',str(root/'visa_student/best.pt'),
            '--output-dir',str(root/'output')])

    def test_manifest_freezes_same_100_and_preserves_overlap_guard_elsewhere(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=self.fixture(root)
            output,spec=entry.setup(args)
            self.assertEqual(spec['sample_count'],100)
            self.assertEqual(spec['selection']['total_test_images'],120)
            samples=load_samples(spec['input_json'],spec['data_root'])
            self.assertEqual(len(samples),100)
            self.assertTrue(all(s.text==GENERIC_PROMPT_V0 and s.gt in (0,1) and s.mask is None for s in samples))
            self.assertTrue(all('train_only' not in s.sample_id for s in samples))
            for group in ('reference','mvtec','visa'):
                child=json.loads((output/group/'run.json').read_text(encoding='utf-8'))
                common.verify_inputs(child)
                self.assertEqual(child['input_json'],spec['input_json'])
                self.assertEqual(child['input_sha256'],spec['input_sha256'])
                self.assertEqual(child['sample_count'],100)
            self.assertEqual(len(spec['jobs']),7)
            self.assertTrue(spec['checkpoints']['visa']['visa_test_used_as_training_source'])
            self.assertEqual(spec['training_overlap']['visa']['train_overlap']+
                             spec['training_overlap']['visa']['validation_overlap'],100)
            with self.assertRaisesRegex(ValueError,'VisA test'): common.checkpoint_identity(args.visa_checkpoint)
            args.resume=True
            self.assertEqual(entry.setup(args)[1],spec)
            child_path=output/'visa/run.json'
            child=json.loads(child_path.read_text(encoding='utf-8'));child['sample_count']=120
            common.save_json(child_path,child)
            with self.assertRaisesRegex(ValueError,'worker configuration changed'): entry.setup(args)

    def test_sequential_jobs_use_fixed_gpu_and_report_only_100_per_trial(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=self.fixture(root)
            calls=[]
            def process(command,**kwargs):
                calls.append(command)
                manifest=Path(command[command.index('--manifest')+1])
                job_id=command[command.index('--worker')+1]
                child=json.loads(manifest.read_text(encoding='utf-8'))
                job=next(j for j in child['jobs'] if j['job_id']==job_id)
                samples=load_samples(child['input_json'],child['data_root'])
                self.assertEqual(len(samples),100)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'5')
                if job['group']=='reference': self.assertIsNone(child['checkpoint'])
                else:
                    self.assertEqual(child['checkpoint']['declared_training_dataset'],job['group'])
                    self.assertEqual(job['method'],'hybrid_baseline')
                folder=manifest.parent/job_id;folder.mkdir()
                tp=sum(s.gt==1 for s in samples);tn=100-tp
                common.save_json(folder/'metrics.json',dict(job_id=job_id,status='success',complete=True,
                    input_sha256=child['input_sha256'],expected_samples=100,evaluated_samples=100,
                    accuracy=1.,tp=tp,tn=tn,fp=0,fn=0,lat_mean_ms=123.,mem_peak_allocated_gib=20.))
                return SimpleNamespace(wait=lambda **kw:0)
            with patch.object(entry.subprocess,'Popen',side_effect=process):
                self.assertEqual(entry.run(args),0)
                self.assertEqual(len(calls),7)
                args.resume=True
                self.assertEqual(entry.run(args),0)
                self.assertEqual(len(calls),7)
            with (root/'output/summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);self.assertEqual(tuple(reader.fieldnames),common.FIELDS);rows=list(reader)
            self.assertEqual(len(rows),7)
            for source in ('mvtec','visa'):
                matches=[r for r in rows if f'student_trained_on={source};' in r['config']]
                self.assertEqual(len(matches),3)
                self.assertTrue(all('tail=9(extra)' in r['config'] and 'input=base_text' in r['config'] for r in matches))
            self.assertTrue(all(float(r['acc'])==100 and float(r['relative_accuracy_pct'])==100 for r in rows))

    def test_wrong_dataset_slot_or_changed_subset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);args=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'MVTec slot'): entry.student_identity(args.visa_checkpoint,'mvtec')
            with self.assertRaisesRegex(ValueError,'VisA slot'): entry.student_identity(args.mvtec_checkpoint,'visa')
            _,spec=entry.setup(args)
            Path(spec['input_json']).write_text('changed',encoding='utf-8')
            args.resume=True
            with self.assertRaisesRegex(ValueError,'Frozen VisA input changed'): entry.setup(args)


if __name__=='__main__': unittest.main()
