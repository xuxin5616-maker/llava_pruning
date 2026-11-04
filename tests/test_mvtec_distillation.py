"""All-image MVTec loading and the real copied-student training CLI without val."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import train_attention_predictor_mvtec as entry
from benchmark_visa_methods import checkpoint_identity
from llava_pruning.mvtec_distillation import load_mvtec_distillation
from llava_pruning.methods import load_method
from llava_pruning.predictor_training import extract_teacher
from llava_pruning.prompts import GENERIC_PROMPT_V0
from test_clse import toy_model, layout


def dataset(root):
    paths = []
    for index in range(10):
        group = 'train/good' if index<3 else 'test/good' if index<5 else 'test/crack'
        path = root/'bottle'/group/f'{index:03d}.png'
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(16,8),'white').save(path)
        paths.append(path.relative_to(root).as_posix())
    masks=root/'bottle/ground_truth/crack';masks.mkdir(parents=True)
    Image.new('L',(16,8),255).save(masks/'005_mask.png')
    return paths


class MVTecDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_loader_includes_all_splits_ignores_masks_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);paths=dataset(root)
            samples,info=load_mvtec_distillation(root)
            self.assertEqual([s.origin_path for s in samples],sorted(paths))
            self.assertEqual(info['original_split_counts'],{'train':3,'test':7})
            self.assertEqual(info['categories'],{'bottle':10})
            self.assertFalse(info['original_visa_test_is_training_source'])
            self.assertTrue(info['original_mvtec_test_is_training_source'])
            self.assertTrue(all(s.gt is None and s.mask is None and s.bbox is None for s in samples))
            self.assertTrue(all(s.text==GENERIC_PROMPT_V0 and 'gt' not in s.source and 'mask' not in s.source for s in samples))
            self.assertEqual(load_mvtec_distillation(root),(samples,info))
            (root/paths[0]).write_bytes(b'changed file identity')
            self.assertNotEqual(load_mvtec_distillation(root)[1]['source_sha256'],info['source_sha256'])

    def test_missing_test_or_wrong_root_fails_instead_of_silently_training_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaisesRegex(ValueError,'Expected'): load_mvtec_distillation(root)
            (root/'bottle/train/good').mkdir(parents=True)
            Image.new('RGB',(4,4)).save(root/'bottle/train/good/0.png')
            with self.assertRaisesRegex(FileNotFoundError,'test'): load_mvtec_distillation(root)

    def test_actual_train_only_cli_saves_last_with_no_validation_or_best_checkpoint(self):
        torch.manual_seed(820)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);paths=dataset(root)
            model,_,_=toy_model();core=model.get_model()
            core.config.num_hidden_layers=6
            for index in (4,5): core.layers.append(type(core.layers[0])(core.config,layer_idx=index))
            class Backend:
                def __init__(self,*args,**kwargs):
                    self.model=model
                    self.inference_config={'model_dtype':'float32','attention':'sdpa_fixture'}
            packed=torch.randn(1,19,32)
            examples=[]
            def example(backend,sample,teacher_layer):
                examples.append(sample.sample_id)
                assert sample.gt is None and sample.mask is None
                return extract_teacher(backend.model.get_model(),packed,(2,16),layout(),teacher_layer=teacher_layer)
            options=['--model-path',str(root),'--data-root',str(root),'--teacher-layer','3',
                '--epochs','2','--student-precision','float32','--gradient-accumulation','4',
                '--output-dir',str(root/'trained')]
            with patch('llava_pruning.backend.LlavaBackend',Backend),patch('benchmark_fastv.validate_single_gpu'),\
                    patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=1),\
                    patch('torch.cuda.manual_seed_all'),patch('torch.cuda.synchronize'),\
                    patch('llava_pruning.predictor_training.prepare_teacher_example',side_effect=example),\
                    redirect_stdout(io.StringIO()) as logs:
                entry.main(options)
            trained=root/'trained'
            self.assertEqual(len(examples),20)
            self.assertEqual(set(examples),set(paths))
            self.assertTrue((trained/'last.pt').is_file())
            self.assertFalse((trained/'best.pt').exists())
            self.assertFalse((trained/'validation.jsonl').exists())
            saved=torch.load(trained/'last.pt',weights_only=True)
            self.assertEqual(saved['epoch'],2)
            self.assertIsNone(saved['validation_kl'])
            self.assertEqual(saved['architecture']['source_layers'],(3,4,5))
            self.assertEqual(saved['provenance']['training_dataset'],'mvtec')
            self.assertFalse(saved['provenance']['validation_enabled'])
            self.assertEqual(load_method('attention_predictor',trained/'method.json').checkpoint,(trained/'last.pt').resolve())
            info=json.loads((trained/'train_run.json').read_text(encoding='utf-8'))
            self.assertEqual(info['training_samples'],10);self.assertEqual(info['validation_samples'],0)
            self.assertFalse(info['validation_enabled']);self.assertFalse(info['original_visa_test_is_training_source'])
            self.assertTrue(info['original_mvtec_test_is_training_source'])
            self.assertEqual(info['final_test_dataset'],'independent VisA')
            self.assertEqual(info['configuration']['validation_fraction'],0.)
            self.assertEqual(info['optimization']['peak_learning_rate'],5e-5)
            self.assertEqual(info['optimization']['betas'],[.9,.98])
            self.assertEqual(info['optimization']['schedule']['total_steps'],6)
            self.assertTrue(checkpoint_identity(trained/'last.pt')['training_manifest'].endswith('train_run.json'))
            train=[json.loads(line) for line in (trained/'train.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual({r['image'] for r in train},set(paths))
            self.assertTrue(all('gt' not in r and 'mask' not in r for r in train))
            with (trained/'history.csv').open(newline='') as stream: history=list(csv.DictReader(stream))
            self.assertEqual([row['split'] for row in history],['train','train'])
            self.assertEqual([int(row['samples']) for row in history],[10,10])
            self.assertEqual([int(row['global_step']) for row in history],[3,6])
            status=json.loads((trained/'training_status.json').read_text(encoding='utf-8'))
            self.assertTrue(status['complete']);self.assertIsNone(status['best_validation_kl'])
            self.assertEqual(status['evaluation_checkpoint'],'last.pt')
            self.assertTrue(all(p.grad is None for p in core.parameters()))
            self.assertIn('loss=',logs.getvalue());self.assertIn('total_ETA=',logs.getvalue())
            self.assertIn('overall=100.0%',logs.getvalue());self.assertIn('No validation',logs.getvalue())


if __name__=='__main__': unittest.main()
