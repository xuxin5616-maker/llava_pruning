"""Verify native Base-only generation, exact masks, and the three PNG outputs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from llava import mm_utils
from llava.model.language_model.attention_predictor import CopiedAttentionPredictor, backbone_signature
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor

import visualize_base_pruning as entry
from llava_pruning.backend import LlavaBackend, configure_image_mode
from llava_pruning.predictor_training import save_student
from llava_pruning.visa_comparison_methods import observe_prefill
from test_mvtec_baseline_compact import fixture
from test_pool_ablations import model_and_options


class BaseOnlyPruningTests(unittest.TestCase):
    def test_native_matched_masks_and_three_images(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                args = fixture(Path(temporary))
                args = entry.parser().parse_args([], namespace=args)
                for path in Path(args.data_root).glob("*.png"):
                    Image.new("RGB", (768, 384), (12, 44, 76)).save(path)
                model, options = model_and_options(28)
                model.config.max_position_embeddings = 4096
                core = model.get_model()
                tower = core.get_vision_tower()
                tower.features = tower.features[:1]
                student = CopiedAttentionPredictor.from_backbone(core)
                save_student(args.checkpoint, student, student_precision="float32", backbone_depth=28,
                             teacher_layer=22, signature=backbone_signature(core.config))
                output, spec = entry.setup(args)
                samples = entry.samples_for(spec)
                configure_image_mode(model.config, "base_only")
                backend = LlavaBackend.__new__(LlavaBackend)
                backend.model, backend.vision_tower = model, tower
                backend.processor = SigLipImageProcessor()
                backend.tokenizer = type("Tokenizer", (), {"batch_decode": lambda *a, **k: ["B"]})()
                native_generate = model.generate

                with patch.object(mm_utils, "tokenizer_image_token", return_value=options["inputs"][0]), \
                        patch.object(model, "generate", side_effect=lambda **kw: native_generate(**dict(kw, max_new_tokens=2))), \
                        patch.object(torch.cuda, "synchronize"), torch.inference_mode():
                    for mode in entry.MODES:
                        method = entry.BaseOnlyPruning(core, mode, args.checkpoint)
                        self.assertEqual(method.target, 179)
                        with observe_prefill(core) as lengths:
                            entry.render_samples(backend, method, samples, spec, output / mode)
                        expected = [733] * 28 if mode == "fastv" else [733] * 2 + [183] * 26
                        self.assertEqual(list(lengths.values()), expected)
                        self.assertEqual(model.config.image_aspect_ratio, "base_only")
                        self.assertNotIn("prepare_inputs_labels_for_multimodal", model.__dict__)
                        rows = entry.restored(output / mode, samples, mode)
                        self.assertTrue(all(row["image_tokens"] == 729 and row["kept_tokens"] == 179 for row in rows))
                        for row in rows:
                            with Image.open(output / mode / row["output"]) as image:
                                rgb = np.asarray(image)
                                self.assertEqual(image.size, (384, 384))
                                self.assertEqual(np.all(rgb == 255, axis=2).sum(), 550 * 14 ** 2)
                                self.assertTrue(np.all(rgb[378:, :, :] == [12, 44, 76]))
                        entry.render_samples(backend, method, samples, spec, output / mode)
                    self.assertEqual(model.generate.call_count, len(samples) * 2)

                entry.comparison_images(output, samples)
                self.assertEqual(len(list(output.rglob("*.png"))), len(samples) * 3)
                with Image.open(output / "comparisons" / "sample_000000000.png") as image:
                    self.assertEqual(image.size, (1600, 472))
                args.resume = True
                self.assertEqual(entry.setup(args), (output, spec))
        finally:
            torch.set_num_threads(threads)
