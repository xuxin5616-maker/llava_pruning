"""Center-reference regression checks on CPU; no real checkpoint/GPU claims."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from matplotlib.figure import Figure

import visualize_patch_means as offline
from test_score_visualization import TinyProbeModel, TinyTower, SigLipImageProcessor
from llava_pruning.score_visualization import (
    LAYERS, STAGES, capture_scores, global_reference_info, prepare_views,
    reference_image, run_score_visualization,
)
from visualize_scores import build_parser


class CenterReferenceTests(unittest.TestCase):
    def setUp(self):
        self.processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        self.config = SimpleNamespace(image_grid_pinpoints=[[28, 28], [56, 28], [28, 56], [56, 56]])

    def image(self, size):
        rng = np.random.RandomState(38)
        return Image.fromarray(rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8))

    def test_integer_crop_and_resize_back_before_processor(self):
        for size, box in (((1024, 1024), [128, 128, 896, 896]),
                          ((61, 30), [8, 4, 53, 26]),
                          ((30, 61), [4, 8, 26, 53]),
                          ((1, 1), [0, 0, 1, 1])):
            with self.subTest(size=size):
                image = self.image(size)
                before = np.array(image)
                info = global_reference_info(size, "center-crop")
                self.assertEqual(info["crop_box_xyxy"], box)
                self.assertEqual(info["resize_back_to"], list(size))
                reference = reference_image(image, "center-crop")
                expected = image.crop(box).resize(size, Image.Resampling.BICUBIC)
                self.assertEqual(reference.size, size)
                np.testing.assert_array_equal(reference, expected)
                np.testing.assert_array_equal(image, before)
                self.assertIs(reference_image(image, "base"), image)

    def test_only_reference_pixels_change_anyres_and_shared_default_unchanged(self):
        for size in ((28, 28), (61, 30), (30, 61), (39, 37)):
            with self.subTest(size=size):
                image = self.image(size)
                old, old_base, tiles, geometry = prepare_views(image, self.processor, self.config, 14)
                explicit, _, _, _ = prepare_views(image, self.processor, self.config, 14, "base")
                new, reference, new_tiles, new_geometry = prepare_views(
                    image, self.processor, self.config, 14, "center-crop")
                self.assertTrue(torch.equal(old, explicit))
                self.assertTrue(torch.equal(new[1:], old[1:]))
                self.assertFalse(torch.equal(new[0], old[0]))
                self.assertEqual(new_geometry, geometry)
                for first, second in zip(tiles, new_tiles):
                    np.testing.assert_array_equal(first, second)
                enlarged = reference_image(image, "center-crop")
                expected = enlarged.resize((28, 28))
                np.testing.assert_array_equal(reference, expected)
                pixels = self.processor.preprocess(expected, return_tensors="pt")["pixel_values"][0]
                self.assertTrue(torch.equal(new[0], pixels))

    def test_each_stage_uses_reencoded_crop_mean_and_local_is_unchanged(self):
        torch.manual_seed(7)
        model = TinyProbeModel().eval()
        image = self.image((61, 30))
        old_pixels, _, _, _ = prepare_views(image, self.processor, self.config, 14)
        new_pixels, _, _, _ = prepare_views(image, self.processor, self.config, 14, "center-crop")
        old_scores, _ = capture_scores(model, old_pixels)
        new_scores, _ = capture_scores(model, new_pixels)
        np.testing.assert_array_equal(new_scores["local"], old_scores["local"])
        self.assertFalse(np.allclose(new_scores["global"], old_scores["global"]))
        with torch.inference_mode():
            states = model.tower.vision_tower(new_pixels, output_hidden_states=True).hidden_states
            projected = model.encode_images(new_pixels)
        for row, features in enumerate([states[layer] for layer in LAYERS] + [projected]):
            unit = features.float() / features.float().norm(dim=-1, keepdim=True)
            reference = unit[0].mean(dim=0)
            expected = -torch.nn.functional.cosine_similarity(unit[1:], reference[None, None], dim=-1)
            np.testing.assert_allclose(new_scores["global"][row], expected.numpy(), atol=2e-6)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_cli_defaults_new_reference_and_supports_legacy(self):
        parser = build_parser()
        argv = ["--model-path", "m", "--input-json", "i", "--data-root", "d", "--output-dir", "o"]
        self.assertEqual(parser.parse_args(argv).global_reference, "center-crop")
        self.assertEqual(parser.parse_args(argv + ["--global-reference", "base"]).global_reference, "base")
        with patch("llava_pruning.backend.LlavaBackend") as loader:
            with self.assertRaisesRegex(ValueError, "global reference"):
                run_score_visualization("m", "i", "d", "o", global_reference="invalid")
            loader.assert_not_called()

    def test_runner_and_offline_preserve_reference_metadata_and_captions(self):
        model = TinyProbeModel().eval()
        model.device, model.config = torch.device("cpu"), self.config
        backend = SimpleNamespace(model=model, vision_tower=model.tower, processor=self.processor,
                                  inference_config={"vision_dtype": "float32"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.image((61, 30))
            source.save(root / "image.png")
            questions = root / "questions.json"
            questions.write_text(json.dumps([{"id": "001", "image": "image.png", "gt": 0}]), encoding="utf-8")
            output = root / "online"
            titles = []
            original_save = Figure.savefig

            def inspect_figure(fig, *args, **kwargs):
                titles.append(fig._suptitle.get_text())
                return original_save(fig, *args, **kwargs)

            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=1), \
                 patch("llava_pruning.backend.LlavaBackend", return_value=backend), \
                 patch("llava.model.multimodal_encoder.siglip_encoder.SigLipVisionTower", TinyTower), \
                 patch.object(model, "encode_images", wraps=model.encode_images) as encoder, \
                 patch.object(Figure, "savefig", inspect_figure):
                run_score_visualization(root, questions, root, output, display_mode="patch-means")
            encoder.assert_called_once()
            expected, _, _, _ = prepare_views(source, self.processor, self.config, 14, "center-crop")
            self.assertTrue(torch.equal(encoder.call_args.args[0], expected.to(torch.float16)))
            self.assertIn("center 75% W/H, enlarged", titles[0])
            folder = output / "000001_001"
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["global_reference"], "center-crop")
            self.assertEqual(meta["global_reference_transform"]["crop_box_xyxy"], [8, 4, 53, 26])
            self.assertEqual(meta["geometry"]["original_size"], [61, 30])
            with np.load(folder / "scores.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["global_reference"].item(), "center-crop")
                self.assertEqual(tuple(archive["stages"]), STAGES)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["global_reference"], "center-crop")
            self.assertFalse(run["llm_generation"])
            redrawn = offline.run(output, root / "offline") / folder.name
            with Image.open(folder / "patch_means.png") as first, Image.open(redrawn / "patch_means.png") as second:
                np.testing.assert_array_equal(first, second)
            self.assertEqual((folder / "patch_means.csv").read_bytes(), (redrawn / "patch_means.csv").read_bytes())
            redrawn_meta = json.loads((redrawn / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(redrawn_meta["global_reference_transform"], meta["global_reference_transform"])


if __name__ == "__main__":
    unittest.main()
