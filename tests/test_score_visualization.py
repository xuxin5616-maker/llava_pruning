"""CPU feature/geometry tests; no checkpoint download or claimed GPU equivalence."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "vendor" / "llava"))
from llava.model.multimodal_encoder.siglip_encoder import (
    SigLipVisionConfig, SigLipVisionModel, SigLipImageProcessor)
from llava.mm_utils import process_anyres_image, resize_and_pad_image
from llava_pruning.score_visualization import (
    LAYERS, STAGES, ScoreGeometry, score_tokens, capture_scores, prepare_views,
    tile_score_map, stitch_score_map, color_limits, load_score_samples,
    save_sample, draw_figures, run_score_visualization)
from visualize_scores import build_parser


class TinyTower(torch.nn.Module):
    """Real SigLIP blocks, small patch grid; same layer truncation/feature selection."""
    def __init__(self):
        super().__init__()
        config = SigLipVisionConfig(hidden_size=8, intermediate_size=16,
                                    num_hidden_layers=27, num_attention_heads=2,
                                    image_size=28, patch_size=14)
        self.vision_tower = SigLipVisionModel(config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = torch.nn.Identity()

    def forward(self, pixels):
        pixels = pixels.to(dtype=next(self.vision_tower.parameters()).dtype)
        return self.vision_tower(pixels, output_hidden_states=True).hidden_states[-1]


class TinyProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = TinyTower()
        self.mm_projector = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.GELU(),
                                                torch.nn.Linear(12, 12))
        self.last_output = None

    def get_vision_tower(self):
        return self.tower

    def get_model(self):
        return self

    def encode_images(self, pixels):
        result = self.mm_projector(self.tower(pixels))
        self.last_output = result.detach().clone()
        return result

    def generate(self, *args, **kwargs):
        raise AssertionError("Must not generate text")


class FeatureScoreTests(unittest.TestCase):
    def test_exact_global_and_local_negative_cosine(self):
        features = torch.tensor([[[1., 0.], [1., 0.]],
                                 [[1., 0.], [0., 1.]],
                                 [[0., 1.], [0., 1.]]], dtype=torch.float16)
        before = features.clone()
        actual = score_tokens(features)
        np.testing.assert_allclose(actual["global"], [[-1, 0], [0, 0]], atol=1e-6)
        np.testing.assert_allclose(actual["local"], [[-2 ** -0.5] * 2, [-1, -1]], atol=1e-6)
        self.assertTrue(torch.equal(features, before))
        self.assertEqual(features.dtype, torch.float16)
        self.assertEqual(actual["local"].dtype, np.float32)

    def test_tile_permutation_preserves_corresponding_scores(self):
        features = torch.randn(4, 6, 8)
        expected = score_tokens(features)
        actual = score_tokens(features[[0, 3, 1, 2]])
        for kind in expected:
            np.testing.assert_allclose(actual[kind], expected[kind][[2, 0, 1]], atol=1e-6)

    def test_undefined_cosine_is_missing_not_zero(self):
        features = torch.tensor([[[1., 0.], [-1., 0.]], [[1., 0.], [0., 0.]]])
        actual = score_tokens(features)
        self.assertTrue(np.isnan(actual["global"]).all())
        self.assertEqual(actual["local"][0, 0], -1.)
        self.assertTrue(np.isnan(actual["local"][0, 1]))
        features[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            score_tokens(features)
        with self.assertRaisesRegex(ValueError, "Base"):
            score_tokens(torch.ones(1, 3, 4))

    def test_real_siglip_layer_indices_and_projector_forward_unchanged(self):
        torch.manual_seed(7)
        model = TinyProbeModel().eval()
        pixels = torch.randn(3, 3, 28, 28)
        with torch.inference_mode():
            states = model.tower.vision_tower(pixels, output_hidden_states=True).hidden_states
            projected = model.encode_images(pixels).clone()
        self.assertEqual(len(states), 27)  # embeddings + 26 actual blocks
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        actual, shapes = capture_scores(model, pixels)
        self.assertTrue(torch.equal(projected, model.last_output))
        for row, layer in enumerate(LAYERS):
            expected = score_tokens(states[layer])  # NOT hidden_states[layer-1]
            for kind in expected:
                np.testing.assert_array_equal(actual[kind][row], expected[kind])
            self.assertEqual(shapes[STAGES[row]], [3, 4, 8])
        for kind, expected in score_tokens(projected).items():
            np.testing.assert_array_equal(actual[kind][-1], expected)
        self.assertEqual(shapes["projector"], [3, 4, 12])
        self.assertTrue(all(torch.equal(value, weights[key]) for key, value in model.state_dict().items()))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_hooks_cleaned_after_error_and_no_layer_substitution(self):
        model = TinyProbeModel().eval()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            capture_scores(model, torch.full((2, 3, 28, 28), float("nan")))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        del model.tower.vision_tower.vision_model.encoder.layers[-1:]
        with self.assertRaisesRegex(ValueError, "26-block"):
            capture_scores(model, torch.randn(2, 3, 28, 28))
        model = TinyProbeModel()
        with self.assertRaisesRegex(ValueError, "eval"):
            capture_scores(model, torch.randn(2, 3, 28, 28))


class ScoreGeometryTests(unittest.TestCase):
    def test_preprocessed_pixels_equal_existing_anyres(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_grid_pinpoints=[[28, 28], [56, 28], [28, 56], [56, 56]])
        rng = np.random.RandomState(5)
        for size in ((28, 28), (61, 30), (30, 61), (39, 37)):
            with self.subTest(size=size):
                image = Image.fromarray(rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8))
                before = np.array(image)
                pixels, base, tiles, geometry = prepare_views(image, processor, config, 14)
                expected = process_anyres_image(image, processor, config.image_grid_pinpoints)
                self.assertTrue(torch.equal(pixels, expected))
                reconstructed = torch.stack([processor.preprocess(view, "pt")["pixel_values"][0]
                                             for view in [base] + tiles])
                self.assertTrue(torch.equal(pixels, reconstructed))
                self.assertEqual(len(tiles), geometry.tile_count)
                np.testing.assert_array_equal(np.array(image), before)

    def test_patch_support_does_not_stretch_378_to_384(self):
        geometry = ScoreGeometry.create((768, 384), (2, 1), 384, 14)
        scores = np.stack([np.full(729, 1.), np.full(729, 2.)])
        result = stitch_score_map(scores, geometry)
        self.assertEqual(result.shape, (384, 768))
        self.assertTrue(np.all(result[:378, :378] == 1))
        self.assertTrue(np.all(result[:378, 384:762] == 2))
        self.assertTrue(np.isnan(result[:, 378:384]).all())
        self.assertTrue(np.isnan(result[:, 762:]).all())
        self.assertTrue(np.isnan(result[378:]).all())

    def test_row_major_token_and_tile_coordinates(self):
        geometry = ScoreGeometry.create((8, 8), (2, 2), 4, 2)
        scores = np.arange(16, dtype=np.float32).reshape(4, 4)
        expected = np.array([[0, 1, 4, 5], [2, 3, 6, 7],
                             [8, 9, 12, 13], [10, 11, 14, 15]], dtype=np.float32)
        np.testing.assert_array_equal(stitch_score_map(scores, geometry), expected.repeat(2, 0).repeat(2, 1))

    def test_padding_matches_actual_resize_with_ceil_and_odd_borders(self):
        for size in ((31, 17), (17, 31), (67, 13), (13, 67)):
            geometry = ScoreGeometry.create(size, (2, 2), 28, 14)
            source = Image.new("RGB", size, "white")
            padded = np.asarray(resize_and_pad_image(source, (56, 56)))[:, :, 0] > 0
            mapped = np.block([[tile_score_map(np.ones(4), geometry, row * 2 + col)
                                for col in range(2)] for row in range(2)])
            np.testing.assert_array_equal(np.isfinite(mapped), padded)
            stitched = stitch_score_map(np.ones((4, 4)), geometry)
            self.assertEqual(stitched.shape, geometry.resized_size[::-1])
            self.assertTrue(np.isfinite(stitched).all())

    def test_shared_color_limits_exclude_invisible_padding(self):
        geometry = ScoreGeometry.create((8, 1), (2, 1), 4, 2)
        scores = {"global": np.full((5, 2, 4), -0.8, dtype=np.float32),
                  "local": np.full((5, 2, 4), -0.6, dtype=np.float32)}
        # Visible content y=1 only overlaps the top row of tokens.
        for values in scores.values():
            values[:, :, 2:] = 1.0
        low, high = color_limits(scores, geometry, "sample")
        self.assertAlmostEqual(low, -0.8)
        self.assertAlmostEqual(high, -0.6)
        self.assertEqual(color_limits(scores, geometry, "fixed"), (-1., 1.))
        for values in scores.values():
            values[:] = -1
        self.assertEqual(color_limits(scores, geometry, "sample"), (-1., -0.99))
        for values in scores.values():
            values[:] = np.nan
        with self.assertRaisesRegex(ValueError, "No finite"):
            color_limits(scores, geometry, "sample")


class ScoreOutputTests(unittest.TestCase):
    def test_cli_defaults_all_records_and_no_pruning_options(self):
        parser = build_parser()
        args = parser.parse_args(["--model-path", "m", "--input-json", "i", "--data-root", "d",
                                  "--output-dir", "o"])
        self.assertIsNone(args.limit)
        self.assertEqual(args.color_scale, "sample")
        self.assertNotIn("method", vars(args))
        self.assertNotIn("prompt_version", vars(args))

    def test_records_no_100_limit_no_annotation_dependency_and_ids_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "imgs").mkdir()
            Image.new("RGB", (4, 4)).save(root / "imgs" / "image.png")
            records = [{"question_id": f"{i:09d}", "image": "image.png", "mask": "missing.png"}
                       for i in range(201)]
            path = root / "questions.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            samples = load_score_samples(path, root)
            self.assertEqual(len(samples), 201)
            self.assertEqual(samples[0]["id"], "000000000")
            self.assertEqual(len(load_score_samples(path, root, limit=3)), 3)
            records[1]["question_id"] = records[0]["question_id"]
            path.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_score_samples(path, root)

    def test_real_png_and_numeric_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            geometry = ScoreGeometry.create((55, 31), (2, 2), 28, 14)
            image = Image.new("RGB", (55, 31), "steelblue")
            base = image.resize((28, 28))
            tiles = [base.copy() for _ in range(4)]
            scores = {"global": np.linspace(-1, 0.1, 80).astype(np.float32).reshape(5, 4, 4),
                      "local": np.linspace(-.9, -.2, 80).astype(np.float32).reshape(5, 4, 4)}
            folder = Path(tmp) / "sample"
            sample = {"id": "000108", "gt": 1, "image": Path(tmp) / "source.png"}
            save_sample(image, base, tiles, scores, {}, geometry, sample, folder, "sample")
            self.assertEqual({p.name for p in folder.iterdir()},
                             {"scores.npz", "metadata.json", "scores_overview.png", "scores_tiles_01.png"})
            with np.load(folder / "scores.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["global_scores"], scores["global"])
                self.assertEqual(tuple(saved["stages"]), STAGES)
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["score_shape"], [5, 4, 4])
            for name in metadata["figures"]:
                with Image.open(folder / name) as png:
                    self.assertGreater(png.width, 1000)
                    self.assertGreater(png.height, 1000)
                    self.assertAlmostEqual(png.info["dpi"][0], 150, places=1)
            with self.assertRaises(FileExistsError):
                save_sample(image, base, tiles, scores, {}, geometry, sample, folder, "sample")

    def test_tile_pages_include_every_tile_once(self):
        from unittest.mock import patch
        from matplotlib import colormaps
        from matplotlib.figure import Figure
        geometry = ScoreGeometry.create((140, 28), (5, 1), 28, 14)
        image = Image.new("RGB", geometry.original_size, "gray")
        base = image.resize((28, 28))
        scores = {kind: np.ones((5, 5, 4), dtype=np.float32) * -0.5 for kind in ("global", "local")}
        titles = []
        def inspect_figure(fig, target, **kwargs):
            titles.extend(ax.get_title() for ax in fig.axes)
            for ax in fig.axes:
                if len(ax.images) == 2:
                    # Fixed [-1, 1] maps this constant -0.5 score to 0.25.
                    rgba = np.asarray(ax.images[1].get_array())
                    np.testing.assert_allclose(rgba[..., :3],
                                               np.broadcast_to(colormaps["jet"](0.25)[:3], rgba[..., :3].shape))
                    np.testing.assert_allclose(rgba[..., 3], 0.70)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, "savefig", inspect_figure):
            saved = draw_figures(image, base, [base] * 5, scores, geometry, Path(tmp), "test", "fixed")
        self.assertEqual(saved["figures"], ["scores_overview.png", "scores_tiles_01.png", "scores_tiles_02.png"])
        for index in range(1, 6):
            for kind in ("Global", "Local"):
                self.assertEqual(titles.count(f"Tile {index} | {kind}"), 1)

    def test_runner_uses_encoder_only_and_writes_compact_provenance(self):
        from unittest.mock import patch
        model = TinyProbeModel().eval()
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(image_grid_pinpoints=[[28, 28], [56, 28]])
        backend = SimpleNamespace(model=model, vision_tower=model.tower,
                                  processor=SigLipImageProcessor(size=(28, 28),
                                            crop_size={"height": 28, "width": 28}),
                                  inference_config={"vision_dtype": "float32", "attention_source": "unused"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (51, 24), "orange").save(root / "image.png")
            questions = root / "questions.jsonl"
            questions.write_text(json.dumps({"question_id": "00001", "image": "image.png", "gt": 0}) + "\n"
                                 + json.dumps({"question_id": "00002", "image": "image.png"}), encoding="utf-8")
            output = root / "output"
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=1), \
                 patch("llava_pruning.backend.LlavaBackend", return_value=backend) as loader, \
                 patch("llava.model.multimodal_encoder.siglip_encoder.SigLipVisionTower", TinyTower), \
                 patch("llava_pruning.score_visualization.draw_figures",
                       return_value={"figures": [], "color_limits": [-1, 1]}):
                run_score_visualization(root, questions, root, output, limit=1, color_scale="fixed")
                loader.assert_called_once_with(root, roi_mode="anyres_max_9")
                self.assertEqual(sorted(path.name for path in output.iterdir()), ["000001_00001", "run.json"])
                saved = json.loads((output / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["stages"], list(STAGES))
                self.assertFalse(saved["pruning"])
                self.assertFalse(saved["llm_generation"])
                self.assertEqual(saved["colormap"], "jet")
                self.assertEqual(saved["overlay_alpha"], 0.70)
                self.assertNotIn("attention_source", saved["loader_config"])
                with np.load(output / "000001_00001" / "scores.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["global_scores"].shape, (5, 2, 4))
                with self.assertRaisesRegex(FileExistsError, "new/empty"):
                    run_score_visualization(root, questions, root, output)
                loader.assert_called_once()

    def test_multi_gpu_refused_before_loading_or_writing(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (4, 4)).save(root / "image.png")
            questions = root / "questions.json"
            questions.write_text(json.dumps([{"id": "1", "image": "image.png"}]), encoding="utf-8")
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=2), \
                 patch("llava_pruning.backend.LlavaBackend") as loader:
                with self.assertRaisesRegex(RuntimeError, "exactly one visible"):
                    run_score_visualization(root, questions, root, root / "output")
                loader.assert_not_called()
                self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
