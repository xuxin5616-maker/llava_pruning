"""Synthetic CPU checks; no checkpoint, GPU or dataset results are claimed."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import run_base_center_weighted_scores as vis
from llava_pruning.score_visualization import ScoreGeometry


def example(geometry=None):
    geometry = geometry or ScoreGeometry.create((420, 240), (3, 3), 384, 14)
    inner, ring, outer = vis.center_masks(geometry)
    features = np.zeros((1 + geometry.tile_count, 729, 3), dtype=np.float32)
    features[..., 2] = 1
    features[0, inner] = [10, 0, 0]
    features[0, ring] = [0, 2, 0]
    # Test tokens aligned with each orthogonal direction, including outside ROI.
    features[1:, :243] = [3, 0, 0]
    features[1:, 243:486] = [0, 4, 0]
    return geometry, features


def stacked(features, geometry):
    stages = [vis.score_with_config(features, geometry) for _ in vis.LAYERS]
    return {key: np.stack([stage[key] for stage in stages]) for key in stages[0]}


class CenterMathTests(unittest.TestCase):
    def test_default_rectangles_are_width_height_not_area(self):
        geometry, _ = example()
        inner, ring, outer = vis.center_masks(geometry)
        self.assertEqual([int(m.sum()) for m in (inner, ring, outer)], [196, 245, 441])
        self.assertFalse((inner & ring).any())
        np.testing.assert_array_equal(inner | ring, outer)
        np.testing.assert_array_equal(vis.center_masks(asdict(geometry))[0], inner)

    def test_normalize_then_separate_means_then_weight(self):
        geometry, features = example()
        before = features.copy()
        score = vis.score_with_config(features, geometry)
        np.testing.assert_allclose(score['reference'], [.8, .2, 0])
        self.assertAlmostEqual(float(score['reference_norm']), np.sqrt(.68), places=6)
        np.testing.assert_allclose(score['reference_unit'], np.array([.8, .2, 0]) / np.sqrt(.68))
        np.testing.assert_array_equal(features, before)
        inner, ring, outer = vis.center_masks(geometry)
        np.testing.assert_allclose(score['base_cosine'][inner], .8 / np.sqrt(.68))
        np.testing.assert_allclose(score['base_cosine'][ring], .2 / np.sqrt(.68))
        np.testing.assert_allclose(score['base_cosine'][~outer], 0)
        # Euclidean distance is against the weighted mean, NOT its unit vector.
        np.testing.assert_allclose(score['base_l2'][inner], np.sqrt(.08))
        self.assertTrue(np.isfinite(score['base_l2'][~outer]).all())

    def test_magnitudes_do_not_affect_scores_and_anyres_cannot_change_reference(self):
        geometry, features = example()
        scores = vis.score_with_config(features, geometry)
        scaled = features * np.linspace(.1, 9, features.shape[0] * 729).reshape(-1, 729, 1)
        actual = vis.score_with_config(scaled, geometry)
        for view in ('base', 'tile'):
            for metric in vis.METRICS:
                np.testing.assert_allclose(actual[f'{view}_{metric}'], scores[f'{view}_{metric}'], atol=1e-6)
        features[1:] = [-1, 0, 0]
        different_tiles = vis.score_with_config(features, geometry)
        np.testing.assert_array_equal(different_tiles['reference'], scores['reference'])
        finite = np.isfinite(different_tiles['tile_cosine'])
        np.testing.assert_allclose(different_tiles['tile_cosine'][finite], -.8 / np.sqrt(.68))

    def test_all_tokens_share_one_reference_and_metrics_keep_their_sign(self):
        geometry = ScoreGeometry.create((384, 384), (1, 1), 384, 14)
        geometry, features = example(geometry)
        features[1] = [1, 0, 0]
        score = vis.score_with_config(features, geometry)
        np.testing.assert_allclose(score['tile_cosine'], .8 / np.sqrt(.68))
        for view in ('base', 'tile'):
            np.testing.assert_array_equal(score[f'{view}_neg_cos'], -score[f'{view}_cosine'])
            np.testing.assert_allclose(score[f'{view}_foreground_score'],
                                       (score[f'{view}_cosine'] + 1) / 2, atol=1e-7)

    def test_normalization_handles_zero_tiny_and_nonfinite_vectors(self):
        values = np.array([[3., 4.], [0., 0.], [1e-14, 0.]])
        unit, norms, valid = vis.unit_token_features(values)
        np.testing.assert_allclose(unit, [[.6, .8], [0, 0], [0, 0]])
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_array_equal(norms, [5, 0, 1e-14])
        for invalid in (np.array([[np.nan, 0]]), np.array([[np.inf, 0]]), np.array([[1, 2]])):
            with self.assertRaises(ValueError):
                vis.unit_token_features(invalid)
        for eps in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                vis.unit_token_features(values, eps)

    def test_zero_tokens_undefined_cosine_but_defined_l2(self):
        geometry, features = example()
        features[0, 0] = 0  # Outside the reference region.
        score = vis.score_with_config(features, geometry)
        self.assertTrue(np.isnan(score['base_cosine'][0]))
        self.assertFalse(score['base_direction_valid'][0])
        self.assertAlmostEqual(float(score['base_l2'][0]), float(score['reference_norm']), places=6)
        with self.assertRaisesRegex(ValueError, 'zero/tiny norm'):
            vis.score_with_config(np.zeros_like(features), geometry)

    def test_padding_and_region_mapping_match_existing_geometry(self):
        geometry, features = example()
        score = vis.score_with_config(features, geometry)
        layout = vis.token_layout(geometry)
        self.assertFalse(layout['tile_valid'].all())
        self.assertTrue(layout['tile_valid'].any())
        self.assertTrue((layout['base_region'] >= 0).all())
        self.assertTrue((layout['base_region'] < geometry.tile_count).all())
        for tile in range(geometry.tile_count):
            left, top, right, bottom = geometry.tile_content_box(tile)
            for token in range(729):
                row, col = divmod(token, 27)
                expected = (left <= col * 14 and (col + 1) * 14 <= right
                            and top <= row * 14 and (row + 1) * 14 <= bottom)
                self.assertEqual(bool(layout['tile_valid'][tile, token]), expected)
        for metric in vis.METRICS:
            self.assertTrue(np.isnan(score[f'tile_{metric}'][~layout['tile_valid']]).all())

    def test_invalid_weights_rectangles_and_geometry(self):
        geometry, features = example()
        for kwargs in ({'inner_weight': .9}, {'inner_weight': -1, 'ring_weight': 2},
                       {'ring_weight': np.nan}, {'inner_fraction': .8, 'outer_fraction': .7},
                       {'inner_fraction': 0}):
            with self.assertRaises(ValueError):
                vis.score_center_weighted(features, geometry, **kwargs)
        with self.assertRaises(ValueError):
            vis.score_with_config(features[1:], geometry)
        with self.assertRaises(ValueError):
            vis.center_masks(ScoreGeometry.create((28, 28), (1, 1), 28, 14))

    def test_region_means_sums_and_no_data_do_not_become_zero(self):
        geometry, features = example()
        scores = stacked(features, geometry)
        for view in ('base', 'tile'):
            means, sums, counts, geometric = vis.region_statistics(scores, 0, view, 'cosine', geometry)
            np.testing.assert_allclose(means[counts > 0], sums[counts > 0] / counts[counts > 0])
            self.assertTrue(np.isnan(means[counts == 0]).all())
            self.assertTrue((counts <= geometric).all())
            self.assertAlmostEqual(float(np.nansum(sums)), float(np.nansum(scores[f'{view}_cosine'][0], dtype=np.float64)))


class CenterOutputTests(unittest.TestCase):
    def test_cli_help_without_torch_and_raw_alias(self):
        args = vis.build_parser().parse_args(['--color-scale', 'raw'])
        self.assertIsNone(args.limit)
        self.assertEqual(args.color_scale, 'raw')
        self.assertEqual(vis.build_parser().parse_args([]).color_scale, 'sample')
        result = subprocess.run([sys.executable, '-B', '-c',
                                 "import run_base_center_weighted_scores as m, sys; "
                                 "assert 'torch' not in sys.modules; m.build_parser().print_help()"],
                                cwd=Path(vis.__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--input-json', result.stdout)

    def test_color_limits_do_not_rescale_scores(self):
        base, tiles = np.array([-.8, -.1, np.nan]), np.array([[.6]])
        saved = base.copy()
        self.assertEqual(vis.metric_limits(base, tiles, 'cosine', 'sample'), (-.8, .6))
        for mode in ('fixed', 'raw'):
            self.assertEqual(vis.metric_limits(base, tiles, 'cosine', mode), (-1., 1.))
            self.assertEqual(vis.metric_limits(base, tiles, 'l2', mode), (0., 2.))
        np.testing.assert_array_equal(base, saved)
        lo, hi = vis.metric_limits(np.array([1.]), np.array([[1.]]), 'cosine', 'sample')
        self.assertLess(lo, hi)

    def test_four_images_csv_npz_and_metadata(self):
        geometry, features = example()
        image = Image.new('RGB', geometry.original_size, '#8f9977')
        scores = stacked(features, geometry)
        shapes = {stage: list(features.shape) for stage in vis.STAGES}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'sample'
            vis.save_center_sample(image, image.resize((384, 384)), scores, shapes, geometry,
                                   {'id': 'SYNTHETIC_TEST', 'image': 'synthetic.png', 'gt': 0}, folder, 'raw')
            self.assertEqual(len(list(folder.glob('*.png'))), 4)
            for stage in vis.STAGES:
                with Image.open(folder / f'{stage}_center_weighted.png') as fig:
                    self.assertEqual(fig.size, (2520, 1260))
            with np.load(folder / 'center_scores.npz', allow_pickle=False) as data:
                self.assertEqual(data['reference'].shape, (4, 3))
                np.testing.assert_array_equal(data['tile_cosine'], scores['tile_cosine'])
                np.testing.assert_array_equal(data['stages'], vis.STAGES)
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            self.assertFalse(metadata['definition']['reference_renormalized_for_l2'])
            self.assertFalse(metadata['definition']['llm_forward'])
            self.assertEqual(metadata['colorbar_limits'][vis.STAGES[0]]['cosine'], [-1, 1])
            with (folder / 'region_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4 * 2 * 3 * geometry.tile_count)
            self.assertIn('mean', rows[0])
            with self.assertRaises(FileExistsError):
                vis.save_center_sample(image, image, scores, shapes, geometry, {'id': 'x'}, folder, 'raw')

    def test_cached_results_reuse_metadata_not_npz_and_do_not_rename(self):
        geometry, _ = example()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / 'original.png'
            Image.new('RGB', geometry.original_size).save(image)
            results = root / 'source'
            folder = results / 'sample'
            folder.mkdir(parents=True)
            (folder / 'metadata.json').write_text(json.dumps({
                'id': 'old', 'image': str(image), 'gt': 1, 'geometry': asdict(geometry)}), encoding='utf-8')
            # Deliberately NOT an NPZ; a reader would fail. Must remain untouched.
            archive = folder / 'scores.npz.log'
            archive.write_bytes(b'synthetic sentinel, not cached feature vectors')
            samples, _, source_root = vis.prepare_inputs(results_dir=results)
            self.assertEqual(samples[0]['id'], 'old')
            self.assertTrue(archive.exists())
            self.assertFalse((folder / 'scores.npz').exists())
            for output in (results, folder, root):
                with self.assertRaises(ValueError):
                    vis.check_output(output, source_root)
            vis.check_output(root / 'new_output', source_root)
            with self.assertRaises(ValueError):
                vis.prepare_inputs(results_dir=results, limit=0)

    def test_run_wires_full_base_and_existing_capture_without_language_forward(self):
        # Mock model operations only to verify control flow, not real GPU inference.
        geometry, features = example()
        image = Image.new('RGB', geometry.original_size, '#8f9977')
        scores = stacked(features, geometry)
        shapes = {stage: list(features.shape) for stage in vis.STAGES}
        fake_torch = ModuleType('torch')
        fake_torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        class Tower:
            device, dtype = 'cpu', 'test-float32'
            vision_tower = SimpleNamespace(vision_model=SimpleNamespace(
                encoder=SimpleNamespace(layers=[None] * 26),
                embeddings=SimpleNamespace(patch_embedding=SimpleNamespace(kernel_size=(14, 14)))))
            def eval(self):
                return self
        backend_module = ModuleType('llava_pruning.backend')
        backend_module.LlavaBackend = Mock(return_value=SimpleNamespace(
            vision_tower=Tower(), processor=object(), model=SimpleNamespace(config=object()),
            inference_config={'model_dtype': 'unchanged'}))
        tower_module = ModuleType('llava.model.multimodal_encoder.siglip_encoder')
        tower_module.SigLipVisionTower = Tower
        pixels = SimpleNamespace(to=lambda **kwargs: object())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            image.save(root / 'source.png')
            records = root / 'questions.json'
            records.write_text(json.dumps([{'question_id': 'test', 'image': 'source.png', 'gt': 0}]), encoding='utf-8')
            modules = {'torch': fake_torch, 'llava_pruning.backend': backend_module,
                       'llava.model.multimodal_encoder.siglip_encoder': tower_module}
            with patch.dict(sys.modules, modules), \
                 patch.object(vis, 'prepare_views', return_value=(pixels, image, [], geometry)) as prepare, \
                 patch.object(vis, 'capture_stage_scores', return_value=(scores, shapes)) as capture, \
                 patch.object(vis, 'save_center_sample') as save:
                vis.run_center(str(checkpoint), root / 'output', input_json=records, data_root=root, color_scale='raw')
                self.assertEqual(prepare.call_args.kwargs['global_reference'], 'base')
                self.assertIs(capture.call_args.kwargs['score_fn'], vis.score_with_config)
                self.assertEqual(save.call_count, 1)
                save.side_effect = RuntimeError('synthetic rendering failure')
                with self.assertRaises(RuntimeError):
                    vis.run_center(str(checkpoint), root / 'failed', input_json=records, data_root=root)
            state = json.loads((root / 'output/run.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['completed_samples'], 1)
            self.assertEqual(state['layers'], [7, 14, 21, 26])
            self.assertFalse(state['llm_forward'])
            failed = json.loads((root / 'failed/run.json').read_text(encoding='utf-8'))
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('synthetic rendering failure', failed['error'])


if __name__ == '__main__':
    unittest.main()
