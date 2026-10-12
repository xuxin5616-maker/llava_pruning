"""Synthetic CPU evidence for Base-quadrant scores, geometry, sums and plots."""
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

import run_base_quadrant_scores as vis
from base_quadrant_math import METRICS, region_totals, score_quadrants, token_layout
from llava_pruning.score_visualization import ScoreGeometry


def stack_stages(features, geometry):
    stages = [score_quadrants(features, geometry) for _ in vis.LAYERS]
    return {key: np.stack([stage[key] for stage in stages]) for key in stages[0]}


def demo_data():
    """Clearly synthetic features/image; NOT model or dataset results."""
    geometry = ScoreGeometry.create((300, 300), (3, 3), 384, 14)
    y, x = np.indices((300, 300))
    pixels = np.stack([x / 300 * 200, y / 300 * 200, np.full(x.shape, 70)], axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels)
    yy, xx = np.indices((27, 27))
    features = np.ones((10, 729, 4), dtype=np.float32)
    for view in range(10):
        features[view, :, 1] = xx.ravel() / 14 + view / 10
        features[view, :, 2] = yy.ravel() / 14
        features[view, :, 3] = (np.sin(xx.ravel() / 3 + view) + 1) * 2
    scores = stack_stages(features, geometry)
    return image, geometry, scores, {stage: list(features.shape) for stage in vis.STAGES}


class QuadrantMathTests(unittest.TestCase):
    def test_native_grid_membership_and_nine_base_regions(self):
        g = ScoreGeometry.create((1000, 1000), (3, 3), 384, 14)
        layout = token_layout(g)
        np.testing.assert_array_equal(np.bincount(layout['base_quadrant']), [196, 182, 182, 169])
        np.testing.assert_array_equal(np.bincount(layout['base_region']), [81] * 9)
        self.assertTrue(layout['tile_valid'].all())
        self.assertEqual(set(layout['tile_quadrant'][4]), {0, 1, 2, 3})

    def test_anyres_uses_base_reference_not_its_own_mean(self):
        g = ScoreGeometry.create((84, 84), (3, 3), 28, 14)
        features = np.zeros((10, 4, 4), dtype=np.float32)
        features[0] = np.eye(4)
        features[1:, :, 0] = 1  # Every crop's OWN mean would yield -1 everywhere.
        actual = score_quadrants(features, g)
        q = actual['tile_quadrant']
        np.testing.assert_array_equal(actual['tile_neg_cos'], np.where(q == 0, -1, 0))
        np.testing.assert_allclose(actual['tile_l2'], np.where(q == 0, 0, np.sqrt(2)), rtol=1e-6)
        np.testing.assert_array_equal(actual['base_l2'], 0)
        np.testing.assert_array_equal(actual['base_neg_cos'], -1)
        # A crop crossing all four quadrants must switch reference PER TOKEN.
        np.testing.assert_array_equal(q[4], [0, 1, 2, 3])

    def test_inclusive_quadrant_mean_is_not_renormalized(self):
        g = ScoreGeometry.create((56, 56), (1, 1), 56, 14)
        features = np.tile([[10., 0.], [0., 2.]], (2, 8, 1)).astype(np.float32)
        actual = score_quadrants(features, g)
        np.testing.assert_allclose(actual['base_l2'], 1 / np.sqrt(2), rtol=1e-6)
        np.testing.assert_allclose(actual['base_neg_cos'], -1 / np.sqrt(2), rtol=1e-6)
        np.testing.assert_allclose(actual['base_quadrant_mean_norms'], 1 / np.sqrt(2), rtol=1e-6)
        # Independent positive feature magnitudes must not affect scores.
        scaled = score_quadrants(features * np.arange(1, 17)[None, :, None], g)
        for metric in METRICS:
            np.testing.assert_allclose(actual['tile_' + metric], scaled['tile_' + metric], rtol=1e-6)

    def test_padding_mapping_and_only_fully_valid_anyres_tokens(self):
        g = ScoreGeometry.create((84, 28), (3, 3), 28, 14)
        features = np.ones((10, 4, 2), dtype=np.float32)
        scores = stack_stages(features, g)
        np.testing.assert_array_equal(scores['tile_valid'][0].any(axis=1),
                                      [False, False, False, True, True, True, False, False, False])
        self.assertTrue(np.isnan(scores['tile_l2'][0, :3]).all())
        self.assertEqual(set(scores['base_region'][0]), {3, 5})
        sums, counts, valid = region_totals(scores, 0, 'tile', 'l2', g)
        self.assertTrue(np.isnan(sums[:3]).all())
        np.testing.assert_array_equal(counts[3:6], [4, 4, 4])
        np.testing.assert_array_equal(sums[3:6], 0)
        # Partial-content cells are also excluded (ceil resize and odd padding).
        g = ScoreGeometry.create((384, 201), (2, 1), 384, 14)
        layout = token_layout(g)
        self.assertTrue(layout['tile_valid'].any())
        self.assertFalse(layout['tile_valid'].all())
        for ti in range(g.tile_count):
            left, top, right, bottom = g.tile_content_box(ti)
            for i in range(729):
                row, col = divmod(i, 27)
                self.assertEqual(bool(layout['tile_valid'][ti, i]),
                                 left <= col * 14 and (col + 1) * 14 <= right and
                                 top <= row * 14 and (row + 1) * 14 <= bottom)

    def test_sums_are_signed_separate_conserved_and_not_display_scaled(self):
        g = ScoreGeometry.create((1000, 1000), (3, 3), 384, 14)
        scores = stack_stages(np.ones((10, 729, 2), dtype=np.float32), g)
        base, nbase, _ = region_totals(scores, 0, 'base', 'neg_cos', g)
        tiles, ntiles, _ = region_totals(scores, 0, 'tile', 'neg_cos', g)
        np.testing.assert_allclose(base, -81)
        np.testing.assert_allclose(tiles, -729)
        np.testing.assert_array_equal(nbase, 81)
        np.testing.assert_array_equal(ntiles, 729)
        self.assertEqual(float(base.sum()), float(scores['base_neg_cos'][0].sum()))
        before = base.copy()
        vis.display_scores(scores['base_neg_cos'][0], scores['tile_neg_cos'][0], 'neg_cos', 'minmax')
        np.testing.assert_array_equal(region_totals(scores, 0, 'base', 'neg_cos', g)[0], before)

    def test_zero_norm_and_invalid_input(self):
        g = ScoreGeometry.create((28, 28), (1, 1), 28, 14)
        zeros = np.zeros((2, 4, 3), dtype=np.float32)
        scores = stack_stages(zeros, g)
        self.assertTrue(np.isnan(scores['base_neg_cos']).all())
        np.testing.assert_array_equal(scores['base_l2'], 0)
        sums, counts, valid = region_totals(scores, 0, 'base', 'neg_cos', g)
        self.assertTrue(np.isnan(sums).all())
        np.testing.assert_array_equal(counts, 0)
        np.testing.assert_array_equal(valid, 4)
        for bad in (zeros[0], zeros[:1], np.full_like(zeros, np.nan)):
            with self.assertRaises(ValueError):
                score_quadrants(bad, g)


class QuadrantOutputTests(unittest.TestCase):
    def test_cli_and_no_model_import_for_help(self):
        args = vis.build_parser().parse_args([])
        self.assertIsNone(args.limit)
        self.assertEqual(args.color_scale, 'raw')
        result = subprocess.run([sys.executable, '-B', '-c',
                                 "import run_base_quadrant_scores, sys; assert 'torch' not in sys.modules; "
                                 "print(run_base_quadrant_scores.SCRIPT_VERSION)"],
                                cwd=Path(vis.__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_color_scales_and_missing_values(self):
        b, t = np.array([-.8, -.2, np.nan]), np.array([[-1., 0.]])
        raw, _, limits, _ = vis.display_scores(b, t, 'neg_cos', 'raw')
        np.testing.assert_array_equal(raw, b)
        self.assertEqual(limits, [-1., 1.])
        scaled, _, limits, _ = vis.display_scores(b, t, 'neg_cos', 'minmax')
        np.testing.assert_allclose(scaled[:2], [.2, .8])
        self.assertTrue(np.isnan(scaled[-1]))
        with self.assertRaises(ValueError):
            vis.display_scores(b, t, 'cosine', 'raw')

    def test_four_figures_configs_csv_and_original_arrays(self):
        image, geometry, scores, shapes = demo_data()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'sample'
            vis.save_sample(image, image.resize((384, 384)), scores, shapes, geometry,
                            {'id': 'SYNTHETIC_TEST', 'image': 'synthetic.png', 'gt': 0},
                            folder, 'minmax', {'model_path': 'synthetic-only'})
            self.assertEqual(len(list(folder.glob('*.png'))), 4)
            with Image.open(folder / 'siglip_7_quadrant_scores.png') as figure:
                self.assertEqual(figure.size, (2250, 1575))
                self.assertAlmostEqual(figure.info['dpi'][0], 150, places=1)
            cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(cfg['base_quadrant_token_counts'], [196, 182, 182, 169])
            self.assertEqual(cfg['geometry'], json.loads(json.dumps(asdict(geometry))))
            self.assertFalse(cfg['definition']['mean_renormalized'])
            self.assertEqual(cfg['definition']['reference_used_by'], ['Base', 'AnyRes'])
            with np.load(folder / 'quadrant_scores.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['base_neg_cos'], scores['base_neg_cos'])
                np.testing.assert_array_equal(saved['tile_l2'], scores['tile_l2'])
            with (folder / 'region_sums.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4 * 2 * 2 * 9)
            self.assertEqual(rows[0]['view'], 'base')
            expected = region_totals(scores, 0, 'base', 'neg_cos', geometry)[0][0]
            self.assertAlmostEqual(float(rows[0]['raw_sum']), expected)
            with self.assertRaises(FileExistsError):
                vis.save_sample(image, image, scores, shapes, geometry,
                                {'id': 'same', 'image': 'synthetic.png'}, folder, 'raw', {})

    def test_run_wires_full_base_and_scorer_and_writes_root_config(self):
        # Control-flow test only. All model/GPU operations are explicitly mocked.
        image, geometry, scores, shapes = demo_data()
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
            path = root / 'source.png'
            image.save(path)
            records = root / 'questions.json'
            records.write_text(json.dumps([{'question_id': 'test', 'image': 'source.png', 'gt': 0}]), encoding='utf-8')
            with patch.dict(sys.modules, {'torch': fake_torch, 'llava_pruning.backend': backend_module,
                                         'llava.model.multimodal_encoder.siglip_encoder': tower_module}), \
                 patch.object(vis, 'prepare_views', return_value=(pixels, image, [], geometry)) as prepare, \
                 patch.object(vis, 'capture_stage_scores', return_value=(scores, shapes)) as capture, \
                 patch.object(vis, 'save_sample') as save:
                vis.run(str(checkpoint), root / 'output', input_json=records, data_root=root)
                self.assertEqual(prepare.call_args.kwargs['global_reference'], 'base')
                self.assertIs(capture.call_args.kwargs['score_fn'], score_quadrants)
                self.assertEqual(save.call_count, 1)
            cfg = json.loads((root / 'output/config.json').read_text(encoding='utf-8'))
            state = json.loads((root / 'output/run.json').read_text(encoding='utf-8'))
            self.assertEqual(cfg['layers'], [7, 14, 21, 26])
            self.assertIsNone(cfg['limit'])
            self.assertFalse(cfg['llm_forward'])
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['completed_samples'], 1)


if __name__ == '__main__':
    unittest.main()
