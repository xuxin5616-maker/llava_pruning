"""Unit-token residual regression tests; no checkpoint or GPU needed."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import runVFlowOpt as raw
import runVFlowOpt_unit as unit
from feature_residual_math import stage_residuals


PROJECT = Path(__file__).resolve().parents[1]


class UnitResidualTests(unittest.TestCase):
    def test_unit_vectors_zero_epsilon_large_values_and_input_unchanged(self):
        values = np.array([[[3., 4.], [0., 0.], [1e-14, 0.], [1e30, -1e30]]],
                          dtype=np.float32)
        before = values.copy()
        result = unit.unit_token_features(values)
        expected = [[.6, .8], [0., 0.], [.01, 0.], [1 / np.sqrt(2), -1 / np.sqrt(2)]]
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)
        np.testing.assert_array_equal(values, before)
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(np.shares_memory(values, result))

    def test_rejects_invalid_token_features(self):
        for values in (np.zeros((0, 2, 3)), np.ones((2, 3)),
                       np.ones((1, 2, 3), dtype=np.int32),
                       np.ones((1, 2, 3), dtype=np.complex64),
                       np.full((1, 2, 3), np.nan), np.full((1, 2, 3), np.inf)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                unit.unit_token_features(values)

    def test_center_excluded_and_neighbor_mean_not_renormalized(self):
        geometry = unit.ScoreGeometry.create((28, 28), (1, 1), 28, 14)
        features = np.tile(np.array([[10., 0.], [1., 0.], [20., 0.], [0., 7.]]),
                           (2, 1, 1)).astype(np.float32)
        result = unit.score_unit_features(features, geometry)
        # Center [1,0], mean of its three neighbors [2/3,1/3].
        np.testing.assert_allclose(result['base_l2'][:, 0], np.sqrt(2) / 3, rtol=1e-6)
        np.testing.assert_allclose(result['tile_l2'][:, 0, 0], np.sqrt(2) / 3, rtol=1e-6)
        self.assertFalse(np.allclose(result['base_l2'], stage_residuals(features, geometry)['base_l2']))

    def test_per_token_scale_invariance_and_raw_scorer_preserved(self):
        geometry = unit.ScoreGeometry.create((56, 28), (2, 1), 28, 14)
        rng = np.random.default_rng(42)
        values = rng.normal(size=(3, 4, 5)).astype(np.float32)
        result = unit.score_unit_features(values, geometry)
        scaled = unit.score_unit_features(values * rng.uniform(.2, 10, size=(3, 4, 1)), geometry)
        expected = stage_residuals(unit.unit_token_features(values), geometry)
        for key in result:
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-6, atol=2e-7)
            np.testing.assert_allclose(result[key], scaled[key], rtol=1e-6, atol=2e-7)
        original = stage_residuals(values, geometry)
        raw_scaled = stage_residuals(values * 2, geometry)
        np.testing.assert_allclose(raw_scaled['base_l2'], original['base_l2'] * 2, rtol=1e-6)
        self.assertEqual(raw.SCRIPT_VERSION, '3.1-integrated-feature-residual')
        self.assertFalse(raw.RESIDUAL_DEFINITION.get('token_normalization', False))

    def test_display_raw_does_not_undo_token_normalization(self):
        base = np.array([[.2, .8, np.nan]])
        tiles = np.array([[[.1, 1.1, np.nan]]])
        b, t, limits, _ = unit.display_scores(base, tiles, 'raw')
        np.testing.assert_array_equal(b, base)
        np.testing.assert_array_equal(t, tiles)
        self.assertEqual(limits, (0, 1.1))
        b, t, limits, _ = unit.display_scores(base, tiles, 'minmax')
        np.testing.assert_allclose(b[0, :2], [.1, .7])
        np.testing.assert_allclose(t[0, 0, :2], [0, 1])
        self.assertEqual(limits, (0, 1))

    def test_cli_and_default_output_do_not_load_torch(self):
        args = unit.build_parser().parse_args([])
        self.assertEqual((args.metric, args.color_scale, args.limit), ('l2', 'raw', None))
        for args in (['--metric', 'entropy'], ['--color-scale', 'softmax']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                unit.build_parser().parse_args(args)
        with patch.object(sys, 'argv', ['runVFlowOpt_unit.py', '--results-dir', 'outputs/source']), \
             patch.object(unit, 'run') as run:
            unit.main()
        self.assertEqual(run.call_args.kwargs['output_dir'].name, 'source_feature_residual_unit')
        code = "import runVFlowOpt_unit, sys; assert 'torch' not in sys.modules; print(runVFlowOpt_unit.SCRIPT_VERSION)"
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('3.2-unit-token-feature-residual', result.stdout)

    def test_save_four_figures_and_unit_metadata_with_raw_arrays(self):
        geometry = unit.ScoreGeometry.create((28, 28), (1, 1), 28, 14)
        values = np.random.default_rng(7).normal(size=(2, 4, 3)).astype(np.float32)
        stages = [unit.score_unit_features(values * (i + 1), geometry) for i in range(4)]
        scores = {key: np.stack([stage[key] for stage in stages]) for key in stages[0]}
        shapes = {stage: list(values.shape) for stage in unit.STAGES}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'sample'
            image = Image.new('RGB', (28, 28), 'gray')
            unit.save_sample(image, image, scores, shapes, geometry,
                             {'id': 'synthetic', 'image': 'synthetic.png', 'gt': 0},
                             folder, 'l2', 'minmax')
            self.assertEqual(len(list(folder.glob('*.png'))), 4)
            with np.load(folder / 'residuals.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['base_l2'], scores['base_l2'])
                np.testing.assert_array_equal(saved['tile_cosine'], scores['tile_cosine'])
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            self.assertFalse(metadata['definition']['token_normalization']['neighbor_mean_renormalized'])
            self.assertFalse(metadata['definition']['token_normalization']['model_forward_modified'])
            self.assertEqual(metadata['script_version'], unit.SCRIPT_VERSION)
            self.assertEqual(metadata['colormap'], 'jet')
            self.assertEqual(len(list(folder.glob('crop_summary.csv'))), 1)
            with self.assertRaises(FileExistsError):
                unit.save_sample(image, image, scores, shapes, geometry,
                                 {'id': 'synthetic', 'image': 'synthetic.png'}, folder, 'l2', 'raw')


if __name__ == '__main__':
    unittest.main()
