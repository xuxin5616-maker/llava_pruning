"""Offline Top-K and native-token rendering checks on synthetic caches."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import visualize_token_topk as vis
from test_patch_mean_visualization import fixture


class TopKMathTests(unittest.TestCase):
    def test_highest_signed_ten_not_absolute_or_percent_or_full_mean(self):
        values = np.linspace(-1, 1, 100).reshape(1, 1, 100)
        means, counts, selected = vis.topk_means(values, 10)
        self.assertAlmostEqual(means[0, 0], values[0, 0, -10:].mean())
        np.testing.assert_array_equal(counts, [[100]])
        np.testing.assert_array_equal(selected, [[10]])
        raw = np.array([[[-.9, -.8, -.2, -.1]]])
        self.assertAlmostEqual(vis.topk_means(raw, 2)[0][0, 0], -.15)

    def test_per_tile_independence_missing_and_fewer_than_k(self):
        values = np.array([[[.1, .3, .5, np.nan], [-1, -.8, np.nan, np.nan], [np.nan] * 4]])
        means, counts, selected = vis.topk_means(values, 10)
        np.testing.assert_allclose(means[0, :2], [.3, -.9])
        self.assertTrue(np.isnan(means[0, 2]))
        np.testing.assert_array_equal(counts, [[3, 2, 0]])
        np.testing.assert_array_equal(selected, counts)

    def test_average_tokens_across_layers_before_topk_not_after(self):
        values = np.zeros((4, 1, 2))
        values[0, 0] = [1, -1]
        values[1, 0] = [-1, 1]
        panels = vis.build_token_panels(values)
        self.assertEqual(panels.shape, (7, 1, 2))
        np.testing.assert_array_equal(panels[4], [[0, 0]])
        means = vis.topk_means(panels, 1)[0]
        np.testing.assert_array_equal(means[:2], [[1], [1]])
        self.assertEqual(means[4, 0], 0)  # Mean(Top1 each layer) would be 1, not 0.

    def test_missing_layer_at_one_position_does_not_use_remaining_layers(self):
        values = np.ones((4, 1, 3))
        values[1, 0, 0] = np.nan
        panels = vis.build_token_panels(values)
        self.assertTrue(np.isnan(panels[4:, 0, 0]).all())
        self.assertTrue(np.isfinite(panels[0, 0, 0]))
        self.assertEqual(vis.topk_means(panels)[1][4, 0], 2)

    def test_no_inplace_score_changes_and_fixed_scaling(self):
        values = np.linspace(-1, 1, 4 * 2 * 15).reshape(4, 2, 15)
        before = values.copy()
        panels = vis.build_token_panels(values)
        original_panels = panels.copy()
        means, _, _ = vis.topk_means(panels)
        np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(panels, original_panels)
        np.testing.assert_allclose(vis.scale_scores_01(means), (means + 1) / 2)

    def test_invalid_values_and_k(self):
        values = np.zeros((4, 1, 4))
        for k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                vis.topk_means(values, k)
        for bad in (np.full_like(values, np.inf), np.full_like(values, 1.1), values[0]):
            with self.assertRaises(ValueError):
                vis.build_token_panels(bad)


class TopKIOTests(unittest.TestCase):
    def test_cli_import_does_not_load_torch(self):
        args = vis.build_parser().parse_args([])
        self.assertEqual(args.top_k, 10)
        result = subprocess.run([sys.executable, '-B', '-c',
                                 "import visualize_token_topk as v, sys; assert 'torch' not in sys.modules; v.build_parser().print_help()"],
                                cwd=Path(vis.__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--source-results-dir', result.stdout)

    def test_run_native_token_heatmap_and_no_cache_mutation_or_rename(self):
        from matplotlib.figure import Figure
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results, folder, _, _ = fixture(root, suffix='scores.npz.log')
            source = folder / 'scores.npz.log'
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            rendered = []
            original_save = Figure.savefig
            def inspect_save(fig, *args, **kwargs):
                heat_axes = [ax for ax in fig.axes if len(ax.images) == 2]
                self.assertEqual(len(heat_axes), 7)
                # A 2x1 crop grid with 2x2 tokens per crop: fine-grained RGBA map,
                # NOT one uniform color rectangle for each crop.
                heat = np.asarray(heat_axes[0].images[1].get_array())
                self.assertEqual(heat.shape, (28, 56, 4))
                self.assertFalse(np.array_equal(heat[2, 2], heat[2, 16]))
                self.assertEqual(heat_axes[0].images[1].get_interpolation(), 'nearest')
                self.assertAlmostEqual(float(heat_axes[0].get_aspect()), .5)
                rendered.append(True)
                return original_save(fig, *args, **kwargs)
            with patch.object(Figure, 'savefig', inspect_save):
                output = vis.run(results, root / 'redraw', top_k=2)
            self.assertEqual(rendered, [True])
            target = output / folder.name
            self.assertEqual({p.name for p in target.iterdir()}, {'token_topk.png', 'token_topk.csv', 'metadata.json'})
            with Image.open(target / 'token_topk.png') as image:
                self.assertEqual(image.size, (2400, 1350))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            self.assertFalse((folder / 'scores.npz').exists())
            state = vis.read_json(output / 'run.json')
            self.assertTrue(state['complete'])
            self.assertFalse(state['model_loaded'])
            with self.assertRaises(FileExistsError):
                vis.run(results, output)

    def test_derived_mean_folder_resolves_original_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results, folder, meta, _ = fixture(root)
            derived = root / 'means'
            child = derived / folder.name
            child.mkdir(parents=True)
            vis.write_json(derived / 'run.json', {'source_results': str(results), 'script_version': '4.2-global-horizontal'})
            vis.write_json(child / 'metadata.json', {**meta, 'source_scores': str(folder / 'scores.npz')})
            _, samples, protected = vis.prepare_samples(derived)
            self.assertEqual(samples[0]['archive'], (folder / 'scores.npz').resolve())
            self.assertIn(results.resolve(), protected)
            for output in (derived, child, results, results / 'overwrite', root):
                with self.assertRaises(ValueError):
                    vis.check_output(output, protected)

    def test_relocation_override_and_missing_cache_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results, folder, meta, _ = fixture(root)
            derived = root / 'means'
            child = derived / folder.name
            child.mkdir(parents=True)
            vis.write_json(derived / 'run.json', {'source_results': str(root / 'missing')})
            vis.write_json(child / 'metadata.json', {**meta, 'source_scores': str(root / 'missing/scores.npz')})
            with self.assertRaisesRegex(FileNotFoundError, 'cannot recover Top-K'):
                vis.prepare_samples(derived)
            _, samples, _ = vis.prepare_samples(derived, source_results_dir=results)
            self.assertEqual(samples[0]['archive'], (folder / 'scores.npz').resolve())
            derived_meta = vis.read_json(child / 'metadata.json')
            derived_meta['id'] = 'wrong'
            vis.write_json(child / 'metadata.json', derived_meta)
            with self.assertRaisesRegex(ValueError, 'IDs disagree'):
                vis.prepare_samples(derived, source_results_dir=results)

    def test_shared_stitching_preserves_unencoded_strips_and_unpads(self):
        g = vis.ScoreGeometry.create((768, 384), (2, 1), 384, 14)
        values = np.arange(2 * 729).reshape(2, 729).astype(float)
        result = vis.stitch_score_map(values, g)
        self.assertEqual(result.shape, (384, 768))
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 14], 1)
        self.assertEqual(result[0, 384], 729)
        self.assertTrue(np.isnan(result[:, 378:384]).all())
        self.assertTrue(np.isnan(result[378:, :]).all())


if __name__ == '__main__':
    unittest.main()
