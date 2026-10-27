"""Standalone probability-mixture integration: CPU only, no model downloads."""

import contextlib
import csv
import io
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import runMyPruningPoolSpatial as pool


class PoolSpatialMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(81)
        cls.features = torch.randn(10, 729, 8, generator=generator)
        cls.detail = pool.pooled_crop_redundancy(cls.features, (3, 3))
        cls.raw = cls.detail["redundancy"].double().numpy()
        cls.bins = cls.detail["base_token_indices"]
        cls.prior = pool.center_spatial_prior(729, cls.bins, 0)["spatial_prior"]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def allocation(self, rate=70, weight=0.02, raw=None, prior=None):
        return pool.allocate_spatial_probability_budgets(
            self.raw if raw is None else raw, self.bins, (3, 3),
            self.prior if prior is None else prior, prune_rate=rate,
            mix_weight=weight, temperature=0.05, minimum_keep=4,
        )

    def test_defaults_aliases_and_layer_validation(self):
        defaults = pool.build_parser().parse_args([])
        settings = pool.settings_from_args(defaults)
        self.assertEqual(settings.redundancy_layers, (26,))
        self.assertEqual(settings.visionzip_layers, (26,))
        self.assertEqual(settings.spatial_strength, 0)
        self.assertEqual(settings.prune_rate, 70)
        self.assertFalse(defaults.save_prune_vis)
        for alias in ("--spatial-mix-weight", "--spatial-strength"):
            args = pool.build_parser().parse_args([
                alias, "0.02", "--redundancy-layers", "7", "14", "21", "26",
                "--visionzip-layers", "21", "--save-prune-vis", "False",
            ])
            settings = pool.settings_from_args(args)
            self.assertEqual(settings.spatial_strength, 0.02)
            self.assertEqual(settings.redundancy_layers, (7, 14, 21, 26))
            self.assertEqual(settings.visionzip_layers, (21,))
        for weight in (-0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                pool.MethodSettings(spatial_strength=weight).validate()

    def test_redundancy_is_unit_token_pool_cosine_without_input_mutation(self):
        original = self.features.clone()
        unit = torch.nn.functional.normalize(self.features, dim=-1)
        base_means = torch.stack([unit[0, indices].mean(0) for indices in self.bins])
        expected = base_means.norm(dim=-1) + unit[1:].mean(1).norm(dim=-1)
        torch.testing.assert_close(self.detail["redundancy"], expected)
        scaled = pool.pooled_crop_redundancy(self.features * 7, (3, 3))
        torch.testing.assert_close(scaled["redundancy"], self.detail["redundancy"])
        torch.testing.assert_close(self.features, original, rtol=0, atol=0)

    def test_disabled_mixture_is_exact_legacy_allocator(self):
        base_map = pool.redundancy_to_base_map(self.raw, self.bins, 729)
        legacy = pool.allocate_crop_budgets(
            base_map, (3, 3), prune_rate=70, temperature=0.05, minimum_keep=4)
        actual, probability = self.allocation(weight=0)
        self.assertEqual(actual, legacy)
        self.assertEqual(probability["mode"], "legacy_disabled")

    def test_probability_fusion_has_no_second_softmax(self):
        allocation, probability = self.allocation(weight=0.2)
        shifted = -self.raw / 0.05
        expected_r = np.exp(shifted - shifted.max())
        expected_r /= expected_r.sum()
        expected_s = self.prior / self.prior.sum()
        expected = 0.8 * expected_r + 0.2 * expected_s
        np.testing.assert_allclose(probability["redundancy_probabilities"], expected_r)
        np.testing.assert_allclose(probability["mixed_probabilities"], expected)
        np.testing.assert_allclose(allocation["allocation_probabilities"], expected)
        self.assertEqual(allocation["allocation_rule"], "bounded_probability_mixture")

    def test_aggregate_budget_and_base_are_separate_at_all_rates(self):
        for rate in (0, 1, 10, 30, 50, 70, 90, 99):
            for weight in (0, 0.02, 0.5, 1):
                with self.subTest(rate=rate, weight=weight):
                    allocation, _ = self.allocation(rate, weight)
                    keeps = [crop["keep"] for crop in allocation["crops"]]
                    expected_anyres = (9 * 729 * (100 - rate) + 99) // 100
                    self.assertEqual(sum(keeps), expected_anyres)
                    self.assertTrue(all(4 <= keep <= 729 for keep in keeps))
                    base_keep = sum(pool.token_budget(729, rate / 100))
                    self.assertEqual(base_keep, (729 * (100 - rate) + 99) // 100)
        allocation, _ = self.allocation(70)
        self.assertEqual(sum(crop["keep"] for crop in allocation["crops"]), 1969)
        self.assertEqual(sum(pool.token_budget(729, 0.7)), 219)

    def test_capacity_redistribution_preserves_tiny_probability_mass(self):
        for weight in (0, 0.02, 1):
            allocation, _ = self.allocation(
                rate=10, weight=weight, raw=np.arange(9, dtype=float) * 1000,
                prior=np.array([1., 0, 0, 0, 0, 0, 0, 0, 0]))
            keeps = [crop["keep"] for crop in allocation["crops"]]
            self.assertEqual(sum(keeps), math.ceil(6561 * 0.9))
            self.assertTrue(all(4 <= keep <= 729 for keep in keeps))

    def test_empty_spatial_prior_falls_back_to_redundancy(self):
        disabled, _ = self.allocation(weight=0)
        actual, probability = self.allocation(weight=0.9, prior=np.zeros(9))
        self.assertEqual(actual, disabled)
        self.assertTrue(probability["spatial_empty"])
        self.assertEqual(probability["effective_mix_weight"], 0)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Identity()
        self.num_heads, self.head_dim = 2, 4

    def forward(self, hidden_states):
        self.k_proj(hidden_states)
        tokens = hidden_states.shape[1]
        received = torch.linspace(-1, 1, tokens).softmax(-1)
        weights = received.view(1, 1, 1, tokens).expand(
            hidden_states.shape[0], self.num_heads, tokens, tokens)
        return hidden_states, weights


class _Block(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.self_attn = _Attention()
        self.index = index

    def forward(self, hidden_states):
        hidden, _ = self.self_attn(hidden_states)
        return (hidden + self.index * 0.0001,)


class _Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(123)
        self.features = torch.randn(10, 729, 8, generator=generator)
        self.blocks = torch.nn.ModuleList([_Block(index) for index in range(1, 27)])
        self.vision_tower = types.SimpleNamespace(vision_model=types.SimpleNamespace(
            encoder=types.SimpleNamespace(layers=self.blocks)))
        self.num_patches = 729
        self.num_patches_per_side = 27
        self.image_size = 384
        self.config = types.SimpleNamespace(patch_size=14)
        self.last_output = None
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        hidden = self.features
        for block in self.blocks:
            hidden = block(hidden)[0]
        self.last_output = hidden
        return hidden


class _Projector(torch.nn.Module):
    def forward(self, hidden):
        self.last_input = hidden
        return hidden


class _Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = _Tower()
        self.mm_projector = _Projector()
        self.image_newline = torch.full((8,), -100.)
        self.config = types.SimpleNamespace(
            image_aspect_ratio="anyres_max_9", image_token_order="base_first",
            mm_patch_merge_type="spatial_unpad", image_grid_pinpoints=[[1152, 1152]])
        self.eval()

    def get_vision_tower(self):
        return self.tower

    def configure_fastv(self, **kwargs):
        assert kwargs == {"enabled": False}


class PoolSpatialIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_adapter_collects_layers_and_compresses_before_projector(self):
        for layers in ((26,), (21,), (7, 14, 21, 26)):
            with self.subTest(layers=layers):
                core = _Core()
                adapter = pool.CustomPruningAdapter(core, pool.MethodSettings(
                    prune_rate=70, redundancy_layers=layers, visionzip_layers=layers,
                    spatial_strength=0.02))
                adapter.configure(70, capture_visualization=False)
                adapter.set_image_context((1024, 1024))
                packed = adapter.encode_images(torch.zeros(10, 3, 1, 1))
                self.assertEqual(core.tower.calls, 1)
                self.assertEqual(adapter.stats["base_tokens_after"], 219)
                self.assertEqual(adapter.stats["anyres_tokens_after"], 1969)
                self.assertEqual(core.mm_projector.last_input.shape, (2188, 8))
                self.assertEqual(packed.shape, (2188 + 81, 8))
                self.assertEqual(int((packed == core.image_newline).all(-1).sum()), 81)
                self.assertFalse(adapter.spatial_details)
                for block in core.tower.blocks:
                    self.assertFalse(block._forward_hooks)
                    self.assertFalse(block.self_attn._forward_hooks)
                    self.assertFalse(block.self_attn.k_proj._forward_hooks)

    def test_zero_executes_scoring_and_preserves_original_projector_tensor(self):
        core = _Core()
        adapter = pool.CustomPruningAdapter(core, pool.MethodSettings(prune_rate=0))
        adapter.configure(0, capture_visualization=True)
        adapter.set_image_context((1024, 1024))
        packed = adapter.encode_images(torch.zeros(10, 3, 1, 1))
        self.assertTrue(adapter.stats["full_grid_identity"])
        self.assertEqual(adapter.stats["base_tokens_after"], 729)
        self.assertEqual(adapter.stats["anyres_tokens_after"], 6561)
        self.assertIs(core.mm_projector.last_input, core.tower.last_output)
        self.assertEqual(core.mm_projector.last_input.shape, (10, 729, 8))
        self.assertEqual(adapter.visionzip_details["layers"].tolist(), [26])
        expected, _ = pool.pack_full_views(
            core.tower.last_output,
            pool.TokenDecisions(729, 729, torch.arange(729).expand(10, -1)),
            (3, 3), (1024, 1024), core.image_newline, "spatial_unpad")
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)

    def test_runner_writes_metrics_budgets_and_audit_without_images(self):
        class Backend:
            def __init__(self, model_path, *, roi_mode, image_token_order):
                assert roi_mode == "anyres_max_9" and image_token_order == "base_first"
                self.core = _Core()
                self.model = types.SimpleNamespace(get_model=lambda: self.core)
                self.vision_tower = self.core.tower

            def generate(self, sample, prompt, method, rate, roi_mode, **kwargs):
                assert not kwargs["do_sample"] and not kwargs["capture_visualization"]
                method.configure(self.core, rate, capture_visualization=False)
                method.adapter.set_image_context((1024, 1024))
                method.adapter.encode_images(torch.zeros(10, 3, 1, 1))
                return dict(answer="A" if sample.gt else "B", generation_seconds=0.1,
                            stats=method.stats(self.core, rate), max_new_tokens=512,
                            allocation=method.allocation_data(self.core))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (16, 16)).save(root / "image.png")
            source = root / "questions.jsonl"
            source.write_text("".join(json.dumps(dict(
                question_id=str(index), image="image.png", gt=index,
                origin_path="screw/test/good/example.png")) + "\n" for index in (0, 1)), encoding="utf-8")
            args = pool.build_parser().parse_args([
                "--input-json", str(source), "--data-root", str(root),
                "--output-dir", str(root / "output"), "--spatial-mix-weight", "0.02",
                "--save-prune-vis", "False"])
            with contextlib.redirect_stdout(io.StringIO()):
                output = pool.run_experiment(args, backend_factory=Backend)
            rate_dir = output / "prune_70"
            metrics = json.loads((rate_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["complete"])
            for metric in ("accuracy", "precision", "recall", "tnr"):
                self.assertEqual(metrics[metric], 1)
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["spatial_fusion_rule"], "probability_convex_mix")
            with (rate_dir / "crop_budgets.csv").open(encoding="utf-8-sig", newline="") as stream:
                crops = list(csv.DictReader(stream))
            self.assertEqual(len(crops), 18)
            self.assertIn("fused_probability", crops[0])
            self.assertTrue((rate_dir / "geometry_audit.csv").is_file())
            self.assertFalse(list(output.rglob("*.png")))
            self.assertFalse(list(output.rglob("*.npz")))


if __name__ == "__main__":
    unittest.main()
