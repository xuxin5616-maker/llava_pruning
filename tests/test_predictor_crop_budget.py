"""Base allocates crop quotas; own AnyRes prediction chooses retained tokens."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_attention_predictor import make_checkpoint
from test_clse import layout, toy_model
from llava.model.language_model.attention_predictor_crops import (
    dense_crop_provenance, validate_crop_provenance, bounded_crop_quotas,
    select_base_crop_tokens)
from llava_pruning.methods import load_method


def geometry(base=2, rows=2, cols=4, raw_grid=(2, 1), raw_side=2,
             unpadded=(2, 4), order="base_first", extra=False):
    result = layout(base=base, rows=rows, cols=cols, order=order, extra=extra)
    result["predictor_crops"] = dense_crop_provenance(result, raw_grid, raw_side, unpadded)
    return result


def spatial(geo):
    return [i for grid in geo["grids"] for i in grid["indices"]]


class PredictorCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_square_raw_grid_and_actual_packing_orders_have_same_crop_identities(self):
        for order in ("base_first", "anyres_first"):
            for extra in (False, True):
                geo = geometry(base=27, rows=81, cols=81, raw_grid=(3, 3),
                               raw_side=27, unpadded=(81, 81), order=order, extra=extra)
                crops = validate_crop_provenance(geo, geo["predictor_crops"])
                self.assertEqual([len(c["indices"]) for c in crops], [729] * 9)
                self.assertEqual([len(c["base_indices"]) for c in crops], [81] * 9)
                self.assertFalse(geo["predictor_crops"]["pre_pruning_applied"])

    def test_unpad_and_max9_mapping_retain_original_tile_ids(self):
        geo = geometry(base=27, rows=48, cols=81, raw_grid=(3, 2),
                       raw_side=27, unpadded=(48, 81))
        metadata = geo["predictor_crops"]
        self.assertEqual(metadata["unpad_offset"], [3, 0])
        self.assertEqual([len(c["indices"]) for c in metadata["crops"]], [648] * 6)
        scaled = geometry(base=27, rows=81, cols=81, raw_grid=(6, 6),
                          raw_side=27, unpadded=(162, 162))
        crops = validate_crop_provenance(scaled, scaled["predictor_crops"])
        self.assertEqual(len(crops), 36)
        self.assertEqual(sum(len(c["indices"]) for c in crops), 81 * 81)
        self.assertTrue(all(len(c["indices"]) < 729 for c in crops))

    def test_softmax_positive_means_matches_oracle_without_minmax_or_second_softmax(self):
        means = torch.tensor([.1, .2, .3], dtype=torch.float64)
        keeps, weights = bounded_crop_quotas(means, [100] * 3, 90, temperature=.1)
        expected = torch.softmax(means / .1, 0)
        torch.testing.assert_close(torch.tensor(weights, dtype=torch.float64), expected)
        quota = expected * 90
        oracle = quota.floor().long()
        order = sorted(range(3), key=lambda i: (-(quota[i] - oracle[i]).item(), i))
        for i in order[:90 - int(oracle.sum())]:
            oracle[i] += 1
        self.assertEqual(keeps, oracle.tolist())
        self.assertGreater(keeps[2], keeps[0])

    def test_capacity_redistribution_accepts_unequal_survivors_and_empty_crops(self):
        for temperature in (0.05, 1e-10):
            keeps, weights = bounded_crop_quotas(
                torch.tensor([1000., 0., 10000.]), [20, 100, 0], 60, temperature=temperature)
            self.assertEqual(keeps, [20, 40, 0])
            self.assertEqual(weights[2], 0)
        for total in range(121):
            keeps, _ = bounded_crop_quotas(torch.tensor([.8, .2]), [20, 100], total)
            self.assertEqual(sum(keeps), total)
            self.assertTrue(0 <= keeps[0] <= 20 and 0 <= keeps[1] <= 100)

    def test_minimum_and_rounding_are_exact_and_infeasible_settings_fail(self):
        keeps, _ = bounded_crop_quotas(torch.ones(3), [10, 10, 10], 8, minimum_keep=1)
        self.assertEqual(keeps, [3, 3, 2])
        with self.assertRaisesRegex(ValueError, "crop_min_keep"):
            bounded_crop_quotas(torch.ones(3), [10] * 3, 2, minimum_keep=1)
        for temperature in (0, -1, True, float("nan")):
            with self.assertRaisesRegex(ValueError, "crop_temperature"):
                bounded_crop_quotas(torch.ones(2), [10, 10], 3, temperature=temperature)

    def test_base_means_set_quotas_but_own_anyres_attention_selects_tokens(self):
        geo = geometry()
        own = torch.tensor([.35, .05, .35, .05, .01, .02, .03, .04, .04, .03, .02, .01])
        chosen, plan = select_base_crop_tokens(own.log(), spatial(geo), 14, .5, geo, temperature=.1)
        self.assertEqual([g["kept_tokens"] for g in plan["groups"]], [2, 3])
        self.assertEqual([c["kept_tokens"] for c in plan["crops"]], [3, 0])
        self.assertEqual(chosen.tolist(), [0, 2, 5, 8, 9, 10, 13])
        # Permuting OWN scores inside a crop changes selection, not its budget.
        changed = own.clone()
        changed[[4, 5, 8, 9]] = changed[[8, 9, 4, 5]]
        other, other_plan = select_base_crop_tokens(changed.log(), spatial(geo), 14, .5, geo, temperature=.1)
        self.assertEqual([c["kept_tokens"] for c in other_plan["crops"]], [3, 0])
        self.assertFalse(torch.equal(chosen, other))
        # Changing the Base spatial distribution transfers crop budgets.
        changed = own.clone()
        changed[:4] = changed[[1, 0, 3, 2]]
        _, other_plan = select_base_crop_tokens(changed.log(), spatial(geo), 14, .5, geo, temperature=.1)
        self.assertEqual([c["kept_tokens"] for c in other_plan["crops"]], [0, 3])

    def test_layer_average_50_70_90_totals_still_match_unpruned_baseline(self):
        geo = geometry(base=27, rows=81, cols=81, raw_grid=(3, 3), raw_side=27, unpadded=(81, 81))
        scores = torch.linspace(-20, -2, len(spatial(geo)))
        for rate in (50, 70, 90):
            ratio = 1 - rate * 28 / 26 / 100
            selected, plan = select_base_crop_tokens(scores, spatial(geo), geo["packed_tokens"], ratio, geo)
            self.assertEqual(len(selected), math.floor(geo["packed_tokens"] * ratio + 1e-10))
            self.assertEqual(sum(c["kept_tokens"] for c in plan["crops"]), plan["groups"][1]["kept_tokens"])
            self.assertTrue(set(plan["protected_indices"]).issubset(set(selected.tolist())))
            self.assertEqual(len(plan["protected_indices"]), 81)
            for group in plan["groups"]:
                self.assertLess(abs(group["kept_tokens"] - group["candidate_tokens"] * plan["spatial_keep_ratio"]), 1 + 1e-10)

    def test_provenance_missing_duplicate_and_prepruned_inputs_fail_explicitly(self):
        geo = geometry()
        with self.assertRaisesRegex(ValueError, "versioned"):
            validate_crop_provenance(geo, None)
        for change in ("duplicate", "bad_sources", "prepruned"):
            metadata = copy.deepcopy(geo["predictor_crops"])
            if change == "duplicate":
                metadata["crops"][1]["indices"][0] = metadata["crops"][0]["indices"][0]
            elif change == "bad_sources":
                metadata["tokens"][0]["source_weights"] = []
            else:
                metadata["pre_pruning_applied"] = True
            with self.assertRaises((ValueError, NotImplementedError)):
                validate_crop_provenance(geo, metadata)

    def test_real_generate_scores_all_views_keeps_original_positions_and_zero_identity(self):
        for order in ("base_first", "anyres_first"):
            for extra in (False, True):
                model, features, options = toy_model(order=order, extra=extra)
                core = model.get_model()
                with tempfile.TemporaryDirectory() as directory, patch.object(model, "encode_images", return_value=features), torch.no_grad():
                    path = Path(directory) / "best.pt"
                    make_checkpoint(path, core)
                    baseline = model.generate(**options)
                    core.configure_attention_predictor(enabled=False, keep_ratio=1)
                    self.assertTrue(torch.equal(baseline, model.generate(**options)))
                    core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path,
                                                       capture_attention=True, capture_visualization=True,
                                                       selection_mode="base_crop")
                    with patch.object(core.attention_predictor, "forward", wraps=core.attention_predictor.forward) as forward:
                        first = model.generate(**options)
                    self.assertEqual(len(forward.call_args.args[2]), 12)
                    self.assertTrue(torch.equal(first, model.generate(**options)))
                    stats, stage = core.get_clse_stats(), core.get_clse_stages()[0]
                    self.assertEqual(stats["selection_mode"], "base_crop")
                    self.assertEqual(len(stats["crop_budgets"]), 2)
                    self.assertEqual(stage["attention"]["score_kind"], "attention_predictor_base_crop")
                    self.assertAlmostEqual(sum(stage["attention"]["scores"]), 1, places=5)
                    self.assertEqual(core.config._attn_implementation, "sdpa")

    def test_old_method_json_alias_and_new_options_are_portable_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best.pt").write_bytes(b"parser-only checkpoint")
            path = root / "method.json"
            for name in (None, "base_guided", "base_crop", "base_interpolated", "joint"):
                config = dict(checkpoint="best.pt", prune_rates=[50, 70, 90])
                if name is not None:
                    config["selection_mode"] = name
                path.write_text(json.dumps(config), encoding="utf-8")
                method = load_method("attention_predictor", path)
                self.assertEqual(method.selection_mode, "base_full" if name in (None, "base_guided") else name)
                self.assertEqual(method.crop_temperature, .05)
                self.assertEqual(method.crop_min_keep, 0)
            for options in (dict(crop_temperature=0), dict(crop_min_keep=True), dict(crop_min_keep=-1)):
                config.update(options)
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_method("attention_predictor", path)
                config = dict(checkpoint="best.pt", prune_rates=[50, 70, 90])

    def test_native_single_crop_and_unpad_plus_max9_pipeline(self):
        for size, raw_grid, side in (((8, 8), (8, 8), 2), ((28, 20), (32, 32), 6)):
            model, features, options = toy_model(size=size, grid=raw_grid)
            core = model.get_model()
            core.vision_tower.num_patches_per_side = side
            features = torch.randn(features.shape[0], side ** 2, 32)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pt"
                make_checkpoint(path, core)
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path, selection_mode="base_crop")
                with patch.object(model, "encode_images", return_value=features), torch.no_grad():
                    model.generate(**options)
                provenance = core._clse_layout["predictor_crops"]
                self.assertEqual(len(provenance["crops"]), 1 if side == 2 else 16)
                if side == 6:
                    self.assertEqual(provenance["unpad_offset"], [3, 0])
                    self.assertEqual(provenance["unpadded_shape"], [18, 24])
                    actual = next(grid for grid in core._clse_layout["grids"] if grid["name"] == "anyres")
                    self.assertEqual((actual["rows"], actual["cols"]), (15, 20))


if __name__ == "__main__":
    unittest.main()
