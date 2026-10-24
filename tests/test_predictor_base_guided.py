"""Base-only score mapping, proportional group quotas and native physical-drop integration."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_attention_predictor import make_checkpoint
from test_clse import toy_model, layout
from llava.model.language_model.attention_predictor_selection import (
    named_spatial_grids, base_guided_log_scores, base_guided_budget, select_base_guided_tokens)
from llava_pruning.methods import load_method


def spatial_indices(geometry):
    return [index for grid in geometry["grids"] for index in grid["indices"]]


class BaseGuidedPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_interpolates_base_probabilities_not_log_scores_with_known_bilinear_oracle(self):
        geometry = layout()
        base = torch.tensor([.1, .2, .3, .4])
        expected_anyres = torch.tensor([.1, .125, .175, .2, .3, .325, .375, .4])
        mapped = base_guided_log_scores(base.log(), geometry)
        expected = torch.cat((base, expected_anyres)) / 3
        torch.testing.assert_close(mapped.exp(), expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(mapped.dtype, torch.float32)

    def test_actual_rectangular_grids_and_packing_orders_do_not_assume_raw_tiles_or_27x27(self):
        base = torch.tensor([-.5, -3., -1., -2.])
        expected = F.interpolate(base.softmax(0).reshape(1, 1, 2, 2), (3, 5),
                                 mode="bilinear", align_corners=False).flatten()
        for order in ("base_first", "anyres_first"):
            for extra in (False, True):
                geometry = layout(rows=3, cols=5, order=order, extra=extra)
                scores = base_guided_log_scores(base, geometry)
                torch.testing.assert_close(scores[4:].softmax(0), expected / expected.sum())
                reversed_geometry = copy.deepcopy(geometry)
                reversed_geometry["grids"].reverse()
                reverse = base_guided_log_scores(base, reversed_geometry)
                torch.testing.assert_close(reverse, torch.cat((scores[4:], scores[:4])), rtol=0, atol=0)

    def test_extreme_log_scores_stay_finite_and_uniform_mass_shifts_do_not_change_mapping(self):
        geometry = layout()
        base = torch.tensor([-1000., -1001., 0., -2000.])
        mapped = base_guided_log_scores(base, geometry)
        shifted = base_guided_log_scores(base + 100., geometry)
        self.assertTrue(torch.isfinite(mapped).all())
        torch.testing.assert_close(mapped, shifted, rtol=0, atol=0)
        self.assertGreater(float(mapped[0]), float(mapped[1]))  # preserve weak Base ranking
        self.assertAlmostEqual(float(mapped.exp().sum()), 1, places=6)

    def test_group_budget_preserves_original_packed_total_and_reserves_newlines_before_proportions(self):
        geometry = layout()
        plan = base_guided_budget(geometry, .5)
        self.assertEqual(plan["target"], 7)
        self.assertEqual(plan["protected_indices"], [8, 13])
        self.assertEqual([group["kept_tokens"] for group in plan["groups"]], [2, 3])
        self.assertEqual(plan["spatial_keep_ratio"], 5 / 12)
        self.assertEqual(sum(group["kept_tokens"] for group in plan["groups"]) + 2, 7)

    def test_full_size_layer_average_targets_have_equal_spatial_ratio_with_less_than_one_token_rounding(self):
        geometry = layout(base=27, rows=81, cols=81)
        for average in (0, 50, 70, 90):
            keep = 1 - average * 28 / 26 / 100
            plan = base_guided_budget(geometry, keep)
            self.assertEqual(plan["target"], max(1, math.floor(geometry["packed_tokens"] * keep + 1e-10)))
            self.assertEqual(len(plan["protected_indices"]), 81)
            for group in plan["groups"]:
                requested = plan["spatial_keep_ratio"] * group["candidate_tokens"]
                self.assertLess(abs(group["kept_tokens"] - requested), 1 + 1e-10)
                self.assertGreater(group["kept_tokens"], 0)
            self.assertEqual(sum(g["kept_tokens"] for g in plan["groups"]) + 81, plan["target"])

    def test_group_attention_mass_cannot_steal_budget_and_ties_follow_original_spatial_order(self):
        geometry = layout()
        scores = torch.tensor([-200., -200., -200., -200.] + [100.] * 8)
        chosen, plan = select_base_guided_tokens(scores, spatial_indices(geometry), 14, .5, geometry)
        self.assertEqual(chosen.tolist(), [0, 1, 4, 5, 6, 8, 13])
        scores[:4] = 10000
        other, _ = select_base_guided_tokens(scores, spatial_indices(geometry), 14, .5, geometry)
        self.assertTrue(torch.equal(chosen, other))

    def test_layout_validation_and_infeasible_groups_fail_without_silent_budget_changes(self):
        geometry = layout()
        with self.assertRaisesRegex(ValueError, "at least one spatial token in each"):
            base_guided_budget(geometry, .2)
        unequal = layout(base=1, rows=3, cols=5)
        with self.assertRaisesRegex(ValueError, "proportional base budget"):
            base_guided_budget(unequal, .3)
        for invalid in (None, {}, {**geometry, "grids": geometry["grids"][:1]}):
            with self.assertRaises(ValueError):
                named_spatial_grids(invalid)
        overlap = copy.deepcopy(geometry)
        overlap["grids"][1]["indices"][0] = 0
        with self.assertRaisesRegex(ValueError, "distinct valid"):
            named_spatial_grids(overlap)
        with self.assertRaisesRegex(ValueError, "finite scores"):
            base_guided_log_scores(torch.tensor([float("nan")] * 4), geometry)

    def test_inference_requests_base_scores_only_and_ignores_arbitrary_anyres_scores(self):
        model, _, _ = toy_model()
        core = model.get_model()
        geometry = layout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            make_checkpoint(path, core)
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
            core.set_clse_image_layout([[(2, 16)]], 19, geometry)
            h2, ids = torch.randn(1, 19, 32), torch.arange(19)[None]
            native_scores = torch.arange(19).float() / 10
            calls = []
            def score(hidden, positions, indices):
                calls.append(list(indices))
                self.assertIs(hidden, h2)  # full sequence, NOT Base-only side input
                return native_scores[indices]
            with patch.object(core.attention_predictor, "forward", side_effect=score):
                first = core._clse_rank(h2, 2, 16, ids, None)
                native_scores[[2 + i for i in geometry["grids"][1]["indices"]]] = -10000
                second = core._clse_rank(h2, 2, 16, ids, None)
            self.assertEqual(calls, [[2, 3, 4, 5]] * 2)
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_explicit_joint_option_preserves_legacy_own_scores_and_joint_topk(self):
        model, _, _ = toy_model()
        core = model.get_model()
        geometry = layout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            make_checkpoint(path, core)
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path, selection_mode="joint")
            core.set_clse_image_layout([[(2, 16)]], 19, geometry)
            own_scores = torch.arange(12).float()
            with patch.object(core.attention_predictor, "forward", return_value=own_scores) as predictor:
                actual = core._clse_rank(torch.randn(1, 19, 32), 2, 16, torch.arange(19)[None], None)
            self.assertIs(actual, own_scores)
            self.assertEqual(predictor.call_args.args[2], [2 + i for i in spatial_indices(geometry)])
            chosen, _, _ = core._clse_select(actual, spatial_indices(geometry), 14)
            self.assertFalse(any(i in chosen.tolist() for i in geometry["grids"][0]["indices"]))

    def test_real_generate_tracks_both_group_counts_and_protects_newlines_in_both_image_orders(self):
        for order in ("base_first", "anyres_first"):
            for extra in (False, True):
                model, features, options = toy_model(order=order, extra=extra)
                core = model.get_model()
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "best.pt"
                    make_checkpoint(path, core)
                    core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path,
                                                       capture_attention=True, capture_visualization=True)
                    with patch.object(model, "encode_images", return_value=features), torch.no_grad():
                        model.generate(**options)
                    stats, stage = core.get_clse_stats(), core.get_clse_stages()[0]
                    self.assertEqual(stats["selection_mode"], "base_guided")
                    self.assertEqual(stage["attention"]["score_kind"], "attention_predictor_base_guided")
                    self.assertEqual(stage["selection_groups"], stats["selection_groups"])
                    actual_layout = core._clse_layout
                    plan = base_guided_budget(actual_layout, .5)
                    keep = stage["mask"]["keep"]
                    self.assertEqual(sum(keep), plan["target"])
                    self.assertTrue(all(keep[index] for index in plan["protected_indices"]))
                    for group in plan["groups"]:
                        indices = named_spatial_grids(actual_layout)[group["name"]]["indices"]
                        self.assertEqual(sum(keep[index] for index in indices), group["kept_tokens"])
                    self.assertAlmostEqual(sum(stage["attention"]["scores"]), 1, places=5)
                    self.assertTrue(all(stage["attention"]["scores"][i] == 0 for i in plan["protected_indices"]))

    def test_existing_method_json_defaults_to_new_policy_and_joint_is_available_for_old_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best.pt").write_bytes(b"not loaded by method parser")
            config = dict(checkpoint="best.pt", prune_rates=[50, 70, 90], visualize_rates=[])
            path = root / "method.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            method = load_method("attention_predictor", path)
            self.assertEqual(method.selection_mode, "base_guided")
            self.assertEqual(method.details()["selection_mode"], "base_guided")
            config["selection_mode"] = "joint"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_method("attention_predictor", path).selection_mode, "joint")
            config["selection_mode"] = "typo"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection_mode"):
                load_method("attention_predictor", path)


if __name__ == "__main__":
    unittest.main()
