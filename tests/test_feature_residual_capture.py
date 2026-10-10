"""Real miniature SigLIP blocks, no checkpoint/GPU. Forward-path regression."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "llava"))
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
from llava_pruning.score_visualization import ScoreGeometry
from feature_residual_capture import LAYERS, STAGES, capture_stage_scores
from feature_residual_math import stage_residuals


class TinyTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = SigLipVisionConfig(hidden_size=8, intermediate_size=16,
                                    num_hidden_layers=27, num_attention_heads=2,
                                    image_size=28, patch_size=14)
        self.vision_tower = SigLipVisionModel(config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = torch.nn.Identity()


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_real_blocks_match_raw_hidden_states_no_head_or_postnorm(self):
        torch.manual_seed(17)
        tower = TinyTower().eval()
        geometry = ScoreGeometry.create((56, 28), (2, 1), 28, 14)
        pixels = torch.randn(3, 3, 28, 28)
        with torch.inference_mode():
            states = tower.vision_tower(pixels, output_hidden_states=True).hidden_states
        weights = {key: value.clone() for key, value in tower.state_dict().items()}
        vision = tower.vision_tower.vision_model
        with patch.object(vision.post_layernorm, "forward", side_effect=AssertionError("postnorm ran")), \
             patch.object(vision.head, "forward", side_effect=AssertionError("head ran")), \
             patch.object(tower, "forward", side_effect=AssertionError("outer tower ran")):
            actual, shapes = capture_stage_scores(tower, pixels, geometry)
        self.assertEqual(set(shapes), set(STAGES))
        for si, layer in enumerate(LAYERS):
            expected = stage_residuals(states[layer].numpy(), geometry)
            self.assertEqual(shapes[STAGES[si]], [3, 4, 8])
            for key in expected:
                np.testing.assert_array_equal(actual[key][si], expected[key])
        self.assertTrue(all(torch.equal(value, weights[key]) for key, value in tower.state_dict().items()))
        self.assertTrue(all(not module._forward_hooks for module in tower.modules()))
        self.assertTrue(all(parameter.grad is None for parameter in tower.parameters()))

    def test_cleanup_and_validation(self):
        tower = TinyTower().eval()
        geometry = ScoreGeometry.create((28, 28), (1, 1), 28, 14)
        pixels = torch.randn(2, 3, 28, 28)
        with patch("feature_residual_capture.stage_residuals", side_effect=ValueError("probe failed")):
            with self.assertRaisesRegex(ValueError, "probe failed"):
                capture_stage_scores(tower, pixels, geometry)
        self.assertTrue(all(not module._forward_hooks for module in tower.modules()))
        with self.assertRaisesRegex(ValueError, "shape"):
            capture_stage_scores(tower, pixels[:1], geometry)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            capture_stage_scores(tower, torch.full_like(pixels, float("nan")), geometry)
        self.assertTrue(all(not module._forward_hooks for module in tower.modules()))
        tower.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            capture_stage_scores(tower, pixels, geometry)
        tower.eval()
        del tower.vision_tower.vision_model.encoder.layers[-1:]
        with self.assertRaisesRegex(ValueError, "26-block"):
            capture_stage_scores(tower, pixels, geometry)


if __name__ == "__main__":
    unittest.main()
