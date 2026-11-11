"""Validate global AnyRes anchors, equal budgets and the pre-LLM-only render."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import torch
from PIL import Image

import runMyPruningPoolSpatial as pool
import visualize_mvtec_pre_llm as entry
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava_pruning.pool_compact_visualization import render_compact
from llava_pruning.visa_comparison_methods import observe_prefill
from test_pool_ablations import model_and_options
from test_pool_spatial import _Tower


class PreLlmVisualizationTests(TestCase):
    @torch.inference_mode()
    def test_both_modes_use_the_same_budget_and_run_all_llm_blocks_on_it(self):
        model, options = model_and_options(28)
        core = model.get_model()
        core.config.image_grid_pinpoints = [[768, 384]]
        core.vision_tower = _Tower()
        core.vision_tower.features = core.vision_tower.features[:3]
        pixels = torch.zeros(3, 3, 384, 384)
        options.update(images=[pixels], image_sizes=[(768, 384)], max_new_tokens=2)
        with TemporaryDirectory() as temporary:
            image = Path(temporary) / "original.png"
            Image.new("RGB", (768, 384), (30, 70, 110)).save(image)
            sample = SimpleNamespace(image=image, sample_id="000000001", gt=0)
            for rate in entry.RATES:
                observed = []
                for mode, title in entry.MODES.items():
                    method = entry.PreLlmMethod(core, mode, rate)
                    method.configure(core, rate, capture_attention=False, capture_visualization=False)
                    with observe_prefill(core) as lengths, method.inference_context(model):
                        model.generate(**options)
                    plan, view_plan = method.adapter.plan, method.adapter.view_plan
                    self.assertFalse(core.attention_predictor_enabled)
                    self.assertEqual(list(lengths.values()), [plan["pre_llm_tokens"] + 4] * 28)
                    canvas, details = render_compact(method, {"pixels": pixels}, SigLipImageProcessor(),
                                                     {"answer": "B"}, sample, rate, 120, pre_llm_label=title)
                    with canvas:
                        self.assertGreater(canvas.width, 360)
                    self.assertEqual(details["base_kept"], view_plan["base_keep"])
                    self.assertEqual(sum(details["anyres_kept_per_crop"]), view_plan["anyres_keep"])
                    observed.append((plan["pre_llm_tokens"], view_plan))
                self.assertEqual(*observed)

    def test_global_topk_can_concentrate_dominant_anchors_in_one_crop(self):
        features = torch.randn(3, 729, 8)
        scores = torch.stack((torch.rand(729), torch.arange(729).float(), -torch.arange(1, 730).float()))
        keys = torch.randn(3, 729, 4)
        core = SimpleNamespace(mm_projector=torch.nn.Identity(), image_newline=torch.zeros(8),
                               config=SimpleNamespace(mm_patch_merge_type="spatial_unpad"))
        adapter = SimpleNamespace(core=core, geometry=SimpleNamespace(grid=(2, 1)), prepared=None,
                                  view_plan={"base_keep": 100, "anyres_keep": 200})
        packed, decisions, roles = entry.compress_global_anyres(adapter, features, scores, keys)
        dominant = round(200 * pool.DOMINANT_RATIO)
        self.assertEqual(decisions[1]["dominant_indices"].tolist(), list(range(729 - dominant, 729)))
        self.assertEqual(len(decisions[2]["dominant_indices"]), 0)
        self.assertEqual(roles.bincount().tolist(), [100, 200, 27])
        self.assertEqual(packed.shape, (327, 8))
