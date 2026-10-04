"""Optional read-only comparisons with the user's downloaded LLaVA checkout.

Set LLAVA_REFERENCE_DIR to its repository root on another machine. These tests
never import the original model loader or download any model weights.
"""

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get("LLAVA_REFERENCE_DIR", PROJECT.parent / "Triad" / "Triad"))
sys.path.insert(0, str(PROJECT / "vendor" / "llava"))
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor


def load_reference(name, relative):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / "LLaVA-NeXT" / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def methods(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


class ToyPacking(LlavaMetaForCausalLM):
    """Exercise real packing code with deterministic vision features."""

    def __init__(self, dtype):
        self.config = SimpleNamespace(
            image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad",
            image_grid_pinpoints=[[8, 8], [16, 8]], tokenizer_model_max_length=128,
            tokenizer_padding_side="right", use_pos_skipping=False,
        )
        self.device = torch.device("cpu")
        self.training = False
        self.dtype = dtype
        self.tower = SimpleNamespace(num_patches_per_side=2, image_size=8)
        self.core = SimpleNamespace(
            embed_tokens=torch.nn.Embedding(32, 4).to(dtype),
            image_newline=torch.arange(4, dtype=dtype),
            get_vision_tower=lambda: self.tower, fastv_enabled=False,
        )
        self.model = self.core

    def get_model(self):
        return self.core

    def encode_images(self, images):
        return torch.arange(images.shape[0] * 4 * 4, dtype=self.dtype).reshape(-1, 4, 4) / 20


@unittest.skipUnless((REFERENCE / "LLaVA-NeXT" / "llava").is_dir(),
                     "Original LLaVA checkout unavailable; set LLAVA_REFERENCE_DIR")
class ReferenceTests(unittest.TestCase):
    def test_chat_prompt_matches_original_qwen_template(self):
        original = load_reference("_llava_reference_conversation", "llava/conversation.py")
        backend = ast.parse((PROJECT / "llava_pruning/backend.py").read_text(encoding="utf-8"))
        expression = next(node.value for node in ast.walk(backend)
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "text"
                                  for target in node.targets))
        for prompt in ("Is there a defect?", "<image>\nChoose A or B."):
            message = prompt if "<image>" in prompt else "<image>\n" + prompt
            state = original.conv_templates["qwen_1_5"].copy()
            state.append_message(state.roles[0], (message, [Image.new("RGB", (8, 8))], "Default"))
            state.append_message(state.roles[1], None)
            actual = eval(compile(ast.Expression(expression), "backend_prompt", "eval"),
                          {"message": message})
            self.assertEqual(actual, state.get_prompt())

    def test_causal_lm_wrapper_matches_original_except_state_reset(self):
        relative = Path("llava/model/language_model/llava_qwen.py")
        original = methods(REFERENCE / "LLaVA-NeXT" / relative, "LlavaQwenForCausalLM")
        current = methods(PROJECT / "vendor/llava" / relative, "LlavaQwenForCausalLM")
        for name in ("__init__", "forward", "prepare_inputs_for_generation", "generate"):
            if name == "generate":
                current[name].body = [node for node in current[name].body
                                      if not (isinstance(node, ast.Expr)
                                              and isinstance(node.value, ast.Call)
                                              and isinstance(node.value.func, ast.Attribute)
                                              and node.value.func.attr == "reset_fastv_state")]
            self.assertEqual(ast.dump(current[name]), ast.dump(original[name]), name)

    def test_anyres_preprocessing_pixels_match_original(self):
        original = load_reference("_llava_reference_mm", "llava/mm_utils.py")
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad",
                                 image_grid_pinpoints=[[28, 28], [56, 28], [28, 56], [56, 56]])
        rng = np.random.RandomState(5)
        for width, height in ((28, 28), (61, 29), (29, 61), (53, 49)):
            image = Image.fromarray(rng.randint(0, 256, (height, width, 3), dtype=np.uint8))
            expected = original.process_images([image], processor, config)
            actual, _ = process_images([image], processor, config, return_pro_data=True)
            self.assertTrue(torch.equal(actual, expected), (width, height))

    def test_anyres_packed_embeddings_masks_and_positions_match_original(self):
        original = load_reference("llava.model._llava_reference_arch", "llava/model/llava_arch.py")
        for dtype in (torch.float16, torch.float32):
            toy = ToyPacking(dtype)
            for dimensions, views in (((8, 8), 2), ((16, 8), 3)):
                images = torch.zeros(1, views, 3, 8, 8, dtype=dtype)
                tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                args = (tokens, None, None, None, None, images)
                expected = original.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal(
                    toy, *args, image_sizes=[dimensions])
                actual = toy.prepare_inputs_labels_for_multimodal(*args, image_sizes=[dimensions])
                for expected_item, actual_item in zip(expected, actual):
                    if expected_item is None:
                        self.assertIsNone(actual_item)
                    else:
                        self.assertTrue(torch.equal(actual_item, expected_item))

    def test_anyres_first_only_moves_visual_blocks_and_keeps_original_tail(self):
        original = load_reference("llava.model._llava_reference_arch", "llava/model/llava_arch.py")
        for dtype in (torch.float16, torch.float32):
            for final_newline in (False, True):
                toy = ToyPacking(dtype)
                toy.config.image_token_order = "anyres_first"
                if final_newline:
                    toy.config.mm_patch_merge_type += "_add_newl"
                for dimensions, views in (((8, 8), 2), ((16, 8), 3)):
                    with self.subTest(dtype=dtype, final_newline=final_newline,
                                      dimensions=dimensions):
                        images = torch.zeros(1, views, 3, 8, 8, dtype=dtype)
                        tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                        args = (tokens, torch.arange(4)[None], torch.ones_like(tokens),
                                None, tokens.clone(), images)
                        expected = list(original.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal(
                            toy, *args, image_sizes=[dimensions]))
                        packed = expected[4]
                        # Keep the last structural image token and following two
                        # text tokens fixed. Move the four base tokens behind the
                        # remaining anyres block; nothing else may change.
                        tail = packed.shape[1] - 3
                        expected[4] = torch.cat((packed[:, :1], packed[:, 5:tail],
                                                 packed[:, 1:5], packed[:, tail:]), dim=1)
                        actual = toy.prepare_inputs_labels_for_multimodal(*args, image_sizes=[dimensions])
                        for expected_item, actual_item in zip(expected, actual):
                            if expected_item is None:
                                self.assertIsNone(actual_item)
                            else:
                                self.assertTrue(torch.equal(actual_item, expected_item))


if __name__ == "__main__":
    unittest.main()
