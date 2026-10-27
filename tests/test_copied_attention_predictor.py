"""Native-layer copies, attention/gradient oracles, portable checkpoints and KV pruning."""

import importlib.util
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from test_fastv_attention import tiny_config
from test_clse import toy_model, layout
from test_priortr import causal_bias
from llava.model.language_model.llava_qwen import LlavaQwenModel
from llava.model.language_model.attention_predictor import (
    CopiedAttentionPredictor, CopiedPredictorConfig, backbone_signature, load_predictor, student_autocast)
from llava_pruning.predictor_training import extract_teacher, distillation_loss, epoch_pass, save_student


def core_model(implementation="sdpa"):
    config = tiny_config(implementation)
    config.num_hidden_layers = 6
    return LlavaQwenModel(config).eval().requires_grad_(False)


class CopiedPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_copies_exact_blocks_3_4_5_without_aliasing_precision_or_backbone_mutation(self):
        core = core_model().half()
        snapshot = {key: value.clone() for key, value in core.state_dict().items()}
        student = CopiedAttentionPredictor.from_backbone(core)
        self.assertEqual(student.config.input_size, 32)
        self.assertEqual(student.config.source_layers, (3, 4, 5))
        self.assertFalse(hasattr(student, "project"))
        self.assertFalse(hasattr(student, "lm_head"))
        self.assertFalse(hasattr(student, "norm"))  # no final output norm on H2
        for original, copied in zip(core.layers[2:5], student.layers):
            self.assertIs(type(original), type(copied))
            self.assertIs(type(original.self_attn), type(copied.self_attn))
            self.assertEqual(original.self_attn.layer_idx, copied.self_attn.layer_idx)
            for (key, a), (other, b) in zip(original.named_parameters(), copied.named_parameters()):
                self.assertEqual(key, other)
                torch.testing.assert_close(a.float(), b, rtol=0, atol=0)
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
                self.assertEqual(b.dtype, torch.float32)
        with torch.no_grad():
            student.layers[0].self_attn.q_proj.weight.add_(1)
        self.assertEqual(core.dtype, torch.float16)
        self.assertEqual(core.config._attn_implementation, "sdpa")
        for key, value in core.state_dict().items():
            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in core.parameters()))

    def test_score_matches_original_block5_eager_attention_with_gqa_and_noncompact_positions(self):
        torch.manual_seed(910)
        core = core_model("eager")
        student = CopiedAttentionPredictor.from_backbone(core).eval()
        hidden = torch.randn(1, 19, 32)
        ids = (torch.arange(19) * 2 + 3)[None]
        indices = torch.tensor([2, 4, 7, 10, 12, 15])
        value = hidden.clone()
        for layer in core.layers[2:4]:
            value = layer(value, attention_mask=causal_bias(value), position_ids=ids,
                          position_embeddings=core.rotary_emb(value, ids),
                          output_attentions=False, use_cache=False)[0]
        last = core.layers[4]
        with torch.no_grad():
            _, native, _ = last.self_attn(last.input_layernorm(value), attention_mask=causal_bias(value),
                                         position_ids=ids, position_embeddings=core.rotary_emb(value, ids),
                                         output_attentions=True, use_cache=False)
        expected = native.mean(1)[0, -1, indices]
        expected /= expected.sum()
        actual = student(hidden, ids, indices)
        torch.testing.assert_close(actual.exp(), expected, rtol=1e-5, atol=1e-7)
        self.assertEqual(student.layers[2].self_attn.num_heads, 4)
        self.assertEqual(student.layers[2].self_attn.num_key_value_heads, 2)

    def test_only_copied_relevant_parameters_have_gradients_no_backbone_or_h2_gradients(self):
        torch.manual_seed(911)
        core = core_model()
        student = CopiedAttentionPredictor.from_backbone(core).train()
        hidden = torch.randn(1, 19, 32, requires_grad=True)
        ids, indices = torch.arange(19)[None], torch.tensor([2, 3, 7, 11])
        unused = []
        hooks = [module.register_forward_pre_hook(lambda *args: unused.append(True)) for module in (
            student.layers[2].self_attn.v_proj, student.layers[2].self_attn.o_proj,
            student.layers[2].mlp, student.layers[2].post_attention_layernorm)]
        try:
            predicted = student(hidden, ids, indices)
            distillation_loss(predicted, torch.tensor([.7, .1, .1, .1])).backward()
        finally:
            for hook in hooks:
                hook.remove()
        for name, parameter in student.named_parameters():
            active = not name.startswith("layers.2.") or name.startswith((
                "layers.2.input_layernorm.", "layers.2.self_attn.q_proj.", "layers.2.self_attn.k_proj."))
            self.assertEqual(parameter.requires_grad, active, name)
            if active:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertFalse(unused)
        self.assertIsNone(hidden.grad)
        self.assertTrue(all(parameter.grad is None for parameter in core.parameters()))

    def test_checkpointing_matches_full_forward_and_backward_with_detached_h2(self):
        torch.manual_seed(912)
        core = core_model()
        a = CopiedAttentionPredictor.from_backbone(core, gradient_checkpointing=True).train()
        b = CopiedAttentionPredictor.from_backbone(core, gradient_checkpointing=False).train()
        hidden, ids = torch.randn(1, 19, 32), torch.arange(19)[None]
        indices, target = torch.tensor([2, 3, 7, 11]), torch.tensor([.7, .1, .1, .1])
        for student in (a, b):
            distillation_loss(student(hidden, ids, indices), target).backward()
        for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            if pa.requires_grad:
                torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0, msg=name)

    def test_initialization_reproduces_teacher_block5_when_source_is_raw_h2(self):
        core = core_model("eager")
        student = CopiedAttentionPredictor.from_backbone(core).eval()
        example = extract_teacher(core, torch.randn(1, 19, 32), (2, 16), layout(), teacher_layer=5)
        prediction = student(example["hidden"], example["position_ids"], example["spatial_indices"])
        torch.testing.assert_close(prediction.exp(), example["target"], rtol=1e-5, atol=1e-7)

    def test_kl_training_reduces_loss_without_changing_copied_frozen_tail_or_teacher(self):
        torch.manual_seed(913)
        core = core_model()
        student = CopiedAttentionPredictor.from_backbone(core)
        teacher = {key: value.clone() for key, value in core.state_dict().items()}
        frozen = {key: value.clone() for key, value in student.named_parameters() if not value.requires_grad}
        example = extract_teacher(core, torch.randn(1, 19, 32), (2, 16), layout(), teacher_layer=6)
        example["target"] = torch.zeros_like(example["target"])
        example["target"][0] = 1
        sample = [types.SimpleNamespace(sample_id="x")]
        optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=.01)
        initial = epoch_pass(student, lambda _: example, sample, device="cpu", precision="float32")["kl"]
        for _ in range(12):
            epoch_pass(student, lambda _: example, sample, device="cpu", precision="float32", optimizer=optimizer)
        final = epoch_pass(student, lambda _: example, sample, device="cpu", precision="float32")["kl"]
        self.assertLess(final, initial / 10)
        for key, value in student.named_parameters():
            if key in frozen:
                torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
        for key, value in core.state_dict().items():
            torch.testing.assert_close(value, teacher[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in core.parameters()))

    def test_v2_checkpoint_roundtrip_has_full_independent_blocks_and_original_indices(self):
        core = core_model()
        student = CopiedAttentionPredictor.from_backbone(core).eval()
        hidden, ids, indices = torch.randn(1, 19, 32), torch.arange(19)[None], [2, 7, 9]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.pt"
            save_student(path, student, student_precision="float32", backbone_depth=6, teacher_layer=6,
                         signature=backbone_signature(core.config))
            loaded, precision, saved = load_predictor(path, "cpu", 32)
            self.assertIsInstance(loaded, CopiedAttentionPredictor)
            self.assertEqual(saved["version"], 2)
            self.assertEqual(saved["architecture"]["source_layers"], (3, 4, 5))
            self.assertEqual(precision, "float32")
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            torch.testing.assert_close(loaded(hidden, ids, indices), student(hidden, ids, indices), rtol=0, atol=0)
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
            core.config.rms_norm_eps *= 2
            core.clear_attention_predictor()
            with self.assertRaisesRegex(ValueError, "Qwen2 architecture/attention backend"):
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)

    def test_rejects_insufficient_depth_instead_of_random_initialization_or_wrong_layer_copies(self):
        with self.assertRaisesRegex(ValueError, "blocks 3, 4, 5"):
            CopiedAttentionPredictor.from_backbone(LlavaQwenModel(tiny_config()))
        config = CopiedAttentionPredictor.from_backbone(core_model()).config
        with self.assertRaisesRegex(ValueError, "original blocks 3, 4, 5"):
            CopiedPredictorConfig(config.qwen_config, "sdpa", source_layers=(2, 3, 4))

    def test_real_generate_zero_identity_and_copied_branch_pruning_original_features_once(self):
        torch.manual_seed(914)
        model, features, options = toy_model()
        core = model.get_model()
        core.config.num_hidden_layers = 6
        for index in (4, 5):
            core.layers.append(type(core.layers[0])(core.config, layer_idx=index))
        model.eval().requires_grad_(False)
        student = CopiedAttentionPredictor.from_backbone(core)
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "encode_images", return_value=features), torch.no_grad():
            path = Path(directory) / "best.pt"
            save_student(path, student, student_precision="float32", backbone_depth=6, teacher_layer=6,
                         signature=backbone_signature(core.config))
            baseline = model.generate(**options)
            core.configure_attention_predictor(enabled=False, keep_ratio=1, checkpoint=path)
            with patch("llava.model.language_model.attention_predictor.load_predictor", side_effect=AssertionError("0 loaded")):
                self.assertTrue(torch.equal(baseline, model.generate(**options)))
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path, capture_visualization=True)
            observed, actual_h2 = [], []
            hooks = [core.attention_predictor.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone())),
                     core.layers[1].register_forward_hook(lambda module, args, result: actual_h2.append(result[0].clone()))]
            try:
                result = model.generate(**options)
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertEqual(len(observed), 1)
            torch.testing.assert_close(observed[0], actual_h2[0], rtol=0, atol=0)
            stats = core.get_clse_stats()
            self.assertEqual([row["image_tokens_in"] for row in stats["layers"]], [14, 14, 7, 7, 7, 7])
            self.assertEqual(stats["student_source_layers"], [3, 4, 5])
            self.assertEqual(stats["student_hidden_size"], 32)
            self.assertEqual(stats["attention_source"], "copied_qwen2_blocks_3_4_5")
            self.assertEqual(stats["protected_newline_tokens"], 2)
            with patch("llava.model.language_model.attention_predictor.load_predictor", side_effect=AssertionError("reloaded")):
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
                self.assertTrue(torch.equal(result, model.generate(**options)))
            core.configure_fastv(enabled=False)
            self.assertIsNone(core.attention_predictor)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flash_attn"),
                     "CUDA + FlashAttention2 required on the inference server")
class CopiedPredictorFlashTests(unittest.TestCase):
    def test_fp16_backbone_bf16_copied_native_flash_training_and_compact_cached_decode(self):
        config = tiny_config("flash_attention_2")
        config.num_hidden_layers, config.torch_dtype = 6, torch.float16
        core = LlavaQwenModel(config).cuda().half().eval().requires_grad_(False)
        student = CopiedAttentionPredictor.from_backbone(core).cuda()
        tokens = torch.arange(1, 20, device="cuda")[None]
        example = extract_teacher(core, core.embed_tokens(tokens), (2, 16), layout(), teacher_layer=6)
        optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=.001)
        metrics = epoch_pass(student, lambda _: example, [types.SimpleNamespace(sample_id="x")],
                             device="cuda", precision="bfloat16", optimizer=optimizer)
        self.assertTrue(math.isfinite(metrics["kl"]))
        with student_autocast("cuda", "bfloat16"):
            score = student(example["hidden"], example["position_ids"], example["spatial_indices"])
        self.assertEqual(score.dtype, torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_student(path, student, student_precision="bfloat16", backbone_depth=6, teacher_layer=6,
                         signature=backbone_signature(core.config))
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
            from llava.model.language_model.attention_predictor_crops import dense_crop_provenance
            geometry = layout()
            geometry["predictor_crops"] = dense_crop_provenance(geometry, (2, 1), 2, (2, 4))
            core.set_clse_image_layout([[(2, 16)]], 19, geometry)
            with torch.no_grad():
                result = core(input_ids=tokens, use_cache=True)
                self.assertEqual([result.past_key_values.get_seq_length(i) for i in range(6)], [12] * 6)
                core(input_ids=torch.tensor([[20]], device="cuda"), past_key_values=result.past_key_values, use_cache=True)
        self.assertEqual(core.dtype, torch.float16)
        self.assertEqual(core.config._attn_implementation, "flash_attention_2")


if __name__ == "__main__":
    unittest.main()
