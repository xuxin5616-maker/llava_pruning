"""Gradient, frozen-teacher oracle, real pruning/cache and labels-free data."""

import csv
import importlib.util
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import Qwen2Model
from transformers.cache_utils import DynamicCache

from test_fastv_attention import tiny_config
from test_clse import toy_model, layout
from test_priortr import causal_bias
from llava.model.language_model.attention_predictor import (AttentionPredictor, PredictorConfig, backbone_signature,
                                                           load_predictor, _rope)
from llava.model.language_model.llava_qwen import LlavaQwenModel
from llava_pruning.predictor_training import (extract_teacher, distillation_loss, epoch_pass, save_student,
                                             TeacherExamples, architecture_signature, prepare_teacher_example)
from llava_pruning.predictor_data import load_visa_distillation, split_samples
from llava_pruning.methods import load_method


def student(input_size=32):
    return AttentionPredictor(PredictorConfig(input_size=input_size, hidden_size=16, num_heads=2, intermediate_size=32))


def make_checkpoint(path, core, predictor=None):
    predictor = student() if predictor is None else predictor
    save_student(path, predictor, student_precision="float32", backbone_depth=len(core.layers), teacher_layer=3,
                 signature=backbone_signature(core.config))
    return predictor


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_third_layer_attention_matches_explicit_qk_and_only_student_has_gradient(self):
        torch.manual_seed(810)
        model = student()
        hidden = torch.randn(1, 19, 32)
        ids = torch.arange(19)[None]
        spatial = torch.tensor([2, 3, 4, 5, 6, 7])
        predicted = model(hidden, ids, spatial)
        value = model.project(model.input_norm(hidden))
        for block in model.layers[:2]:
            value = block(value, ids)
        last = model.layers[2]
        value = last.norm(value)
        q = last.q(value[:, -1:]).view(1, 1, 2, 8).transpose(1, 2)
        k = last.k(value).view(1, 19, 2, 8).transpose(1, 2)
        q, k = _rope(q, ids[:, -1:], model.config.rope_theta), _rope(k, ids, model.config.rope_theta)
        probability = ((q @ k.transpose(-2, -1)) / (8 ** .5)).softmax(-1).mean(1)[0, 0, spatial]
        probability /= probability.sum()
        torch.testing.assert_close(predicted.exp(), probability)
        target = torch.tensor([.6, .1, .1, .1, .05, .05])
        distillation_loss(predicted, target).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsNone(hidden.grad)
        self.assertFalse(hasattr(last, "v"))

    def test_teacher_matches_actual_eager_attention_and_raw_h2_without_running_teacher_output(self):
        torch.manual_seed(811)
        core = LlavaQwenModel(tiny_config("eager")).eval().requires_grad_(False)
        packed = torch.randn(1, 19, 32)
        called, outputs = [], []
        handles = [layer.register_forward_hook(lambda layer, args, output: (
            called.append(layer.self_attn.layer_idx), outputs.append(output[0].clone())) and None) for layer in core.layers]
        try:
            example = extract_teacher(core, packed, (2, 16), layout(), teacher_layer=3)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(called, [0, 1])  # Q/K of block3, but NO block3 output / answer generation
        torch.testing.assert_close(example["hidden"], outputs[1], rtol=0, atol=0)
        ids = torch.arange(19)[None]
        layer = core.layers[2]
        with torch.no_grad():
            _, attention, _ = layer.self_attn(layer.input_layernorm(outputs[1]), attention_mask=causal_bias(outputs[1]),
                                             position_ids=ids, position_embeddings=core.rotary_emb(outputs[1], ids),
                                             output_attentions=True, use_cache=False)
        expected = attention.mean(1)[0, -1, example["spatial_indices"]]
        expected /= expected.sum()
        torch.testing.assert_close(example["target"], expected, rtol=1e-5, atol=1e-7)
        self.assertTrue(all(not value.requires_grad for value in example.values()))
        self.assertTrue(all(p.grad is None for p in core.parameters()))

    def test_student_can_overfit_small_distillation_target_and_backbone_stays_frozen(self):
        torch.manual_seed(812)
        model = student()
        core = LlavaQwenModel(tiny_config("sdpa")).eval().requires_grad_(False)
        example = extract_teacher(core, torch.randn(1, 19, 32), (2, 16), layout(), teacher_layer=3)
        example["target"] = torch.zeros_like(example["target"])
        example["target"][0] = 1.
        samples = [types.SimpleNamespace(sample_id=str(i)) for i in range(3)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        initial = epoch_pass(model, lambda sample: example, samples, device="cpu", precision="float32")["kl"]
        for _ in range(12):
            epoch_pass(model, lambda sample: example, samples, device="cpu", precision="float32",
                       optimizer=optimizer, accumulation=2)
        final = epoch_pass(model, lambda sample: example, samples, device="cpu", precision="float32")["kl"]
        self.assertLess(final, initial / 10)
        self.assertTrue(all(p.grad is None for p in core.parameters()))

    def test_partial_accumulation_matches_actual_group_mean(self):
        torch.manual_seed(813)
        a, b = student(), student()
        b.load_state_dict(a.state_dict())
        example = dict(hidden=torch.randn(1, 19, 32), position_ids=torch.arange(19)[None],
                       spatial_indices=torch.tensor([2, 3, 4]), target=torch.tensor([.8, .1, .1]))
        samples = [types.SimpleNamespace(sample_id="x")]
        oa, ob = torch.optim.SGD(a.parameters(), lr=.1), torch.optim.SGD(b.parameters(), lr=.1)
        epoch_pass(a, lambda sample: example, samples, device="cpu", precision="float32", optimizer=oa, accumulation=4)
        epoch_pass(b, lambda sample: example, samples, device="cpu", precision="float32", optimizer=ob, accumulation=1)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)

    def test_real_generate_zero_identity_side_branch_once_at_h2_and_all_kv_compacted(self):
        torch.manual_seed(814)
        model, features, options = toy_model()
        core = model.get_model()
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "encode_images", return_value=features), torch.no_grad():
            path = Path(directory) / "best.pt"
            make_checkpoint(path, core)
            baseline = model.generate(**options)
            core.configure_attention_predictor(enabled=False, keep_ratio=1, checkpoint=path)
            with patch("llava.model.language_model.attention_predictor.load_predictor", side_effect=AssertionError("zero loaded")):
                zero = model.generate(**options)
            self.assertTrue(torch.equal(baseline, zero))
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path,
                                               capture_attention=True, capture_visualization=True)
            observed, native_h2 = [], []
            first = core.attention_predictor.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
            second = core.layers[1].register_forward_hook(lambda module, args, result: native_h2.append(result[0].clone()))
            try:
                output = model.generate(**options)
            finally:
                first.remove()
                second.remove()
            self.assertEqual(len(observed), 1)
            torch.testing.assert_close(observed[0], native_h2[0], rtol=0, atol=0)
            stats = core.get_clse_stats()
            self.assertEqual([row["image_tokens_in"] for row in stats["layers"]], [14, 14, 7, 7])
            self.assertEqual(stats["pruning_layer"], 2)
            self.assertEqual(core.get_clse_stages()[0]["after_layer"], 2)
            self.assertEqual(stats["protected_newline_tokens"], 2)
            with patch("llava.model.language_model.attention_predictor.load_predictor", side_effect=AssertionError("reloaded")):
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
                self.assertTrue(torch.equal(output, model.generate(**options)))
            self.assertEqual(core.config._attn_implementation, "sdpa")
            from llava_pruning.benchmark_timing import generate_with_benchmark
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                    patch("torch.cuda.max_memory_allocated", return_value=100), patch("torch.cuda.max_memory_reserved", return_value=200):
                measured, timing = generate_with_benchmark(model, options)
            self.assertTrue(torch.equal(output, measured))
            self.assertEqual(timing["generated_tokens"], 4)
            core.configure_fastv(enabled=False)
            self.assertIsNone(core.attention_predictor)
            self.assertFalse(core.attention_predictor_enabled)

    def test_cached_decode_original_ids_and_independent_frozen_selection_oracle(self):
        torch.manual_seed(815)
        core = LlavaQwenModel(tiny_config("sdpa")).eval()
        reference = Qwen2Model(tiny_config("eager")).eval()
        reference.load_state_dict(core.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            make_checkpoint(path, core)
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path,
                                               capture_visualization=True)
            from llava.model.language_model.attention_predictor_crops import dense_crop_provenance
            geometry = layout()
            geometry["predictor_crops"] = dense_crop_provenance(geometry, (2, 1), 2, (2, 4))
            core.set_clse_image_layout([[(2, 16)]], 19, geometry)
            tokens = torch.arange(1, 20)[None]
            with torch.no_grad():
                result = core(input_ids=tokens, use_cache=True)
                self.assertEqual([result.past_key_values.get_seq_length(i) for i in range(4)], [12] * 4)
                mask = torch.tensor(core.get_clse_stages()[0]["mask"]["keep"])
                removed = (~mask).nonzero().flatten() + 2
                for token in (20, 21, 22):
                    result = core(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
                    tokens = torch.cat((tokens, torch.tensor([[token]])), 1)
                    hidden, ids = reference.embed_tokens(tokens), torch.arange(tokens.shape[1])[None]
                    for index, layer in enumerate(reference.layers):
                        bias = causal_bias(hidden)
                        if index < 2:
                            bias[:, :, 19:, removed] = torch.finfo(hidden.dtype).min
                        hidden = layer(hidden, attention_mask=bias, position_ids=ids,
                                       position_embeddings=reference.rotary_emb(hidden, ids),
                                       use_cache=False, output_attentions=False)[0]
                        if index == 1:
                            selection = torch.cat((torch.arange(2), mask.nonzero().flatten() + 2,
                                                   torch.arange(16, hidden.shape[1])))
                            hidden, ids = hidden[:, selection], ids[:, selection]
                    torch.testing.assert_close(result.last_hidden_state[:, -1], reference.norm(hidden)[:, -1], rtol=1e-5, atol=1e-6)

    def test_all_mode_switches_drop_side_weights_and_checkpoint_signature_is_checked(self):
        core = LlavaQwenModel(tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            make_checkpoint(path, core)
            for mode in ("fastv", "clse", "priortr", "vico", "pyramiddrop"):
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
                kwargs = dict(enabled=False)
                if mode == "clse":
                    kwargs["keep_ratio"] = 1
                if mode in {"vico", "pyramiddrop"}:
                    kwargs.update(layers=(1, 2, 3), keep_ratios=(.8, .5, .2))
                getattr(core, "configure_" + mode)(**kwargs)
                self.assertFalse(core.attention_predictor_enabled)
                self.assertIsNone(core.attention_predictor)
            core.config.image_token_order = "anyres_first"
            with self.assertRaisesRegex(ValueError, "packing/order"):
                core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)

    def test_cache_identity_reuses_only_matching_input_and_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "x.png"
            image.write_bytes(b"image")
            sample = types.SimpleNamespace(image=image)
            example = dict(hidden=torch.ones(1, 5, 32), position_ids=torch.arange(5)[None],
                           spatial_indices=torch.tensor([1, 2]), target=torch.tensor([.5, .5]))
            examples = TeacherExamples(None, teacher_layer=22, cache_dir=root / "cache", identity={"model": "a"})
            with patch("llava_pruning.predictor_training.prepare_teacher_example", return_value=example) as extract:
                examples(sample)
                examples(sample)
                self.assertEqual(extract.call_count, 1)
                examples.identity = {"model": "b"}
                examples(sample)
                self.assertEqual(extract.call_count, 2)

    def test_public_visualization_has_only_two_png_types_and_spatial_probability_scores(self):
        from PIL import Image
        model, features, options = toy_model()
        core = model.get_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root / "best.pt", core)
            (root / "method.json").write_text(json.dumps(dict(checkpoint="best.pt", prune_rates=[30], visualize_rates=[30])), encoding="utf-8")
            method = load_method("attention_predictor", root / "method.json")
            method.configure(core, 30, capture_attention=True, capture_visualization=True)
            with patch.object(model, "encode_images", return_value=features), torch.no_grad():
                model.generate(**options)
            stages = method.visualization_data(core)["stages"]
            scores = stages[0]["attention"]["scores"]
            self.assertAlmostEqual(sum(scores), 1, places=5)
            self.assertEqual(scores[8], 0)
            self.assertEqual(scores[13], 0)
            result = dict(image=Image.new("RGB", (16, 8), "white"), answer="B", stats=method.stats(core, 30),
                          stages=stages, crop_metadata=[dict(mode="anyres_max_9", original_size=[16, 8],
                                                             grid_patches=[2, 1], roi_boxes=[], final_newline=False)])
            tower = types.SimpleNamespace(num_patches_per_side=2, config=types.SimpleNamespace(patch_size=4))
            paths = method.visualize(result=result, vision_tower=tower, output_dir=root / "vis", sample_id="000000108",
                                     save_prune=True, save_attention=True, ground_truth=0)
            folder = Path(paths[0]["directory"])
            self.assertEqual({p.name for p in folder.glob("*.png")}, {"comparison.png", "attention_overlay.png"})
            metadata = json.loads((folder / "decisions.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["stages"][0]["after_layer"], 2)
            self.assertEqual(metadata["stages"][0]["pruned_newline_tokens"], 0)
            self.assertNotIn("scores", metadata["stages"][0])

    def test_real_teacher_preprocessing_packing_uses_generic_prompt_without_gt_or_generation(self):
        from PIL import Image
        from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
        from llava_pruning.prompts import GENERIC_PROMPT_V0
        model, features, options = toy_model()
        model.requires_grad_(False).eval()
        seen = []
        class Tokenizer:
            bos_token_id = 0
            def __call__(self, value):
                seen.append(value)
                return types.SimpleNamespace(input_ids=[1, 3])
        backend = types.SimpleNamespace(model=model, tokenizer=Tokenizer(),
                                        processor=SigLipImageProcessor(size=(8, 8), crop_size={"height": 8, "width": 8}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (16, 8)).save(path)
            with patch.object(model, "encode_images", return_value=features), \
                    patch.object(model, "generate", side_effect=AssertionError("teacher generated an answer")):
                example = prepare_teacher_example(backend, types.SimpleNamespace(image=path), teacher_layer=3)
            self.assertTrue(any(GENERIC_PROMPT_V0 in value for value in seen))
            self.assertEqual(example["target"].shape, (12,))
            self.assertAlmostEqual(float(example["target"].sum()), 1, places=6)
            self.assertFalse(model.get_model()._attention_teacher_capture)

    def test_visa_all_rows_ignores_gt_and_saved_split_is_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "split_csv").mkdir()
            path = root / "split_csv/1cls.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["object", "image", "split", "label", "mask"])
                writer.writeheader()
                for category in ("candle", "capsules"):
                    (root / category).mkdir()
                    for index in range(10):
                        (root / category / f"{index}.png").write_bytes(b"image")
                        writer.writerow(dict(object=category, image=f"{category}/{index}.png", split="train" if index < 5 else "test",
                                             label="DO_NOT_READ_GT", mask="missing.png"))
            samples, metadata = load_visa_distillation(root)
            self.assertEqual(len(samples), 20)
            self.assertTrue(all(s.gt is None and s.mask is None for s in samples))
            train, val = split_samples(samples, seed=42)
            self.assertEqual((len(train), len(val)), (18, 2))
            self.assertEqual(split_samples(samples, seed=42), (train, val))
            self.assertFalse(set(s.image for s in train) & set(s.image for s in val))
            self.assertFalse(metadata["gt_used"])
            self.assertTrue(any("/9.png" in s.sample_id for s in samples))


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flash_attn"),
                     "CUDA + FlashAttention2 required on the inference server")
class PredictorFlashTests(unittest.TestCase):
    def test_native_fp16_backbone_bf16_side_training_and_sparse_cached_decode(self):
        config = tiny_config("flash_attention_2")
        config.torch_dtype = torch.float16
        core = LlavaQwenModel(config).cuda().half().eval().requires_grad_(False)
        inputs = torch.arange(1, 20, device="cuda")[None]
        with torch.no_grad():
            baseline = core(input_ids=inputs, use_cache=True)
            core.configure_attention_predictor(enabled=False, keep_ratio=1)
            zero = core(input_ids=inputs, use_cache=True)
            self.assertTrue(torch.equal(baseline.last_hidden_state, zero.last_hidden_state))
        example = extract_teacher(core, core.embed_tokens(inputs), (2, 16), layout(), teacher_layer=3)
        predictor = student().cuda()
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
        sample = types.SimpleNamespace(sample_id="x")
        metrics = epoch_pass(predictor, lambda _: example, [sample], device="cuda", precision="bfloat16", optimizer=optimizer)
        self.assertTrue(math.isfinite(metrics["kl"]))
        from llava.model.language_model.attention_predictor import student_autocast
        with student_autocast("cuda", "bfloat16"):
            score = predictor(example["hidden"], example["position_ids"], example["spatial_indices"])
        self.assertEqual(score.dtype, torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_student(path, predictor, student_precision="bfloat16", backbone_depth=4, teacher_layer=3,
                         signature=backbone_signature(core.config))
            core.configure_attention_predictor(enabled=True, keep_ratio=.5, checkpoint=path)
            from llava.model.language_model.attention_predictor_crops import dense_crop_provenance
            geometry = layout()
            geometry["predictor_crops"] = dense_crop_provenance(geometry, (2, 1), 2, (2, 4))
            core.set_clse_image_layout([[(2, 16)]], 19, geometry)
            with torch.no_grad():
                result = core(input_ids=inputs, use_cache=True)
                self.assertEqual([result.past_key_values.get_seq_length(i) for i in range(4)], [12] * 4)
                core(input_ids=torch.tensor([[20]], device="cuda"), past_key_values=result.past_key_values, use_cache=True)
        self.assertEqual(core.dtype, torch.float16)
        self.assertEqual(core.config._attn_implementation, "flash_attention_2")


if __name__ == "__main__":
    unittest.main()
