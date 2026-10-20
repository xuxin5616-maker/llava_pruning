"""CPU/mock-model checks; no checkpoint, downloads or real GPU inference."""
import contextlib
import csv
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

import runVisionZip as vz
from llava_pruning.prompts import GENERIC_PROMPT_V0


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_defaults_and_zero_metadata(self):
        args = vz.build_parser().parse_args([])
        self.assertEqual(args.prune_rate, 70)
        self.assertTrue(args.sparse_row_newlines)
        self.assertFalse(vz.build_parser().parse_args(["--no-sparse-row-newlines"]).sparse_row_newlines)
        self.assertTrue(args.save_prune_vis)
        self.assertEqual(vz.build_parser().parse_args(["--save-prune-vis", "False"]).save_prune_vis, False)
        for rate, expected in ((1, (609, 113)), (50, (308, 57)), (70, (185, 34)),
                               (90, (62, 11)), (99, (7, 1))):
            self.assertEqual(vz.token_budget(729, rate / 100), expected)
        self.assertEqual(vz.token_budget(10, .7), (3, 0))
        self.assertEqual(vz.token_budget(729, 0), (729, 0))
        self.assertEqual(vz.source_metadata(0)["mode"], "visionzip_siglip_adaptation")
        for value in (-1, 100, True, .7):
            with self.assertRaises(ValueError):
                vz.validate_prune_rate(value)

    def test_compression_matches_anchor_plus_mean_others(self):
        torch.manual_seed(3)
        features, metric = torch.randn(2, 16, 8), torch.randn(2, 16, 4)
        scores = torch.rand(2, 16)
        output, decisions = vz.compress_tokens(features, scores, metric, 7, 3, capture=True)
        audit = decisions.export()
        for view in range(2):
            dominant = sorted(scores[view].topk(7).indices.tolist())
            remaining = [i for i in range(16) if i not in dominant]
            anchors = remaining[::len(remaining) // 3][:3]
            other = [i for i in remaining if i not in anchors]
            keys = metric[view] / metric[view].norm(dim=-1, keepdim=True)
            assignment = (keys[other] @ keys[anchors].T).argmax(-1)
            expected = [features[view, i] for i in dominant]
            for slot, anchor in enumerate(anchors):
                selected = [token for token, cluster in zip(other, assignment.tolist()) if cluster == slot]
                merged = features[view, selected].mean(0) if selected else torch.zeros(8)
                expected.append(features[view, anchor] + merged)
            torch.testing.assert_close(output[view], torch.stack(expected))
            self.assertEqual(audit[view]["dominant_indices"], dominant)
            self.assertEqual(audit[view]["contextual_indices"], anchors)
            self.assertEqual(len(vz.token_visibility(audit[view], 16)), 16)

        # Disabling capture changes no feature math and keeps no detailed audit.
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("No CPU transfer in compression")):
            plain, compact = vz.compress_tokens(features, scores, metric, 7, 3)
        torch.testing.assert_close(plain, output, rtol=0, atol=0)
        self.assertIsNone(compact.audit)
        with self.assertRaisesRegex(ValueError, "not captured"):
            compact.export()

    def test_packing_uses_global_spatial_order_without_cpu_transfer(self):
        torch.manual_seed(4)
        zipped, decisions = vz.compress_tokens(torch.randn(5, 16, 8), torch.rand(5, 16),
                                                torch.randn(5, 16, 4), 3, 2)
        positions = []
        for tile in range(4):
            row, col = divmod(tile, 2)
            for slot, index in enumerate(decisions.dominant_indices[tile + 1].tolist()):
                pos = (row * 4 + index // 4) * 8 + col * 4 + index % 4
                positions.append((pos, tile + 1, slot))
        expected = torch.cat((zipped[0], torch.stack([zipped[t, s] for _, t, s in sorted(positions)]),
                              zipped[1:, 3:].reshape(-1, 8)))
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("No packing CPU transfer")):
            actual = vz.pack_sparse_views(zipped, decisions, (2, 2), 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (25, 8))

    def test_validation_and_zero_context(self):
        features, scores, metric = torch.ones(1, 4, 2), torch.arange(4.).view(1, 4), torch.ones(1, 4, 2)
        result, decisions = vz.compress_tokens(features, scores, metric, 2, 0, capture=True)
        self.assertEqual(result.shape, (1, 2, 2))
        self.assertEqual(decisions.export()[0]["discarded_indices"], [0, 1])
        for bad in (metric * float("nan"), metric * 0):
            with self.assertRaises(ValueError):
                vz.compress_tokens(features, scores, bad, 1, 1)

    def test_zero_runs_selection_and_returns_exact_order_and_values(self):
        torch.manual_seed(31)
        features, scores, metric = torch.randn(3, 729, 8), torch.rand(3, 729), torch.zeros(3, 729, 4)
        counts = vz.token_budget(729, 0)
        selected_counts = []
        original_topk = torch.Tensor.topk
        def observe_topk(tensor, k, **kwargs):
            selected_counts.append(k)
            return original_topk(tensor, k, **kwargs)
        with patch.object(torch.Tensor, "topk", observe_topk), \
                patch.object(torch, "cat", side_effect=AssertionError("Identity must preserve original storage")):
            output, decisions = vz.compress_tokens(features, scores, metric, *counts, capture=True)
        self.assertEqual(selected_counts, [729])  # Still run scoring-based selection at full budget.
        self.assertIs(output, features)
        self.assertTrue(torch.equal(output, features))
        self.assertEqual(output.dtype, features.dtype)
        self.assertTrue(torch.equal(decisions.dominant_indices, torch.arange(729).expand(3, -1)))
        for view in decisions.export():
            self.assertEqual(view["contextual_indices"], [])
            self.assertEqual(view["merged_indices"], [])
            self.assertTrue((vz.token_visibility(view, 729) == 1).all())

    def test_full_budget_preserves_noncontiguous_storage_and_offset(self):
        # SigLIP residuals can retain [views, channels, tokens] storage with an
        # exposed [views, tokens, channels] shape. Equal values alone are not
        # sufficient: Linear dispatch also depends on input contiguity.
        for dtype in (torch.float32, torch.float16):
            for capture in (False, True):
                with self.subTest(dtype=dtype, capture=capture):
                    features = torch.randn(3, 8, 733).to(dtype).transpose(1, 2)[:, 2:731]
                    self.assertFalse(features.is_contiguous())
                    self.assertGreater(features.storage_offset(), 0)
                    output, decisions = vz.compress_tokens(
                        features, torch.rand(3, 729), torch.zeros(3, 729, 4),
                        729, 0, capture=capture)
                    self.assertIs(output, features)
                    self.assertEqual(output.stride(), features.stride())
                    self.assertEqual(output.storage_offset(), features.storage_offset())
                    self.assertEqual(output.data_ptr(), features.data_ptr())
                    self.assertEqual(decisions.retained_count, 729)
                    self.assertEqual(decisions.audit is not None, capture)

    def test_repacking_preserves_values_dtype_and_matches_reference_memory_order(self):
        source_shape, selected_shape = (2, 11, 8), (2, 7, 8)
        for dtype in (torch.float32, torch.float16):
            for order in itertools.permutations(range(3)):
                with self.subTest(dtype=dtype, order=order):
                    inverse = tuple(order.index(dim) for dim in range(3))
                    reference = torch.empty(tuple(source_shape[dim] for dim in order),
                                            dtype=dtype).permute(inverse)
                    selected = torch.randn(selected_shape).to(dtype)
                    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("No CPU transfer")):
                        actual = vz._match_feature_memory_order(selected, reference)
                    strides, step = [0] * 3, 1
                    for dim in reversed(order):
                        strides[dim] = step
                        step *= selected_shape[dim]
                    self.assertEqual(actual.stride(), tuple(strides))
                    self.assertTrue(torch.equal(actual, selected))
                    self.assertEqual(actual.dtype, selected.dtype)
                    self.assertEqual(actual.device, selected.device)
                    self.assertEqual(actual.untyped_storage().nbytes(),
                                     selected.numel() * selected.element_size())
                    if order == (0, 1, 2):
                        self.assertEqual(actual.data_ptr(), selected.data_ptr())

    def test_nonzero_compression_matches_old_values_and_keeps_siglip_memory_order(self):
        torch.manual_seed(91)
        features = torch.randn(2, 8, 729).transpose(1, 2)
        scores, metric = torch.rand(2, 729), torch.randn(2, 729, 4)
        for rate in (1, 50, 70, 90, 99):
            with self.subTest(rate=rate):
                counts = vz.token_budget(729, rate / 100)
                # Disable only the new layout copy to recover the pre-fix path.
                with patch.object(vz, "_match_feature_memory_order", side_effect=lambda x, reference: x):
                    previous, previous_decisions = vz.compress_tokens(
                        features, scores, metric, *counts, capture=True)
                output, decisions = vz.compress_tokens(features, scores, metric, *counts, capture=True)
                retained = sum(counts)
                self.assertEqual(output.shape, (2, retained, 8))
                self.assertEqual(output.stride(), (8 * retained, 1, retained))
                self.assertFalse(output.is_contiguous())
                self.assertTrue(torch.equal(output, previous))
                self.assertEqual(decisions.export(), previous_decisions.export())

    def test_complete_grid_packing_matches_actual_reference_for_geometry_and_dtypes(self):
        from test_reference import ToyPacking
        from llava.constants import IMAGE_TOKEN_INDEX
        cases = [((2, 1), (16, 8)), ((1, 2), (8, 16)), ((2, 2), (11, 9)),
                 ((2, 2), (9, 11)), ((1, 1), (8, 8)), ((6, 6), (43, 41))]
        for dtype in (torch.float32, torch.float16):
            for grid, size in cases:
                if dtype == torch.float16 and grid == (6, 6):
                    continue  # This CPU Torch cannot bilinear-resize Half; CUDA checked on server.
                for merge in ("spatial_unpad", "spatial_unpad_add_newl"):
                    with self.subTest(dtype=dtype, grid=grid, size=size, merge=merge):
                        toy = ToyPacking(dtype)
                        toy.config.image_grid_pinpoints = [[grid[0] * 8, grid[1] * 8]]
                        toy.config.mm_patch_merge_type = merge
                        views = 1 + grid[0] * grid[1]
                        features = torch.arange(views * 4 * 4).reshape(views, 4, 4).to(dtype) / 20
                        # The complete selector's row-major output; no Half topk needed for packing test.
                        decisions = vz.TokenDecisions(4, 4, torch.arange(4).expand(views, -1))
                        packed, count = vz.pack_full_views(features, decisions, grid, size,
                                                         toy.core.image_newline, merge)
                        ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                        with patch.object(toy, "encode_images", return_value=features):
                            reference = toy.prepare_inputs_labels_for_multimodal(
                                ids, None, None, None, None, torch.zeros(1, views, 3, 8, 8), image_sizes=[size])
                        self.assertTrue(torch.equal(packed, reference[4][0, 1:-2]))
                        self.assertEqual(packed.shape[0], vz.baseline_packed_count(size, grid, 2, merge))
                        self.assertGreater(count, 0)

    def test_packed_rate_is_not_raw_encoder_rate(self):
        before = vz.baseline_packed_count((1000, 600), (3, 2), 27, "spatial_unpad")
        self.assertEqual(before, 4665)
        self.assertGreater(7 * sum(vz.token_budget(729, .01)), before)
        after = 7 * sum(vz.token_budget(729, .7))
        self.assertAlmostEqual(100 * (1 - after / before), 67.13826366559485)

    def test_encoder_capture_keeps_dtype_and_cleans_hooks(self):
        class Attention(torch.nn.Module):
            num_heads, head_dim = 2, 4

            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(8, 8, bias=False)
                self.fail = False

            def forward(self, hidden):
                key = self.k_proj(hidden)
                if self.fail:
                    raise RuntimeError("synthetic encoder failure")
                weights = torch.softmax(torch.ones(hidden.shape[0], 2, 729, 729), -1)
                return key, weights

        attention = Attention()
        hidden = torch.randn(2, 729, 8)

        class Tower:
            num_patches = 729
            vision_tower = types.SimpleNamespace(vision_model=types.SimpleNamespace(
                encoder=types.SimpleNamespace(layers=[None] * 25 + [types.SimpleNamespace(self_attn=attention)])))

            def __call__(self, images):
                return attention(hidden)[0]

        core = types.SimpleNamespace(get_vision_tower=lambda: Tower(), training=False,
                                     config=types.SimpleNamespace(image_aspect_ratio="anyres_max_9",
                                                                  mm_patch_merge_type="spatial_unpad"),
                                     mm_projector=torch.nn.Identity())
        adapter = vz.VisionZipAdapter(core)
        adapter.configure(70)
        result = adapter.encode_images(torch.zeros(2, 3, 1, 1))
        self.assertEqual(result.shape, (2, 219, 8))
        self.assertEqual(result.dtype, hidden.dtype)
        self.assertIsNone(adapter.decisions.audit)
        adapter.configure(0)
        with patch.object(vz, "compress_tokens", wraps=vz.compress_tokens) as selector:
            zero_result = adapter.encode_images(torch.zeros(2, 3, 1, 1))
        selector.assert_called_once()
        self.assertEqual(selector.call_args.args[-2:], (729, 0))
        self.assertTrue(torch.equal(zero_result, attention(hidden)[0]))
        for module in (attention, attention.k_proj):
            self.assertFalse(module._forward_hooks)
        attention.fail = True
        with self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
            adapter.encode_images(torch.zeros(2, 3, 1, 1))
        for module in (attention, attention.k_proj):
            self.assertFalse(module._forward_hooks)


class PrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self):
        class Model:
            training = False
            device = torch.device("cpu")
            config = types.SimpleNamespace(tokenizer_model_max_length=32)

            def __init__(self):
                self.core = types.SimpleNamespace(embed_tokens=torch.nn.Embedding(10, 4))

            def get_model(self):
                return self.core

            def prepare_inputs_labels_for_multimodal(self, *args, **kwargs):
                return ("ORIGINAL", args, kwargs)

        return Model()

    def test_zero_uses_adapter_never_original_prefill(self):
        from llava.constants import IMAGE_TOKEN_INDEX
        model = self.model()
        original = Mock(side_effect=AssertionError("Original prefill must not be used"))
        model.prepare_inputs_labels_for_multimodal = original
        adapter = types.SimpleNamespace(rate=0, encode_images=Mock(return_value="selected"),
                                        pack=Mock(return_value=[torch.zeros(3, 4)]))
        with vz.visionzip_generation(model, adapter):
            result = model.prepare_inputs_labels_for_multimodal(
                torch.tensor([[1, IMAGE_TOKEN_INDEX, 2]]), None, None, None, None,
                [torch.zeros(2, 3, 8, 8)], image_sizes=[(8, 8)])
        self.assertEqual(result[4].shape, (1, 5, 4))
        adapter.encode_images.assert_called_once()
        adapter.pack.assert_called_once()
        original.assert_not_called()
        self.assertIs(model.prepare_inputs_labels_for_multimodal, original)

    def test_prefill_splices_visual_only_and_restores(self):
        from llava.constants import IMAGE_TOKEN_INDEX
        model = self.model()
        visual = torch.ones(3, 4)
        adapter = types.SimpleNamespace(enabled=True, encode_images=Mock(return_value="encoded"),
                                        pack=Mock(return_value=[visual]))
        ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        with vz.visionzip_generation(model, adapter):
            result = model.prepare_inputs_labels_for_multimodal(
                ids, None, None, None, None, [torch.zeros(2, 3, 2, 2)], image_sizes=[(8, 8)])
            self.assertIsNone(result[1])
            self.assertIsNone(result[2])
            expected = torch.cat((model.core.embed_tokens(ids[0, :1]), visual,
                                  model.core.embed_tokens(ids[0, 2:])))
            torch.testing.assert_close(result[4][0], expected)
            decoded = model.prepare_inputs_labels_for_multimodal(
                torch.tensor([[2]]), None, None, "cache", None, None)
            self.assertTrue(torch.equal(decoded[0], torch.tensor([[2]])))
            self.assertEqual(decoded[3], "cache")
            self.assertIsNone(decoded[4])
        self.assertNotIn("prepare_inputs_labels_for_multimodal", model.__dict__)
        adapter.encode_images.assert_called_once()

    def test_exception_restores_existing_local_method(self):
        model = self.model()
        original = Mock(return_value="original override")
        model.prepare_inputs_labels_for_multimodal = original
        with self.assertRaises(RuntimeError):
            with vz.visionzip_generation(model, types.SimpleNamespace(enabled=True)):
                raise RuntimeError("fail")
        self.assertIs(model.prepare_inputs_labels_for_multimodal, original)

    def test_real_tiny_qwen_baseline_and_sparse_cached_generation(self):
        from transformers import StoppingCriteriaList
        from test_fastv_attention import tiny_config
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava_pruning.benchmark_timing import FirstTokenObserver
        torch.manual_seed(19)
        config = tiny_config("sdpa")  # CPU test only; production loader stays FlashAttention2.
        config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
        config.mm_patch_merge_type, config.image_aspect_ratio = "spatial_unpad", "anyres_max_9"
        config.image_grid_pinpoints = [[8, 8], [16, 8]]
        model = LlavaQwenForCausalLM(config).eval()
        model.get_model().configure_fastv(enabled=False)
        model.get_model().image_newline = torch.nn.Parameter(torch.zeros(config.hidden_size))
        tower = types.SimpleNamespace(num_patches_per_side=2, image_size=8)
        features = torch.randn(3, 4, config.hidden_size)
        ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        options = dict(inputs=ids, images=torch.zeros(1, 3, 3, 8, 8), image_sizes=[(16, 8)],
                       do_sample=False, max_new_tokens=3, use_cache=True)
        with patch.object(model.get_model(), "get_vision_tower", return_value=tower), \
                patch.object(model, "encode_images", return_value=features), torch.inference_mode():
            before = model.generate(**options)
            full_adapter = vz.VisionZipAdapter(model.get_model())
            full_adapter.rate = 0
            def encode_all(crops):
                output, full_adapter.decisions = vz.compress_tokens(
                    features, torch.ones(3, 4), torch.ones(3, 4, 2), *vz.token_budget(4, 0))
                full_adapter.dominant_count = 4
                return output
            with patch.object(full_adapter, "encode_images", side_effect=encode_all) as encode, \
                    patch.object(model, "prepare_inputs_labels_for_multimodal",
                                 side_effect=AssertionError("No original prefill")) as original, \
                    vz.visionzip_generation(model, full_adapter):
                after = model.generate(**options)
            encode.assert_called_once()
            original.assert_not_called()
            self.assertTrue(torch.equal(before, after))
            self.assertTrue(full_adapter.stats["full_grid_identity"])

            sparse = torch.randn(5, config.hidden_size)
            adapter = types.SimpleNamespace(enabled=True, encode_images=Mock(return_value="encoded"),
                                            pack=Mock(return_value=[sparse]))
            observer = FirstTokenObserver(lambda: None, lambda: 1.0)
            observer.start = 0.0
            with vz.visionzip_generation(model, adapter):
                result = model.generate(**options, stopping_criteria=StoppingCriteriaList([observer]))
            manual = torch.cat((model.get_model().embed_tokens(ids[0, :1]), sparse,
                                model.get_model().embed_tokens(ids[0, 2:])))
            expected = super(LlavaQwenForCausalLM, model).generate(
                inputs_embeds=manual[None], do_sample=False, max_new_tokens=3, use_cache=True)
            self.assertTrue(torch.equal(result, expected))
            self.assertEqual(observer.tokens, 3)
            self.assertEqual(observer.latency, 1.0)
            adapter.encode_images.assert_called_once()
            self.assertNotIn("prepare_inputs_labels_for_multimodal", model.__dict__)


class RunnerTests(unittest.TestCase):
    def fixture(self, root, rate=70, count=103, vis=False):
        Image.new("RGB", (8, 8)).save(root / "image.png")
        source = root / "questions.jsonl"
        source.write_text("".join(json.dumps({
            "question_id": str(i), "image": "image.png", "gt": i % 2,
            "origin_path": "screw/test/good/one.png",
        }) + "\n" for i in range(count)), encoding="utf-8")
        return vz.build_parser().parse_args([
            "--input-json", str(source), "--data-root", str(root), "--prune-rate", str(rate),
            "--output-dir", str(root / f"out_{rate}"), "--save-prune-vis", str(vis),
        ])

    @contextlib.contextmanager
    def fake_cuda(self):
        with patch.object(vz, "validate_single_gpu"), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=1), \
                patch("torch.cuda.synchronize"), \
                patch("torch.cuda.get_device_properties",
                      return_value=types.SimpleNamespace(total_memory=10 * 1024**3)):
            yield

    def backend(self, calls, fail_on=None, fail_after=0):
        class Backend:
            def __init__(self, *args, **kwargs):
                self.vision_tower = types.SimpleNamespace(
                    num_patches=729,
                    vision_tower=types.SimpleNamespace(vision_model=types.SimpleNamespace(
                        encoder=types.SimpleNamespace(layers=[None] * 26))))
                self.core = types.SimpleNamespace(
                    get_vision_tower=lambda: self.vision_tower, configure_fastv=Mock(),
                    config=types.SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad"))
                self.model = types.SimpleNamespace(
                    get_model=lambda: self.core, prepare_inputs_labels_for_multimodal=Mock())
                self.inference_config = {"attention": "fake_test", "vision_dtype": "fake_test"}

            def generate(self, sample, prompt, method, rate, mode, **kwargs):
                method.configure(self.core, rate, capture_attention=False, capture_visualization=False)
                calls.append((sample.sample_id, rate, kwargs, method.adapter.capture_decisions))
                if len(calls) > fail_after and sample.sample_id == fail_on:
                    raise RuntimeError("synthetic generation error")
                stats = method.stats(self.core, rate)
                dominant, contextual = vz.token_budget(729, rate / 100)
                stats.update(tokens_per_view_before=729, tokens_per_view_after=dominant + contextual,
                             dominant_per_view=dominant, contextual_per_view=contextual)
                return dict(answer="unknown" if sample.sample_id == "0" else ("A" if sample.gt else "B"),
                            lat_seconds=.1, generation_seconds=.2, generated_tokens=3,
                            peak_allocated_bytes=1024**3, peak_reserved_bytes=2 * 1024**3,
                            stats=stats)
        return Backend

    def test_full_dataset_warmup_baseline_and_metrics(self):
        for rate in (0, 70):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                root, calls = Path(directory).resolve(), []
                args = self.fixture(root, rate)
                with self.fake_cuda(), contextlib.redirect_stdout(io.StringIO()):
                    output = vz.run_experiment(args, backend_factory=self.backend(calls))
                report = json.loads((output / f"prune_{rate:02d}" / "metrics.json").read_text())
                self.assertEqual(len(calls), 113)
                self.assertTrue(all(row[2]["benchmark_metrics"] for row in calls[:10]))
                self.assertTrue(all(row[2]["benchmark_metrics"] for row in calls[10:]))
                self.assertTrue(all(not row[3] for row in calls))
                self.assertEqual(report["evaluated_samples"], 103)
                self.assertEqual(report["warmup_samples"], 10)
                self.assertEqual(report["accuracy"], 102 / 103)
                self.assertEqual(report["unparsed"], 1)
                self.assertEqual((report["precision"], report["recall"], report["tnr"]), (1, 1, 1))
                self.assertEqual(report["lat_mean_ms"], 100)
                self.assertEqual(report["mem_peak_allocated_pct"], 10)
                self.assertTrue(report["complete"])
                self.assertEqual(report["status"], "success")
                self.assertFalse(list(output.rglob("*.png")))
                for _, _, options, _ in calls:
                    self.assertFalse(options["do_sample"])
                    self.assertFalse(options["capture_visualization"])
                    self.assertFalse(options["capture_attention"])
                    self.assertTrue(options["include_pruning_time"])
                with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 1)
                with self.fake_cuda(), self.assertRaises(FileExistsError):
                    vz.run_experiment(args)

    def test_sparse_row_newline_switch_reaches_adapter_and_saved_config(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root, calls, observed = Path(directory).resolve(), [], []
                args = self.fixture(root, rate=1, count=2)
                args.sparse_row_newlines = enabled
                parent = self.backend(calls)

                class ObserveBackend(parent):
                    def generate(self, sample, prompt, method, *positional, **kwargs):
                        observed.append(method.adapter.sparse_row_newlines)
                        result = super().generate(sample, prompt, method, *positional, **kwargs)
                        result["stats"]["newline_tokens_after"] = 27 if enabled else 0
                        return result

                with self.fake_cuda(), contextlib.redirect_stdout(io.StringIO()):
                    output = vz.run_experiment(args, backend_factory=ObserveBackend)
                self.assertEqual(observed, [enabled] * 4)  # two warmups + all two samples
                meta = json.loads((output / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["sparse_row_newlines"], enabled)
                self.assertEqual(meta["method_details"]["sparse_row_newlines"], enabled)
                with (output / "prune_01/classification_results.csv").open(encoding="utf-8-sig") as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual([int(row["newline_tokens_after"]) for row in records],
                                 [27 if enabled else 0] * 2)

    def test_visualization_is_after_generate_including_zero_but_not_warmup(self):
        for rate in (0, 70):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                root, calls, rendered = Path(directory).resolve(), [], []
                args = self.fixture(root, rate, count=2, vis=True)
                def render(backend, method, sample, result, output):
                    self.assertEqual(calls[-1][0], sample.sample_id)
                    self.assertTrue(calls[-1][2]["benchmark_metrics"])
                    rendered.append(sample.sample_id)
                with self.fake_cuda(), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(vz, "save_sample_visualization", side_effect=render):
                    vz.run_experiment(args, backend_factory=self.backend(calls))
                self.assertEqual(rendered, ["0", "1"])
                self.assertTrue(all(not row[3] for row in calls[:2]))
                self.assertTrue(all(row[3] for row in calls[2:]))

    def test_failed_run_keeps_partial_results_and_failed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory).resolve(), []
            args = self.fixture(root, 70, count=3)
            with self.fake_cuda(), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "synthetic generation error"):
                vz.run_experiment(args, backend_factory=self.backend(calls, fail_on="1", fail_after=3))
            report = json.loads((root / "out_70" / "prune_70" / "metrics.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["complete"])
            self.assertEqual(report["evaluated_samples"], 1)
            self.assertIn("synthetic generation error", report["error"])

    def test_visa_uses_test_csv_and_generic_v0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "candle").mkdir()
            for name in ("normal", "anomaly", "train"):
                Image.new("RGB", (8, 8)).save(root / "candle" / f"{name}.png")
            split = root / "split_csv"
            split.mkdir()
            (split / "1cls.csv").write_text(
                "object,split,label,image,mask\n"
                "candle,test,normal,candle/normal.png,\n"
                "candle,test,anomaly,candle/anomaly.png,\n"
                "candle,train,normal,candle/train.png,\n", encoding="utf-8")
            args = vz.build_parser().parse_args(["--dataset", "visa", "--data-root", str(root)])
            samples, prompts, spec = vz.prepare_inputs(args)
            self.assertEqual(len(samples), 2)
            self.assertEqual(sorted(sample.gt for sample in samples), [0, 1])
            self.assertEqual(prompts, [GENERIC_PROMPT_V0] * 2)
            self.assertEqual(spec["prompt_version"], "v0_generic")
            args.visa_split_csv = str(root / "missing.csv")
            with self.assertRaises(FileNotFoundError):
                vz.prepare_inputs(args)

    def test_bad_preflight_does_not_load_model_or_make_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = self.fixture(root, 70, count=1)
            args.warmup = 0
            factory = Mock()
            with self.assertRaisesRegex(ValueError, "warmup"):
                vz.run_experiment(args, backend_factory=factory)
            factory.assert_not_called()
            self.assertFalse(args.output_dir.exists())


class ZeroVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def backend(self):
        from test_fastv_attention import tiny_config
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention, SigLipVisionConfig
        torch.manual_seed(73)

        class TinyTower(torch.nn.Module):
            """Real native SigLIP attention on synthetic patch features; no checkpoint."""
            num_patches, num_patches_per_side, image_size = 729, 27, 384

            def __init__(self):
                super().__init__()
                self.attention = SigLipAttention(SigLipVisionConfig(hidden_size=8, num_attention_heads=2))
                self.vision_tower = types.SimpleNamespace(vision_model=types.SimpleNamespace(
                    encoder=types.SimpleNamespace(layers=[None] * 25 + [types.SimpleNamespace(self_attn=self.attention)])))
                self.register_buffer("features", torch.randn(1, 729, 8))

            def forward(self, pixels):
                hidden = self.features.expand(pixels.shape[0], -1, -1)
                # Match SigLIP's noncontiguous embedding/residual layout rather
                # than testing only contiguous attention outputs.
                hidden = hidden.transpose(1, 2).contiguous().transpose(1, 2)
                return hidden + self.attention(hidden)[0]

        config = tiny_config("sdpa")
        config.num_hidden_layers = 1
        config.max_position_embeddings = config.tokenizer_model_max_length = 4096
        config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
        config.mm_patch_merge_type, config.image_aspect_ratio = "spatial_unpad", "anyres_max_9"
        config.image_grid_pinpoints = [[384, 384]]
        model = LlavaQwenForCausalLM(config)
        core = model.get_model()
        core.vision_tower = TinyTower()
        core.mm_projector = torch.nn.Sequential(
            torch.nn.Linear(8, config.hidden_size), torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size))
        core.image_newline = torch.nn.Parameter(torch.randn(config.hidden_size))
        model.eval()

        class Backend:
            inference_config = {"attention": "sdpa_cpu_test_only", "vision_dtype": "torch.float32"}

            def __init__(self):
                self.model = model
                self.vision_tower = core.vision_tower
                self.calls = []

            def generate(self, sample, prompt, method, rate, mode, **options):
                self.calls.append(type(method).__name__)
                method.configure(core, rate)
                with torch.inference_mode():
                    generated = model.generate(
                        inputs=torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]]),
                        images=torch.zeros(1, 2, 3, 8, 8), image_sizes=[(384, 384)],
                        do_sample=False, max_new_tokens=3, use_cache=True)
                return {"answer": str(generated.tolist()), "stats": method.stats(core, rate)}

        return Backend()

    def test_real_attention_and_qwen_exact_zero_pipeline_and_trace(self):
        backend = self.backend()
        method = vz.SingleFileVisionZip(backend.model.get_model())
        samples = [types.SimpleNamespace(sample_id=str(i)) for i in range(2)]
        original_prepare = backend.model.prepare_inputs_labels_for_multimodal
        original_generate = backend.model.generate
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(vz, "compress_tokens", wraps=vz.compress_tokens) as select:
            result = vz.verify_zero_equivalence(backend, method, samples, ["test"] * 2, 42, directory)
            rows = [json.loads(line) for line in
                    (Path(directory) / "zero_equivalence.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["matched_samples"], 2)
        self.assertTrue(result["complete"])
        self.assertEqual(select.call_count, 2)  # Candidate runs the selector; reference does not.
        self.assertEqual(backend.calls, ["SingleFileVisionZip", "ReferenceOnlyMethod"] * 2)
        self.assertTrue(all(len(row["stages"]) == len(vz.TRACE_FIELDS) for row in rows))
        self.assertTrue(all(all(stage["equal"] for stage in row["stages"].values()) for row in rows))
        for row in rows:
            selected = row["stages"]["selected_features"]
            self.assertFalse(selected["reference_layout"]["is_contiguous"])
            self.assertFalse(selected["candidate_layout"]["is_contiguous"])
            self.assertTrue(selected["layout_equal"])
        self.assertEqual(backend.model.prepare_inputs_labels_for_multimodal, original_prepare)
        self.assertEqual(backend.model.generate, original_generate)
        self.assertFalse(backend.model.get_model().mm_projector._forward_hooks)

    def test_verifier_detects_tensor_change_even_if_answer_happens_to_match(self):
        backend = self.backend()
        method = vz.SingleFileVisionZip(backend.model.get_model())
        pack = vz.pack_full_views
        def altered(*args, **kwargs):
            result, count = pack(*args, **kwargs)
            result[0, 0] += 0.01
            return result, count
        with tempfile.TemporaryDirectory() as directory, patch.object(vz, "pack_full_views", side_effect=altered), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Zero-rate mismatch"):
                vz.verify_zero_equivalence(backend, method, [types.SimpleNamespace(sample_id="mismatch")],
                                           ["test"], 42, directory)
            report = json.loads((Path(directory) / "zero_equivalence.json").read_text())
            row = json.loads((Path(directory) / "zero_equivalence.jsonl").read_text())
            self.assertFalse(row["stages"]["llm_inputs_embeds"]["equal"])
            self.assertTrue(row["stages"]["projected_features"]["equal"])
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["complete"])
            self.assertEqual(report["matched_samples"], 0)
        self.assertNotIn("prepare_inputs_labels_for_multimodal", backend.model.__dict__)
        self.assertNotIn("generate", backend.model.__dict__)
        self.assertFalse(backend.model.get_model().mm_projector._forward_hooks)

    def test_one_percent_runs_sparse_prefill_with_original_feature_memory_order(self):
        backend = self.backend()
        method = vz.SingleFileVisionZip(backend.model.get_model())
        trace = {}
        result = vz.generate_sample(backend, method, types.SimpleNamespace(sample_id="one_percent"),
                                    "test", 1, 42, measure=False, capture=False, trace=trace)
        layout = trace["tensor_layouts"]["selected_features"]
        self.assertEqual(trace["selected_features"].shape, (2, 722, 8))
        self.assertEqual(layout["stride"], [8 * 722, 1, 722])
        self.assertFalse(layout["is_contiguous"])
        self.assertEqual(result["stats"]["tokens_per_view_after"], 722)
        self.assertEqual(result["stats"]["final_llm_visual_tokens"], 1471)
        self.assertEqual(result["stats"]["newline_tokens_after"], 27)
        self.assertFalse(result["stats"]["full_grid_identity"])
        self.assertGreater(trace["generated_token_ids"].numel(), 0)

        legacy_method = vz.SingleFileVisionZip(backend.model.get_model(), sparse_row_newlines=False)
        legacy_trace = {}
        legacy = vz.generate_sample(backend, legacy_method, types.SimpleNamespace(sample_id="legacy"),
                                    "test", 1, 42, measure=False, capture=False, trace=legacy_trace)
        self.assertEqual(legacy["stats"]["final_llm_visual_tokens"], 1444)
        self.assertEqual(legacy["stats"]["newline_tokens_after"], 0)
        for field in ("selected_features", "projected_features", "pixels", "input_ids"):
            self.assertTrue(torch.equal(trace[field], legacy_trace[field]), field)

    def test_missing_shape_or_dtype_difference_cannot_pass(self):
        base = {key: torch.ones(2) for key in vz.TRACE_FIELDS}
        for other in ({}, {**base, "input_ids": torch.ones(3)},
                      {**base, "input_ids": torch.ones(2, dtype=torch.float64)}):
            self.assertFalse(all(row["equal"] for row in vz.compare_traces(base, other).values()))

    def test_trace_records_original_layout_before_cpu_snapshot(self):
        features = torch.arange(24.).view(2, 3, 4).transpose(1, 2)
        projector = torch.nn.Linear(3, 5)
        model = types.SimpleNamespace(
            get_model=lambda: types.SimpleNamespace(mm_projector=projector),
            prepare_inputs_labels_for_multimodal=Mock(), generate=Mock())
        reference, candidate = {}, {}
        with vz.capture_generation_trace(model, reference):
            projector(features)
        with vz.capture_generation_trace(model, candidate):
            projector(features.contiguous())
        stages = vz.compare_traces(reference, candidate)
        selected = stages["selected_features"]
        self.assertTrue(selected["equal"])  # Layout is diagnostic, not a new tolerance rule.
        self.assertFalse(selected["layout_equal"])
        self.assertEqual(selected["reference_layout"]["stride"], [12, 1, 4])
        self.assertEqual(selected["candidate_layout"]["stride"], [12, 3, 1])
        self.assertFalse(projector._forward_hooks)

    def test_verification_mode_writes_no_benchmark_report(self):
        backend = self.backend()
        fixtures = RunnerTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = fixtures.fixture(root, 0, count=1)
            args.verify_zero = True
            with fixtures.fake_cuda(), contextlib.redirect_stdout(io.StringIO()):
                result = vz.run_experiment(args, backend_factory=lambda *a, **kw: backend)
            self.assertTrue((result / "zero_equivalence.json").is_file())
            self.assertFalse((result / "summary.csv").exists())
            self.assertFalse(list(result.rglob("*.png")))
            self.assertEqual(len(backend.calls), 2)
            args.prune_rate = 70
            with self.assertRaisesRegex(ValueError, "requires --prune-rate 0"):
                vz.prepare_inputs(args)


if __name__ == "__main__":
    unittest.main()
