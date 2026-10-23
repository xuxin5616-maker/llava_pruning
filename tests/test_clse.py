"""CLSE pinned scoring oracle, actual packing, physical drop, native baseline."""

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Model
from transformers.cache_utils import DynamicCache

from test_fastv_attention import tiny_config
from test_priortr import causal_bias
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_qwen import LlavaQwenModel, LlavaQwenForCausalLM
from llava.model.language_model.clse import spatial_spectral_energy, evolution_scores, select_joint_tokens
from llava.model.language_model.fastv_budget import base_anyres_groups
from llava_pruning.methods import load_method


PROJECT = Path(__file__).resolve().parents[1]


def oracle():
    spec = importlib.util.spec_from_file_location("_clse_oracle", PROJECT / "third_party/clse/tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layout(base=2, rows=2, cols=4, order="base_first", extra=False):
    item = base_anyres_groups(base ** 2, rows, cols, order=order, final_newline=extra)
    item["grids"] = [dict(group, rows=base if group["name"] == "base" else rows,
                          cols=base if group["name"] == "base" else cols) for group in item["groups"]]
    return item


def configured_core(implementation="sdpa", keep=.5, dtype=torch.float32):
    core = LlavaQwenModel(tiny_config(implementation)).to(dtype).eval()
    core.configure_clse(enabled=True, keep_ratio=keep, capture_attention=True, capture_visualization=True)
    grid = layout()
    core.set_clse_image_layout([[(2, 16)]], 19, grid)
    return core


def toy_model(size=(16, 8), grid=(16, 8), order="base_first", extra=False, *, dtype=torch.float32):
    config = tiny_config("sdpa")
    config.eos_token_id, config.pad_token_id = None, 0
    config.image_aspect_ratio, config.mm_patch_merge_type = "anyres_max_9", "spatial_unpad"
    if extra:
        config.mm_patch_merge_type += "_add_newl"
    config.image_grid_pinpoints, config.image_token_order = [list(grid)], order
    config.tokenizer_model_max_length, config.tokenizer_padding_side = 4096, "right"
    config.use_pos_skipping = False
    model = LlavaQwenForCausalLM(config).to(dtype).eval()
    core = model.get_model()
    core.vision_tower = types.SimpleNamespace(num_patches_per_side=2, image_size=8)
    core.image_newline = torch.nn.Parameter(torch.full((32,), -7., dtype=dtype))
    views = 1 + grid[0] // 8 * (grid[1] // 8)
    features = torch.randn(views, 4, 32, dtype=dtype)
    options = dict(inputs=torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3, 4]]),
                   images=[torch.zeros(views, 3, 8, 8, dtype=dtype)], image_sizes=[size],
                   do_sample=False, max_new_tokens=4, use_cache=True)
    return model, features, options


class CLSETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_chunked_fft_matches_pinned_source_on_square_and_rectangular_grids(self):
        torch.manual_seed(790)
        upstream = oracle()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows, cols in ((2, 4), (7, 5), (27, 27)):
                hidden = torch.randn(1, rows * cols + 3, 13).to(dtype)
                indices = list(range(1, rows * cols + 1))
                before = hidden.clone()
                expected = upstream.spatial_spectral_score_2d(hidden[:, indices], t=1, h=rows, w=cols)[0]
                for chunk in (1, 5, 128):
                    actual = spatial_spectral_energy(hidden, indices, rows, cols, channel_chunk=chunk)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=3e-7)
                self.assertTrue(torch.equal(before, hidden))

    def test_evolution_matches_upstream_pure_clse_and_is_joint(self):
        torch.manual_seed(791)
        hidden, after = torch.randn(1, 24, 8), torch.randn(1, 24, 8)
        upstream = oracle()
        expected = upstream.calculate_evolution_score(hidden, after, None, (1, 4, 6), "clse")[0]
        reference = spatial_spectral_energy(hidden, list(range(24)), 4, 6)
        current = spatial_spectral_energy(after, list(range(24)), 4, 6)
        torch.testing.assert_close(evolution_scores(reference, current), expected, rtol=1e-5, atol=2e-6)
        zero = evolution_scores(reference, reference)
        self.assertTrue(torch.equal(zero, torch.full_like(zero, .5)))

    def test_joint_topk_can_choose_only_anyres_and_always_protects_newlines(self):
        grids = layout()
        indices = [i for group in grids["grids"] for i in group["indices"]]
        scores = torch.tensor([0.] * 4 + list(range(1, 9)))
        selected, target, protected = select_joint_tokens(scores, indices, 14, .5)
        self.assertEqual(target, 7)
        self.assertEqual(protected, 2)
        self.assertFalse(any(i in selected.tolist() for i in grids["groups"][0]["indices"]))
        self.assertTrue(all(i in selected.tolist() for i in (8, 13)))
        with self.assertRaisesRegex(ValueError, "structural"):
            select_joint_tokens(scores, indices, 14, .1)

    def test_actual_packer_geometry_after_unpad_max9_and_token_order(self):
        for size, grid in (((16, 8), (16, 8)), ((16, 8), (16, 16)), ((32, 32), (32, 32))):
            for order in ("base_first", "anyres_first"):
                for extra in (False, True):
                    model, features, options = toy_model(size, grid, order, extra)
                    core = model.get_model()
                    core.configure_clse(enabled=True, keep_ratio=.5)
                    with patch.object(model, "encode_images", return_value=features):
                        packed = model.prepare_inputs_labels_for_multimodal(
                            options["inputs"], None, None, None, None, options["images"], image_sizes=[size])[4]
                    [(start, end)] = core._clse_spans[0]
                    meta = core._clse_layout
                    self.assertEqual(meta["packed_tokens"], end - start)
                    indices = [i for group in meta["grids"] for i in group["indices"]]
                    newline = [i for i in range(end - start) if i not in indices]
                    self.assertTrue(torch.equal(packed[0, torch.tensor(newline) + start],
                                                core.image_newline[None].expand(len(newline), -1)))
                    for group in meta["grids"]:
                        self.assertEqual(len(group["indices"]), group["rows"] * group["cols"])
                    if grid == (32, 32):
                        self.assertEqual((meta["grids"][1]["rows"], meta["grids"][1]["cols"]), (6, 6))
                    if size == (16, 8) and grid == (16, 16):
                        self.assertEqual((meta["grids"][1]["rows"], meta["grids"][1]["cols"]), (2, 4))

    def test_real_decoder_scores_only_after_first_block_compacts_all_caches_and_preserves_ids(self):
        for implementation in ("sdpa", "eager"):
            torch.manual_seed(792)
            core = configured_core(implementation)
            observed = []
            handles = [layer.register_forward_pre_hook(
                lambda layer, args, kwargs: observed.append((kwargs["position_ids"].clone(), kwargs["output_attentions"])),
                with_kwargs=True) for layer in core.layers]
            try:
                with torch.no_grad(), patch("llava.model.language_model.clse.evolution_scores", wraps=evolution_scores) as score:
                    result = core(input_ids=torch.arange(1, 20)[None], use_cache=True)
                    self.assertEqual(score.call_count, 1)
                    self.assertEqual(result.last_hidden_state.shape[1], 12)
                    self.assertEqual([result.past_key_values.get_seq_length(i) for i in range(4)], [12] * 4)
                    stats = core.get_clse_stats()
                    self.assertEqual([r["image_tokens_in"] for r in stats["layers"]], [14, 7, 7, 7])
                    self.assertEqual(stats["protected_newline_tokens"], 2)
                    stage = core.get_clse_stages()[0]
                    keep = torch.tensor(stage["mask"]["keep"])
                    expected_ids = torch.cat((torch.arange(2), keep.nonzero().flatten() + 2, torch.arange(16, 19)))[None]
                    self.assertTrue(torch.equal(observed[0][0], torch.arange(19)[None]))
                    self.assertTrue(all(torch.equal(ids, expected_ids) for ids, _ in observed[1:]))
                    for token, position in ((20, 19), (21, 20), (22, 21)):
                        observed.clear()
                        result = core(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
                        self.assertTrue(all(ids.tolist() == [[position]] for ids, _ in observed))
                        self.assertTrue(all(not requested for _, requested in observed))
                    self.assertEqual(score.call_count, 1)
                    self.assertEqual(core.config._attn_implementation, implementation)
            finally:
                for handle in handles:
                    handle.remove()

    def test_cached_decode_matches_independent_full_prefix_with_frozen_selection(self):
        torch.manual_seed(793)
        core = configured_core()
        native = Qwen2Model(tiny_config("eager")).eval()
        native.load_state_dict(core.state_dict())
        tokens = torch.arange(1, 20)[None]
        with torch.no_grad():
            result = core(input_ids=tokens, use_cache=True)
            mask = torch.tensor(core.get_clse_stages()[0]["mask"]["keep"])
            removed = (~mask).nonzero().flatten() + 2
            for token in (20, 21, 22):
                result = core(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
                tokens = torch.cat((tokens, torch.tensor([[token]])), 1)
                hidden = native.embed_tokens(tokens)
                ids = torch.arange(tokens.shape[1])[None]
                for index, layer in enumerate(native.layers):
                    bias = causal_bias(hidden)
                    if index == 0:
                        bias[:, :, 19:, removed] = torch.finfo(hidden.dtype).min
                    hidden = layer(hidden, attention_mask=bias, position_ids=ids,
                                   position_embeddings=native.rotary_emb(hidden, ids),
                                   use_cache=False, output_attentions=False)[0]
                    if index == 0:
                        selection = torch.cat((torch.arange(2), mask.nonzero().flatten() + 2, torch.arange(16, hidden.shape[1])))
                        hidden, ids = hidden[:, selection], ids[:, selection]
                torch.testing.assert_close(result.last_hidden_state[:, -1], native.norm(hidden)[:, -1], rtol=1e-5, atol=1e-6)

    def test_stage_scores_use_embedding_and_raw_block1_output_with_one_joint_calibration(self):
        torch.manual_seed(796)
        core = configured_core()
        snapshots = []
        before = core.layers[0].register_forward_pre_hook(lambda layer, args: snapshots.append(args[0].clone()))
        after = core.layers[0].register_forward_hook(lambda layer, args, result: snapshots.append(result[0].clone()))
        try:
            with torch.no_grad():
                core(input_ids=torch.arange(1, 20)[None], use_cache=True)
        finally:
            before.remove()
            after.remove()
        upstream = oracle()
        energies = []
        spatial = []
        for hidden in snapshots:
            parts = []
            for grid in core._clse_layout["grids"]:
                parts.append(upstream.spatial_spectral_score_2d(
                    hidden[:, torch.tensor(grid["indices"]) + 2], 1, grid["rows"], grid["cols"])[0])
            energies.append(torch.cat(parts))
        for grid in core._clse_layout["grids"]:
            spatial.extend(grid["indices"])
        expected = upstream.get_evolution_factor(energies[0][None], energies[1][None])[0]
        actual = torch.tensor(core.get_clse_stages()[0]["attention"]["scores"])[spatial]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    def test_zero_core_hidden_and_cache_are_native(self):
        for implementation in ("eager", "sdpa"):
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(797)
                core = LlavaQwenModel(tiny_config(implementation)).to(dtype).eval()
                native = Qwen2Model(tiny_config(implementation)).to(dtype).eval()
                native.load_state_dict(core.state_dict())
                core.configure_clse(enabled=False, keep_ratio=1)
                with torch.no_grad(), patch("llava.model.language_model.clse.spatial_spectral_energy",
                                            side_effect=AssertionError("zero cannot score")):
                    actual = core(input_ids=torch.arange(1, 20)[None], past_key_values=DynamicCache(), use_cache=True)
                    expected = native(input_ids=torch.arange(1, 20)[None], past_key_values=DynamicCache(), use_cache=True)
                self.assertTrue(torch.equal(actual.last_hidden_state, expected.last_hidden_state))
                for index in range(4):
                    self.assertTrue(torch.equal(actual.past_key_values.key_cache[index], expected.past_key_values.key_cache[index]))
                    self.assertTrue(torch.equal(actual.past_key_values.value_cache[index], expected.past_key_values.value_cache[index]))

    def test_instrumented_pruning_timer_counts_spectral_and_selection_sections(self):
        from llava.model.language_model.pruning_timing import PruningTimer
        core = configured_core()
        core._pruning_timer = PruningTimer()
        with torch.no_grad():
            core(input_ids=torch.arange(1, 20)[None], use_cache=True)
        self.assertEqual(core._pruning_timer.sections, 2)
        self.assertGreater(core._pruning_timer.seconds, 0)

    def test_zero_native_generate_identity_no_fft_nonzero_repeatable_and_measured_generation_unchanged(self):
        torch.manual_seed(794)
        # This CPU PyTorch build has no FP16 linear kernels. Test BF16 here;
        # native FP16 + FlashAttention2 has a separate optional CUDA test.
        model, features, options = toy_model(dtype=torch.bfloat16)
        core = model.get_model()
        with torch.no_grad(), patch.object(model, "encode_images", return_value=features):
            baseline = model.generate(**options)
            core.configure_clse(enabled=False, keep_ratio=1)
            with patch("llava.model.language_model.clse.spatial_spectral_energy", side_effect=AssertionError("0% FFT")):
                zero = model.generate(**options)
            self.assertTrue(torch.equal(baseline, zero))
            self.assertEqual(core.get_clse_stats()["final_llm_visual_tokens"], 14)
            core.configure_clse(enabled=True, keep_ratio=.5)
            output = model.generate(**options)
            self.assertTrue(torch.equal(output, model.generate(**options)))
            from llava_pruning.benchmark_timing import generate_with_benchmark
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                    patch("torch.cuda.max_memory_allocated", return_value=100), patch("torch.cuda.max_memory_reserved", return_value=200):
                measured, timing = generate_with_benchmark(model, options)
            self.assertTrue(torch.equal(output, measured))
            self.assertEqual(timing["generated_tokens"], 4)
            self.assertLessEqual(timing["lat_seconds"], timing["generation_seconds"])
            self.assertEqual(core.dtype, torch.bfloat16)
            self.assertEqual(core.config._attn_implementation, "sdpa")
            for altered in (dict(num_beams=2), dict(do_sample=True)):
                with self.assertRaisesRegex(ValueError, "greedy"):
                    model.generate(**dict(options, **altered))

    def test_config_average_targets_and_mutual_exclusion(self):
        method = load_method("clse")
        self.assertTrue({50, 70, 90}.issubset(method.rates))  # Preserve user-added rate experiments.
        self.assertEqual(method.layer, 1)
        for rate in method.rates:
            self.assertAlmostEqual(method.resolve_budget(rate, 28)["requested_layer_prune_rate"] * 27 / 28, rate)
        with self.assertRaisesRegex(ValueError, "unreachable"):
            method.resolve_budgets(4)
        core = LlavaQwenModel(tiny_config()).eval()
        for other in ("fastv", "priortr", "vico", "pyramiddrop"):
            core.configure_clse(enabled=True, keep_ratio=.5)
            kwargs = (dict(enabled=False, layers=(1, 2, 3), keep_ratios=(.8, .5, .2))
                      if other in {"vico", "pyramiddrop"} else dict(enabled=False))
            getattr(core, "configure_" + other)(**kwargs)
            self.assertFalse(core.clse_configured)
            self.assertFalse(core.clse_enabled)
            core.configure_clse(enabled=True, keep_ratio=.5)
            self.assertFalse(getattr(core, other + "_enabled"))
        for changes in ({"score_type": "clse_attn"}, {"layer": 2}, {"rate_semantics": "final_cumulative"},
                        {"channel_chunk": True}, {"temperature": 0}):
            config = json.loads((PROJECT / "configs/clse.json").read_text(encoding="utf-8"))
            config.update(changes)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_method("clse", path)

    def test_partial_image_grid_is_rejected(self):
        core = configured_core()
        core.set_clse_image_layout([[(2, 15)]], 18, layout())
        with self.assertRaisesRegex(ValueError, "truncated"):
            core(input_ids=torch.arange(1, 19)[None], use_cache=True)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flash_attn"),
                     "CUDA + FlashAttention2 required; run on inference server")
class CLSEFlashTests(unittest.TestCase):
    def test_fp16_flash_zero_identity_and_nonzero_sparse_decode(self):
        config = tiny_config("flash_attention_2")
        config.torch_dtype = torch.float16
        model = LlavaQwenModel(config).cuda().half().eval()
        tokens = torch.arange(1, 20, device="cuda")[None]
        with torch.no_grad():
            baseline = model(input_ids=tokens, use_cache=True)
            model.configure_clse(enabled=False, keep_ratio=1)
            zero = model(input_ids=tokens, use_cache=True)
            self.assertTrue(torch.equal(baseline.last_hidden_state, zero.last_hidden_state))
            model.configure_clse(enabled=True, keep_ratio=.5)
            model.set_clse_image_layout([[(2, 16)]], 19, layout())
            result = model(input_ids=tokens, use_cache=True)
            self.assertEqual(result.past_key_values.get_seq_length(), 12)
            model(input_ids=torch.tensor([[20]], device="cuda"), past_key_values=result.past_key_values, use_cache=True)
            self.assertEqual(model.dtype, torch.float16)
            self.assertEqual(model.config._attn_implementation, "flash_attention_2")


if __name__ == "__main__":
    unittest.main()
