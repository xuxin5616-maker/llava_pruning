"""CPU fixtures verify observer transparency; real GPU timings need the server."""

import contextlib
import types
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, StoppingCriteriaList, LogitsProcessorList

from llava_pruning.benchmark_timing import FirstTokenObserver, generate_with_benchmark
from llava_pruning.backend import LlavaBackend


class ObserverTests(unittest.TestCase):
    def test_passive_first_token_only_and_all_tokens_counted(self):
        sync, clock = Mock(), Mock(return_value=12.5)
        observer = FirstTokenObserver(sync, clock)
        observer.start = 10.0
        for length in (5, 6, 7):
            self.assertIs(observer(torch.zeros(1, length), None), False)
        self.assertEqual(observer.latency, 2.5)
        self.assertEqual(observer.tokens, 3)
        sync.assert_called_once_with()
        clock.assert_called_once_with()

    def test_reject_missing_start_batching_and_skipped_tokens(self):
        observer = FirstTokenObserver(Mock(), Mock(return_value=2))
        with self.assertRaisesRegex(RuntimeError, "not started"):
            observer(torch.zeros(1, 3), None)
        observer.start = 1
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            observer(torch.zeros(2, 3), None)
        observer(torch.zeros(1, 3), None)
        with self.assertRaisesRegex(RuntimeError, "autoregressive"):
            observer(torch.zeros(1, 5), None)


class BenchmarkTimingTests(unittest.TestCase):
    def setUp(self):
        self.previous = object()
        self.core = types.SimpleNamespace(_pruning_timer=self.previous)
        self.options = {"inputs": torch.tensor([[1, 2]]), "do_sample": False,
                        "max_new_tokens": 3, "use_cache": True}
        self.model = types.SimpleNamespace(
            get_model=lambda: self.core,
            generation_config=types.SimpleNamespace(do_sample=False, num_beams=1, num_return_sequences=1))
        self.model.generate = Mock(side_effect=self.fake_generate)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.sync = stack.enter_context(patch("torch.cuda.synchronize"))
        self.reset = stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
        stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=1024))
        stack.enter_context(patch("torch.cuda.max_memory_reserved", return_value=2048))
        self.clock = stack.enter_context(patch("llava_pruning.benchmark_timing.time.perf_counter",
                                              side_effect=[10, 12, 15]))

    def fake_generate(self, **kwargs):
        self.assertIsNone(self.core._pruning_timer)
        self.assertTrue(torch.is_inference_mode_enabled())
        self.assertEqual(kwargs["max_new_tokens"], 3)
        for length in (3, 4, 5):
            stopped = kwargs["stopping_criteria"](torch.ones(1, length, dtype=torch.long), None)
            self.assertFalse(stopped.any())
        return torch.tensor([[1, 2, 3, 4, 5]])

    def test_boundaries_memory_and_options_preserved(self):
        original_criterion = Mock(return_value=False)
        self.options["stopping_criteria"] = StoppingCriteriaList([original_criterion])
        result, metrics = generate_with_benchmark(self.model, self.options)
        self.assertEqual(result.tolist(), [[1, 2, 3, 4, 5]])
        self.assertEqual(metrics, {"lat_seconds": 2, "generation_seconds": 5,
                                  "generated_tokens": 3, "peak_allocated_bytes": 1024,
                                  "peak_reserved_bytes": 2048})
        self.assertEqual(len(self.options["stopping_criteria"]), 1)
        self.assertEqual(original_criterion.call_count, 3)
        self.assertEqual(self.sync.call_count, 3)  # before generate, first token, end
        self.reset.assert_called_once_with(0)
        self.assertIs(self.core._pruning_timer, self.previous)

    def test_timer_restored_when_generation_fails(self):
        self.model.generate.side_effect = RuntimeError("boom")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            generate_with_benchmark(self.model, self.options)
        self.assertIs(self.core._pruning_timer, self.previous)

    def test_no_observer_fails_instead_of_substituting_full_duration(self):
        self.model.generate.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "never called"):
            generate_with_benchmark(self.model, self.options)

    def test_invalid_durations_fail(self):
        for times in ([10, 9, 15], [10, 16, 15], [10, float("nan"), 15]):
            self.clock.side_effect = times
            with self.assertRaisesRegex(RuntimeError, "Invalid"):
                generate_with_benchmark(self.model, self.options)

    def test_sampling_and_beams_rejected(self):
        for options in ({"do_sample": True}, {"num_beams": 2}, {"num_return_sequences": 2}):
            with self.assertRaisesRegex(ValueError, "greedy"):
                generate_with_benchmark(self.model, dict(self.options, **options))
        self.model.generate.assert_not_called()

    def test_backend_rejects_incompatible_flags_before_preprocessing(self):
        backend = LlavaBackend.__new__(LlavaBackend)
        defaults = dict(capture_visualization=False, capture_attention=False,
                        random_seed=42, benchmark_metrics=True)
        for changes in ({"do_sample": True}, {"include_pruning_time": False},
                        {"capture_attention": True}, {"capture_visualization": True}):
            with self.assertRaisesRegex(ValueError, "Benchmark requires"):
                backend.generate(None, "", None, 50, "anyres_max_9", **dict(defaults, **changes))


class RealGenerationTests(unittest.TestCase):
    def test_actual_llava_anyres_packing_and_fastv_outputs_unchanged(self):
        from test_fastv_attention import tiny_config
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.language_model.fastv_attention import last_prompt_attention
        from llava_pruning.backend import configure_image_mode

        for rate in (0, 50, 70, 90):
            with self.subTest(rate=rate):
                config = tiny_config("sdpa")
                config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
                config.mm_patch_merge_type = "spatial_unpad"
                config.image_grid_pinpoints = [[8, 8], [16, 8]]
                configure_image_mode(config, "anyres_max_9")
                model = LlavaQwenForCausalLM(config).eval()
                core = model.get_model()
                core.image_newline = torch.nn.Parameter(torch.zeros(config.hidden_size))
                core.configure_fastv(enabled=rate != 0, layer=2, keep_ratio=1 - rate / 100,
                                     capture_attention=False, preserve_image_newline=True)
                tower = types.SimpleNamespace(num_patches_per_side=2, image_size=8)
                features = torch.arange(4 * config.hidden_size).reshape(1, 4, config.hidden_size).float() / 128
                options = dict(inputs=torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]]),
                               images=torch.zeros(1, 3, 3, 8, 8), image_sizes=[(16, 8)],
                               do_sample=False, max_new_tokens=3, use_cache=True)
                events = []

                def encode(images):
                    events.append("vision")
                    return features.repeat(images.shape[0], 1, 1)

                with patch.object(model, "get_vision_tower", return_value=tower), \
                     patch.object(model, "encode_images", side_effect=encode), \
                     patch("llava.model.language_model.llava_qwen.last_prompt_attention",
                           wraps=last_prompt_attention) as score:
                    with torch.inference_mode():
                        expected = model.generate(**options)
                    events.clear()
                    readings = iter([10, 12, 15])

                    def clock():
                        events.append("clock")
                        return next(readings)

                    with patch("torch.cuda.synchronize", side_effect=lambda _: events.append("sync")), \
                         patch("torch.cuda.reset_peak_memory_stats", side_effect=lambda _: events.append("reset")), \
                         patch("torch.cuda.max_memory_allocated", return_value=1), \
                         patch("torch.cuda.max_memory_reserved", return_value=2), \
                         patch("llava_pruning.benchmark_timing.time.perf_counter", side_effect=clock):
                        actual, timing = generate_with_benchmark(model, options)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual.shape, (1, 3))
                self.assertEqual(score.call_count, 0 if rate == 0 else 2)
                self.assertEqual(timing["generated_tokens"], 3)
                self.assertEqual(events, ["sync", "reset", "clock", "vision", "sync", "clock", "sync", "clock"])

    def test_tiny_qwen_outputs_and_eos_identical_with_observer(self):
        torch.manual_seed(7)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=20, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )).eval()
        wrapper = types.SimpleNamespace(get_model=lambda: model.model, generate=model.generate,
                                        generation_config=model.generation_config)

        class ForcedToken:
            def __init__(self, token):
                self.token = token

            def __call__(self, input_ids, scores):
                scores.fill_(-float("inf"))
                scores[:, self.token] = 0
                return scores

        for token, count in ((3, 4), (2, 1)):
            for embeds in (False, True):
                with self.subTest(token=token, embeds=embeds):
                    options = {"inputs": torch.tensor([[1, 4, 5]]), "max_new_tokens": 4,
                               "do_sample": False, "use_cache": True,
                               "logits_processor": LogitsProcessorList([ForcedToken(token)])}
                    if embeds:
                        options["inputs_embeds"] = model.get_input_embeddings()(options["inputs"])
                    with torch.inference_mode():
                        expected = model.generate(**options)
                    with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                         patch("torch.cuda.max_memory_allocated", return_value=1), \
                         patch("torch.cuda.max_memory_reserved", return_value=2):
                        actual, timing = generate_with_benchmark(wrapper, options)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(timing["generated_tokens"], count)
                    self.assertLessEqual(timing["lat_seconds"], timing["generation_seconds"])


if __name__ == "__main__":
    unittest.main()
