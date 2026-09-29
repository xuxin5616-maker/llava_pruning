"""Pruning accounting must not alter decisions, decoder outputs, or caches."""

import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Model

from test_fastv_attention import tiny_config
from llava.model.language_model.fastv_attention import last_prompt_attention
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenModel
from llava.model.language_model.pruning_timing import PruningTimer, measure_pruning


TIMER_MODULE = "llava.model.language_model.pruning_timing"


def configure(core, method, enabled=True):
    if method == "fastv":
        core.configure_fastv(enabled=enabled, layer=2, keep_ratio=0.5,
                             preserve_image_newline=False, capture_attention=True)
        core.set_fastv_image_spans([[(1, 9)]])
    else:
        core.configure_vico(enabled=enabled, layers=[1, 2, 3],
                            keep_ratios=[0.75, 0.5, 0.25] if enabled else [1, 1, 1],
                            capture_visualization=True, capture_attention=True)
        core.set_vico_image_spans([[(1, 9)]], 11)


def decisions(core, method):
    if method == "fastv":
        return (core.get_fastv_stats(), core.get_fastv_image_masks(),
                core.get_fastv_image_attentions())
    return core.get_vico_stats(), core.get_vico_stages()


def score_target(method):
    module = "llava_qwen" if method == "fastv" else "vico"
    return f"llava.model.language_model.{module}.last_prompt_attention"


class PruningTimerTests(unittest.TestCase):
    def test_cuda_drains_prior_work_before_clock_and_finishes_block_before_stop(self):
        events = []
        wall_time = [0.0]
        sync_costs = iter([100.0, 2.0])
        device = torch.device("cuda:0")

        def synchronize(actual_device):
            self.assertEqual(actual_device, device)
            events.append("sync")
            wall_time[0] += next(sync_costs)

        def clock():
            events.append("clock")
            return wall_time[0]

        timer = PruningTimer()
        core = SimpleNamespace(_pruning_timer=timer)
        with patch(f"{TIMER_MODULE}.torch.cuda.synchronize", side_effect=synchronize), \
             patch(f"{TIMER_MODULE}.perf_counter", side_effect=clock):
            with measure_pruning(core, device):
                events.append("prune")
                wall_time[0] += 3.0
        self.assertEqual(events, ["sync", "clock", "prune", "sync", "clock"])
        self.assertEqual(timer.seconds, 5.0)  # The preceding 100 s are excluded.
        self.assertEqual(timer.sections, 1)

    def test_disabled_timer_has_no_clock_cuda_or_device_work(self):
        with patch(f"{TIMER_MODULE}.perf_counter", side_effect=AssertionError("clock")), \
             patch(f"{TIMER_MODULE}.torch.cuda.synchronize", side_effect=AssertionError("sync")):
            for core in (SimpleNamespace(), SimpleNamespace(_pruning_timer=None)):
                with measure_pruning(core, "not-a-device"):
                    core.ran = True
                self.assertTrue(core.ran)

    def test_nested_sections_count_outer_wall_time_once_and_accumulate(self):
        timer = PruningTimer()
        core = SimpleNamespace(_pruning_timer=timer)
        with patch(f"{TIMER_MODULE}.perf_counter", side_effect=[1.0, 4.0, 8.0, 10.0]) as clock, \
             patch(f"{TIMER_MODULE}.torch.cuda.synchronize") as sync:
            with measure_pruning(core, "cpu"):
                with measure_pruning(core, "cpu"):
                    pass
            with measure_pruning(core, "cpu"):
                pass
        self.assertEqual(clock.call_count, 4)
        sync.assert_not_called()
        self.assertEqual((timer.seconds, timer.sections), (5.0, 2))

    def test_invalid_intervals_do_not_corrupt_totals_or_leave_active_section(self):
        for finish in (-1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(finish=finish):
                timer = PruningTimer()
                core = SimpleNamespace(_pruning_timer=timer)
                with patch(f"{TIMER_MODULE}.perf_counter", side_effect=[0.0, finish]):
                    with self.assertRaisesRegex(RuntimeError, "wall-clock interval"):
                        with measure_pruning(core, "cpu"):
                            pass
                self.assertEqual((timer.seconds, timer.sections), (0.0, 0))
                self.assertFalse(timer._active)

    def test_body_exception_clears_active_section_for_later_measurement(self):
        timer = PruningTimer()
        core = SimpleNamespace(_pruning_timer=timer)
        with patch(f"{TIMER_MODULE}.perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]):
            with self.assertRaisesRegex(ValueError, "pruning failed"):
                with measure_pruning(core, "cpu"):
                    raise ValueError("pruning failed")
            self.assertFalse(timer._active)
            with measure_pruning(core, "cpu"):
                pass
        self.assertEqual((timer.seconds, timer.sections), (2.0, 2))

    def test_timer_rejects_different_devices_without_counting_second_block(self):
        timer = PruningTimer()
        core = SimpleNamespace(_pruning_timer=timer)
        with patch(f"{TIMER_MODULE}.perf_counter", side_effect=[0.0, 1.0]), \
             patch(f"{TIMER_MODULE}.torch.cuda.synchronize") as sync:
            with measure_pruning(core, "cuda:0"):
                pass
            with self.assertRaisesRegex(ValueError, "one device"):
                with measure_pruning(core, "cuda:1"):
                    self.fail("A second device was accepted")
        self.assertEqual(sync.call_count, 2)
        self.assertEqual((timer.seconds, timer.sections), (1.0, 1))


class PruningTimingDecoderTests(unittest.TestCase):
    def assert_output_identical(self, expected, actual):
        self.assertEqual(expected.last_hidden_state.dtype, actual.last_hidden_state.dtype)
        torch.testing.assert_close(expected.last_hidden_state, actual.last_hidden_state, rtol=0, atol=0)
        self.assertEqual(len(expected.past_key_values), len(actual.past_key_values))
        for expected_layer, actual_layer in zip(expected.past_key_values, actual.past_key_values):
            for expected_tensor, actual_tensor in zip(expected_layer, actual_layer):
                self.assertEqual(expected_tensor.dtype, actual_tensor.dtype)
                torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)

    def test_cpu_timing_preserves_prefill_decode_cache_decisions_and_score_counts(self):
        for method in ("fastv", "vico"):
            with self.subTest(method=method):
                torch.manual_seed(103)
                untimed = LlavaQwenModel(tiny_config("sdpa")).eval()
                timed = LlavaQwenModel(tiny_config("sdpa")).eval()
                timed.load_state_dict(untimed.state_dict())
                configure(untimed, method)
                configure(timed, method)
                timer = timed._pruning_timer = PruningTimer()
                histories = []
                counts = []

                def check_decoder_outside_timer(module, args):
                    self.assertFalse(timer._active, "decoder forward was counted as pruning")

                handles = [layer.register_forward_pre_hook(check_decoder_outside_timer)
                           for layer in timed.layers]
                try:
                    with patch(f"{TIMER_MODULE}.perf_counter", side_effect=itertools.count()), \
                         patch(f"{TIMER_MODULE}.torch.cuda.synchronize", side_effect=AssertionError("CPU sync")), \
                         torch.no_grad():
                        for model in (untimed, timed):
                            history = []
                            with patch(score_target(method), wraps=last_prompt_attention) as score:
                                output = model(input_ids=torch.arange(1, 12)[None], use_cache=True)
                                history.append(output)
                                for token in (12, 13):
                                    output = model(input_ids=torch.tensor([[token]]),
                                                   past_key_values=output.past_key_values, use_cache=True)
                                    history.append(output)
                                counts.append(score.call_count)
                            histories.append(history)
                finally:
                    for handle in handles:
                        handle.remove()
                for expected, actual in zip(*histories):
                    self.assert_output_identical(expected, actual)
                self.assertEqual(decisions(untimed, method), decisions(timed, method))
                expected_scores = 1 if method == "fastv" else 3
                self.assertEqual(counts, [expected_scores, expected_scores])
                expected_sections = 7 if method == "fastv" else 3
                self.assertEqual((timer.seconds, timer.sections), (float(expected_sections), expected_sections))

    def test_zero_rate_uses_original_forward_and_never_opens_a_timing_block(self):
        for method in ("fastv", "vico"):
            with self.subTest(method=method):
                torch.manual_seed(107)
                reference = Qwen2Model(tiny_config("sdpa")).eval()
                model = LlavaQwenModel(tiny_config("sdpa")).eval()
                model.load_state_dict(reference.state_dict())
                configure(model, method, enabled=False)
                timer = model._pruning_timer = PruningTimer()
                with patch(f"{TIMER_MODULE}.perf_counter", side_effect=AssertionError("zero-rate clock")), \
                     patch(score_target(method), side_effect=AssertionError("zero-rate scoring")), \
                     torch.no_grad():
                    expected = reference(input_ids=torch.arange(1, 12)[None])
                    actual = model(input_ids=torch.arange(1, 12)[None])
                    self.assert_output_identical(expected, actual)
                    expected = reference(input_ids=torch.tensor([[12]]), past_key_values=expected.past_key_values)
                    actual = model(input_ids=torch.tensor([[12]]), past_key_values=actual.past_key_values)
                    self.assert_output_identical(expected, actual)
                self.assertEqual((timer.seconds, timer.sections), (0.0, 0))

    def test_real_generate_keeps_timer_through_resets_and_isolates_repeated_calls(self):
        for method in ("fastv", "vico"):
            with self.subTest(method=method):
                torch.manual_seed(109)
                config = tiny_config("sdpa")
                config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
                model = LlavaQwenForCausalLM(config).eval()
                core = model.get_model()
                configure(core, method)

                def pack(inputs, positions, mask, past, labels, images, modalities, image_sizes=None):
                    if images is None:
                        return inputs, positions, mask, past, None, labels
                    if method == "fastv":
                        core.set_fastv_image_spans([[(1, 9)]])
                    else:
                        core.set_vico_image_spans([[(1, 9)]], inputs.shape[1])
                    return None, None, None, None, core.embed_tokens(inputs), None

                timers = []
                with patch.object(model, "prepare_inputs_labels_for_multimodal", side_effect=pack), \
                     patch(f"{TIMER_MODULE}.perf_counter", side_effect=itertools.count()), torch.no_grad():
                    tokens = torch.arange(1, 12)[None]
                    expected = model.generate(inputs=tokens, images=torch.zeros(1),
                                              do_sample=False, max_new_tokens=4, use_cache=True)
                    expected_decisions = decisions(core, method)
                    for _ in range(2):
                        timer = core._pruning_timer = PruningTimer()
                        with patch(score_target(method), wraps=last_prompt_attention) as score:
                            actual = model.generate(inputs=tokens, images=torch.zeros(1),
                                                    do_sample=False, max_new_tokens=4, use_cache=True)
                        self.assertTrue(torch.equal(expected, actual))
                        self.assertEqual(decisions(core, method), expected_decisions)
                        self.assertIs(core._pruning_timer, timer)
                        self.assertEqual(score.call_count, 1 if method == "fastv" else 3)
                        timers.append(timer)
                sections = 9 if method == "fastv" else 3
                self.assertEqual([(timer.seconds, timer.sections) for timer in timers],
                                 [(float(sections), sections)] * 2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real synchronized timing")
class PruningTimingCudaTests(unittest.TestCase):
    assert_output_identical = PruningTimingDecoderTests.assert_output_identical

    def test_cuda_timing_preserves_prefill_decode_and_pruning(self):
        for method in ("fastv", "vico"):
            with self.subTest(method=method):
                torch.manual_seed(113)
                model = LlavaQwenModel(tiny_config("sdpa")).cuda().half().eval()
                tokens = torch.arange(1, 12, device="cuda")[None]
                outputs = []
                snapshots = []
                with torch.no_grad():
                    for timed in (False, True):
                        configure(model, method)
                        model._pruning_timer = PruningTimer() if timed else None
                        output = model(input_ids=tokens, use_cache=True)
                        output = model(input_ids=tokens[:, -1:], past_key_values=output.past_key_values,
                                       use_cache=True)
                        outputs.append(output)
                        snapshots.append(decisions(model, method))
                self.assert_output_identical(*outputs)
                self.assertEqual(*snapshots)
                self.assertTrue(math.isfinite(model._pruning_timer.seconds))
                self.assertGreater(model._pruning_timer.seconds, 0.0)
                self.assertEqual(model._pruning_timer.sections, 5 if method == "fastv" else 3)


if __name__ == "__main__":
    unittest.main()
