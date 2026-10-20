"""CPU-only Base pixels/packing, tiny decoder, and results-only CLI tests."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import run as cli
from llava_pruning.backend import LlavaBackend, configure_image_mode
from llava_pruning.methods import load_method
from llava_pruning.roi import choose_roi
from llava_pruning.runner import run
from test_fastv_attention import tiny_config
from test_reference import ToyPacking
from test_runner import FakeBackend
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.mm_utils import process_images
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor


PROJECT = Path(__file__).resolve().parents[1]


class BaseOnlyPixelsTests(unittest.TestCase):
    def test_config_preserves_precision_attention_and_checkpoint_merge(self):
        config = SimpleNamespace(image_aspect_ratio="anyres_max_9",
                                 mm_patch_merge_type="spatial_unpad",
                                 torch_dtype=torch.float16,
                                 _attn_implementation="flash_attention_2")
        original = vars(config).copy()
        configure_image_mode(config, "base_only")
        self.assertEqual(config.image_aspect_ratio, "base_only")
        self.assertEqual(config.image_token_order, "base_first")
        for field in ("mm_patch_merge_type", "torch_dtype", "_attn_implementation"):
            self.assertEqual(getattr(config, field), original[field])
        self.assertEqual(choose_roi(object(), "base_only"), ("base_only", None, None))
        with self.assertRaisesRegex(ValueError, "anyres_max_9"):
            configure_image_mode(config, "base_only", "anyres_first")

    def test_exact_anyres_base_pixels_without_computing_tiles_or_roi(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad",
                                 image_grid_pinpoints=[[28, 28], [56, 28], [28, 56], [56, 56]])
        rng = np.random.RandomState(43)
        for width, height in ((28, 28), (61, 29), (29, 61), (53, 49)):
            image = Image.fromarray(rng.randint(0, 256, (height, width, 3), dtype=np.uint8))
            source = np.array(image)
            reference = process_images([image], processor, config)
            base_config = SimpleNamespace(image_aspect_ratio="base_only")  # no tile config needed
            with patch("llava.mm_utils.process_anyres_image", side_effect=AssertionError("tiles")), \
                    patch("llava.mm_utils.process_randomroi_image", side_effect=AssertionError("ROI")), \
                    patch.object(processor, "preprocess", wraps=processor.preprocess) as preprocess:
                actual, info = process_images([image], processor, base_config, masks=[object()],
                                              boxes_list=[object()], return_pro_data=True)
            self.assertEqual(preprocess.call_count, 1)
            self.assertEqual(tuple(actual.shape), (1, 3, 28, 28))
            self.assertTrue(torch.equal(actual[0], reference[0, 0]))
            self.assertEqual(info[0]["base_view_count"], 1)
            self.assertEqual(info[0]["anyres_view_count"], 0)
            self.assertEqual(info[0]["roi_boxes"], [])
            self.assertFalse(info[0]["final_newline"])
            np.testing.assert_array_equal(np.array(image), source)
            self.assertTrue(torch.equal(process_images([image], processor, base_config), actual))

    def test_multiple_images_produce_one_view_each(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="base_only")
        images = [Image.new("RGB", (61, 29), "red"), Image.new("RGB", (29, 61), "blue")]
        actual, info = process_images(images, processor, config, return_pro_data=True)
        self.assertEqual(tuple(actual.shape), (2, 3, 28, 28))
        self.assertFalse(torch.equal(actual[0], actual[1]))
        self.assertEqual([entry["original_size"] for entry in info], [[61, 29], [29, 61]])


class BaseOnlyPackingTests(unittest.TestCase):
    @staticmethod
    def pack(toy, images, modalities=("image",)):
        tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        return toy.prepare_inputs_labels_for_multimodal(
            tokens, torch.arange(4)[None], torch.ones_like(tokens, dtype=torch.bool),
            None, tokens.clone(), images, modalities=modalities, image_sizes=[(16, 8)])

    def test_preserves_base_features_order_text_and_positions_without_newline(self):
        for dtype in (torch.float16, torch.float32):
            toy = ToyPacking(dtype)
            reference = self.pack(toy, torch.zeros(1, 3, 3, 8, 8, dtype=dtype))
            configure_image_mode(toy.config, "base_only")
            pixels = torch.zeros(1, 3, 8, 8, dtype=dtype)
            with patch.object(toy, "encode_images", wraps=toy.encode_images) as encoder:
                packed = self.pack(toy, pixels)
            encoder.assert_called_once_with(pixels)
            self.assertEqual(packed[4].shape, (1, 7, 4))  # 4 Base + 3 text, no newline
            self.assertTrue(torch.equal(packed[4][0, 1:5], reference[4][0, 1:5]))
            self.assertTrue(torch.equal(packed[4][0, 1:5], toy.encode_images(pixels)[0]))
            self.assertTrue(torch.equal(packed[4][0, [0, 5, 6]], toy.core.embed_tokens(torch.tensor([1, 2, 3]))))
            self.assertEqual(packed[1][0].tolist(), list(range(7)))
            self.assertTrue(packed[2].all())
            self.assertEqual(packed[5][0].tolist(), [1] + [IGNORE_INDEX] * 4 + [2, 3])

    def test_multiview_inputs_and_nonzero_pruning_cannot_silently_enter_base_only(self):
        toy = ToyPacking(torch.float32)
        configure_image_mode(toy.config, "base_only")
        for bad in (torch.zeros(1, 1, 3, 8, 8), torch.zeros(1, 3, 3, 8, 8),
                    [torch.zeros(1, 3, 8, 8)], torch.zeros(2, 3, 8, 8)):
            with self.assertRaisesRegex(ValueError, "base_only requires"):
                self.pack(toy, bad)
        with self.assertRaisesRegex(ValueError, "base_only requires"):
            self.pack(toy, torch.zeros(1, 3, 8, 8), ("video",))
        backend = LlavaBackend.__new__(LlavaBackend)
        for rate, vis, attention in ((10, False, False), (0, True, False), (0, False, True)):
            with self.assertRaisesRegex(ValueError, "base_only requires"):
                backend.generate(None, "", None, rate, "base_only", capture_visualization=vis,
                                 capture_attention=attention, random_seed=42)

    def test_real_tiny_decoder_with_cache_has_no_pruning_scoring(self):
        config = tiny_config("sdpa")
        config.bos_token_id, config.eos_token_id, config.pad_token_id = 1, None, 0
        config.mm_patch_merge_type = "spatial_unpad"
        configure_image_mode(config, "base_only")
        model = LlavaQwenForCausalLM(config).eval()
        method = load_method("fastv", PROJECT / "configs" / "baseline.json")
        method.configure(model.get_model(), 0, capture_attention=False)
        self.assertFalse(model.get_model().fastv_enabled)
        features = torch.arange(4 * config.hidden_size).reshape(1, 4, -1).float() / 128
        with patch.object(model, "get_vision_tower", return_value=SimpleNamespace(num_patches_per_side=2)), \
                patch.object(model, "encode_images", return_value=features) as encoder, \
                patch("llava.model.language_model.llava_qwen.last_prompt_attention",
                      side_effect=AssertionError("unexpected pruning score")), \
                patch("llava.model.language_model.vico.last_prompt_attention",
                      side_effect=AssertionError("unexpected ViCo score")), torch.no_grad():
            options = dict(inputs=torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]]),
                           images=torch.zeros(1, 3, 8, 8), image_sizes=[(16, 8)],
                           do_sample=False, max_new_tokens=3, use_cache=True)
            actual = model.generate(**options)
            # Same weights, native single-image baseline; only image mode differs.
            model.config.image_aspect_ratio = "square"
            expected = model.generate(**options)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.shape, (1, 3))
        self.assertEqual(encoder.call_count, 2)  # once per prefill, not every decode token
        self.assertTrue(all(tuple(call.args[0].shape) == (1, 3, 8, 8) for call in encoder.call_args_list))


class ResultsOnlyCliTests(unittest.TestCase):
    def test_no_plots_disables_dependency_check_sample_capture_and_final_plot(self):
        basic = ["run.py", "--model-path", "checkpoint", "--input-json", "input.jsonl",
                 "--data-root", "dataset", "--save-prune-vis", "--save-attention-vis"]
        for args in (["--roi-mode", "base_only"], ["--roi-mode", "anyres_max_9", "--no-plots"]):
            with patch("sys.argv", basic + args), patch.object(cli, "check_plot_dependencies") as deps, \
                    patch.object(cli, "save_metric_plot") as plot, \
                    patch.object(cli, "run", return_value=Path("result")) as runner:
                cli.main()
            deps.assert_not_called()
            plot.assert_not_called()
            self.assertFalse(runner.call_args.kwargs["save_prune_vis"])
            self.assertFalse(runner.call_args.kwargs["save_attention_vis"])

    def test_full_cli_uses_zero_only_evaluates_all_samples_and_creates_no_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "image.png")
            source = root / "input.json"
            source.write_text(json.dumps([
                {"question_id": str(i), "image": "image.png", "gt": i % 2,
                 "origin_path": "screw/test/example.png"} for i in range(101)
            ]), encoding="utf-8")
            output = root / "result"
            args = ["run.py", "--model-path", str(root), "--input-json", str(source),
                    "--data-root", str(root), "--roi-mode", "base_only",
                    "--no-plots", "--no-sample", "--seed", "42", "--output-dir", str(output)]
            FakeBackend.calls.clear()
            with patch("sys.argv", args), patch("llava_pruning.runner.LlavaBackend", FakeBackend), \
                    patch.object(cli, "check_plot_dependencies", side_effect=AssertionError("matplotlib")), \
                    patch.object(cli, "save_metric_plot", side_effect=AssertionError("plot")), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(FakeBackend.calls, [(0, 42 + i, False) for i in range(101)])
            expected = {"run.json", "summary.csv", "prune_00/predictions.jsonl", "prune_00/metrics.json"}
            self.assertEqual({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}, expected)
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["method_config"]["prune_rates"], [0])
            self.assertEqual(metadata["roi_mode"], "base_only")
            self.assertFalse(metadata["save_prune_vis"])
            self.assertFalse(metadata["save_attention_vis"])
            rows = list(csv.DictReader((output / "summary.csv").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prune_rate"], "0")
            self.assertEqual(rows[0]["evaluated_samples"], "101")
            self.assertEqual(rows[0]["complete"], "True")

    def test_nonzero_config_fails_before_data_or_model_loading(self):
        with patch("llava_pruning.runner.LlavaBackend") as model, \
                patch("llava_pruning.runner.load_samples") as data, \
                self.assertRaisesRegex(ValueError, "prune_rates must be"):
            run(model_path="unused", input_json="unused", data_root="unused",
                prompt_version="v0", method_name="fastv", method_config=PROJECT / "configs/fastv.json",
                roi_mode="base_only", save_prune_vis=False, save_attention_vis=False, output_dir="unused")
        model.assert_not_called()
        data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
