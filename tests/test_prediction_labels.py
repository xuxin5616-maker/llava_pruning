"""Result labels must agree with metrics and leave visualization pixels intact."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from llava_pruning import visualization
from llava_pruning.metrics import Accuracy
from llava_pruning.runner import run
from test_vico_visualization import sample_stages


class LabelBackend:
    """Synthetic outputs for checking rendering, not model accuracy."""

    def __init__(self, model_path, roi_mode="randomroi", image_token_order="base_first"):
        self.vision_tower = type("Tower", (), {
            "num_patches_per_side": 2,
            "config": type("Config", (), {"patch_size": 4})(),
        })()

    def generate(self, sample, prompt, method, rate, roi_mode, *,
                 capture_visualization, capture_attention, random_seed,
                 do_sample=False, include_pruning_time=True):
        answer = {"normal": "A. Yes" if rate == 50 else "B. No",
                  "defect": " b\n", "unknown": "Cannot decide"}[sample.sample_id]
        result = {"answer": answer, "generation_seconds": 0.1,
                  "max_new_tokens": 512, "stats": {}}
        if not capture_visualization:
            return result
        if roi_mode == "ex_base_copy":
            size, count = (16, 8), 13
            metadata = {"mode": roi_mode, "original_size": list(size), "roi_boxes": [],
                        "base_view_count": 3, "final_newline": True,
                        "processed_view_sizes": [[8, 8]] * 3}
        elif roi_mode in {"anyres_max_9", "anyres_only"}:
            size, count = (16, 8), 10 if roi_mode == "anyres_only" else 14
            metadata = {"mode": roi_mode, "original_size": list(size),
                        "roi_boxes": [], "grid_patches": [2, 1]}
        else:
            size, count = (8, 8), 9
            metadata = {"mode": roi_mode, "original_size": list(size),
                        "roi_boxes": [[0, 0, 4, 4]],
                        "processed_view_sizes": [[8, 8], [8, 8]]}
        result.update(image=Image.new("RGB", size, (80, 120, 200)),
                      crop_metadata=[metadata])
        if method.name == "fastv":
            result.update(masks=[{"span": [0, count], "keep": [True] * count}],
                          stats={"fastv_layer": method.layer, "keep_ratio": 1 - rate / 100})
            if capture_attention:
                result["attentions"] = [{"span": [0, count], "scores": [0.1] * count}]
        else:
            result.update(stages=sample_stages(count), stats={"layers": []})
        return result


class PredictionLabelTests(unittest.TestCase):
    def test_class_names_and_missing_values(self):
        cases = (
            (0, "A", "GT: Normal | Pred: Abnormal"),
            (1, "B", "GT: Abnormal | Pred: Normal"),
            (1, " a. Yes", "GT: Abnormal | Pred: Abnormal"),
            (0, "\nb: no", "GT: Normal | Pred: Normal"),
            (None, "A", "GT: Unknown | Pred: Abnormal"),
            (0, "Yes", "GT: Normal | Pred: Unparsed"),
            (1, "Answer: B", "GT: Abnormal | Pred: Unparsed"),
            (0, "AB", "GT: Normal | Pred: Unparsed"),
            (None, None, "GT: Unknown | Pred: Unparsed"),
            (None, "", "GT: Unknown | Pred: Unparsed"),
        )
        for gt, answer, expected in cases:
            with self.subTest(gt=gt, answer=answer):
                self.assertEqual(visualization.prediction_caption(gt, answer), expected)

    def test_caption_parser_matches_evaluation(self):
        for answer in ("A", "B", " a. Yes", "\tb) No", "AB", "Yes", "Answer: A", ""):
            for gt in (0, 1):
                with self.subTest(answer=answer, gt=gt):
                    metric = Accuracy(1)
                    metric.add(gt, answer)
                    label = visualization.prediction_caption(gt, answer)
                    prediction = ("Unparsed" if metric.unparsed else
                                  "Abnormal" if metric.tp + metric.fp else "Normal")
                    self.assertTrue(label.endswith(f"Pred: {prediction}"))

    def test_header_is_legible_and_does_not_cover_or_resize_existing_pixels(self):
        for size in ((8, 8), (1920, 120)):
            with self.subTest(size=size):
                source = Image.new("RGB", size, (80, 120, 200))
                original = np.array(source)
                label = "GT: Abnormal | Pred: Unparsed"
                draw_text = ImageDraw.ImageDraw.text
                with patch.object(ImageDraw.ImageDraw, "text", autospec=True,
                                  side_effect=draw_text) as draw:
                    labelled = visualization._add_prediction_header(source, label)
                draw.assert_called_once()
                self.assertEqual(draw.call_args.args[2], label)
                self.assertEqual(draw.call_args.kwargs["font"].size, 24)
                header_height = labelled.height - source.height
                self.assertGreater(header_height, 24)
                np.testing.assert_array_equal(np.asarray(labelled)[header_height:, :source.width], original)
                np.testing.assert_array_equal(np.asarray(source), original)
                header = np.asarray(labelled)[:header_height]
                self.assertTrue((header < 255).any())
                self.assertTrue((header[:10] == 255).all())
                self.assertTrue((header[:, -10:] == 255).all())

    def test_optional_label_keeps_legacy_rendering_unchanged(self):
        image = Image.new("RGB", (8, 8))
        self.assertIs(visualization._add_prediction_header(image, None), image)

    def test_runner_labels_both_methods_all_roi_modes_and_save_switches(self):
        for method in ("fastv", "vico"):
            for roi_mode in ("anyres_max_9", "randomroi", "randompatch", "ex_base_copy", "anyres_only"):
                for save_prune, save_attention in ((True, True), (True, False),
                                                   (False, True), (False, False)):
                    with self.subTest(method=method, mode=roi_mode,
                                      prune=save_prune, attention=save_attention), \
                            tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        Image.new("RGB", (16, 8)).save(root / "image.png")
                        source = root / "input.json"
                        source.write_text(json.dumps([
                            {"question_id": name, "image": "image.png", "gt": gt,
                             "origin_path": "screw/test/good/001.png"}
                            for name, gt in (("normal", 0), ("defect", 1), ("unknown", None))
                        ]), encoding="utf-8")
                        config = root / "method.json"
                        config.write_text(json.dumps({
                            **({"layer": 2} if method == "fastv" else {"layers": [8, 16, 24]}),
                            "prune_rates": [0, 50, 90], "visualize_rates": [50, 90],
                        }), encoding="utf-8")
                        output = root / "run"
                        with patch("llava_pruning.runner.LlavaBackend", LabelBackend), \
                                patch.object(visualization, "_add_prediction_header",
                                             wraps=visualization._add_prediction_header) as headers, \
                                contextlib.redirect_stdout(io.StringIO()):
                            run(model_path=root, input_json=source, data_root=root,
                                prompt_version="v0", method_name=method, method_config=config,
                                roi_mode=roi_mode, save_prune_vis=save_prune,
                                save_attention_vis=save_attention, output_dir=output, seed=42)
                        expected_labels = []
                        for rate in (50, 90):
                            normal_pred = "Abnormal" if rate == 50 else "Normal"
                            for label in (f"GT: Normal | Pred: {normal_pred}",
                                          "GT: Abnormal | Pred: Normal",
                                          "GT: Unknown | Pred: Unparsed"):
                                expected_labels.extend([label] * (save_prune + save_attention))
                            records = [json.loads(line) for line in
                                       (output / f"prune_{rate:02d}" / "predictions.jsonl")
                                       .read_text(encoding="utf-8").splitlines()]
                            self.assertTrue(all(len(record) == 9 for record in records))
                            self.assertEqual([r["gt"] for r in records], [0, 1, None])
                        self.assertEqual([call.args[1] for call in headers.call_args_list], expected_labels)
                        self.assertFalse((output / "prune_00" / "visualizations").exists())
                        pngs = list(output.rglob("*.png"))
                        self.assertEqual(len(pngs), len(expected_labels))
                        expected_names = ({"comparison.png"} if save_prune else set()) | \
                                         ({"attention_overlay.png"} if save_attention else set())
                        self.assertEqual({path.name for path in pngs}, expected_names)
                        for path in pngs:
                            with Image.open(path) as image:
                                self.assertTrue((np.asarray(image)[12:40] < 255).any())


if __name__ == "__main__":
    unittest.main()
