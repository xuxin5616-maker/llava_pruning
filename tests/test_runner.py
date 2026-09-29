import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from llava_pruning.metrics import METRIC_RULE
from llava_pruning.runner import run


class FakeBackend:
    calls = []

    def __init__(self, model_path, roi_mode="randomroi"):
        self.vision_tower = type("Tower", (), {
            "num_patches_per_side": 2,
            "config": type("Config", (), {"patch_size": 4})(),
        })()

    def generate(self, sample, prompt, method, rate, roi_mode, *,
                 capture_visualization, capture_attention, random_seed,
                 do_sample=False, include_pruning_time=True):
        self.calls.append((rate, random_seed, do_sample))
        result = {
            "answer": "A", "generation_seconds": 0.1,
            "max_new_tokens": 192,
            "input_token_ids": list(range(100)), "generated_token_ids": [65],
            "roi_source": "anyres" if roi_mode == "anyres_max_9" else "mask",
            "roi_boxes": [],
            "stats": {"fastv_layer": method.layer, "keep_ratio": 1 - rate / 100},
        }
        if not include_pruning_time:
            pruning = 0.0 if rate == 0 else 0.02
            result.update(generation_with_pruning_seconds=0.1, pruning_seconds=pruning,
                          generation_seconds=0.1 - pruning)
        if capture_visualization:
            if roi_mode == "anyres_max_9":
                result.update({
                    "image": Image.new("RGB", (16, 8), "white"),
                    "crop_metadata": [{"mode": "anyres_max_9", "original_size": [16, 8],
                                       "roi_boxes": [], "grid_patches": [2, 1],
                                       "final_newline": False}],
                    "masks": [{"span": [0, 14], "keep": [True] * 14}],
                })
            else:
                result.update({
                    "image": Image.new("RGB", (8, 8), "white"),
                    "crop_metadata": [{"original_size": [8, 8], "roi_boxes": [],
                                       "processed_view_sizes": [[8, 8]]}],
                    "masks": [{"span": [0, 5],
                               "keep": [True, False, True, False, True]}],
                })
        if capture_attention:
            if roi_mode == "anyres_max_9":
                result["attentions"] = [{"span": [0, 14], "scores": [0.1] * 14}]
            else:
                result["attentions"] = [{"span": [0, 5],
                                         "scores": [0.1, 0.2, 0.3, 0.4, 0.0]}]
        return result


class RunnerTests(unittest.TestCase):
    def test_ten_rates_five_visualizations_and_metrics(self):
        FakeBackend.calls.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "imgs").mkdir()
            (root / "musc").mkdir()
            Image.new("RGB", (8, 8)).save(root / "imgs" / "000000108.png")
            Image.new("L", (8, 8)).save(root / "musc" / "000000108.png")
            source = root / "questions.jsonl"
            source.write_text(json.dumps({
                "question_id": "000000108", "image": "000000108.png",
                "mask": "musc/000000108.png",
                "origin_path": "screw/test/thread_top/005.png", "gt": 1,
            }) + "\n", encoding="utf-8")
            output = root / "result"
            config = Path(__file__).resolve().parents[1] / "configs" / "fastv.json"
            with patch("llava_pruning.runner.LlavaBackend", FakeBackend):
                run(model_path=root, input_json=source, data_root=root,
                    prompt_version="v0", method_name="fastv", method_config=config,
                    roi_mode="randomroi", save_prune_vis=True,
                    save_attention_vis=True, output_dir=output, seed=42)
            for rate in range(0, 100, 10):
                row = json.loads((output / f"prune_{rate:02d}" / "predictions.jsonl")
                                 .read_text(encoding="utf-8"))
                metrics = json.loads((output / f"prune_{rate:02d}" / "metrics.json")
                                     .read_text(encoding="utf-8"))
                self.assertEqual(row, {
                    "question_id": "000000108",
                    "image": str((root / "imgs" / "000000108.png").resolve()),
                    "origin_path": "screw/test/thread_top/005.png", "gt": 1,
                    "answer": "A", "prune_rate": rate, "method": "fastv",
                    "generation_seconds": 0.1, "max_new_tokens": 192,
                })
                self.assertTrue(metrics["complete"])
                self.assertEqual(metrics["expected_samples"], 1)
                self.assertEqual(metrics["evaluated_samples"], 1)
                self.assertEqual(metrics["correct"], 1)
                self.assertEqual(metrics["accuracy"], 1.0)
                self.assertEqual(metrics["precision"], 1.0)
                self.assertEqual(metrics["recall"], 1.0)
                self.assertIsNone(metrics["tnr"])
                self.assertNotIn("rule", metrics)
                vis_dir = output / f"prune_{rate:02d}" / "visualizations"
                if rate % 20 == 10:
                    image_dir = vis_dir / "sample_000000108" / "image_0"
                    self.assertTrue((image_dir / "comparison.png").is_file())
                    self.assertTrue((image_dir / "attention_overlay.png").is_file())
                else:
                    self.assertFalse(vis_dir.exists())
            self.assertEqual(len((output / "summary.csv").read_text(encoding="utf-8").splitlines()), 11)
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["do_sample"])
            self.assertEqual(metadata["seed"], 42)
            self.assertIsNotNone(datetime.fromisoformat(metadata["created_utc"]).tzinfo)
            self.assertEqual(metadata["method_config"], json.loads(config.read_text(encoding="utf-8")))
            self.assertEqual(metadata["metric_rule"], METRIC_RULE)
            self.assertEqual(metadata["roi_mode"], "randomroi")
            self.assertEqual(FakeBackend.calls[0], (0, 42, False))

    def test_anyres_mode_writes_visualizations_and_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 8)).save(root / "image.png")
            source = root / "questions.jsonl"
            source.write_text(json.dumps({
                "question_id": "one", "image": "image.png",
                "origin_path": "screw/test/bad.png", "gt": 1,
            }) + "\n", encoding="utf-8")
            config = root / "fastv.json"
            config.write_text(json.dumps({
                "layer": 2, "prune_rates": [0, 10], "visualize_rates": [10],
            }), encoding="utf-8")
            output = root / "results"
            with patch("llava_pruning.runner.LlavaBackend", FakeBackend):
                run(model_path=root, input_json=source, data_root=root,
                    prompt_version="v0", method_name="fastv", method_config=config,
                    roi_mode="anyres_max_9", save_prune_vis=True,
                    save_attention_vis=True, output_dir=output, seed=42)
            row = json.loads((output / "prune_10" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("roi_source", row)
            self.assertNotIn("visualizations", row)
            image_dir = output / "prune_10" / "visualizations" / "sample_one" / "image_0"
            self.assertTrue((image_dir / "comparison.png").is_file())
            self.assertTrue((image_dir / "attention_overlay.png").is_file())
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["roi_mode"], "anyres_max_9")
            self.assertEqual(json.loads((output / "prune_10" / "metrics.json").read_text(encoding="utf-8"))["accuracy"], 1.0)

    def test_no_sample_and_generated_run_seed(self):
        FakeBackend.calls.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "image.png")
            source = root / "questions.jsonl"
            source.write_text(json.dumps({
                "question_id": "one", "image": "image.png",
                "origin_path": "screw/test/bad.png", "gt": 1,
            }) + "\n", encoding="utf-8")
            config = root / "fastv.json"
            config.write_text(json.dumps({
                "layer": 2, "prune_rates": [0], "visualize_rates": [],
            }), encoding="utf-8")
            output = root / "results"
            with patch("llava_pruning.runner.LlavaBackend", FakeBackend), \
                 patch("llava_pruning.runner.secrets.randbelow", return_value=12345):
                run(model_path=root, input_json=source, data_root=root,
                    prompt_version="v0", method_name="fastv", method_config=config,
                    roi_mode="randomroi", save_prune_vis=False,
                    save_attention_vis=False, output_dir=output,
                    seed=None, no_sample=True)
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed"], 12345)
            self.assertEqual(metadata["seed_origin"], "generated")
            self.assertFalse(metadata["do_sample"])
            self.assertEqual(FakeBackend.calls, [(0, 12345, False)])


if __name__ == "__main__":
    unittest.main()
