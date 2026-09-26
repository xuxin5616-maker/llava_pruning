import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from triad_pruning.runner import run


class FakeBackend:
    calls = []

    def __init__(self, model_path, roi_mode="randomroi"):
        self.vision_tower = type("Tower", (), {
            "num_patches_per_side": 2,
            "config": type("Config", (), {"patch_size": 4})(),
        })()

    def generate(self, sample, prompt, method, rate, roi_mode, *,
                 capture_visualization, capture_attention, random_seed,
                 do_sample=True):
        self.calls.append((rate, random_seed, do_sample))
        result = {
            "answer": "A", "generation_seconds": 0.1,
            "roi_source": "anyres" if roi_mode == "anyres_max_9" else "mask",
            "roi_boxes": [],
            "stats": {"fastv_layer": method.layer, "keep_ratio": 1 - rate / 100},
        }
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
            with patch("triad_pruning.runner.TriadBackend", FakeBackend):
                run(model_path=root, input_json=source, data_root=root,
                    prompt_version="v0", method_name="fastv", method_config=config,
                    roi_mode="randomroi", save_prune_vis=True,
                    save_attention_vis=True, output_dir=output, seed=42)
            for rate in range(0, 100, 10):
                row = json.loads((output / f"prune_{rate:02d}" / "predictions.jsonl")
                                 .read_text(encoding="utf-8"))
                metrics = json.loads((output / f"prune_{rate:02d}" / "metrics.json")
                                     .read_text(encoding="utf-8"))
                self.assertEqual(row["question_id"], "000000108")
                self.assertEqual(row["prune_rate"], rate)
                self.assertTrue(metrics["complete"])
                self.assertEqual(metrics["correct"], 1)
                self.assertEqual(metrics["accuracy"], 1.0)
                if rate % 20 == 10:
                    paths, = row["visualizations"]
                    self.assertTrue(Path(paths["comparison"]).is_file())
                    self.assertTrue(Path(paths["attention_overlay"]).is_file())
                else:
                    self.assertIsNone(row["visualizations"])
            self.assertEqual(len((output / "summary.csv").read_text(encoding="utf-8").splitlines()), 11)
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["do_sample"])
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(FakeBackend.calls[0], (0, 42, True))

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
            with patch("triad_pruning.runner.TriadBackend", FakeBackend):
                run(model_path=root, input_json=source, data_root=root,
                    prompt_version="v0", method_name="fastv", method_config=config,
                    roi_mode="anyres_max_9", save_prune_vis=True,
                    save_attention_vis=True, output_dir=output, seed=42)
            row = json.loads((output / "prune_10" / "predictions.jsonl").read_text(encoding="utf-8"))
            paths, = row["visualizations"]
            self.assertEqual(row["roi_source"], "anyres")
            self.assertTrue(Path(paths["attention_overlay"]).is_file())
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
            with patch("triad_pruning.runner.TriadBackend", FakeBackend), \
                 patch("triad_pruning.runner.secrets.randbelow", return_value=12345):
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
