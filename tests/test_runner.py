import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from triad_pruning.runner import run


class FakeBackend:
    def __init__(self, model_path):
        self.vision_tower = type("Tower", (), {
            "num_patches_per_side": 2,
            "config": type("Config", (), {"patch_size": 4})(),
        })()

    def generate(self, sample, prompt, method, rate, roi_mode, *,
                 capture_visualization, capture_attention, random_seed):
        result = {
            "answer": "A", "generation_seconds": 0.1,
            "roi_source": "mask", "roi_boxes": [],
            "stats": {"fastv_layer": method.layer, "keep_ratio": 1 - rate / 100},
        }
        if capture_visualization:
            result.update({
                "image": Image.new("RGB", (8, 8), "white"),
                "crop_metadata": [{"original_size": [8, 8], "roi_boxes": [],
                                   "processed_view_sizes": [[8, 8]]}],
                "masks": [{"span": [0, 5],
                           "keep": [True, False, True, False, True]}],
            })
        if capture_attention:
            result["attentions"] = [{"span": [0, 5],
                                     "scores": [0.1, 0.2, 0.3, 0.4, 0.0]}]
        return result


class RunnerTests(unittest.TestCase):
    def test_nine_rates_and_five_visualizations(self):
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
            for rate in range(10, 100, 10):
                row = json.loads((output / f"prune_{rate:02d}" / "predictions.jsonl")
                                 .read_text(encoding="utf-8"))
                self.assertEqual(row["question_id"], "000000108")
                self.assertEqual(row["prune_rate"], rate)
                if rate % 20 == 10:
                    paths, = row["visualizations"]
                    self.assertTrue(Path(paths["comparison"]).is_file())
                    self.assertTrue(Path(paths["attention_overlay"]).is_file())
                else:
                    self.assertIsNone(row["visualizations"])


if __name__ == "__main__":
    unittest.main()
