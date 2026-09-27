"""Run the real 28-layer pruning logic with tiny CPU weights and fixture images."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from llava_pruning.runner import run
from test_fastv_attention import tiny_config
from test_vico import LlavaQwenModel


class TinyViCoBackend:
    def __init__(self, model_path, roi_mode="anyres_max_9"):
        config = tiny_config("sdpa")
        config.num_hidden_layers = 28
        self.core = LlavaQwenModel(config).eval()
        self.vision_tower = type("Tower", (), {"num_patches_per_side": 2,
                                             "config": type("Config", (), {"patch_size": 4})()})()

    def generate(self, sample, prompt, method, rate, roi_mode, *, capture_visualization,
                 capture_attention, random_seed, do_sample=False):
        method.configure(self.core, rate, capture_visualization=capture_visualization,
                         capture_attention=capture_attention)
        self.core.set_vico_image_spans([[(1, 15)]], 17)
        with torch.no_grad():
            self.core(input_ids=torch.arange(1, 18)[None])
        result = {"answer": "A", "generation_seconds": 0.1, "roi_source": "anyres", "roi_boxes": [],
                  "stats": method.stats(self.core, rate)}
        if capture_visualization:
            result.update(method.visualization_data(self.core, capture_attention=capture_attention))
            result.update(image=Image.new("RGB", (16, 8), "white"),
                          crop_metadata=[{"mode": "anyres_max_9", "original_size": [16, 8],
                                          "roi_boxes": [], "grid_patches": [2, 1], "final_newline": False}])
        return result


class ViCoRunnerTests(unittest.TestCase):
    def test_28_layers_baseline_and_three_stage_csv_predictions_and_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 8)).save(root / "image.png")
            source = root / "input.json"
            source.write_text(json.dumps([{"question_id": "0001", "image": "image.png",
                                           "origin_path": "screw/test/bad.png", "gt": 1}]), encoding="utf-8")
            config = root / "vico.json"
            config.write_text(json.dumps({"layers": [8, 16, 24], "prune_rates": [0, 90],
                                          "visualize_rates": [90]}), encoding="utf-8")
            output = root / "run"
            with patch("llava_pruning.runner.LlavaBackend", TinyViCoBackend), contextlib.redirect_stdout(io.StringIO()):
                run(model_path=root, input_json=source, data_root=root, prompt_version="v0",
                    method_name="vico", method_config=config, roi_mode="anyres_max_9",
                    save_prune_vis=True, save_attention_vis=True, output_dir=output, seed=42)
            for rate in (0, 90):
                folder = output / f"prune_{rate:02d}"
                with (folder / "layer_tokens.csv").open(encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 28)
                self.assertTrue(all(row["question_id"] == "0001" and row["method"] == "vico" for row in rows))
                record = json.loads((folder / "predictions.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(record["method"], "vico")
                if rate == 0:
                    self.assertTrue(all(row["image_tokens_in"] == row["image_tokens_out"] == "14" for row in rows))
                    self.assertEqual(record["pruning_stats"]["mode"], "disabled_baseline")
                    self.assertFalse((folder / "visualizations").exists())
                else:
                    self.assertEqual([int(row["layer"]) for row in rows if int(row["removed_after_layer"])], [8, 16, 24])
                    self.assertEqual([int(rows[i]["image_tokens_in"]) for i in (0, 8, 16, 24)], [14, 7, 4, 2])
                    self.assertEqual(int(rows[-1]["sequence_tokens_out"]), 5)
                    self.assertEqual({p.name for p in folder.rglob("*.png")}, {"comparison.png", "attention_overlay.png"})
                    self.assertEqual(len(record["pruning_stats"]["stages"]), 3)
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["method"], "vico")
            self.assertFalse(metadata["do_sample"])


if __name__ == "__main__":
    unittest.main()
