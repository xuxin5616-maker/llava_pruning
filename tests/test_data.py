import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from triad_pruning.data import load_samples
from triad_pruning.prompts import resolve_prompt
from triad_pruning.roi import choose_roi, load_mask


class DataTests(unittest.TestCase):
    def test_user_record_and_img_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "imgs").mkdir()
            (root / "musc").mkdir()
            (root / "imgs" / "000000108.png").touch()
            (root / "musc" / "000000108.png").touch()
            source = root / "questions.jsonl"
            source.write_text(json.dumps({
                "question_id": "000000108", "image": "000000108.png",
                "text": "Is there any defect?", "gt": 1,
                "origin_path": "screw/test/thread_top/005.png",
                "mask": "musc/000000108.png", "musc_scores": 0.62,
            }) + "\n", encoding="utf-8")
            sample, = load_samples(source, root)
            self.assertEqual(sample.sample_id, "000000108")
            self.assertEqual(sample.category, "screw")
            self.assertEqual(sample.image, (root / "imgs" / "000000108.png").resolve())
            self.assertEqual(sample.mask, (root / "musc" / "000000108.png").resolve())
            self.assertEqual(sample.gt, 1)
            self.assertEqual(choose_roi(sample, "randomroi")[0], "mask")
            self.assertEqual(choose_roi(sample, "randompatch")[0], "random")

    def test_prompt_versions_and_fallback(self):
        for version in ("v0", "v1", "v2", "v3"):
            prompt, source = resolve_prompt(version, "screw", "Ignored text")
            self.assertIn("screw", prompt)
            self.assertEqual(source, f"mvtec_{version}")
        self.assertEqual(resolve_prompt("v0", "unknown", "Question?")[0], "Question?")

    def test_bbox_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").touch()
            source = root / "questions.json"
            source.write_text(json.dumps([{
                "question_id": 1, "image": "image.png", "bbox": [[1, 2, 3, 4]],
                "text": "Question?",
            }]), encoding="utf-8")
            sample, = load_samples(source, root)
            self.assertEqual(sample.bbox, [[1, 2, 3, 4]])
            self.assertIsNone(sample.mask)
            self.assertEqual(choose_roi(sample, "randomroi")[0], "bbox")

    def test_legacy_single_image_bbox_and_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "mvtec"
            root.mkdir()
            (root / "image.png").touch()
            source = root / "questions.jsonl"
            source.write_text(json.dumps({
                "question_id": "1", "image": "image.png",
                "origin_path": "mvtec/screw/test/bad.png",
                "bbox": [[[1, 2, 3, 4]]], "text": "Question?",
            }) + "\n", encoding="utf-8")
            sample, = load_samples(source, root)
            self.assertEqual(sample.category, "screw")
            self.assertEqual(sample.bbox, [[1, 2, 3, 4]])

    def test_numpy_anomaly_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.npz"
            np.savez(path, anomaly_map=np.ones((3, 4), dtype=np.float32))
            self.assertEqual(load_mask(path).shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
