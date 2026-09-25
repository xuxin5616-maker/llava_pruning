import tempfile
import unittest
from pathlib import Path

from PIL import Image

from triad_pruning.visualization import save_fastv_visualizations


class VisualizationTests(unittest.TestCase):
    def test_independent_save_switches(self):
        image = Image.new("RGB", (8, 8), "white")
        crop = {"original_size": [8, 8], "roi_boxes": [],
                "processed_view_sizes": [[8, 8]]}
        mask = {"span": [0, 5], "keep": [True, False, True, False, True]}
        attention = {"span": [0, 5], "scores": [0.1, 0.2, 0.3, 0.4, 0.0]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            prune, = save_fastv_visualizations(
                [image], [crop], [mask], base_grid=2, patch_size=4,
                output_dir=output, sample_id="108", fastv_layer=2,
                keep_ratio=0.5, save_prune=True, save_attention=False,
            )
            self.assertTrue(Path(prune["comparison"]).is_file())
            self.assertIsNone(prune["attention_overlay"])
            attention_only, = save_fastv_visualizations(
                [image], [crop], [mask], base_grid=2, patch_size=4,
                output_dir=output, sample_id="108", fastv_layer=2,
                keep_ratio=0.5, image_attentions=[attention],
                save_prune=False, save_attention=True,
            )
            self.assertIsNone(attention_only["comparison"])
            self.assertTrue(Path(attention_only["attention_overlay"]).is_file())
            self.assertTrue(Path(attention_only["directory"]).name == "image_0")


if __name__ == "__main__":
    unittest.main()
