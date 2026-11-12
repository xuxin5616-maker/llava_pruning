from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image
import torch

import visualize_fastv_visionzip as entry
from test_pool_ablations import model_and_options
from test_pool_spatial import _Tower


class PruningVisualizationTests(TestCase):
    @torch.inference_mode()
    def test_overlay_uses_actual_method_decisions(self):
        model, options = model_and_options(28)
        core = model.get_model()
        core.config.image_grid_pinpoints = [[768, 384]]
        core.vision_tower = _Tower()
        core.vision_tower.features = core.vision_tower.features[:3]
        options.update(images=[torch.zeros(3, 3, 384, 384)], image_sizes=[(768, 384)], max_new_tokens=2)
        metadata = dict(grid_patches=[2, 1], original_size=[768, 384], final_newline=False)
        pixels = np.full((3, 384, 384, 3), (11, 51, 101), dtype=np.uint8)
        methods = (entry.AverageFastV(50), entry.VisionZipMethod(core, 50, capture_anchors=True),
                   entry.make_method("clse", core, 50, None))
        for method in methods:
            with self.subTest(method=method.name), TemporaryDirectory() as temporary:
                if method.name == "clse":
                    method.configure(core, 50, capture_visualization=True)
                else:
                    method.configure(core, 50)
                    method.prepare_inputs((768, 384), options["inputs"])
                context = method.inference_context(model) if method.name == "visionzip" else nullcontext()
                with context:
                    model.generate(**options)
                stats = method.stats(core, 50)
                if method.name == "visionzip":
                    keeps = entry.visionzip_view_anchors(method.adapter, 27)
                    self.assertEqual(keeps.sum(axis=1).tolist(), stats["raw_view_keeps"])
                else:
                    if method.name == "clse":
                        stage, = method.visualization_data(core)["stages"]
                        self.assertEqual(stage["after_layer"], 1)
                        image_mask = stage["mask"]
                        retained = stats["final_llm_visual_tokens"]
                    else:
                        (image_mask,), = core.get_fastv_image_masks()
                        retained = stats["retained_visible_image_tokens"]
                    keeps = entry.packed_view_keeps(image_mask, metadata, 27)
                    row_newlines = np.asarray(image_mask["keep"][729:]).reshape(27, 55)[:, -1]
                    self.assertEqual(keeps.sum() + row_newlines.sum(), retained)
                directory = Path(temporary)
                entry.save_views(pixels, keeps, (2, 1), 14, directory)
                self.assertEqual({path.name for path in directory.iterdir()}, {"base.png", "anyres.png"})
                with Image.open(directory / "base.png") as base, Image.open(directory / "anyres.png") as anyres:
                    self.assertEqual(base.size, (384, 384))
                    self.assertEqual(anyres.size, (768, 384))
                    for view in range(3):
                        image = base if view == 0 else anyres
                        x_offset = max(0, view - 1) * 384
                        for index, kept in enumerate(keeps[view]):
                            row, col = divmod(index, 27)
                            expected = (11, 51, 101) if kept else (133, 153, 178)
                            self.assertEqual(image.getpixel((x_offset + col * 14, row * 14)), expected)
                        self.assertEqual(image.getpixel((x_offset, 383)), (11, 51, 101))

    def test_packed_mask_leaves_unpacked_padding_unshaded_and_projects_max9(self):
        for width, height, image_size in ((2, 1, (768, 256)), (4, 4, (1536, 1536))):
            side = 27
            from llava.model.llava_arch import unpad_image
            rows, cols = unpad_image(torch.zeros(1, height * side, width * side), image_size).shape[1:]
            scale = ((rows * cols) / (9 * side**2))**.5
            pooled_rows, pooled_cols = (int(rows // scale), int(cols // scale)) if scale > 1.1 else (rows, cols)
            length = side**2 + pooled_rows * (pooled_cols + 1) + 1
            metadata = dict(grid_patches=[width, height], original_size=image_size, final_newline=True)
            keeps = entry.packed_view_keeps({"span": [7, 7 + length], "keep": [False] * length}, metadata, side)
            self.assertFalse(keeps[0].any())
            self.assertEqual((~keeps[1:]).sum(), rows * cols)
