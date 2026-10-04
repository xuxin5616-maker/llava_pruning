"""Exercise selectable image-token order through the real multimodal packer."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from test_reference import ToyPacking
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor


def packing_fixture(order=None, *, final_newline=False, fastv=False):
    toy = ToyPacking(torch.float32)
    if order is not None:
        toy.config.image_token_order = order
    if final_newline:
        toy.config.mm_patch_merge_type += "_add_newl"
    toy.core.fastv_enabled = fastv
    captured = []
    toy.core.set_fastv_image_spans = lambda spans: captured.append(spans)
    return toy, captured


def pack(toy, *, images=None, modalities=None):
    tokens = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
    if images is None:
        images = torch.zeros(1, 3, 3, 8, 8)
    return toy.prepare_inputs_labels_for_multimodal(
        tokens, torch.arange(4)[None], torch.ones_like(tokens, dtype=torch.bool),
        None, tokens.clone(), images, modalities=modalities or ["image"],
        image_sizes=[(16, 8)],
    )


def expected_blocks(toy):
    features = toy.encode_images(torch.zeros(3, 3, 8, 8))
    newline = toy.core.image_newline
    # Two 2x2 crops become two rows of four patches, each followed by newline.
    anyres = torch.stack((features[1, 0], features[1, 1], features[2, 0], features[2, 1], newline,
                          features[1, 2], features[1, 3], features[2, 2], features[2, 3], newline))
    return features[0], anyres


class ImageTokenOrderPackingTests(unittest.TestCase):
    def test_default_and_both_orders_preserve_exact_token_identities_and_tail(self):
        for order in (None, "base_first", "anyres_first"):
            for final_newline in (False, True):
                for fastv in (False, True):
                    with self.subTest(order=order, final_newline=final_newline, fastv=fastv):
                        toy, captured = packing_fixture(order, final_newline=final_newline, fastv=fastv)
                        output = pack(toy)
                        base, anyres = expected_blocks(toy)
                        if order == "anyres_first" and not final_newline:
                            parts = [anyres[:-1], base, anyres[-1:]]
                        else:
                            parts = [anyres, base] if order == "anyres_first" else [base, anyres]
                            if final_newline:
                                parts.append(toy.core.image_newline[None])
                        expected = torch.cat(parts)
                        torch.testing.assert_close(output[4][0, 1:-2], expected, rtol=0, atol=0)
                        torch.testing.assert_close(output[4][0, -3], toy.core.image_newline, rtol=0, atol=0)
                        torch.testing.assert_close(output[1][0], torch.arange(17 + int(final_newline)))
                        self.assertTrue(output[2].all())
                        self.assertEqual(captured, [[[(1, 15 + int(final_newline))]]] if fastv else [])

    def test_switching_order_leaves_text_masks_positions_and_labels_unchanged(self):
        for final_newline in (False, True):
            with self.subTest(final_newline=final_newline):
                toy, _ = packing_fixture(final_newline=final_newline)
                default = pack(toy)
                toy.config.image_token_order = "base_first"
                base_first = pack(toy)
                for expected, observed in zip(default, base_first):
                    if expected is None:
                        self.assertIsNone(observed)
                    else:
                        self.assertTrue(torch.equal(observed, expected))
                toy.config.image_token_order = "anyres_first"
                swapped = pack(toy)
                for index in (0, 1, 2, 3, 5):
                    if default[index] is None:
                        self.assertIsNone(swapped[index])
                    else:
                        self.assertTrue(torch.equal(swapped[index], default[index]))
                self.assertTrue(torch.equal(swapped[4][:, :1], default[4][:, :1]))
                self.assertTrue(torch.equal(swapped[4][:, -3:], default[4][:, -3:]))

    def test_anyres_first_rejects_unsupported_packing_instead_of_silently_ignoring_order(self):
        for aspect, merge, modality in (
                ("randomroi", "spatial_unpad", "image"),
                ("anyres_max_9_randomroi", "spatial_unpad", "image"),
                ("anyres", "spatial_unpad", "image"),
                ("anyres_max_9", "flat", "image"),
                ("anyres_max_9", "spatial_unpad_nobase", "image"),
                ("anyres_max_9", "spatial_unpad", "video")):
            with self.subTest(aspect=aspect, merge=merge, modality=modality):
                toy, _ = packing_fixture("anyres_first")
                toy.config.image_aspect_ratio = aspect
                toy.config.mm_patch_merge_type = merge
                with self.assertRaisesRegex(ValueError, "image_token_order=anyres_first requires"):
                    pack(toy, modalities=[modality])
        for images in (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 3, 8, 8)):
            toy, _ = packing_fixture("anyres_first")
            with self.assertRaisesRegex(ValueError, "requires a base view and anyres views"):
                pack(toy, images=images)
        toy, _ = packing_fixture("unknown")
        with self.assertRaisesRegex(ValueError, "Unknown image_token_order"):
            pack(toy)

    def test_preprocessing_records_order_without_changing_the_input_pixels(self):
        processor = SigLipImageProcessor(size=(28, 28), crop_size={"height": 28, "width": 28})
        config = SimpleNamespace(image_aspect_ratio="anyres_max_9", mm_patch_merge_type="spatial_unpad",
                                 image_grid_pinpoints=[[28, 28], [56, 28]])
        pixels = np.arange(28 * 56 * 3, dtype=np.uint8).reshape(28, 56, 3)
        image = Image.fromarray(pixels)
        reference, metadata = process_images([image], processor, config, return_pro_data=True)
        self.assertEqual(metadata[0]["image_token_order"], "base_first")
        for order in ("base_first", "anyres_first"):
            config.image_token_order = order
            actual, metadata = process_images([image], processor, config, return_pro_data=True)
            self.assertTrue(torch.equal(actual, reference))
            self.assertEqual(metadata[0]["image_token_order"], order)


if __name__ == "__main__":
    unittest.main()
