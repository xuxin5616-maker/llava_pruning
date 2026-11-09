"""Check crop provenance through native AnyRes packing and actual generation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from llava import mm_utils
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor

import benchmark_crop_ablation as entry
from llava_pruning.backend import LlavaBackend
from llava_pruning.crop_ablation import remove_crop
from runMyPruningPoolSpatial import TokenDecisions, pack_full_views
from test_ficoco_benchmark import fixture
from test_pool_ablations import model_and_options


class CropAblationTests(unittest.TestCase):
    def test_crop_features_do_not_survive_max9_interpolation(self):
        grid, side, image_size = (4, 4), 27, (1536, 1536)
        features = torch.zeros(17, side ** 2, 1)
        features[6] = 1
        decisions = TokenDecisions(side ** 2, side ** 2, torch.arange(side ** 2).expand(17, -1))
        packed, _ = pack_full_views(features, decisions, grid, image_size, torch.tensor([-1.]),
                                    "spatial_unpad_add_newl")
        kept = remove_crop(packed, grid, side, image_size, 5, True)
        self.assertGreater(int(((packed > 0) & (packed < 1)).sum()), 0)
        self.assertTrue(torch.equal(kept, packed[packed[:, 0] <= 0]))

    def test_unpadding_and_single_crop_removal(self):
        side = 3
        for grid, size, crop, expected in [((1, 2), (384, 512), 0, 2), ((1, 1), (384, 384), 0, 0)]:
            with self.subTest(grid=grid):
                views = 1 + grid[0] * grid[1]
                features = torch.zeros(views, side ** 2, 1)
                features[1 + crop] = 1
                decisions = TokenDecisions(side ** 2, side ** 2, torch.arange(side ** 2).expand(views, -1))
                packed, _ = pack_full_views(features, decisions, grid, size, torch.tensor([-1.]), "spatial_unpad")
                kept = remove_crop(packed, grid, side, size, crop, False)
                self.assertEqual(len(kept), side ** 2 + expected * (side + 1))
                self.assertFalse((kept > 0).any())

    def test_two_data_scopes_all_modes_and_actual_prefill_lengths(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                legacy = fixture(root)
                args = entry.parser().parse_args([
                    "--model-path", legacy.model_path, "--mvtec-input-json", legacy.mvtec_input_json,
                    "--mvtec-data-root", legacy.mvtec_data_root, "--mmad-root", legacy.mmad_root,
                    "--output-dir", str(legacy.output_dir), "--warmup", "0",
                ])
                args.mmad_mvtec_root = None
                for image in Path(args.mvtec_data_root).glob("*.png"):
                    Image.new("RGB", (768, 384)).save(image)
                output, spec = entry.setup(args)
                self.assertEqual(spec["datasets"]["mmad"]["info"]["excluded_sources"], ["VisA"])
                self.assertEqual({name: ds["gpu"] for name, ds in spec["datasets"].items()}, {"mvtec": "5", "mmad": "6"})
                model, options = model_and_options(2)
                core = model.get_model()
                model.config.max_position_embeddings = 4096
                model.config.image_grid_pinpoints = [[768, 384]]
                tower = core.get_vision_tower()
                features = torch.randn(3, 729, 8, generator=torch.Generator().manual_seed(1))
                features[1] = 1  # most redundant; least important
                backend = LlavaBackend.__new__(LlavaBackend)
                backend.model, backend.vision_tower = model, tower
                backend.processor = SigLipImageProcessor()
                backend.inference_config = {"gpu": "CPU test fixture"}
                backend.tokenizer = type("Tokenizer", (), {"batch_decode": lambda *a, **k: ["B"]})()
                samples = entry.frozen_samples(spec["datasets"]["mvtec"], "mvtec")
                visual_counts = {"base_anyres": 2214, "base_only": 729, "anyres_only": 1485,
                                 "drop_most_important": 1485, "drop_least_important": 1485}
                native_generate = model.generate
                lengths = []
                handle = core.layers[0].register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
                try:
                    with patch.object(tower, "forward", side_effect=lambda pixels: features[:len(pixels)]), \
                            patch.object(mm_utils, "tokenizer_image_token", return_value=options["inputs"][0]), \
                            patch.object(model, "generate", side_effect=lambda **kw: native_generate(**dict(kw, max_new_tokens=1))), \
                            patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "reset_peak_memory_stats"):
                        for mode, count in visual_counts.items():
                            folder = output / "mvtec" / mode
                            folder.mkdir(parents=True)
                            job = dict(dataset="mvtec", mode=mode, prune_rate=0)
                            result = entry.evaluate(backend, samples, spec, job, folder, 2**30)
                            self.assertEqual(lengths[-2:], [count + 4] * 2)
                            self.assertEqual(result["budget"]["input_image_tokens"], count)
                            if mode in entry.DROP_MODES:
                                row = json.loads((folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
                                self.assertEqual(row["budget"]["crop_index"], int(mode == "drop_most_important"))
                            entry.common.save_json(folder / "metrics.json", dict(result, status="success"))
                finally:
                    handle.remove()
                entry.write_reports(output, spec)
                with (output / "summary.csv").open(encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 10)
                self.assertEqual([row["status"] for row in rows], ["success"] * 5 + ["pending"] * 5)
                self.assertFalse(list(output.rglob("*.png")))
        finally:
            torch.set_num_threads(threads)
