"""Layer orchestration and sample selection without GPU/checkpoint inference."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import run_fastv_layer_sweep as entry
from llava_pruning.runner import run
from test_runner import FakeBackend


class LayerSweepTests(unittest.TestCase):
    def fixture(self, root, ids):
        Image.new("RGB", (16, 8), "white").save(root / "image.png")
        source = root / "questions.json"
        source.write_text(json.dumps([{"question_id": sample_id, "image": "image.png", "gt": 1,
                                       "origin_path": "screw/test/bad.png"} for sample_id in ids]), encoding="utf-8")
        return source

    def test_default_twenty_independent_workers_and_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [sample.removeprefix("sample_") for sample in entry.DEFAULT_VISUALIZE] + ["other"]
            source = self.fixture(root, ids)
            args = entry.build_parser().parse_args(["--input-json", str(source), "--data-root", str(root),
                                                    "--output-dir", str(root / "output"), "--gpu", "4"])
            self.assertEqual(args.layers, list(range(2, 22)))
            commands = []

            def child(command, **kwargs):
                commands.append(command)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "4")
                self.assertTrue(kwargs["check"])
                layer = int(command[command.index("--worker-layer") + 1])
                config = json.loads((root / "output/configs" / f"layer_{layer:02d}.json").read_text())
                self.assertEqual(config["layer"], layer)
                self.assertEqual(config["prune_rates"], [70])
                self.assertEqual(config["budget_mode"], "base_anyres")
                folder = root / "output" / f"layer_{layer:02d}"
                folder.mkdir()
                (folder / "summary.csv").write_text("prune_rate,accuracy\n70,0.9\n", encoding="utf-8")

            with patch.object(entry.subprocess, "run", side_effect=child), contextlib.redirect_stdout(io.StringIO()):
                output = entry.run_sweep(args)
            self.assertEqual(len(commands), 20)
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["fastv_layer"]) for row in rows], list(range(2, 22)))
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(metadata["samples"], 9)
            self.assertEqual(len(metadata["visualize_samples"]), 8)

    def test_runner_filters_pngs_only_and_exports_both_budgets_for_every_sample(self):
        captures = []

        class Backend(FakeBackend):
            def generate(self, sample, prompt, method, rate, roi_mode, **kwargs):
                captures.append((sample.sample_id, kwargs["capture_visualization"], kwargs["capture_attention"]))
                result = super().generate(sample, prompt, method, rate, roi_mode, **kwargs)
                result["stats"].update(budget_mode="base_anyres", batch=[{"groups": [
                    {"image_index": 0, "name": name, "candidate_tokens": 10,
                     "kept_tokens": 3, "masked_tokens": 7, "actual_prune_rate": 70}
                    for name in ("base", "anyres")]}])
                return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root, ["000000000", "000000270", "000000001"])
            with patch("llava_pruning.runner.LlavaBackend", Backend), contextlib.redirect_stdout(io.StringIO()):
                output = run(model_path=root, input_json=source, data_root=root, prompt_version="v0",
                             method_name="fastv", method_config=entry.PROJECT / "configs/fastv70_base_anyres.json",
                             roi_mode="anyres_max_9", save_prune_vis=True, save_attention_vis=True,
                             visualize_samples=["sample_000000000", "000000270"], output_dir=root / "output", seed=42)
            self.assertEqual(captures, [("000000000", True, True), ("000000270", True, True), ("000000001", False, False)])
            folder = output / "prune_70"
            self.assertEqual(len((folder / "predictions.jsonl").read_text().splitlines()), 3)
            self.assertEqual(json.loads((folder / "metrics.json").read_text())["evaluated_samples"], 3)
            self.assertEqual({path.parent.parent.name for path in (folder / "visualizations").rglob("comparison.png")},
                             {"sample_000000000", "sample_000000270"})
            with (folder / "view_budgets.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["name"] for row in rows}, {"base", "anyres"})

    def test_missing_visualization_id_fails_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root, ["000000000"])
            with patch("llava_pruning.runner.LlavaBackend") as backend, self.assertRaisesRegex(ValueError, "not found"):
                run(model_path=root, input_json=source, data_root=root, prompt_version="v0",
                    method_name="fastv", method_config=entry.PROJECT / "configs/fastv70_base_anyres.json",
                    roi_mode="anyres_max_9", save_prune_vis=True, save_attention_vis=True,
                    visualize_samples=["sample_000000270"], output_dir=root / "output")
            backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
