"""Random image subsets keep native preset options and persist across resume."""

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import benchmark_mmad_binary as mm
import benchmark_mode as mode
import run_pool_distill as mv
from test_mmad_binary import fixture


class BenchmarkModeSamplingTests(unittest.TestCase):
    def test_mvtec_freezes_100_distinct_images_for_the_sweep(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "input.jsonl"
            rows = []
            for index in range(140):
                image = f"{index}.png"
                (root / image).touch()
                rows.append(dict(question_id=str(index), image=image, gt=index % 2))
            rows.append(dict(rows[0], question_id="duplicate_image"))
            questions.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            checkpoint = root / "best.pt"
            checkpoint.touch()
            output = root / "results"
            preset = mode.parse_args(["mvtecmode", "--samples", "100", "--output-dir", str(output),
                "--input-json", str(questions), "--data-root", str(root), "--model-path", str(root),
                "--checkpoint", str(checkpoint)])
            args = mv.build_parser().parse_args(mode.command_for(preset)[2:])
            calls = []

            def interrupt(command, **kwargs):
                calls.append(mv.build_parser().parse_args(command[2:]))
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                mv.run_sweep(args, sample_count=preset.samples, sample_seed=preset.sample_seed, run_process=interrupt)
            manifest = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sampling"]["eligible_images"], 140)
            self.assertEqual(manifest["prune_rates"], [50, 70, 90])
            subset = calls[0].input_json
            selected = [json.loads(line) for line in subset.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(selected), 100)
            self.assertEqual(len({row["image"] for row in selected}), 100)
            self.assertNotEqual(selected, rows[:100])
            resume_cli = ["--resume", "--output-dir", str(output)]
            resumed = mv.restore_sweep_args(mv.build_parser().parse_args(resume_cli), resume_cli)
            self.assertEqual(resumed.input_json, subset)
            with subset.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "sample list changed"):
                mv.run_sweep(resumed, run_process=interrupt)

    def test_mmad_freezes_unique_images_and_stretches_only_the_subset(self):
        with TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp))
            model = Path(paths.model_path)
            mm.common.save_json(model / "config.json", dict(num_hidden_layers=28, image_grid_pinpoints=[[1152, 1152]]))
            output = paths.output_dir.resolve()
            preset = mode.parse_args(["mmadmode", "--samples", "5", "--sample-seed", "8", "--output-dir", str(output),
                "--data-root", paths.data_root, "--model-path", paths.model_path, "--checkpoint", paths.checkpoint,
                "--mvtec-root", str(Path(paths.data_root) / "MVTec-AD")])
            args = mm.parser().parse_args(mode.command_for(preset)[2:])
            a, info = mm.load_evaluation_data(args)
            self.assertEqual((a, info), mm.load_evaluation_data(args))
            self.assertEqual(info["sampling"], dict(seed=8, images=5, eligible_images=6))
            self.assertEqual(info["source_counts"], dict(Counter(row["dataset"] for row in a)))
            self.assertEqual(len({row["resolved_image"] for row in a}), 5)
            self.assertNotIn("VisA", info["source_counts"])
            _, spec = mm.setup(args)
            self.assertEqual(spec["sample_count"], 5)
            self.assertEqual([j["prune_rate"] for j in spec["jobs"]], [0, 50])
            self.assertEqual(len(list((output / "prepared_images").glob("*.png"))), 5)
            frozen, _ = mm.common.read_prefix(spec["input_json"])
            self.assertEqual([r["question_id"] for r in frozen], [r["question_id"] for r in a])
            resume_args = mm.parser().parse_args(["--resume", "--output-dir", str(output)])
            self.assertEqual(mm.setup(resume_args), (output, spec))
