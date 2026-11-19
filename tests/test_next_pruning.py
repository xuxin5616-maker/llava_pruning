"""Verify the NeXT deployment contract and its train-to-benchmark handoff."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import benchmark_prefill_kv as benchmark
import run_next_pruning as pipeline
from llava_pruning import next_interleave
from llava_pruning.backend import open_image
from test_mmad_binary import fixture


def make_model(folder):
    config = dict(
        model_type="llava_qwen", num_hidden_layers=32, hidden_size=4096,
        num_attention_heads=32, num_key_value_heads=32, mm_hidden_size=1152,
        mm_patch_merge_type="spatial_unpad", mm_vision_select_layer=-2,
        mm_vision_select_feature="patch", image_aspect_ratio="anyres",
        mm_vision_tower=next_interleave.VISION_REPO,
        image_grid_pinpoints=sorted(next_interleave.NATIVE_PINPOINTS),
    )
    pipeline.common.save_json(folder / "config.json", config)
    pipeline.common.save_json(folder / "model.safetensors.index.json",
                              {"weight_map": {"model.weight": "model-00001.safetensors"}})
    for name in ("model-00001.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (folder / name).write_bytes(b"test")


class NextPruningTests(unittest.TestCase):
    def test_training_and_independent_evaluation_start_before_student_attachment(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = fixture(root)
            model = Path(paths.model_path)
            make_model(model)
            visa = root / "VisA"
            visa.mkdir()
            input_json = root / "questions.jsonl"
            input_json.write_text("{}", encoding="utf-8")
            output = root / "pipeline"
            args = pipeline.parser().parse_args([
                "--output-dir", str(output), "--model-path", str(model),
                "--train-gpu", "0", "--eval-gpus", "1", "2", "3",
                "--visa-root", str(visa), "--mvtec-data-root", str(root),
                "--mvtec-input-json", str(input_json), "--mmad-root", paths.data_root,
                "--mmad-mvtec-root", str(Path(paths.data_root) / "MVTec-AD"),
            ])
            calls = []

            def launch(command, **options):
                script = Path(command[2]).name
                calls.append(script)
                self.assertIn("--next-interleave", command)
                train = script == "train_attention_predictor.py"
                if train:
                    self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "0")
                else:
                    self.assertIn("--without-student", command)
                    start = command.index("--gpus") + 1
                    self.assertEqual(command[start:start + 3], ["1", "2", "3"])
                state = []

                def wait():
                    self.assertEqual(len(calls), 2)
                    if train:
                        worker_output = output / "student"
                        worker_output.mkdir()
                        (worker_output / "best.pt").write_bytes(b"test")
                        pipeline.common.save_json(worker_output / "training_status.json", {"complete": True})
                    else:
                        for mode in ("mvtecmode", "mmadmode"):
                            worker_output = output / mode
                            worker_output.mkdir()
                            pipeline.common.save_json(worker_output / "run.json", {})
                    state.append(0)
                    return 0

                return SimpleNamespace(wait=wait, poll=lambda: state[-1] if state else None)

            def finish(command, **options):
                calls.append("attach")
                self.assertIn("--resume", command)
                self.assertIn("--attach-student", command)
                self.assertEqual(Path(command[command.index("--checkpoint") + 1]), output / "student" / "best.pt")
                return SimpleNamespace(returncode=0)

            with patch.object(pipeline.subprocess, "Popen", side_effect=launch), \
                    patch.object(pipeline.subprocess, "run", side_effect=finish):
                self.assertEqual(pipeline.run(args), 0)
                args.resume = True
                self.assertEqual(pipeline.run(args), 0)
            self.assertEqual(calls, ["train_attention_predictor.py", "benchmark_prefill_kv.py", "attach", "attach"])

    def test_frozen_evaluation_uses_next_geometry_and_preserves_image_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = fixture(root)
            model = Path(paths.model_path)
            make_model(model)
            image = root / "bottle.png"
            Image.new("RGB", (89, 193)).save(image)
            question = root / "questions.jsonl"
            question.write_text('{"question_id":"1","image":"bottle.png","category":"bottle","gt":0}\n',
                                encoding="utf-8")
            checkpoint = Path(paths.checkpoint)
            pipeline.common.save_json(checkpoint.parent / "train_run.json", dict(
                configuration=dict(dataset="visa", model_path=str(model)),
                inference=dict(model_profile=next_interleave.MODEL_NAME, vision_tower=next_interleave.VISION_REPO,
                               image_resize=[1152, 1152], native_anyres_grid=[2, 2])))
            for mode in ("mvtecmode", "mmadmode"):
                folder = root / mode
                args = benchmark.parser().parse_args([
                    mode, "--output-dir", str(folder), "--next-interleave", "--gpu", "5", "--without-student",
                    "--model-path", str(model), "--checkpoint", str(checkpoint),
                    "--data-root", str(root) if mode == "mvtecmode" else paths.data_root,
                    "--input-json", str(question), "--mvtec-root", str(Path(paths.data_root) / "MVTec-AD"),
                ])
                spec = benchmark.setup(args, folder)
                self.assertEqual(benchmark.load_run(folder, mode), spec)
                self.assertIsNone(spec["checkpoint"]["path"])
                self.assertEqual(len(spec["jobs"]), 25)
                pending = [job for job in spec["jobs"] if job["status"] == "pending"]
                self.assertEqual(len(pending), 24)
                self.assertTrue(all(job["gpu"] == 5 and "grid=2x2" in job["config"] for job in pending))
                pyramid = [job for job in spec["jobs"] if job["method"] == "pyramiddrop"]
                self.assertTrue(all("layers=8/16/24" in job["config"] for job in pyramid))
                self.assertEqual(pyramid[-1]["status"], "unreachable")
                first = spec["jobs"][0]
                hybrid = [job for job in pending if job["method"] == "hybrid_baseline"]
                self.assertEqual(len(hybrid), 3)
                self.assertTrue(all(benchmark.awaiting_student(spec, job) for job in hybrid))
                self.assertFalse(benchmark.awaiting_student(spec, first))
                benchmark.write_reports(folder, spec)
                self.assertIn("waiting_for_student", (folder / "summary.csv").read_text())
                args.without_student, args.attach_student, args.resume = False, True, True
                with patch.object(benchmark, "run_trials", return_value=0):
                    self.assertEqual(benchmark.run(args), 0)
                attached = benchmark.load_run(folder, mode)
                self.assertEqual(attached["input_sha256"], spec["input_sha256"])
                self.assertEqual(attached["jobs"], spec["jobs"])
                self.assertEqual(attached["checkpoint"]["path"], str(checkpoint))
                self.assertTrue(all(not benchmark.awaiting_student(attached, job) for job in hybrid))
                if mode == "mmadmode":
                    samples = benchmark.mm.load_frozen(spec["input_json"], exclude_sources=("VisA",))
                    self.assertEqual(len(samples), 8)
                else:
                    samples = benchmark.samples_for(spec)
                    self.assertEqual(len(samples), 1)
                for sample in samples:
                    self.assertEqual(sample.source["image_size"], [1152, 1152])
                    self.assertEqual(open_image(sample.image, next_interleave.INPUT_SIZE).size, (1152, 1152))
                self.assertFalse((folder / "prepared_images").exists())
            self.assertEqual(open_image(image).size, (89, 193))

    def test_pipeline_refuses_an_incomplete_download(self):
        with TemporaryDirectory() as tmp:
            model = Path(tmp)
            make_model(model)
            self.assertEqual(next_interleave.verify_model_download(model), model.resolve())
            (model / "model-00001.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Finish the NeXT checkpoint download"):
                next_interleave.verify_model_download(model)
