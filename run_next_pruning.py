"""Train a VisA student while benchmarking independent NeXT pruning methods."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys

import benchmark_visa_methods as common
from benchmark_fastv import DEFAULT_DATA_ROOTS, DEFAULT_MVTEC_JSON
from llava_pruning.next_interleave import DEFAULT_MODEL_PATH, verify_model_download, vision_tower_path
from run_mvtec_visa_pipeline import interrupt


PROJECT = Path(__file__).resolve().parent
DATA = Path("/home/yz/xxy/data/datasets")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-gpu", type=int, default=0, help="Physical GPU for VisA distillation")
    p.add_argument("--eval-gpus", type=int, nargs="+", default=[1, 2, 3],
                   help="Evaluation GPUs. The first half serves MVTec; the rest serves MMAD.")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--vision-tower-path", type=Path, help="Override the local SigLIP directory")
    p.add_argument("--visa-root", type=Path, default=DATA / "VisA")
    p.add_argument("--visa-split-csv", type=Path)
    p.add_argument("--mvtec-data-root", type=Path, default=Path(DEFAULT_DATA_ROOTS["mvtec"]))
    p.add_argument("--mvtec-input-json", type=Path, default=Path(DEFAULT_MVTEC_JSON))
    p.add_argument("--mmad-root", type=Path, default=DATA / "MMAD")
    p.add_argument("--mmad-annotations", type=Path)
    p.add_argument("--mmad-image-root", type=Path)
    p.add_argument("--mmad-mvtec-root", type=Path, default=DATA / "MVTec-AD")
    p.add_argument("--resume", action="store_true",
                   help="Reuse completed training and continue incomplete evaluation")
    return p


def command(script, options):
    return [sys.executable, "-u", str(PROJECT / script), *options]


def run_independent(output, training, evaluation, train_gpu, *, train_required):
    processes = {}
    with ExitStack() as logs:
        try:
            if train_required:
                log = output / "training.log"
                processes["training"] = subprocess.Popen(
                    command("train_attention_predictor.py", training), cwd=PROJECT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(train_gpu)),
                    stdout=logs.enter_context(log.open("a", encoding="utf-8")),
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=os.name == "posix",
                )
                print(f"VisA distillation started on GPU {train_gpu}. Log: {log}", flush=True)
            log = output / "independent_evaluation.log"
            processes["evaluation"] = subprocess.Popen(
                command("benchmark_prefill_kv.py", evaluation), cwd=PROJECT,
                stdout=logs.enter_context(log.open("a", encoding="utf-8")),
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=os.name == "posix",
            )
            print(f"Independent evaluation started. Log: {log}", flush=True)
            return {name: process.wait() for name, process in processes.items()}
        finally:
            for process in processes.values():
                interrupt(process)


def run(args):
    gpus = [args.train_gpu, *args.eval_gpus]
    if any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("Provide distinct, nonnegative training and evaluation GPU IDs")
    model = verify_model_download(args.model_path)
    tower = vision_tower_path(model, args.vision_tower_path)
    output = args.output_dir.expanduser().resolve()
    student_dir = output / "student"
    checkpoint = student_dir / "best.pt"
    status_path = student_dir / "training_status.json"
    launch = output / "pipeline.json"

    for path in (args.visa_root, args.mvtec_data_root, args.mmad_root, args.mmad_mvtec_root):
        if not path.is_dir():
            raise NotADirectoryError(f"Data directory not found: {path}")
    from llava_pruning.mmad_data import find_annotations
    annotations = find_annotations(args.mmad_root, args.mmad_annotations)
    if not args.mvtec_input_json.is_file():
        raise FileNotFoundError(f"MVTec question file not found: {args.mvtec_input_json}")

    model_args = ["--next-interleave", "--model-path", str(model)]
    if tower is not None:
        model_args.extend(("--vision-tower-path", tower))
    training_args = [*model_args, "--dataset", "visa", "--data-root", str(args.visa_root.resolve()),
                     "--output-dir", str(student_dir)]
    if args.visa_split_csv is not None:
        training_args.extend(("--visa-split-csv", str(args.visa_split_csv.resolve())))

    eval_args = ["both", *model_args, "--output-dir", str(output),
                 "--gpus", *map(str, args.eval_gpus),
                 "--data-root", str(args.mvtec_data_root.resolve()),
                 "--input-json", str(args.mvtec_input_json.resolve()),
                 "--mmad-data-root", str(args.mmad_root.resolve()),
                 "--annotations", str(annotations.resolve()),
                 "--mvtec-root", str(args.mmad_mvtec_root.resolve())]
    if args.mmad_image_root is not None:
        eval_args.extend(("--image-root", str(args.mmad_image_root.resolve())))
    spec = dict(train_gpu=args.train_gpu, eval_gpus=args.eval_gpus, training=training_args,
                evaluation=eval_args, checkpoint=str(checkpoint),
                timing="Independent trials may overlap training on a separate GPU; hybrid trials run after distillation.")
    if args.resume:
        if json.loads(launch.read_text(encoding="utf-8")) != spec:
            raise ValueError("Pipeline settings changed. Resume with the original command and --resume.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use a new/empty output directory or --resume: {output}")
        output.mkdir(parents=True, exist_ok=True)
        common.save_json(launch, spec)

    training_complete = status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("complete")
    eval_manifests = [output / mode / "run.json" for mode in ("mvtecmode", "mmadmode")]
    if not training_complete and student_dir.exists() and any(student_dir.iterdir()):
        raise ValueError(f"Training is incomplete. After its worker stops, move {student_dir} aside and rerun with --resume.")
    if not training_complete or not all(path.is_file() for path in eval_manifests):
        independent = [*eval_args, "--without-student"]
        if any(path.is_file() for path in eval_manifests):
            independent.append("--resume")
        codes = run_independent(output, training_args, independent, args.train_gpu,
                                train_required=not training_complete)
        if codes.get("training", 0):
            raise ValueError(f"VisA distillation failed. See {output / 'training.log'}")
        training_complete = json.loads(status_path.read_text(encoding="utf-8")).get("complete")
        if codes["evaluation"] and not all(path.is_file() for path in eval_manifests):
            raise ValueError(f"Evaluation setup failed. See {output / 'independent_evaluation.log'}")
    if not training_complete or not checkpoint.is_file():
        raise ValueError("Distillation must finish and produce best.pt before evaluation")

    print(f"Distillation complete. Attaching {checkpoint} and running unfinished trials.", flush=True)
    options = [*eval_args, "--resume", "--attach-student", "--checkpoint", str(checkpoint)]
    result = subprocess.run(command("benchmark_prefill_kv.py", options), cwd=PROJECT, check=False)
    for mode in ("mvtecmode", "mmadmode"):
        summary = output / mode / "summary.csv"
        if summary.is_file():
            print(f"{mode} summary: {summary}", flush=True)
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(run(parser().parse_args()))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Error: {error}")
    except KeyboardInterrupt:
        sys.exit(130)
