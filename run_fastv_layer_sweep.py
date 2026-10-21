"""FastV 70% per-Base/AnyRes budgets, independent layers 2..21, selected PNGs.

Each layer uses a fresh process on the same GPU. All input questions are
evaluated; --visualize-samples controls image export only. Default server paths
match the existing MVTec experiments. No sample limit or final single-rate plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent
DEFAULT_VISUALIZE = (
    "sample_000000000", "sample_000000270", "sample_000000340", "sample_000000422",
    "sample_000000538", "sample_000000723", "sample_000000940", "sample_000001038",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=5, help="Physical GPU index; one layer experiment runs at a time")
    parser.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    parser.add_argument("--input-json", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl")
    parser.add_argument("--data-root", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/")
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(2, 22)))
    parser.add_argument("--visualize-samples", nargs="+", default=list(DEFAULT_VISUALIZE))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output" / "fastv70_layers2_21")
    parser.add_argument("--worker-layer", type=int, help=argparse.SUPPRESS)
    return parser


def run_worker(args):
    # Parent restricts CUDA visibility BEFORE the worker imports the backend.
    from llava_pruning.runner import run

    root = args.output_dir.expanduser().resolve()
    return run(
        model_path=args.model_path, input_json=args.input_json, data_root=args.data_root,
        prompt_version=args.prompt_version, method_name="fastv",
        method_config=root / "configs" / f"layer_{args.worker_layer:02d}.json",
        roi_mode="anyres_max_9", save_prune_vis=True, save_attention_vis=True,
        visualize_samples=args.visualize_samples,
        output_dir=root / f"layer_{args.worker_layer:02d}", seed=args.seed,
        no_sample=True, include_pruning_time=True,
    )


def run_sweep(args):
    from llava_pruning.data import load_samples
    from llava_pruning.selection import resolve_visualization_ids

    if args.gpu < 0:
        raise ValueError("gpu must be nonnegative")
    if (not args.layers or any(layer < 1 for layer in args.layers)
            or len(set(args.layers)) != len(args.layers)):
        raise ValueError("layers must be distinct positive integers")
    samples = load_samples(args.input_json, args.data_root)
    selected = resolve_visualization_ids(args.visualize_samples, samples)
    if not samples or not 0 <= args.seed <= 2**32 - len(samples):
        raise ValueError("Input must contain samples and seed + index must fit NumPy's range")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir()
    config = json.loads((PROJECT / "configs" / "fastv70_base_anyres.json").read_text(encoding="utf-8"))
    for layer in args.layers:
        effective = dict(config, layer=layer)
        (output / "configs" / f"layer_{layer:02d}.json").write_text(
            json.dumps(effective, indent=2), encoding="utf-8")
    metadata = {"method": "fastv", "prune_rate": 70, "layers": args.layers,
                "budget_mode": "base_anyres", "budget_scope": "Base and ALL AnyRes spatial tokens separately; newlines protected",
                "samples": len(samples), "visualize_samples": sorted(selected), "gpu": args.gpu,
                "model_path": args.model_path, "input_json": args.input_json, "data_root": args.data_root,
                "prompt_version": args.prompt_version, "seed": args.seed, "do_sample": False,
                "execution": "one fresh process per independent layer; all input samples evaluated",
                "timing": "per-sample full generate seconds in predictions.jsonl; Lat/Mem benchmark is not enabled"}
    (output / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    summary_fields = None
    print(f"FastV: 70% per Base/AnyRes group; {len(samples)} evaluation samples; "
          f"{len(selected)} visualization IDs; layers={args.layers}", flush=True)
    for layer in args.layers:
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--worker-layer", str(layer), "--output-dir", str(output),
                   "--model-path", args.model_path, "--input-json", args.input_json,
                   "--data-root", args.data_root, "--prompt-version", args.prompt_version,
                   "--seed", str(args.seed), "--visualize-samples", *args.visualize_samples]
        print(f"Starting independent layer={layer} on GPU {args.gpu}", flush=True)
        subprocess.run(command, cwd=PROJECT, env=environment, check=True)
        with (output / f"layer_{layer:02d}" / "summary.csv").open(encoding="utf-8", newline="") as stream:
            [row] = list(csv.DictReader(stream))
        row = {"fastv_layer": layer, "budget_mode": "base_anyres", **row}
        with (output / "summary.csv").open("a", encoding="utf-8", newline="") as stream:
            if summary_fields is None:
                summary_fields = list(row)
                csv.DictWriter(stream, fieldnames=summary_fields).writeheader()
            csv.DictWriter(stream, fieldnames=summary_fields).writerow(row)
    return output


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = run_worker(args) if args.worker_layer is not None else run_sweep(args)
        print(f"Saved results to {output}", flush=True)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
