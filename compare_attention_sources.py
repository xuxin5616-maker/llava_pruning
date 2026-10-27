"""One command: own-AnyRes vs Base-mapped scores, same Base-full budgets.

Use isolated SEQUENTIAL processes on one GPU so visualization/allocator caches
from one trial do not enter the other trial's peaks. Each child loads once,
evaluates ALL input records at30/50/70, and optionally visualizes ONLY70 after
all its clean measurements. Existing backbone and side weights are unchanged.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent
RATES = (30, 50, 70)
TRIALS = (("anyres_self", "own_predicted_anyres_attention"),
          ("base_full", "base_predicted_attention_mapped_to_anyres"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=5, help="One physical GPU ID; trials run sequentially, NOT concurrently")
    parser.add_argument("--model-path", type=Path, default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/"))
    parser.add_argument("--input-json", type=Path, default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/"))
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "output/attention_predictor_twig_train_01/best.pt")
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    parser.add_argument("--image-token-order", choices=("base_first", "anyres_first"), default="base_first")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup images per rate, excluded from metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize-samples", nargs="+", default=None, help="Optional IDs; ALL records still evaluated")
    parser.add_argument("--no-visualizations", action="store_true", help="Metrics only; default saves BOTH PNG types at70 only")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output/attention_sources_30_50_70_01")
    return parser


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_trial(path, mode, source):
    with (path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 3 or [float(row["prune_rate"]) for row in records] != list(RATES):
        raise RuntimeError(f"Unexpected rates in {path / 'summary.csv'}")
    required = {"accuracy", "precision", "recall", "tnr", "lat_mean_ms", "mem_peak_allocated_gib"}
    for row in records:
        if not required.issubset(row) or row.get("complete", "").lower() != "true":
            raise RuntimeError(f"Missing complete classification/Lat/Mem results for {mode}")
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if metadata["method_details"]["selection_mode"] != mode:
        raise RuntimeError(f"Wrong selection policy recorded for {mode}; sync all updated code")
    return [dict(selection_mode=mode, attention_source=source, **row) for row in records]


def run_comparison(args, *, run_process=None):
    run_process = subprocess.run if run_process is None else run_process
    if args.gpu < 0 or args.warmup < 0 or not 0 <= args.seed < 2 ** 32:
        raise ValueError("GPU/warmup must be nonnegative; seed must fit uint32")
    paths = {name: Path(getattr(args, name)).expanduser().resolve()
             for name in ("model_path", "input_json", "data_root", "checkpoint", "output_dir")}
    for name in ("model_path", "data_root"):
        if not paths[name].is_dir():
            raise NotADirectoryError(f"{name} not found: {paths[name]}")
    for name in ("input_json", "checkpoint"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name} not found: {paths[name]}; use the existing training output, no retraining needed")
    output = paths["output_dir"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / "configs"
    config_dir.mkdir()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                    gpu=args.gpu, sequential_isolated_processes=True, backbone_loads_requested=2,
                    checkpoint=str(paths["checkpoint"]), model_path=str(paths["model_path"]),
                    input_json=str(paths["input_json"]), data_root=str(paths["data_root"]),
                    prune_rates=list(RATES), visualize_rates=[] if args.no_visualizations else [70],
                    seed=args.seed, warmup_per_rate=args.warmup,
                    base_policy="all Base and newlines protected in BOTH trials",
                    budget="same WHOLE-image LLM-layer-average target, same original packed denominator",
                    attention="three-layer distilled prediction in BOTH trials; NOT native teacher attention",
                    timing="clean measurements before extra visualization calls in each child; loading/extra calls not included in Lat/Mem",
                    trials=[])
    manifest_path = output / "comparison.json"
    _write_json(manifest_path, manifest)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    combined = []
    for ordinal, (mode, source) in enumerate(TRIALS, 1):
        config = config_dir / f"{mode}.json"
        _write_json(config, dict(checkpoint=str(paths["checkpoint"]), layer=2, rate_semantics="layer_average",
                                 selection_mode=mode, prune_rates=list(RATES),
                                 visualize_rates=[] if args.no_visualizations else [70]))
        trial_output = output / mode
        command = [sys.executable, str(PROJECT / "run.py"), "--model-path", str(paths["model_path"]),
                   "--input-json", str(paths["input_json"]), "--data-root", str(paths["data_root"]),
                   "--prompt-version", args.prompt_version, "--roi-mode", "anyres_max_9",
                   "--image-token-order", args.image_token_order, "--method", "attention_predictor",
                   "--method-config", str(config), "--no-sample", "--benchmark-metrics",
                   "--no-metric-plot",
                   "--warmup", str(args.warmup), "--seed", str(args.seed), "--output-dir", str(trial_output)]
        if args.no_visualizations:
            command.append("--no-plots")
        else:
            command.extend(("--save-prune-vis", "--save-attention-vis"))
            if args.visualize_samples:
                command.extend(("--visualize-samples", *args.visualize_samples))
        trial = dict(selection_mode=mode, attention_source=source, directory=str(trial_output),
                     command=command, status="running")
        manifest["trials"].append(trial)
        _write_json(manifest_path, manifest)
        print(f"Trial {ordinal}/2: {mode}; rates30/50/70; visualizations70 only; GPU{args.gpu}", flush=True)
        try:
            completed = run_process(command, cwd=str(PROJECT), env=environment, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{mode} exited with code {completed.returncode}; inspect the traceback above. "
                                   "Infeasible Base-full budgets are never relaxed or filtered.")
            combined.extend(_collect_trial(trial_output, mode, source))
            with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
                writer.writeheader()
                writer.writerows(combined)
        except Exception as error:
            trial.update(status="failed", error=str(error))
            _write_json(manifest_path, manifest)
            raise
        trial["status"] = "complete"
        _write_json(manifest_path, manifest)
    manifest["complete"] = True
    _write_json(manifest_path, manifest)
    print(f"Completed SIX configurations; combined ACC/PRE/Recall/TNR/Lat/Mem: {output / 'summary.csv'}", flush=True)
    return output


def main():
    run_comparison(build_parser().parse_args())


if __name__ == "__main__":
    main()
