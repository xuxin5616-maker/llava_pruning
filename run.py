"""Small public CLI for the Triad pruning study."""

import argparse
from pathlib import Path

from triad_pruning.methods import METHODS
from triad_pruning.runner import run


PROJECT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Triad OneVision/Qwen2 pruning sweep")
    parser.add_argument("--model-path", required=True, help="Triad checkpoint directory")
    parser.add_argument("--input-json", required=True, help="JSON list or JSONL with image/mask records")
    parser.add_argument("--data-root", required=True, help="Root directory for image and mask paths")
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    parser.add_argument("--method", choices=tuple(METHODS), default="fastv")
    parser.add_argument("--method-config", type=Path, default=PROJECT / "configs" / "fastv.json")
    parser.add_argument("--roi-mode", choices=("randomroi", "randompatch"), default="randomroi")
    parser.add_argument("--save-prune-vis", action="store_true")
    parser.add_argument("--save-attention-vis", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs" / "run")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    output = run(
        model_path=args.model_path, input_json=args.input_json,
        data_root=args.data_root, prompt_version=args.prompt_version,
        method_name=args.method, method_config=args.method_config,
        roi_mode=args.roi_mode, save_prune_vis=args.save_prune_vis,
        save_attention_vis=args.save_attention_vis, output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
