"""Visualize SigLIP/MLP scores: per-token L2 before view means; no pruning or LLM."""

import argparse
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=positive_int, default=None,
                        help="Optional first N records; default: all records, no fixed limit")
    parser.add_argument("--display-mode", choices=("tokens", "patch-means"), default="tokens",
                        help="tokens (default): original token maps; patch-means: one 2x4 Global "
                             "tile-mean figure with layers 7/14/21/26 and first 2/3/4 layer means")
    parser.add_argument("--color-scale", choices=("sample", "fixed"), default=None,
                        help="Default: sample for tokens, fixed for patch-means. "
                             "fixed: [-1,1] token scores or [0,1] scaled patch means; "
                             "sample: one shared observed range within each image")
    return parser


def main():
    args = build_parser().parse_args()
    from llava_pruning.score_visualization import run_score_visualization
    run_score_visualization(**vars(args))


if __name__ == "__main__":
    main()
