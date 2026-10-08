"""Visualize SigLIP/MLP feature scores without pruning or language generation."""

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
    parser.add_argument("--color-scale", choices=("sample", "fixed"), default="sample",
                        help="sample: one shared range for all five layers and both scores; "
                             "fixed: [-1, 1], also comparable across images")
    return parser


def main():
    args = build_parser().parse_args()
    from llava_pruning.score_visualization import run_score_visualization
    run_score_visualization(**vars(args))


if __name__ == "__main__":
    main()
