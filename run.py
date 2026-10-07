"""Public CLI for LLaVA visual-token pruning experiments."""

import argparse
from pathlib import Path

from llava_pruning.methods import METHODS
from llava_pruning.metric_plot import check_plot_dependencies, save_metric_plot
from llava_pruning.runner import run


PROJECT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="LLaVA OneVision/Qwen2 pruning sweep")
    parser.add_argument("--model-path", required=True, help="LLaVA checkpoint directory")
    parser.add_argument("--input-json", required=True, help="JSON list or JSONL with image/mask records")
    parser.add_argument("--data-root", required=True, help="Root directory for image and mask paths")
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    parser.add_argument("--method", choices=tuple(METHODS), default="fastv")
    parser.add_argument("--method-config", type=Path, default=None,
                        help="Defaults to configs/<method>.json")
    parser.add_argument("--roi-mode", choices=("randomroi", "randompatch", "anyres_max_9", "ex_base_copy", "anyres_only"),
                        default="randomroi", help="ex_base_copy: three Base views; anyres_only: anyres tiles without Base")
    parser.add_argument("--image-token-order", choices=("base_first", "anyres_first"),
                        default="base_first",
                        help="Anyres view order; anyres_first keeps the final newline at the end")
    parser.add_argument("--save-prune-vis", action="store_true")
    parser.add_argument("--save-attention-vis", action="store_true")
    decoding = parser.add_mutually_exclusive_group()
    decoding.add_argument("--no-sample", dest="no_sample", action="store_true",
                          help="Greedy decoding (default, matching the reference baseline)")
    decoding.add_argument("--sample", dest="no_sample", action="store_false",
                          help="Opt into sampling; not an exact greedy baseline comparison")
    parser.set_defaults(no_sample=True)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--include-pruning-time", dest="include_pruning_time", action="store_true",
                        help="Include pruning in generation_seconds (default; normal timing)")
    timing.add_argument("--exclude-pruning-time", dest="include_pruning_time", action="store_false",
                        help="Subtract separately timed pruning blocks; diagnostic timing, pruning still runs")
    parser.set_defaults(include_pruning_time=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs" / "run")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed seed for a repeatable run; default generates a new seed")
    return parser


def main():
    args = build_parser().parse_args()
    check_plot_dependencies()
    output = run(
        model_path=args.model_path, input_json=args.input_json,
        data_root=args.data_root, prompt_version=args.prompt_version,
        method_name=args.method, method_config=args.method_config,
        roi_mode=args.roi_mode, save_prune_vis=args.save_prune_vis,
        image_token_order=args.image_token_order,
        save_attention_vis=args.save_attention_vis, output_dir=args.output_dir,
        seed=args.seed, no_sample=args.no_sample, include_pruning_time=args.include_pruning_time,
    )
    print(f"Saved results to {output}")
    try:
        figure = save_metric_plot(output / "summary.csv")
    except Exception as error:
        raise RuntimeError(
            f"Evaluation finished; predictions and summary.csv are saved, but plotting failed. "
            f'Retry without inference: python -m llava_pruning.metric_plot "{output}"'
        ) from error
    print(f"Saved ACC/PRE/Recall/TNR curves to {figure}")


if __name__ == "__main__":
    main()
