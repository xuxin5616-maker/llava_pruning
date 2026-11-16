"""Run the named MVTec or MMAD hybrid pruning benchmark."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    mvtec = modes.add_parser("mvtecmode", help="MVTec AD, Base-guided, tail 8")
    mm = modes.add_parser("mmadmode", help="MMAD without VisA, joint self scores, tail 8, 3x3 AnyRes")
    for mode, gpu in ((mvtec, 5), (mm, 3)):
        mode.add_argument("--gpu", type=int, default=gpu, help=f"Physical GPU (default {gpu})")
        mode.add_argument("--output-dir", type=Path, required=True)
        mode.add_argument("--model-path", type=Path)
        mode.add_argument("--data-root", type=Path)
        mode.add_argument("--checkpoint", type=Path, help="VisA-trained student checkpoint")
        mode.add_argument("--samples", type=int, help="Uniform random sample of this many distinct query images")
        mode.add_argument("--sample-seed", type=int, default=42, help="Random image sampling seed (default 42)")
        mode.add_argument("--resume", action="store_true", help="Resume the saved run and its frozen image list")
    mvtec.add_argument("--input-json", type=Path)
    mvtec.add_argument("--prune-rates", type=float, nargs="+", default=[50, 70, 90],
                       help="LLM layer-average pruning percentages (default 50 70 90)")
    mm.add_argument("--prune-rates", type=int, nargs="+", choices=(50, 70, 90),
                    help="LLM layer-average pruning percentages (required for a new run)")
    mm.add_argument("--annotations", type=Path)
    mm.add_argument("--image-root", type=Path)
    mm.add_argument("--mvtec-root", type=Path, default=Path("/home/yz/xxy/data/datasets/MVTec-AD"))
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative physical GPU ID")
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")
    if not 0 <= args.sample_seed < 2**32:
        parser.error("--sample-seed must be an integer in [0, 2**32)")
    if args.mode == "mmadmode" and not args.resume and args.prune_rates is None:
        mm.error("--prune-rates is required for a new run")
    return args


def command_for(args):
    script = "run_pool_distill.py" if args.mode == "mvtecmode" else "benchmark_mmad_binary.py"
    command = [sys.executable, str(PROJECT / script)]
    if args.resume:
        return [*command, "--resume", "--output-dir", str(args.output_dir)]
    if args.mode == "mvtecmode":
        options = ["--prune-rates", *map(str, args.prune_rates), "--lambdas", "0.7",
                   "--prompt-version", "v0", "--selection-modes", "base_full",
                   "--predictor-input", "base_text", "--base-mapping", "fast",
                   "--base-policy", "anyres_first", "--pruning-audit", "lean",
                   "--tail-drop-layers", "8", "--summary-format", "compact"]
        paths = ("input_json",)
    else:
        options = ["--experiment-group", "joint_self", "--prune-rates", *map(str, args.prune_rates),
                   "--joint-self-tails", "8", "--resize-mode", "stretch"]
        if args.samples is not None:
            options.extend(("--samples", str(args.samples), "--sample-seed", str(args.sample_seed)))
        paths = ("annotations", "image_root", "mvtec_root")
    options.extend(("--gpu", str(args.gpu), "--output-dir", str(args.output_dir)))
    for name in ("model_path", "data_root", "checkpoint", *paths):
        path = getattr(args, name)
        if path is not None:
            options.extend(("--" + name.replace("_", "-"), str(path)))
    return [*command, *options]


def main():
    args = parse_args()
    command = command_for(args)
    if args.mode == "mvtecmode" and args.samples is not None and not args.resume:
        from run_pool_distill import build_parser, run_sweep
        run_sweep(build_parser().parse_args(command[2:]), sample_count=args.samples, sample_seed=args.sample_seed)
        return 0
    if os.name == "posix":
        os.execv(command[0], command)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")
    except KeyboardInterrupt:
        sys.exit(130)
