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
    mm = modes.add_parser("mmadmode", help="MMAD without VisA, joint self scores, 50 percent, tail 8, 3x3 AnyRes")
    for mode, gpu in ((mvtec, 5), (mm, 3)):
        mode.add_argument("--gpu", type=int, default=gpu, help=f"Physical GPU (default {gpu})")
        mode.add_argument("--output-dir", type=Path, required=True)
        mode.add_argument("--model-path", type=Path)
        mode.add_argument("--data-root", type=Path)
        mode.add_argument("--checkpoint", type=Path, help="VisA-trained student checkpoint")
    mvtec.add_argument("--input-json", type=Path)
    mvtec.add_argument("--prune-rates", type=float, nargs="+", default=[50, 70, 90],
                       help="LLM layer-average pruning percentages (default 50 70 90)")
    mm.add_argument("--annotations", type=Path)
    mm.add_argument("--image-root", type=Path)
    mm.add_argument("--mvtec-root", type=Path, default=Path("/home/yz/xxy/data/datasets/MVTec-AD"))
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative physical GPU ID")
    return args


def command_for(args):
    if args.mode == "mvtecmode":
        script = "run_pool_distill.py"
        options = ["--prune-rates", *map(str, args.prune_rates), "--lambdas", "0.7",
                   "--prompt-version", "v0", "--selection-modes", "base_full",
                   "--predictor-input", "base_text", "--base-mapping", "fast",
                   "--base-policy", "anyres_first", "--pruning-audit", "lean",
                   "--tail-drop-layers", "8", "--summary-format", "compact"]
        paths = ("input_json",)
    else:
        script = "benchmark_mmad_binary.py"
        options = ["--experiment-group", "joint_self", "--prune-rates", "50",
                   "--joint-self-tails", "8", "--resize-mode", "stretch"]
        paths = ("annotations", "image_root", "mvtec_root")
    options.extend(("--gpu", str(args.gpu), "--output-dir", str(args.output_dir)))
    for name in ("model_path", "data_root", "checkpoint", *paths):
        path = getattr(args, name)
        if path is not None:
            options.extend(("--" + name.replace("_", "-"), str(path)))
    return [sys.executable, str(PROJECT / script), *options]


def main():
    command = command_for(parse_args())
    if os.name == "posix":
        os.execv(command[0], command)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
