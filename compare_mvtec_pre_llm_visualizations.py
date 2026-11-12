"""Stack the saved crop-quota and global-AnyRes PNGs for each MVTec sample."""

import argparse
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile

from PIL import Image


SAMPLE_NAME = re.compile(r"sample_[0-9]{9}\.png")
RATES = (50, 70)


def paired_names(top_dir, bottom_dir):
    names = []
    for folder in (top_dir, bottom_dir):
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        names.append({path.name for path in folder.iterdir() if path.is_file() and SAMPLE_NAME.fullmatch(path.name)})
    top, bottom = names
    if top != bottom:
        missing_top = ", ".join(sorted(bottom - top)[:5]) or "none"
        missing_bottom = ", ".join(sorted(top - bottom)[:5]) or "none"
        raise ValueError(f"{top_dir.parent.name}: missing from crop_quota ({len(bottom - top)}): {missing_top}; "
                         f"missing from global_anyres ({len(top - bottom)}): {missing_bottom}")
    if not top:
        raise ValueError(f"No sample PNGs found in {top_dir}")
    return sorted(top)


def stack_pair(top_path, bottom_path, output_path):
    gap = 12
    with Image.open(top_path) as top, Image.open(bottom_path) as bottom:
        width = max(top.width, bottom.width)
        with Image.new("RGB", (width, top.height + gap + bottom.height), "white") as canvas:
            canvas.paste(top, ((width - top.width) // 2, 0))
            canvas.paste(bottom, ((width - bottom.width) // 2, top.height + gap))
            temporary = None
            try:
                with NamedTemporaryFile(dir=output_path.parent, suffix=".png.tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    canvas.save(stream, format="PNG", compress_level=2)
                temporary.replace(output_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def compare(results_dir):
    jobs = []
    for rate in RATES:
        rate_folder = f"prune_{rate}"
        top = results_dir / "crop_quota" / rate_folder / "visualizations"
        bottom = results_dir / "global_anyres" / rate_folder / "visualizations"
        destination = results_dir / "comparisons" / rate_folder
        jobs.append((rate, top, bottom, destination, paired_names(top, bottom)))
    for rate, top, bottom, destination, names in jobs:
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            stack_pair(top / name, bottom / name, destination / name)
        print(f"{rate}% | {len(names)} pairs | {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Completed visualize_mvtec_pre_llm.py output directory")
    args = parser.parse_args()
    try:
        compare(args.results_dir.expanduser().resolve())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
