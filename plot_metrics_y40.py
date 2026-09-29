"""Redraw saved ACC/PRE/Recall/TNR curves on a 40--100% axis, without inference.

Edit SUMMARY_CSV below, then run: python plot_metrics_y40.py
An optional command-line path overrides that setting. Only Python + Matplotlib
are needed; no model or project-package imports are performed.
Existing CSV, predictions and figures are never overwritten.
"""

# ===== 在这里填写已有 summary.csv 的路径，然后直接运行本文件 =====
# 可以写绝对路径；相对路径以本脚本所在目录为起点，不受启动目录影响。
SUMMARY_CSV = r"output/vico_anyres_01/summary.csv"

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import warnings


Y_MIN = 40
SERIES = (
    ("accuracy", "ACC", "#0876B9", "o", "-"),
    ("precision", "PRE (Precision)", "#E5262D", "s", "-"),
    ("recall", "Recall", "#0876B9", "^", "--"),
    ("tnr", "TNR", "#E5262D", "D", "--"),
)


def configured_input():
    if not SUMMARY_CSV.strip():
        raise ValueError("Set SUMMARY_CSV at the top of this script to your saved summary.csv path")
    source = Path(SUMMARY_CSV).expanduser()
    if not source.is_absolute():
        source = Path(__file__).resolve().parent / source
    return source.resolve()


def load_rows(source_text):
    reader = csv.DictReader(io.StringIO(source_text.lstrip("\ufeff")))
    required = {"prune_rate", "complete", *(key for key, *_ in SERIES)}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"summary.csv is missing metric columns: {sorted(missing)}")
    rows = []
    seen = set()
    for line, record in enumerate(reader, start=2):
        try:
            rate = int(record["prune_rate"])
            complete = record["complete"].strip().lower()
            if not 0 <= rate < 100 or rate in seen:
                raise ValueError("prune_rate must be unique and between 0 and 99")
            if complete not in {"true", "false"}:
                raise ValueError("complete must be True or False")
            row = {"prune_rate": rate, "complete": complete == "true"}
            for key, *_ in SERIES:
                raw = record[key].strip()
                value = float(raw) if raw else None
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"{key} must be a fraction in [0, 1] or an empty field")
                row[key] = value
        except (TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"Invalid summary.csv row {line}: {error}") from error
        rows.append(row)
        seen.add(rate)
    if not rows:
        raise ValueError("summary.csv has no metric rows")
    return sorted(rows, key=lambda row: row["prune_rate"])


def build_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Install the plotting dependency: python -m pip install "matplotlib>=3.7,<4"') from error

    # Preserve the original metric_plot.py figure style; only the y range and
    # its out-of-range notices change. No smoothing or metric recomputation.
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
             "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
             "axes.labelsize": 12, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(8.8, 5.4))
        figure.subplots_adjust(left=0.10, right=0.975, bottom=0.25, top=0.83)
        hidden, undefined = 0, 0
        for key, label, color, marker, linestyle in SERIES:
            values = [row[key] * 100 if row["complete"] and row[key] is not None else math.nan for row in rows]
            hidden += sum(math.isfinite(value) and value < Y_MIN for value in values)
            undefined += sum(not math.isfinite(value) for value in values)
            axis.plot([row["prune_rate"] for row in rows], values, label=label,
                      color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=5)
        upper = max(90, math.ceil(max(row["prune_rate"] for row in rows) / 10) * 10)
        axis.set(title="Classification Metrics vs. Pruning Rate", xlabel="Pruning rate (%)",
                 ylabel="Score (%)", xlim=(-2, upper + 2), ylim=(Y_MIN, 100))
        axis.set_xticks(list(range(0, upper + 1, 10)))
        axis.set_yticks(list(range(Y_MIN, 101, 5)))
        axis.grid(True, linestyle=":", linewidth=0.6, color="#B8B8B8")
        axis.set_axisbelow(True)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False,
                    fontsize=10, handlelength=2.5, columnspacing=1.2)
        note = "Positive class: defect. ACC: all labeled samples; PRE/Recall/TNR: parsed A/B answers only."
        if hidden:
            note += f"\n{hidden} point(s) below {Y_MIN}% are outside the displayed range; exact values remain in summary.csv."
            warnings.warn(f"{hidden} metric point(s) below {Y_MIN}%; values were NOT clamped or changed.",
                          UserWarning, stacklevel=2)
        if undefined:
            note += "\nUndefined or incomplete points are omitted, not replaced by zero."
        figure.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=8.5, color="#444444")
    return figure


def redraw(input_path=None):
    source = configured_input() if input_path is None else Path(input_path).expanduser().resolve()
    if source.is_dir():
        source = source / "summary.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Saved summary not found: {source}. Edit SUMMARY_CSV at the top of this script, "
            "or pass your existing run directory or summary.csv; "
            "this script does not run inference."
        )
    raw = source.read_bytes()
    rows = load_rows(raw.decode("utf-8-sig"))
    figure = build_figure(rows)
    import matplotlib.pyplot as plt

    # Finish rendering in memory first. Exclusive file creation preserves both
    # the old 50%-axis figure and previous exports of this 40%-axis version.
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=300, facecolor="white",
                       metadata={"Source": str(source), "SourceSHA256": hashlib.sha256(raw).hexdigest(),
                                 "Description": "Saved summary metrics times 100; y-axis 40--100%; no inference or smoothing."})
    finally:
        plt.close(figure)
    suffix = 1
    while True:
        tail = "" if suffix == 1 else f"_{suffix}"
        destination = source.parent / f"metrics_vs_pruning_y40{tail}.png"
        try:
            with destination.open("xb") as stream:
                stream.write(buffer.getvalue())
            return destination
        except FileExistsError:
            suffix += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=None,
                        help="Optional run directory or summary.csv; otherwise use SUMMARY_CSV at the top of this script")
    args = parser.parse_args(argv)
    try:
        destination = redraw(args.input)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved: {destination}\nY-axis: 40--100%. No inference. Original data and figures preserved.")


if __name__ == "__main__":
    main()
