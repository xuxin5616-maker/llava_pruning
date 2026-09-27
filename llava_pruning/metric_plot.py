"""Plot the run.py classification summary, with a fixed 50--100% vertical axis."""

import csv
import math
import warnings
from pathlib import Path


SERIES = (
    ("accuracy", "ACC", "#0876B9", "o", "-"),
    ("precision", "PRE (Precision)", "#E5262D", "s", "-"),
    ("recall", "Recall", "#0876B9", "^", "--"),
    ("tnr", "TNR", "#E5262D", "D", "--"),
)


def check_plot_dependencies():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
    except ImportError as error:
        raise RuntimeError('Plotting requires Matplotlib. Install: python -m pip install "matplotlib>=3.7,<4"') from error


def load_metric_rows(summary_path):
    with Path(summary_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"prune_rate", "complete", *(key for key, *_ in SERIES)}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary.csv lacks the new metric columns: {sorted(missing)}")
        rows = []
        for record in reader:
            rate = int(record["prune_rate"])
            if not 0 <= rate < 100:
                raise ValueError(f"Invalid pruning rate: {rate}")
            row = {"prune_rate": rate, "complete": record["complete"].lower() == "true"}
            for key, *_ in SERIES:
                raw = record[key].strip()
                value = float(raw) if raw else None
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"{key} at rate {rate} must be a fraction in [0, 1] or empty")
                row[key] = value
            rows.append(row)
    if not rows or len({row["prune_rate"] for row in rows}) != len(rows):
        raise ValueError("summary.csv must have at least one row and unique pruning rates")
    return sorted(rows, key=lambda row: row["prune_rate"])


def build_metric_figure(rows):
    import matplotlib.pyplot as plt

    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
             "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
             "axes.labelsize": 12, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        fig.subplots_adjust(left=0.10, right=0.975, bottom=0.25, top=0.83)
        hidden, undefined = 0, 0
        for key, label, color, marker, linestyle in SERIES:
            values = [row[key] * 100 if row["complete"] and row[key] is not None else math.nan for row in rows]
            hidden += sum(math.isfinite(value) and value < 50 for value in values)
            undefined += sum(not math.isfinite(value) for value in values)
            ax.plot([row["prune_rate"] for row in rows], values, label=label,
                    color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=5)
        upper = max(90, math.ceil(max(row["prune_rate"] for row in rows) / 10) * 10)
        ax.set(title="Classification Metrics vs. Pruning Rate", xlabel="Pruning rate (%)",
               ylabel="Score (%)", xlim=(-2, upper + 2), ylim=(50, 100))
        ax.set_xticks(list(range(0, upper + 1, 10)))
        ax.set_yticks(list(range(50, 101, 5)))
        ax.grid(True, linestyle=":", linewidth=0.6, color="#B8B8B8")
        ax.set_axisbelow(True)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False,
                  fontsize=10, handlelength=2.5, columnspacing=1.2)
        note = "Positive class: defect. ACC: all labeled samples; PRE/Recall/TNR: parsed A/B answers only."
        if hidden:
            note += f"\n{hidden} point(s) below 50% are outside the displayed range; exact values remain in summary.csv."
            warnings.warn(f"{hidden} metric point(s) are below the 50% plot limit; no values have been clamped or changed.",
                          UserWarning, stacklevel=2)
        if undefined:
            note += "\nUndefined or incomplete points are omitted, not replaced by zero."
        fig.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=8.5, color="#444444")
    return fig


def save_metric_plot(summary_path):
    summary_path = Path(summary_path)
    rows = load_metric_rows(summary_path)
    check_plot_dependencies()
    import matplotlib.pyplot as plt

    figure = build_metric_figure(rows)
    destination = summary_path.parent / "metrics_vs_pruning.png"
    try:
        figure.savefig(destination, dpi=300, facecolor="white")
    finally:
        plt.close(figure)
    return destination


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Run output directory or summary.csv; no inference is performed")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    if source.is_dir():
        source = source / "summary.csv"
    print(f"Saved metric curves to {save_metric_plot(source)}")


if __name__ == "__main__":
    main()
