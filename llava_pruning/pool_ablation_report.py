"""Presentation-only hybrid summaries; no changes to inference or raw metrics."""

FIELDS = ("config", "tp", "fp", "tn", "fn", "acc", "Lat", "mem")


def compact_row(row):
    rate = float(row["prune_rate"])
    share = float(row["distill_share"])
    mode = row["selection_mode"]
    branch = row.get("predictor_input", "full")
    tail = int(row.get("tail_drop_layers", 0))
    return dict(config=f"prune={rate:g}%;lambda={share:g};{mode};input={branch};tail={tail}",
                tp=row["tp"], fp=row["fp"], tn=row["tn"], fn=row["fn"],
                acc=None if row["accuracy"] is None else 100 * float(row["accuracy"]),
                Lat=row["lat_mean_ms"], mem=row["mem_peak_allocated_gib"])


def summary_rows(rows, style):
    if style not in {"full", "compact"}:
        raise ValueError("Unknown hybrid summary format")
    return [compact_row(row) for row in rows] if style == "compact" else rows
