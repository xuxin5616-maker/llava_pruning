"""Presentation-only hybrid summaries; no changes to inference or raw metrics."""

from llava_pruning.extended_metrics import extended_metrics


LEGACY_FIELDS = ("config", "tp", "fp", "tn", "fn", "acc", "Lat", "mem")
FIELDS = (*LEGACY_FIELDS, "bacc", "f1", "relative_accuracy_pct")


def compact_metric_columns(metrics, baseline=.9118):
    """Percentage columns from exact counts; raw metrics keep fractional units."""
    values = extended_metrics(metrics, baseline)
    return dict(bacc=None if values["bacc"] is None else 100 * values["bacc"],
                f1=None if values["f1"] is None else 100 * values["f1"],
                relative_accuracy_pct=values["relative_accuracy_pct"])


def compact_row(row):
    rate = float(row["prune_rate"])
    share = float(row["distill_share"])
    mode = row["selection_mode"]
    branch = row.get("predictor_input", "full")
    tail = int(row.get("tail_drop_layers", 0))
    mapping=";mapping=fast" if row.get("base_mapping")=="fast" and mode=="base_full" else ""
    policy=";base_policy=anyres_first" if row.get("stage2_base_policy")=="anyres_first" else ""
    return compact_result_row(row, f"prune={rate:g}%;lambda={share:g};{mode};input={branch};tail={tail}{mapping}{policy}")


def compact_result_row(row, configuration):
    """Shared eleven-column presentation for distilled and independent methods."""
    return dict(config=configuration,
                tp=row["tp"], fp=row["fp"], tn=row["tn"], fn=row["fn"],
                acc=None if row["accuracy"] is None else 100 * float(row["accuracy"]),
                Lat=row["lat_mean_ms"], mem=row["mem_peak_allocated_gib"],
                **compact_metric_columns(row, float(row.get("baseline_accuracy", .9118))))


def summary_rows(rows, style):
    if style not in {"full", "compact"}:
        raise ValueError("Unknown hybrid summary format")
    return [compact_row(row) for row in rows] if style == "compact" else rows
