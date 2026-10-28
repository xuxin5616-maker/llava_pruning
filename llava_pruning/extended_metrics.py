"""Shared count-based F1/BACC and accuracy relative to a supplied baseline.

No model, GPU, plotting or spreadsheet dependencies. Units match the existing
predictor ablation report. Undefined denominators remain None, never zero-fill.
"""

import math


EXTRA_FIELDS = ("f1", "bacc", "baseline_accuracy", "relative_accuracy_pct")
METRIC_DEFINITIONS = {
    "f1": "2*TP/(2*TP+FP+FN), defect positive class; parsed labeled A/B answers only; zero denominator=null.",
    "bacc": "0.5*(TP/(TP+FN)+TN/(TN+FP)); parsed labeled A/B only; either class absent=null.",
    "baseline_accuracy": "User-supplied unpruned reference ACC as a fraction; not a newly measured baseline.",
    "relative_accuracy_pct": "100*accuracy/baseline_accuracy; may exceed100; this is accuracy retention, NOT latency/memory/token ratio.",
    "extended_units": "F1/BACC/baseline_accuracy are fractions0..1; relative_accuracy_pct is percent.",
}


def extended_metrics(metrics, baseline):
    """Compute from integer confusion counts, not rounded PRE/Recall columns."""
    if not math.isfinite(baseline) or not 0 < baseline <= 1:
        raise ValueError("Baseline must be a fraction in (0,1]")
    counts = [metrics[key] for key in ("tp", "fp", "tn", "fn")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, fp, tn, fn = counts
    accuracy = metrics.get("accuracy")
    if accuracy is not None and (type(accuracy) not in (int, float) or not math.isfinite(accuracy)
                                 or not 0 <= accuracy <= 1):
        raise ValueError("ACC must be a fraction in [0,1], not a percentage")
    recall = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return dict(metrics, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                bacc=(recall + tnr) / 2 if recall is not None and tnr is not None else None,
                baseline_accuracy=baseline,
                relative_accuracy_pct=100 * accuracy / baseline if accuracy is not None else None)
