"""Strict image-level A/B accuracy for the MVTec prompt templates."""

from __future__ import annotations

import re
from dataclasses import dataclass


OPTION = re.compile(r"^\s*([AB])\b", re.IGNORECASE)
METRIC_RULE = (
    "A=defect (1), B=no defect (0); accuracy includes all labeled samples and counts unparsed answers as incorrect; "
    "precision/recall/tnr and tp/fp/tn/fn use only parsed labeled A/B answers; zero denominators are null"
)


@dataclass
class Accuracy:
    expected_samples: int
    evaluated_samples: int = 0
    labeled_samples: int = 0
    correct: int = 0
    unparsed: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, gt: int | None, answer: str) -> None:
        self.evaluated_samples += 1
        if gt is None:
            return
        self.labeled_samples += 1
        match = OPTION.match(answer)
        if match is None:
            self.unparsed += 1
            return
        prediction = 1 if match.group(1).upper() == "A" else 0
        self.correct += prediction == gt
        if gt == 1:
            if prediction == 1:
                self.tp += 1
            else:
                self.fn += 1
        elif prediction == 1:
            self.fp += 1
        else:
            self.tn += 1

    def result(self, prune_rate: int, *, complete: bool) -> dict:
        return {
            "prune_rate": prune_rate,
            "complete": complete and self.evaluated_samples == self.expected_samples,
            "expected_samples": self.expected_samples,
            "evaluated_samples": self.evaluated_samples,
            "labeled_samples": self.labeled_samples,
            "correct": self.correct,
            "incorrect": self.labeled_samples - self.correct,
            "unparsed": self.unparsed,
            "parsed_samples": self.labeled_samples - self.unparsed,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": self.correct / self.labeled_samples if self.labeled_samples else None,
            "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else None,
            "recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else None,
            "tnr": self.tn / (self.tn + self.fp) if self.tn + self.fp else None,
            "rule": METRIC_RULE,
        }
