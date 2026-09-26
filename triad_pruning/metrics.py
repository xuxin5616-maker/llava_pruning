"""Strict image-level A/B accuracy for the MVTec prompt templates."""

import re
from dataclasses import dataclass


OPTION = re.compile(r"^\s*([AB])\b", re.IGNORECASE)


@dataclass
class Accuracy:
    expected_samples: int
    evaluated_samples: int = 0
    labeled_samples: int = 0
    correct: int = 0
    unparsed: int = 0

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
            "accuracy": self.correct / self.labeled_samples if self.labeled_samples else None,
            "rule": "A=defect (1), B=no defect (0); unparsed labeled answers count as incorrect",
        }
