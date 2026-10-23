"""All-VisA distillation pool, category-stratified internal split; no GT or masks.

Official train/test membership is intentionally NOT an evaluation split here.
All rows in the user's real CSV are used as the training source, with a new
internal split. The resulting predictor must be evaluated on independent MVTec,
not reported as zero-shot or held-out evaluation on this same VisA population.
"""

from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
import random

from .data import Sample
from .prompts import GENERIC_PROMPT_V0
from .visa_data import CATEGORIES, _relative_path


def load_visa_distillation(data_root, split_csv=None):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"VisA data root not found: {root}")
    if split_csv is None:
        path = next((p for p in (root / "split_csv/1cls.csv", root / "1cls.csv") if p.is_file()), None)
    else:
        path = Path(split_csv).expanduser().resolve()
        if path.is_dir():
            path = path / "1cls.csv"
    if path is None or not path.is_file():
        raise FileNotFoundError("Distillation requires your FULL VisA CSV; use --visa-split-csv /.../split_csv/1cls.csv. "
                                "The bundled test-only list is never used for training.")
    samples, seen, resolved_seen = [], set(), set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"object", "image"}.issubset(reader.fieldnames or []):
            raise ValueError("VisA source CSV requires object,image columns; label/gt/mask are not read")
        for number, row in enumerate(reader, 2):
            relative = _relative_path(row.get("image", ""))
            category = (row.get("object") or "").strip().lower()
            if category not in CATEGORIES or Path(relative).parts[0].lower() != category:
                raise ValueError(f"Invalid VisA category/image in row {number}: {category}, {relative}")
            image = (root / relative).resolve()
            if not image.is_relative_to(root):
                raise ValueError(f"VisA image escapes root: {relative}")
            if relative in seen or image in resolved_seen:
                raise ValueError(f"Duplicate VisA image (including symlink alias): {relative}")
            if not image.is_file():
                raise FileNotFoundError(f"VisA CSV image missing: {image}; restore CSV paths or supply a matching CSV. "
                                        "No fallback to test-only folders or silent skips.")
            seen.add(relative)
            resolved_seen.add(image)
            source = dict(question_id=relative, image=relative, origin_path=relative, text=GENERIC_PROMPT_V0)
            samples.append(Sample(relative, image, category, relative, GENERIC_PROMPT_V0, None, None, None, source))
    if not samples:
        raise ValueError("VisA distillation source is empty")
    return sorted(samples, key=lambda s: s.sample_id), dict(
        source_csv=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_policy="ALL CSV rows; original train/test ignored; internal split only; final evaluation on independent MVTec",
        gt_used=False, masks_used=False)


def split_samples(samples, validation_fraction=.1, seed=42):
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    groups = defaultdict(list)
    for sample in sorted(samples, key=lambda item: item.sample_id):
        groups[sample.category].append(sample)
    train, validation = [], []
    rng = random.Random(seed)
    for category in sorted(groups):
        group = groups[category]
        if len(group) < 2:
            raise ValueError(f"Need at least two images per category for an internal split: {category}")
        rng.shuffle(group)
        count = max(1, min(len(group) - 1, round(len(group) * validation_fraction)))
        validation.extend(group[:count])
        train.extend(group[count:])
    return train, validation
