"""Read VisA TEST images only, without moving data or loading annotation masks.

Supports prepared <category>/test/{good,bad,<defect_type>} and raw VisA with a split CSV.
Official layout/split: https://github.com/amazon-science/spot-diff
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
from pathlib import Path, PurePosixPath

from .data import Sample
from .prompts import GENERIC_PROMPT_V0


CATEGORIES = frozenset(("candle", "capsules", "cashew", "chewinggum", "fryum",
                        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"))
IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))
DIRECTORY_LABELS = {"good": 0, "normal": 0, "bad": 1, "anomaly": 1, "abnormal": 1}
# These are not defect names. Fail rather than mislabel masks or unlabeled data.
NON_LABEL_DIRECTORIES = frozenset(("ground_truth", "mask", "masks", "annotation", "annotations",
                                   "labels", "images", "train", "test", "val", "validation",
                                   "unknown", "unlabeled", "unlabelled"))
BUNDLED_SPLIT = Path(__file__).resolve().parent / "assets" / "visa_1cls_test.csv"


def _relative_path(value):
    raw = str(value).strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or ":" in raw:
        raise ValueError(f"VisA image path must stay relative to the data root: {raw!r}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported VisA image extension: {raw}")
    return path.as_posix()


def _sample(root, relative, category, gt):
    relative = _relative_path(relative)
    image = (root / relative).resolve()
    if not image.is_relative_to(root):
        raise ValueError(f"VisA image escapes the data root: {relative}")
    if not image.is_file():
        raise FileNotFoundError(f"VisA test image not found: {image}")
    record = {"question_id": relative, "image": relative, "origin_path": relative,
              "category": category, "gt": gt, "text": GENERIC_PROMPT_V0}
    return Sample(relative, image, category, relative, GENERIC_PROMPT_V0, gt,
                  None, None, record)


def _from_csv(root, path):
    samples, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"object", "split", "label", "image"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"VisA split CSV requires columns {sorted(required)}: {path}. "
                             "Use split_csv/1cls.csv, not the image_anno.csv annotation file.")
        for number, row in enumerate(reader, 2):
            if any(row.get(key) is None for key in required):
                raise ValueError(f"Incomplete VisA CSV row {number}")
            category = row["object"].strip().lower()
            split = row["split"].strip().lower()
            label = row["label"].strip().lower()
            if category not in CATEGORIES or split not in {"train", "test", "val", "validation"}:
                raise ValueError(f"Invalid category or split in VisA CSV row {number}: {category}, {split}")
            if label not in {"normal", "anomaly"}:
                raise ValueError(f"Invalid VisA label at row {number}: {label!r}")
            relative = _relative_path(row["image"])
            if PurePosixPath(relative).parts[0].lower() != category:
                raise ValueError(f"CSV category does not match image path at row {number}")
            if relative in seen:
                raise ValueError(f"Duplicate image or train/test overlap in VisA CSV: {relative}")
            seen.add(relative)
            if split != "test":
                continue  # Never load training/validation images or any mask.
            parent_label = PurePosixPath(relative).parent.name.lower()
            gt = int(label == "anomaly")
            if parent_label in DIRECTORY_LABELS and DIRECTORY_LABELS[parent_label] != gt:
                raise ValueError(f"CSV label conflicts with image directory: {relative}")
            try:
                samples.append(_sample(root, relative, category, gt))
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"{error}\nSelected by CSV row {number}: {path}. "
                    "CSV image paths are relative to --data-root. Restore the listed image paths "
                    "or provide a verified split CSV matching your layout. "
                    "No images are skipped or guessed by filename; no fallback to test/ folders."
                ) from error
    return samples


def _prepared_base(root):
    candidates = []
    for base in (root, root / "1cls"):
        if not base.is_dir():
            continue
        categories = sorted((p for p in base.iterdir() if p.is_dir() and p.name.lower() in CATEGORIES),
                            key=lambda p: p.name)
        if any((p / "test").is_dir() for p in categories):
            missing = [p.name for p in categories if not (p / "test").is_dir()]
            if missing:
                raise ValueError(f"Incomplete VisA test layout: missing test/ for {missing}")
            candidates.append((base, categories))
    if len(candidates) > 1:
        raise ValueError("Multiple VisA test layouts found; point --data-root at the intended category root")
    return candidates[0] if candidates else None


def _from_directories(root, categories):
    samples = []
    for category_path in categories:
        test = category_path / "test"
        category_samples = []
        for folder in sorted(test.iterdir()):
            if not folder.is_dir():
                if folder.suffix.lower() in IMAGE_SUFFIXES:
                    raise ValueError(f"Unlabeled image directly under test/: {folder}")
                continue
            label = folder.name.lower()
            if label in NON_LABEL_DIRECTORIES or label.startswith("."):
                raise ValueError(f"Unknown VisA test label directory: {folder}; expected good/normal "
                                 "or an anomaly/defect-type directory, not masks or unlabeled data")
            # MVTec-style exports retain specific defect names, e.g.
            # candle/test/chunk_of_wax_missing. They are positive class samples.
            gt = DIRECTORY_LABELS.get(label, 1)
            images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                raise ValueError(f"Empty VisA test image directory: {folder}")
            for image in images:
                category_samples.append(_sample(root, image.relative_to(root).as_posix(),
                                                category_path.name.lower(), gt))
        if not category_samples:
            raise ValueError(f"No labeled VisA images in {test}")
        samples.extend(category_samples)
    return samples


def load_visa_test(data_root, split_csv=None):
    """Return deterministic Sample objects and a compact, auditable split description.

    Explicit CSV takes precedence, then local split_csv/1cls.csv or 1cls.csv,
    then existing test folders, then the bundled official 1-class test list.
    Once a CSV is selected, its test rows are authoritative: a broken/missing
    path is an error, never a reason to switch to a different test population.
    No random split, no network, no mask loading, and no silently skipped images.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"VisA root not found: {root}")
    if split_csv is not None:
        path = Path(split_csv).expanduser().resolve()
        if path.is_dir():
            path = path / "1cls.csv"
    else:
        path = next((p for p in (root / "split_csv" / "1cls.csv", root / "1cls.csv")
                     if p.exists()), None)
    prepared = _prepared_base(root) if path is None else None
    metadata = {"split": "test", "mask_input": False}
    if prepared is not None:
        base, categories = prepared
        samples = _from_directories(root, categories)
        metadata.update(layout="test_directories", category_root=str(base), split_csv=None,
                        directory_label_rule="good/normal=0; other defect folders=1; "
                                             "reserved/unlabeled folders rejected")
    else:
        if path is None:
            path = BUNDLED_SPLIT
        if not path.is_file():
            raise FileNotFoundError(f"VisA split CSV missing: {path}. Sync llava_pruning/assets/ "
                                    "or provide --visa-split-csv /path/to/1cls.csv")
        samples = _from_csv(root, path)
        metadata.update(layout="raw_csv", split_csv=str(path),
                        split_source="bundled_official_1cls_test" if path == BUNDLED_SPLIT else "local_csv",
                        split_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not samples:
        raise ValueError("VisA test selection is empty")
    if len({item.image for item in samples}) != len(samples):
        raise ValueError("Duplicate resolved VisA image paths (including symlink aliases)")
    samples.sort(key=lambda item: item.source["image"])
    counts = Counter(item.category for item in samples)
    metadata.update(samples=len(samples), normal=sum(item.gt == 0 for item in samples),
                    abnormal=sum(item.gt == 1 for item in samples), categories=dict(sorted(counts.items())))
    return samples, metadata
