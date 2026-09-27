"""Read the question JSON/JSONL format without dataset-specific path strings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Path
    category: str
    origin_path: str
    text: str | None
    gt: int | None
    mask: Path | None
    bbox: list[list[int]] | None
    source: dict[str, Any]


def _resolve(root: Path, raw: str, *, image: bool) -> Path:
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(f"Data paths must be relative to --data-root: {raw}")
    candidates = [root / relative]
    if image and len(relative.parts) == 1:
        candidates.append(root / "imgs" / relative)
    resolved_root = root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"Data path escapes --data-root: {raw}")
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Cannot find {raw} under {root}")


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
    else:
        raise ValueError("--input-json must end in .json or .jsonl")
    if not isinstance(records, list):
        raise ValueError("Input must contain a list of JSON objects or JSONL objects")
    return records


def load_samples(input_json: str | Path, data_root: str | Path) -> list[Sample]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Data root not found: {root}")
    samples = []
    seen_ids = set()
    for index, record in enumerate(_records(Path(input_json))):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object")
        sample_id = str(record.get("question_id", record.get("id", "")))
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Record {index} has an empty or duplicate question_id: {sample_id!r}")
        seen_ids.add(sample_id)
        image_name = record.get("image")
        if not isinstance(image_name, str):
            raise ValueError(f"Record {index} needs a single image path")
        origin = str(record.get("origin_path", image_name))
        origin_parts = origin.replace("\\", "/").split("/")
        inferred_category = origin_parts[0]
        if inferred_category == root.name and len(origin_parts) > 1:
            inferred_category = origin_parts[1]
        category = str(record.get("category") or record.get("subset") or
                       record.get("sub_dataset") or inferred_category)
        mask_name = record.get("mask")
        if isinstance(mask_name, list) and len(mask_name) == 1:
            mask_name = mask_name[0]
        mask = _resolve(root, mask_name, image=False) if isinstance(mask_name, str) and mask_name else None
        if mask_name is not None and not isinstance(mask_name, str):
            raise ValueError(f"Record {index} mask must be a string path")
        bbox = record.get("bbox")
        # Legacy LLaVA records can include an outer per-image list.
        if isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list) and bbox[0] and isinstance(bbox[0][0], list):
            bbox = bbox[0]
        if bbox is not None:
            if (not isinstance(bbox, list) or any(
                not isinstance(box, list) or len(box) != 4 or
                any(not isinstance(value, int) for value in box)
                for box in bbox
            )):
                raise ValueError(f"Record {index} bbox must be a list of four-integer boxes")
        gt = record.get("gt")
        if gt is not None and gt not in (0, 1):
            raise ValueError(f"Record {index} gt must be 0 or 1")
        samples.append(Sample(
            sample_id=sample_id,
            image=_resolve(root, image_name, image=True),
            category=category,
            origin_path=origin,
            text=record.get("text"),
            gt=gt,
            mask=mask,
            bbox=bbox,
            source=record,
        ))
    return samples
