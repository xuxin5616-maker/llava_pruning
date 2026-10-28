"""CPU-only checkpoint validation for hybrid inference, including old JSONL.

One flushed journal row commits answer + timing + budget together. A torn last
row can be ignored; corrupt interior records, changed IDs/labels/configuration
are errors. Legacy predictions have no budgets; callers explicitly reconstruct
and label those counts rather than inventing observed measurements.
"""

import hashlib
import json
import math
from pathlib import Path


TIMING_FIELDS = ("lat_seconds", "generation_seconds", "generated_tokens",
                 "peak_allocated_bytes", "peak_reserved_bytes")
COMMON_FIELDS = ("model_path", "input_json", "data_root", "checkpoint", "prompt_version",
                 "redundancy_layers", "visionzip_layers", "allocation_temperature",
                 "spatial_mix_weight", "warmup", "seed")
PATH_FIELDS = frozenset(("model_path", "input_json", "data_root", "checkpoint"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_setting(name, left, right):
    if name in PATH_FIELDS:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    return left == right


def validate_configuration(args, saved, rate, share, mode):
    expected = {name: getattr(args, name) for name in COMMON_FIELDS}
    expected.update(prune_rates=[rate], lambdas=[share], selection_modes=[mode])
    for name, value in expected.items():
        if name not in saved or not same_setting(name, value, saved[name]):
            raise ValueError(f"Resume configuration differs for {name}; use the original experiment settings")


def read_prefix(path):
    """Return valid JSON objects and byte length; only torn final rows are ignored."""
    path = Path(path)
    if not path.is_file():
        return [], 0
    lines = path.read_bytes().splitlines(keepends=True)
    rows, valid_bytes = [], 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("Checkpoint row is not an object")
        except (ValueError, UnicodeDecodeError):
            if i == len(lines)-1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"Corrupt checkpoint record {i+1}: {path}")
        rows.append(row)
        valid_bytes += len(line)
    return rows, valid_bytes


def validate_prediction(row, sample_id, gt, index):
    if str(row.get("question_id")) != str(sample_id) or row.get("gt") != gt:
        raise ValueError(f"Resume ID/GT/order mismatch at sample {index+1}")
    if not isinstance(row.get("answer"), str):
        raise ValueError(f"Missing saved answer at sample {index+1}")
    for name in TIMING_FIELDS:
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid saved {name} at sample {index+1}")
    if row["lat_seconds"] > row["generation_seconds"]:
        raise ValueError("Saved Lat exceeds full generation duration")
    for name in ("generated_tokens", "peak_allocated_bytes", "peak_reserved_bytes"):
        if type(row[name]) is not int:
            raise ValueError(f"Saved {name} must be an integer")


def load_records(output, samples, recover_budget, *, rate, share, mode):
    """Migrate legacy predictions only when no nonempty modern journal exists."""
    output = Path(output)
    journal = output / "progress.jsonl"
    modern, _ = read_prefix(journal)
    legacy = not modern
    rows = read_prefix(output / "predictions.jsonl")[0] if legacy else modern
    if len(rows) > len(samples):
        raise ValueError("Saved sample count exceeds the current input")
    records = []
    for index, (row, sample) in enumerate(zip(rows, samples)):
        prediction = row if legacy else row.get("prediction", {})
        validate_prediction(prediction, sample.sample_id, sample.gt, index)
        if legacy:
            budget = dict(question_id=sample.sample_id, **recover_budget(sample))
        else:
            if row.get("index") != index:
                raise ValueError("Journal indices are not a contiguous prefix")
            budget = row["budget"]
        if (str(budget.get("question_id")) != str(sample.sample_id)
                or budget.get("selection_mode") != mode or budget.get("distill_share") != share
                or budget.get("target_average_prune_rate") != rate):
            raise ValueError("Saved budget belongs to a different sample or hybrid configuration")
        records.append(dict(index=index, prediction=prediction, budget=budget))
    return records
