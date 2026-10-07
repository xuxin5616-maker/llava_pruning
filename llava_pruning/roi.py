"""Choose the annotation source independently of image preprocessing."""

from pathlib import Path

import numpy as np
from PIL import Image

from .data import Sample


def choose_roi(sample: Sample, mode: str):
    if mode == "anyres_only":
        return "anyres_only", None, None
    if mode == "ex_base_copy":
        return "ex_base_copy", None, None
    if mode == "anyres_max_9":
        return "anyres", None, None
    if mode == "randompatch":
        return "random", None, None
    if mode != "randomroi":
        raise ValueError(f"Unknown ROI mode: {mode}")
    if sample.mask is not None:
        return "mask", sample.mask, None
    if sample.bbox is not None:
        return "bbox", None, sample.bbox
    return "random", None, None


def load_mask(path: Path):
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as opened:
            return opened.convert("L")
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "anomaly_map" not in archive:
                raise ValueError(f"Mask archive has no anomaly_map: {path}")
            return archive["anomaly_map"]
    raise ValueError(f"Unsupported mask format: {path}")
