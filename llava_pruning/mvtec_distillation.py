"""Read all raw MVTec-AD train/test RGB images for attention distillation.

No class labels, anomaly masks, question JSON or VisA data enter this loader.
Official MVTec test membership is intentionally a TRAINING source here.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import Sample
from .prompts import GENERIC_PROMPT_V0
from .visa_data import IMAGE_SUFFIXES


CATEGORIES = frozenset(("bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"))


def load_mvtec_distillation(data_root):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir(): raise NotADirectoryError(f"MVTec-AD root not found: {root}")
    directories = sorted((p for p in root.iterdir() if p.is_dir() and
                          (p.name in CATEGORIES or (p/"train").is_dir() or (p/"test").is_dir())),key=lambda p:p.name)
    if not directories:
        raise ValueError(f"Expected <category>/train/ and <category>/test/ under {root}")
    samples,seen,identity = [],set(),[]
    split_counts = Counter()
    for category in directories:
        if category.name not in CATEGORIES: raise ValueError(f"Unknown MVTec-AD category: {category.name}")
        for split in ("train","test"):
            folder = category/split
            if not folder.is_dir(): raise FileNotFoundError(f"Missing MVTec-AD image split: {folder}")
            images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not images: raise ValueError(f"MVTec-AD image split is empty: {folder}")
            for path in images:
                image = path.resolve()
                if not image.is_relative_to(root): raise ValueError(f"MVTec image escapes data root: {path}")
                if image in seen: raise ValueError(f"Duplicate MVTec image/symlink alias: {path}")
                seen.add(image)
                relative = path.relative_to(root).as_posix()
                source = dict(question_id=relative,image=relative,origin_path=relative,text=GENERIC_PROMPT_V0)
                samples.append(Sample(relative,image,category.name,relative,GENERIC_PROMPT_V0,None,None,None,source))
                stat = image.stat()
                identity.append([relative,stat.st_size,stat.st_mtime_ns])
                split_counts[split] += 1
    return sorted(samples,key=lambda s:s.sample_id),dict(dataset="mvtec",data_root=str(root),
        source_sha256=hashlib.sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest(),
        fingerprint="relative image paths, file sizes and mtimes; not a content hash",
        source_policy="ALL MVTec-AD category/train + category/test images; evaluate on independent VisA",
        samples=len(samples),original_split_counts=dict(split_counts),
        categories=dict(sorted(Counter(s.category for s in samples).items())),gt_used=False,masks_used=False,
        original_mvtec_test_is_training_source=True,original_visa_test_is_training_source=False)
