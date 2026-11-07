"""Load the original MPDD category/test layout, with no training images or masks."""

from collections import Counter
from pathlib import Path

from .data import Sample
from .prompts import GENERIC_PROMPT_V0


CATEGORIES = ("bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes")
IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
RESERVED = frozenset(("ground_truth", "mask", "masks", "annotation", "annotations", "labels",
                      "train", "test", "val", "validation", "images", "unknown", "unlabeled", "unlabelled"))


def load_mpdd_test(data_root):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"MPDD root not found: {root}")
    missing = [name for name in CATEGORIES if not (root/name/"test").is_dir()]
    if missing:
        raise ValueError(f"Incomplete MPDD root: missing category/test for {missing}. "
                         f"Point --data-root at the directory containing {', '.join(CATEGORIES)}.")
    unknown = [p.name for p in root.iterdir() if p.is_dir() and (p/"test").is_dir() and p.name not in CATEGORIES]
    if unknown:
        raise ValueError(f"Unexpected MPDD test categories: {sorted(unknown)}")
    from PIL import Image
    samples, seen, sizes, category_counts = [], set(), Counter(), {}
    for category in CATEGORIES:
        counts = Counter()
        for folder in sorted((root/category/"test").iterdir()):
            if not folder.is_dir():
                if folder.suffix.lower() in IMAGE_SUFFIXES:
                    raise ValueError(f"Unlabeled MPDD image directly under test/: {folder}")
                continue
            if folder.name.lower() in RESERVED:
                raise ValueError(f"Reserved/unlabeled MPDD test directory: {folder}")
            gt = int(folder.name.lower() not in {"good", "normal"})
            for path in sorted(folder.iterdir()):
                if path.is_dir():
                    raise ValueError(f"Unexpected nested MPDD test directory: {path}; expected defect_type/image.png")
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"MPDD test image escapes data root: {path}")
                if resolved in seen:
                    raise ValueError(f"Duplicate resolved MPDD test image: {path}")
                seen.add(resolved)
                with Image.open(resolved) as image:
                    size = list(image.size)
                stat = resolved.stat()
                relative = path.relative_to(root).as_posix()
                record = dict(question_id=relative, image=relative, origin_path=relative, category=category,
                    gt=gt, text=GENERIC_PROMPT_V0, image_size=size,
                    image_bytes=stat.st_size, image_mtime_ns=stat.st_mtime_ns)
                samples.append(Sample(relative,resolved,category,relative,GENERIC_PROMPT_V0,gt,None,None,record))
                counts[gt] += 1
                sizes[f"{size[0]}x{size[1]}"] += 1
        if not counts[0] or not counts[1]:
            raise ValueError(f"MPDD category {category} needs both normal and abnormal test images; "
                             f"found normal={counts[0]}, abnormal={counts[1]}")
        category_counts[category] = dict(samples=sum(counts.values()),normal=counts[0],abnormal=counts[1])
    samples.sort(key=lambda s:s.sample_id)
    return samples, dict(dataset="MPDD",layout="category/test/label",split="test",samples=len(samples),
        normal=sum(s.gt==0 for s in samples),abnormal=sum(s.gt==1 for s in samples),
        categories=category_counts,image_sizes=dict(sorted(sizes.items())),mask_input=False,
        label_rule="good/normal=0; other defect directories=1; reserved/unlabeled directories rejected")
