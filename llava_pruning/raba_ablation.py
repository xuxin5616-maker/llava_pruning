"""Crop quota ablation and post-generation diagnostic measurements."""

import hashlib
from pathlib import Path, PurePosixPath
import random
from statistics import mean, pstdev


STRATEGIES = ("uniform", "global_only", "local_only", "global_local", "shuffled")
SHUFFLE_SEEDS = (0, 1, 2)
FIELDS = ("crop_keep_rate_std_pct", "shuffle_unchanged_pct", "abnormal_samples",
          "annotated_samples", "covered_mask_samples", "defect_token_retention_pct")
SAMPLE_FIELDS = ("abnormal_samples", "annotated_samples", "covered_mask_samples")
METRICS = ("acc", "bacc", "defect_token_retention_pct", "crop_keep_rate_std_pct", "shuffle_unchanged_pct")
SUMMARY_FIELDS = ("prune_rate", "raba_strategy", "expected_runs", "completed_runs", *SAMPLE_FIELDS,
                  *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")))


def schedule(jobs):
    for job in jobs:
        if job["method"] != "hybrid_baseline":
            continue
        for strategy in STRATEGIES:
            seeds = SHUFFLE_SEEDS if strategy == "shuffled" else (None,)
            for seed in seeds:
                job_id = f"{job['job_id']}_{strategy}"
                config = f"{job['config']};raba={strategy}"
                if seed is not None:
                    job_id += f"_seed{seed}"
                    config += f";shuffle_seed={seed}"
                yield dict(job, job_id=job_id, config=config, raba_strategy=strategy, raba_seed=seed)


class QuotaAblation:
    def __init__(self, adapter, strategy, seed):
        if strategy not in STRATEGIES or (strategy == "shuffled") != (seed is not None):
            raise ValueError("Invalid RABA strategy or shuffle seed")
        if not adapter.lean or adapter.settings.spatial_strength != 0:
            raise ValueError("The RABA ablation uses the lean baseline with spatial prior disabled")
        self.adapter = adapter
        self.strategy = strategy
        self.seed = seed
        self.image_key = None
        self.unchanged = None
        adapter.crop_redundancy_component = {
            "global_only": "base_redundancy", "local_only": "anyres_redundancy",
        }.get(strategy, "redundancy")
        adapter.transform_crop_quotas = self.transform

    def transform(self, allocation):
        crops = allocation["crops"]
        original = [crop["keep"] for crop in crops]
        if self.strategy == "uniform":
            quotient, remainder = divmod(allocation["target_anyres_keep"], len(crops))
            keeps = [quotient + (index < remainder) for index in range(len(crops))]
        elif self.strategy == "shuffled":
            if self.image_key is None:
                raise RuntimeError("RABA shuffle requires a stable image identity")
            digest = hashlib.sha256(f"{self.seed}\n{self.image_key}".encode()).digest()
            keeps = original.copy()
            random.Random(int.from_bytes(digest, "big")).shuffle(keeps)
            self.unchanged = keeps == original
        else:
            return
        if sum(keeps) != allocation["target_anyres_keep"] or any(not 4 <= k <= 729 for k in keeps):
            raise RuntimeError("RABA allocation violated the fixed AnyRes budget")
        for crop, keep in zip(crops, keeps):
            crop["keep"] = keep

    def observe(self, sample):
        keeps = [crop["keep"] for crop in self.adapter.allocation["crops"]]
        result = dict(crop_keeps=keeps, unchanged=self.unchanged)
        if sample.gt == 1:
            result.update(measure_defect_tokens(sample.source, self.adapter.geometry,
                                               self.adapter.device_decisions[1:]))
        return result


def mask_paths(row, mvtec_root):
    relative = PurePosixPath(str(row.get("origin_path") or "").replace("\\", "/"))
    if len(relative.parts) < 4:
        return []
    category, layout, defect, filename = relative.parts[-4:]
    if layout not in {"image", "test"} or any(part in {".", ".."} for part in (category, defect, filename)):
        return []
    source = row.get("dataset", "MVTec-AD")
    image = Path(row.get("original_resolved_image", row["resolved_image"]))
    category_root = Path(mvtec_root) / category if source == "MVTec-AD" else image.parent.parent.parent
    stem = PurePosixPath(filename).stem
    for directory_name in ("ground_truth", "mask", "masks"):
        directory = category_root / directory_name / defect
        if source == "MVTec-LOCO":
            paths = sorted((directory / stem).glob("*.png"))
            if paths:
                return paths
        for name in (stem + "_mask.png", stem + ".png"):
            path = directory / name
            if path.is_file():
                return [path]
    return []


def freeze_masks(rows, mvtec_root):
    from PIL import Image

    for row in rows:
        if row["gt"] != 1:
            continue
        paths = mask_paths(row, mvtec_root)
        row["raba_gt_masks"] = []
        if not paths:
            continue
        size = row.get("transform", {}).get("original_size") or row.get("image_size")
        if size is None:
            with Image.open(row["resolved_image"]) as image:
                size = list(image.size)
        row["raba_mask_size"] = size
        for path in paths:
            stat = path.stat()
            row["raba_gt_masks"].append(dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))


def project_defect_tokens(mask, geometry):
    """Project an original-size binary mask to the valid raw AnyRes patch grid."""
    import numpy as np
    from PIL import Image

    nearest = Image.Resampling.NEAREST
    resized = Image.fromarray(mask).resize(geometry.original_size, nearest).resize(geometry.resized_size, nearest)
    width, height = geometry.grid
    tile, patch, side = geometry.tile_size, geometry.patch_size, geometry.token_side
    support = side * patch
    canvas = Image.new("1", (width * tile, height * tile))
    canvas.paste(resized, geometry.paste_xy)
    pixels = np.asarray(canvas, dtype=bool)
    for index in range(width * height):
        row, col = divmod(index, width)
        y, x = row * tile, col * tile
        crop = pixels[y:y+support, x:x+support]
        yield crop.reshape(side, patch, side, patch).any(axis=(1, 3)).reshape(-1)


def measure_defect_tokens(source, geometry, decisions):
    import numpy as np
    from PIL import Image
    import torch

    files = source.get("raba_gt_masks", [])
    if not files:
        return dict(defect_mask_status="missing")
    size = tuple(source["raba_mask_size"])
    mask = np.zeros((size[1], size[0]), dtype=bool)
    for file in files:
        path = Path(file["path"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (file["bytes"], file["mtime_ns"]):
            raise ValueError(f"Frozen GT mask changed: {path}")
        try:
            with Image.open(path) as image:
                if image.size != size:
                    return dict(defect_mask_status="size_mismatch")
                mask |= np.asarray(image.convert("L")) != 0
        except OSError:
            return dict(defect_mask_status="unreadable")
    anchors = [torch.cat((view["dominant_indices"], view["contextual_indices"])) for view in decisions]
    indices = torch.cat(anchors).cpu().numpy()
    total = kept = offset = 0
    for tokens, anchor in zip(project_defect_tokens(mask, geometry), anchors, strict=True):
        total += int(tokens.sum())
        end = offset + anchor.numel()
        kept += int(tokens[indices[offset:end]].sum())
        offset = end
    return dict(defect_mask_status="covered" if total else "no_defect_patch",
                defect_tokens=total, kept_defect_tokens=kept)


def summarize(records):
    observations = [record["raba"] for record in records]
    covered = [row for row in observations if row.get("defect_tokens", 0)]
    unchanged = [row["unchanged"] for row in observations if row["unchanged"] is not None]
    return dict(crop_keep_rate_std_pct=mean(pstdev(row["crop_keeps"]) * 100 / 729 for row in observations),
                shuffle_unchanged_pct=100 * mean(unchanged) if unchanged else None,
                abnormal_samples=sum(row["gt"] == 1 for row in records),
                annotated_samples=sum(row.get("defect_mask_status") not in {None, "missing"} for row in observations),
                covered_mask_samples=len(covered),
                defect_token_retention_pct=(100 * mean(row["kept_defect_tokens"] / row["defect_tokens"] for row in covered)
                                            if covered else None))


def summarize_trials(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["prune_rate"], row["raba_strategy"]), []).append(row)
    for (rate, strategy), trials in groups.items():
        completed = [trial for trial in trials if trial["status"] == "success"]
        summary = dict(prune_rate=rate, raba_strategy=strategy, expected_runs=len(trials), completed_runs=len(completed))
        if len(completed) == len(trials):
            summary.update({key: completed[0][key] for key in SAMPLE_FIELDS})
            for metric in METRICS:
                values = [row[metric] for row in completed if row.get(metric) is not None]
                if len(values) == len(completed):
                    summary[metric + "_mean"] = mean(values)
                    summary[metric + "_std"] = pstdev(values)
        yield summary
