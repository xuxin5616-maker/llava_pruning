"""Calibrate and evaluate SparseVLM on MVTec AD or MMAD binary questions."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import benchmark_visa_methods as common
from benchmark_fastv import DEFAULT_DATA_ROOTS, DEFAULT_MVTEC_JSON
from llava_pruning.data import Sample, load_samples
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.pool_ablation_report import FIELDS, compact_result_row
from llava_pruning.prompts import resolve_prompt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mvtec", "mmad"), default="mvtec",
                   help="MMAD uses 0-shot binary Anomaly Detection, excluding VisA, with 1152x1152 stretch")
    p.add_argument("--model-path", type=Path,
                   default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov"))
    p.add_argument("--input-json", type=Path, help="MVTec question list or optional MMAD annotation file")
    p.add_argument("--data-root", type=Path, help="Image root for the selected dataset")
    p.add_argument("--mvtec-root", type=Path, default=Path("/home/yz/xxy/data/datasets/MVTec-AD"),
                   help="Raw MVTec-AD images for the MMAD source of that name")
    p.add_argument("--average-prune-rate", type=float, default=50)
    p.add_argument("--after-blocks", type=int, nargs="+",
                   help="Completed 1-based decoder blocks, default 3 7 16; use 2 7 16 for a 90%% target")
    p.add_argument("--calibration-samples", type=int, default=12)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-acc", type=float,
                   help="Matching unpruned ACC in percent; defaults to 91.18 for MVTec, blank Ratio for MMAD")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--resume", action="store_true", help="Continue with the saved calibration and sample order")
    return p


def stretch_mmad(rows, original_images, output, model_config, *, resume):
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from benchmark_mmad_binary import prepare_stretched_images
    from benchmark_visa_grid3 import CANVAS, validate_grid

    validate_grid(SimpleNamespace(**model_config))
    cache_key = hashlib.sha256(json.dumps(
        ["bicubic-stretch-v1", CANVAS, [(str(path), info) for path, info in original_images.items()]],
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    parent = output.parent / ".sparsevlm_mmad_stretch"
    cache = parent / cache_key
    manifest = cache / "images.json"
    if not manifest.is_file():
        if resume:
            raise FileNotFoundError(f"Missing prepared MMAD image cache: {cache}")
        parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=parent) as temp:
            staging = Path(temp) / "cache"
            copies = [dict(row) for row in rows]
            prepare_stretched_images(copies, staging, model_config)
            metadata = {}
            for item in copies:
                source = item["original_resolved_image"]
                metadata[source] = {key: item[key] for key in (
                    "image_size", "image_bytes", "image_mtime_ns", "transform",
                )}
                metadata[source]["filename"] = Path(item["resolved_image"]).name
            common.save_json(staging / "images.json", metadata)
            try:
                staging.rename(cache)
            except OSError:
                if not manifest.is_file():
                    raise

    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if set(metadata) != {str(path) for path in original_images}:
        raise ValueError("Prepared MMAD image set does not match the original files")
    prepared = {}
    for source, info in metadata.items():
        filename = info["filename"]
        if Path(filename).name != filename or info["image_size"] != [CANVAS, CANVAS]:
            raise ValueError("Invalid prepared MMAD image metadata")
        path = cache / "prepared_images" / filename
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (info["image_bytes"], info["image_mtime_ns"]):
            raise ValueError(f"Prepared MMAD image changed: {path}")
        prepared[source] = dict(info, original_resolved_image=source, resolved_image=str(path))
    for row in rows:
        row.update(prepared[row["resolved_image"]])
        row.pop("filename")


def prepare_inputs(args, output, model_config):
    if args.dataset == "mvtec":
        data_root = args.data_root or Path(DEFAULT_DATA_ROOTS["mvtec"])
        input_json = args.input_json or Path(DEFAULT_MVTEC_JSON)
        samples = load_samples(input_json, data_root)
        if not samples or any(sample.gt is None for sample in samples):
            raise ValueError("SparseVLM benchmark requires a nonempty MVTec binary test set")
        samples = [replace(sample, text=resolve_prompt("v0", sample.category, sample.text)[0]) for sample in samples]
        identity = dict(data_root=str(data_root.resolve()), input_json=str(input_json.resolve()),
                        input_sha256=common.digest(input_json))
        return samples, identity, None, None

    from PIL import Image
    from llava_pruning.mmad_data import load_binary

    data_root = args.data_root or Path("/home/yz/xxy/data/datasets/MMAD")
    rows, info = load_binary(data_root, args.input_json, seed=args.seed, exclude_sources=("VisA",),
                             mvtec_root=args.mvtec_root)
    images = {}
    for row in rows:
        path = Path(row["resolved_image"])
        if path not in images:
            with Image.open(path) as image:
                size = list(image.size)
            stat = path.stat()
            images[path] = dict(image_size=size, image_bytes=stat.st_size, image_mtime_ns=stat.st_mtime_ns)
    stretch_mmad(rows, images, output, model_config, resume=args.resume)
    samples = []
    for row in rows:
        path = Path(row["resolved_image"])
        samples.append(Sample(row["question_id"], path, row["category"], row["origin_path"],
                              row["text"], row["gt"], None, None, row))
    frozen = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    identity = dict(dataset="mmad", resize_mode="stretch", data_root=str(data_root.resolve()), input_json=info["annotations"],
                    input_sha256=info["annotations_sha256"], mvtec_root=info["mvtec_root"],
                    frozen_questions_sha256=hashlib.sha256(frozen).hexdigest())
    return samples, identity, info, frozen


def check_budget(blocks, depth, target):
    if (type(depth) is not int or not blocks or tuple(sorted(set(blocks))) != tuple(blocks)
            or any(not 0 < block < depth for block in blocks)):
        raise ValueError("Pruning blocks must be distinct completed blocks below the decoder depth")
    ceiling = 100 * (depth - blocks[0]) / depth
    if target >= ceiling:
        raise ValueError(f"{target:g}% is unreachable after block {blocks[0]}: "
                         f"layer-average pruning must be below {ceiling:.3f}%")


def calibrate(measure, target):
    """Search the global Eq. 7 scale using measured, not nominal, pruning."""
    trials = []

    def trial(scale):
        actual = measure(scale)
        trials.append(dict(scale=scale, measured_average_prune_rate=actual))
        print(f"Calibration scale={scale:.6f} observed={actual:.3f}%", flush=True)
        return actual

    low, high = 0.0, 0.25
    actual = trial(high)
    while actual < target and high < 1:
        low, high = high, min(1.0, high * 2)
        actual = trial(high)
    if actual < target - .25:
        raise ValueError(f"Target {target:g}% was not reached on calibration samples; "
                         f"scale 1 measured {actual:.3f}%")
    for _ in range(10):
        if any(abs(t["measured_average_prune_rate"] - target) <= .25 for t in trials):
            break
        middle = (low + high) / 2
        if trial(middle) < target:
            low = middle
        else:
            high = middle
    best = min(trials, key=lambda t: abs(t["measured_average_prune_rate"] - target))
    return dict(best, trials=trials)


def run(args):
    from llava_pruning.backend import LlavaBackend
    from llava_pruning.sparsevlm import AFTER_BLOCKS, PAPER, REFERENCE, SparseVLMMethod
    import torch

    target = args.average_prune_rate
    baseline = args.baseline_acc
    if baseline is None and args.dataset == "mvtec":
        baseline = 91.18
    if (not 0 < target < 100 or (baseline is not None and not 0 < baseline <= 100)
            or args.calibration_samples < 1 or args.warmup < 0 or not 0 <= args.seed < 2**32):
        raise ValueError("Invalid target rate, baseline ACC, calibration size, warmup or seed")
    output = (args.output_dir or Path(f"output/{args.dataset}_sparsevlm_avg{target:g}_01")).resolve()
    run_path = output / "run.json"
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty directory or --resume: {output}")

    blocks = tuple(args.after_blocks or AFTER_BLOCKS)
    config_path = args.model_path / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    check_budget(blocks, model_config.get("num_hidden_layers"), target)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one server GPU with CUDA_VISIBLE_DEVICES")
    samples, data_identity, dataset_info, frozen = prepare_inputs(args, output, model_config)
    is_mmad = args.dataset == "mmad"
    if is_mmad:
        import benchmark_mmad_binary as mm
        from llava_pruning.mmad_data import canonical_answer
        mm.describe(dataset_info)
    frozen_path = output / "mmad_binary.jsonl"
    identity = dict(
        **data_identity, model_path=str(args.model_path.resolve()), model_config_sha256=common.digest(config_path),
        sample_count=len(samples),
        target_average_prune_rate=target, seed=args.seed, calibration_samples=args.calibration_samples,
        warmup=args.warmup, baseline_acc=baseline, after_blocks=list(blocks),
        adapter_sha256=common.digest(Path(__file__).parent / "llava_pruning" / "sparsevlm.py"),
    )
    if args.resume:
        spec = json.loads(run_path.read_text(encoding="utf-8"))
        if spec["configuration"] != identity:
            raise ValueError("Saved experiment differs from the current arguments or input files")
        if is_mmad and common.digest(frozen_path) != identity["frozen_questions_sha256"]:
            raise ValueError("Saved MMAD question manifest changed")

    backend = LlavaBackend(args.model_path, "anyres_max_9", image_token_order="base_first")
    common.validate_single_gpu(backend.model, torch)
    core = backend.model.get_model()
    if is_mmad:
        from benchmark_visa_grid3 import validate_grid
        validate_grid(core.config, core.get_vision_tower().image_size)
    device = torch.cuda.get_device_properties(0)
    job = dict(prune_rate=target)

    def generate(index, method, *, measured):
        sample = samples[index]
        return backend.generate(
            sample, sample.text, method, target, "anyres_max_9",
            capture_visualization=False, capture_attention=False,
            random_seed=(args.seed + index) % 2**32, do_sample=False, benchmark_metrics=measured,
        )

    if not args.resume:
        selected = common.warmup_indices(len(samples), args.calibration_samples)

        def measure(scale):
            method = SparseVLMMethod(core, scale, after_blocks=blocks)
            return statistics.mean(generate(index, method, measured=False)["stats"]["actual_average_prune_rate"]
                                   for index in selected)

        calibration = calibrate(measure, target)
        calibration["question_ids"] = [samples[index].sample_id for index in selected]
        spec = dict(
            configuration=identity, calibration=calibration, paper=PAPER, stage_reference=REFERENCE,
            definitions=dict(
                {key: common.DEFINITIONS[key] for key in ("lat", "memory", "accuracy")},
                budget="Mean per-image 100 * (1 - sum(visual inputs across LLM blocks) / "
                       "(LLM depth * native packed image tokens)); recycled tokens count as retained.",
                adaptation=f"Paper rank rule and sum-based recycling; pruning after blocks {blocks}; v1.5 cluster sizes; "
                           "Qwen2 GQA; unchanged native attention backend; layer-specific dense cache positions.",
                calibration="Global rank scale selected from budget only. These samples also appear in full evaluation.",
            ),
        )
        output.mkdir(parents=True, exist_ok=True)
        if is_mmad:
            frozen_path.write_bytes(frozen)
            spec["dataset_info"] = dataset_info
            spec["definitions"]["accuracy"] = "MMAD binary Anomaly Detection, 0-shot, frozen question and option mapping."
            spec["definitions"]["preprocessing"] = (
                "RGB BICUBIC stretch to 1152x1152 before native AnyRes, for one Base plus nine crops. "
                "Prepared files are shared across rates; CPU resize and file I/O are outside timing."
            )
        common.save_json(run_path, spec)

    common.verify_runtime(output, backend.inference_config, device.name, device.total_memory)
    method = SparseVLMMethod(core, spec["calibration"]["scale"], after_blocks=blocks)
    path = output / "predictions.jsonl"
    records = mm.restore_records(path, samples) if is_mmad else common.restore_records(path, samples)
    start = len(records)
    if records and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    if start < len(samples):
        for index in common.warmup_indices(len(samples), args.warmup):
            generate(index, method, measured=True)
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as stream:
        for index in range(start, len(samples)):
            result = generate(index, method, measured=True)
            sample = samples[index]
            record = dict(question_id=sample.sample_id, gt=sample.gt, answer=result["answer"],
                          budget=result["stats"], **{key: result[key] for key in common.TIMING})
            if is_mmad:
                answer, option = canonical_answer(sample.source, record["answer"])
                record.update(canonical_answer=answer, selected_option=option)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            if index == start or (index + 1) % 25 == 0 or index + 1 == len(samples):
                elapsed = time.perf_counter() - started
                eta = elapsed * (len(samples) - index - 1) / (index - start + 1)
                print(f"SparseVLM {index + 1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    config = f"sparsevlm;dataset={args.dataset};target_layer_average={target:g}%;after_blocks={','.join(map(str, blocks))}"
    denominator = baseline / 100 if baseline is not None else None
    fields = FIELDS
    if is_mmad:
        config += ";resize=stretch;canvas=1152;grid=3x3"
        report = mm.summarize_records(records, samples, job, device.total_memory)
        row = mm.metric_row(report, config, denominator)
        fields = mm.FIELDS
        source_rows = [
            dict(mm.metric_row(result, config, None), source=source,
                 questions=result["evaluated_samples"], normal=result["normal"], abnormal=result["abnormal"],
                 actual_average_prune_rate=result["budget"]["actual_average_prune_rate"])
            for source, result in report["sources"].items()
        ]
        common.save_csv(output / "summary_by_source.csv", source_rows,
                        (*mm.SOURCE_FIELDS, "actual_average_prune_rate"))
    else:
        report = extended_metrics(common.summarize_records(records, len(samples), job, device.total_memory),
                                  denominator)
        row = compact_result_row(report, config)
    actual = report["budget"]["actual_average_prune_rate"]
    report.update(calibration=spec["calibration"], actual_average_prune_rate=actual)
    common.save_json(output / "metrics.json", report)
    row["actual_average_prune_rate"] = actual
    common.save_csv(output / "summary.csv", [row], (*fields, "actual_average_prune_rate"))
    print(f"Measured layer-average pruning {actual:.3f}% | summary {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    run(parser().parse_args())
