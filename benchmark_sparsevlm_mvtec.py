"""Calibrate and evaluate SparseVLM on the full MVTec AD question set."""

import argparse
import json
from pathlib import Path
import statistics
import time

import benchmark_visa_methods as common
from benchmark_fastv import DEFAULT_DATA_ROOTS, DEFAULT_MVTEC_JSON
from llava_pruning.data import load_samples
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.pool_ablation_report import FIELDS, compact_result_row
from llava_pruning.prompts import resolve_prompt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path,
                   default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov"))
    p.add_argument("--input-json", type=Path, default=Path(DEFAULT_MVTEC_JSON))
    p.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOTS["mvtec"]))
    p.add_argument("--average-prune-rate", type=float, default=50)
    p.add_argument("--calibration-samples", type=int, default=12)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-acc", type=float, default=91.18, help="Reference unpruned ACC in percent")
    p.add_argument("--output-dir", type=Path, default=Path("output/mvtec_sparsevlm_avg50_01"))
    p.add_argument("--resume", action="store_true", help="Continue with the saved calibration and sample order")
    return p


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
    if (not 0 < target < 100 or not 0 < args.baseline_acc <= 100
            or args.calibration_samples < 1 or args.warmup < 0 or not 0 <= args.seed < 2**32):
        raise ValueError("Invalid target rate, baseline ACC, calibration size, warmup or seed")
    output = args.output_dir.resolve()
    run_path = output / "run.json"
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty directory or --resume: {output}")

    samples = load_samples(args.input_json, args.data_root)
    if not samples or any(sample.gt is None for sample in samples):
        raise ValueError("SparseVLM benchmark requires a nonempty MVTec binary test set")
    prompts = [resolve_prompt("v0", sample.category, sample.text)[0] for sample in samples]
    identity = dict(
        model_path=str(args.model_path.resolve()), model_config_sha256=common.digest(args.model_path / "config.json"),
        input_json=str(args.input_json.resolve()), input_sha256=common.digest(args.input_json),
        data_root=str(args.data_root.resolve()), sample_count=len(samples),
        target_average_prune_rate=target, seed=args.seed, calibration_samples=args.calibration_samples,
        warmup=args.warmup, baseline_acc=args.baseline_acc, after_blocks=list(AFTER_BLOCKS),
        adapter_sha256=common.digest(Path(__file__).parent / "llava_pruning" / "sparsevlm.py"),
    )
    if args.resume:
        spec = json.loads(run_path.read_text(encoding="utf-8"))
        if spec["configuration"] != identity:
            raise ValueError("Saved experiment differs from the current arguments or input files")

    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one server GPU with CUDA_VISIBLE_DEVICES")
    backend = LlavaBackend(args.model_path, "anyres_max_9", image_token_order="base_first")
    common.validate_single_gpu(backend.model, torch)
    core = backend.model.get_model()
    device = torch.cuda.get_device_properties(0)
    job = dict(prune_rate=target)

    def generate(index, method, *, measured):
        return backend.generate(
            samples[index], prompts[index], method, target, "anyres_max_9",
            capture_visualization=False, capture_attention=False,
            random_seed=(args.seed + index) % 2**32, do_sample=False, benchmark_metrics=measured,
        )

    if not args.resume:
        selected = common.warmup_indices(len(samples), args.calibration_samples)

        def measure(scale):
            method = SparseVLMMethod(core, scale)
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
                adaptation="Paper rank rule and sum-based recycling, v1.5 pruning locations and cluster sizes; "
                           "Qwen2 GQA; unchanged native attention backend; layer-specific dense cache positions.",
                calibration="Global rank scale selected from budget only. These samples also appear in full evaluation.",
            ),
        )
        output.mkdir(parents=True, exist_ok=True)
        common.save_json(run_path, spec)

    common.verify_runtime(output, backend.inference_config, device.name, device.total_memory)
    method = SparseVLMMethod(core, spec["calibration"]["scale"])
    path = output / "predictions.jsonl"
    records = common.restore_records(path, samples)
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
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            if index == start or (index + 1) % 25 == 0 or index + 1 == len(samples):
                elapsed = time.perf_counter() - started
                eta = elapsed * (len(samples) - index - 1) / (index - start + 1)
                print(f"SparseVLM {index + 1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    report = extended_metrics(common.summarize_records(records, len(samples), job, device.total_memory),
                              args.baseline_acc / 100)
    actual = report["budget"]["actual_average_prune_rate"]
    report.update(calibration=spec["calibration"], actual_average_prune_rate=actual)
    common.save_json(output / "metrics.json", report)
    row = compact_result_row(report, f"sparsevlm;dataset=mvtec;target_layer_average={target:g}%")
    row["actual_average_prune_rate"] = actual
    common.save_csv(output / "summary.csv", [row], (*FIELDS, "actual_average_prune_rate"))
    print(f"Measured layer-average pruning {actual:.3f}% | summary {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    run(parser().parse_args())
