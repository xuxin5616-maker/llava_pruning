"""Five visual-input ablations on the full MVTec and non-VisA MMAD binary sets."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import benchmark_visa_methods as common
from benchmark_ficoco_v import frozen_samples, parse, prepare_data, summarize
from benchmark_mmad_binary import FIELDS, SOURCE_FIELDS, metric_row
from llava_pruning.crop_ablation import CropAblation, DROP_MODES


PROJECT = Path(__file__).resolve().parent
DATASETS = ("mvtec", "mmad")
MODES = ("base_anyres", "base_only", "anyres_only", *DROP_MODES)
VERSION = "1.0-crop-ablation"
TOKEN_FIELDS = ("original_image_tokens", "input_image_tokens", "removed_image_tokens",
                "reduction_vs_full_pct", "raw_anyres_crops")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(datasets=DATASETS, include_visa=False)
    p.add_argument("--gpus", nargs="+", type=int, default=[5, 6], help="One or two physical GPUs")
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--mvtec-input-json", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl")
    p.add_argument("--mvtec-data-root", default="/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/")
    p.add_argument("--mmad-root", default="/home/yz/xxy/data/datasets/MMAD")
    p.add_argument("--mmad-annotations")
    p.add_argument("--mmad-image-root")
    p.add_argument("--mmad-mvtec-root", default="/home/yz/xxy/data/datasets/MVTec-AD")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("output/mvtec_mmad_crop_ablation_01"))
    p.add_argument("--check-data", action="store_true", help="Resolve and count images without loading the model")
    p.add_argument("--resume", action="store_true", help="Continue the saved configuration and samples")
    p.add_argument("--worker", nargs="+", choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def verify(spec):
    if spec["version"] != VERSION or spec["source_identity"] != common.source_identity():
        raise ValueError("Benchmark source changed. Use a new output directory")
    if common.digest(Path(spec["model_path"]) / "config.json") != spec["model_config_sha256"]:
        raise ValueError("Model configuration changed")
    for dataset in spec["datasets"].values():
        if common.digest(dataset["input_json"]) != dataset["input_sha256"]:
            raise ValueError("Frozen test manifest changed")


def setup(args):
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        spec = json.loads((output / "run.json").read_text(encoding="utf-8"))
        verify(spec)
        return output, spec
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        raise ValueError("Provide one or two distinct nonnegative GPU IDs")
    if args.warmup < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid warmup or seed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty directory or --resume: {output}")
    model = Path(args.model_path).expanduser().resolve()
    config_hash = common.digest(model / "config.json")
    data = prepare_data(args)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for index, (name, value) in enumerate(data.items()):
        path = output / f"input_{name}.jsonl"
        rows = value["rows"]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        datasets[name] = dict(info=value["info"], input_json=str(path), input_sha256=common.digest(path),
                              samples=len(rows), gpu=str(args.gpus[index % len(args.gpus)]))
    spec = dict(
        version=VERSION, model_path=str(model), model_config_sha256=config_hash,
        datasets=datasets, modes=list(MODES), warmup=args.warmup, seed=args.seed,
        source_identity=common.source_identity(),
        definitions=dict(
            common.DEFINITIONS,
            crop_score="MyPool mean pooled-cosine redundancy at SigLIP block 26; smaller means more important.",
            crop_removal="Remove one AnyRes crop before LLM prefill. Keep Base and the native raster order. "
                         "Remove any packed token that includes the selected crop in its bilinear support, "
                         "and newlines from empty rows. All encoder views are used for scoring.",
            scope="Zero-shot greedy, MVTec v0 and MMAD binary Anomaly Detection excluding VisA.",
            relative_accuracy="Measured Base+AnyRes accuracy for the same dataset and source.",
        ),
    )
    common.save_json(output / "run.json", spec)
    return output, spec


def evaluate(backend, samples, spec, job, folder, total_memory):
    from llava_pruning.backend import configure_image_mode
    from llava_pruning.visa_comparison_methods import make_method, native_counts

    mode = job["mode"]
    core = backend.model.get_model()
    roi = mode if mode in {"base_only", "anyres_only"} else "anyres_max_9"
    configure_image_mode(backend.model.config, roi)
    inference = dict(backend.inference_config, image_aspect_ratio=roi, condition=mode,
                     attention_source="disabled_no_attention_scoring")
    common.verify_runtime(folder, inference, inference["gpu"], total_memory)
    method = CropAblation(core, mode) if mode in DROP_MODES else make_method("llava_unpruned", core, 0, None)
    path = folder / "predictions.jsonl"
    records = common.restore_records(path, samples)
    for row, sample in zip(records, samples):
        if row.get("mode") != mode or (row["canonical_answer"], row["selected_option"]) != parse(sample.source, row["answer"]):
            raise ValueError("Saved predictions do not match the condition or option mapping")
    if records and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    resumed = len(records)

    def generate(index):
        sample = samples[index]
        grid, counts = native_counts(core, sample.source["image_size"])
        full = sum(counts)
        result = backend.generate(
            sample, sample.text, method, 0, roi, capture_visualization=False,
            capture_attention=False, random_seed=(spec["seed"] + index) % 2**32,
            do_sample=False, include_pruning_time=True, benchmark_metrics=True,
        )
        details = result["stats"] if mode in DROP_MODES else {}
        input_tokens = {"base_only": counts[0], "anyres_only": full - counts[0]}.get(mode, full)
        input_tokens = details.get("input_image_tokens", input_tokens)
        budget = dict(
            original_image_tokens=full, input_image_tokens=input_tokens, removed_image_tokens=full - input_tokens,
            reduction_vs_full_pct=100 * (1 - input_tokens / full), raw_anyres_crops=grid[0] * grid[1],
            **{key: value for key, value in details.items() if key != "input_image_tokens"},
        )
        canonical, option = parse(sample.source, result["answer"])
        return dict(question_id=sample.sample_id, gt=sample.gt, mode=mode, answer=result["answer"],
                    canonical_answer=canonical, selected_option=option, budget=budget,
                    **{key: result[key] for key in common.TIMING})

    if resumed < len(samples):
        for index in common.warmup_indices(len(samples), spec["warmup"]):
            generate(index)
    started = time.perf_counter()
    label = f'{job["dataset"]}/{mode}'
    with path.open("a", encoding="utf-8") as stream:
        for index in range(resumed, len(samples)):
            row = generate(index)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(row)
            done = index + 1
            if done == resumed + 1 or done % 25 == 0 or done == len(samples):
                elapsed = time.perf_counter() - started
                eta = elapsed * (len(samples) - done) / (done - resumed)
                print(f"{label} {done}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    return dict(summarize(records, samples, job, total_memory), inference=inference)


def worker(manifest, names):
    import torch
    from llava_pruning.backend import LlavaBackend

    spec = json.loads(manifest.read_text(encoding="utf-8"))
    verify(spec)
    device = torch.cuda.get_device_properties(0)
    torch.manual_seed(spec["seed"])
    backend = LlavaBackend(spec["model_path"], "anyres_max_9")
    common.validate_single_gpu(backend.model, torch)
    failed = False
    for name in names:
        dataset = spec["datasets"][name]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != dataset["gpu"]:
            raise ValueError("Worker GPU does not match the manifest")
        samples = frozen_samples(dataset, name)
        for mode in spec["modes"]:
            folder = manifest.parent / name / mode
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / "metrics.json"
            old = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            if old.get("complete") and old.get("status") == "success":
                print(f"Reuse {name}/{mode}", flush=True)
                continue
            job = dict(dataset=name, mode=mode, prune_rate=0)
            print(f'Start {name}/{mode} on GPU {dataset["gpu"]}', flush=True)
            try:
                torch.cuda.empty_cache()
                result = evaluate(backend, samples, spec, job, folder, device.total_memory)
                result["status"] = "success"
            except Exception as error:
                failed = True
                result = dict(status="failed", complete=False, error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            common.save_json(result_path, result)
    return int(failed)


def write_reports(output, spec):
    rows, source_rows = [], []
    for name, dataset in spec["datasets"].items():
        results = {}
        for mode in spec["modes"]:
            path = output / name / mode / "metrics.json"
            results[mode] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        baseline = results["base_anyres"]
        for mode, result in results.items():
            config = f'dataset={name};condition={mode};shot=0;gpu={dataset["gpu"]}'
            status = result.get("status", "pending")
            row = dict(config=config, status=status, error=result.get("error", ""))
            if status == "success" and result["complete"]:
                row.update(metric_row(result, config, baseline.get("accuracy")))
                row.update({field: result["budget"][field] for field in TOKEN_FIELDS})
                for source, metrics in result["sources"].items():
                    reference = baseline.get("sources", {}).get(source, {}).get("accuracy")
                    source_rows.append(dict(metric_row(metrics, config, reference), source=source,
                                            questions=metrics["evaluated_samples"], normal=metrics["normal"],
                                            abnormal=metrics["abnormal"]))
            rows.append(row)
    common.save_csv(output / "summary.csv", rows, (*FIELDS, *TOKEN_FIELDS, "status", "error"))
    common.save_csv(output / "summary_by_source.csv", source_rows, SOURCE_FIELDS)


def run(args):
    if args.check_data:
        prepare_data(args)
        return 0
    output, spec = setup(args)
    assignments = {}
    for name, data in spec["datasets"].items():
        assignments.setdefault(data["gpu"], []).append(name)
    processes = []
    try:
        for gpu, names in assignments.items():
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--manifest", str(output / "run.json"),
                       "--worker", *names]
            processes.append(subprocess.Popen(command, cwd=PROJECT,
                                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
        codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        write_reports(output, spec)
        print(f"Summary: {output / 'summary.csv'}", flush=True)
    return int(any(codes))


def main():
    args = parser().parse_args()
    try:
        return worker(args.manifest, args.worker) if args.worker else run(args)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
