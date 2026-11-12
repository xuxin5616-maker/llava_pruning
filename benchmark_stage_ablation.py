"""Evaluate MyPool/VisionZip × distillation/FastV physical pruning."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import benchmark_mmad_binary as mm
import benchmark_visa_methods as common
from compare_distilled_visa100 import student_identity
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.mmad_source import scope_identity, working_files
from llava_pruning.pool_distill_budget import hybrid_budget, pre_view_budget
from visualize_mvtec_baseline import freeze, samples_for


PROJECT = Path(__file__).resolve().parent
RATES = (50, 70, 90)
DEFAULT_ROOT = Path("/home/yz/xxy/data/datasets")
STAGES = (("mypool", "distill"), ("visionzip", "distill"),
          ("mypool", "fastv_score_physical"), ("visionzip", "fastv_score_physical"))
BUDGET_FIELDS = (*mm.BUDGET_FIELDS, "stage2_llm_visual_tokens")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("mvtecmode", "mmadmode"))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--gpu", type=int, help="Physical GPU. Defaults to 5 for MVTec, 3 for MMAD.")
    p.add_argument("--model-path", type=Path, default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov"))
    p.add_argument("--checkpoint", type=Path, default=PROJECT / "output/attention_predictor_twig_train_01/best.pt")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--input-json", type=Path, default=DEFAULT_ROOT / "Traid_eval_data/mvtec/question_musc.jsonl")
    p.add_argument("--annotations", type=Path, help="MMAD annotation JSON")
    p.add_argument("--image-root", type=Path, help="MMAD image container")
    p.add_argument("--mvtec-root", type=Path, default=DEFAULT_ROOT / "MVTec-AD", help="External MVTec-AD images used by MMAD")
    p.add_argument("--resume", action="store_true", help="Continue from the same frozen inputs and saved GPU")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    return p


def schedule(mode, gpu):
    is_mmad = mode == "mmadmode"
    prefix = "dataset=MMAD;exclude=VisA;shot=0;resize=stretch;canvas=1152;grid=3x3" if is_mmad else "dataset=MVTec-AD;roi=anyres_max_9"
    jobs = []
    if is_mmad:
        jobs.append(dict(job_id="llava_unpruned_00", method="llava_unpruned", prune_rate=0,
                         config=f"{prefix};method=llava_unpruned;prune=0%;gpu={gpu}", status="pending", reason=""))
    for rate in RATES:
        for pre, post in STAGES:
            use_student = post == "distill"
            context = "full" if is_mmad else "base_text"
            policy = "joint" if is_mmad else "anyres_first"
            ranking = ("anyres_self" if is_mmad else "base_full") if use_student else "fastv_backbone_attention"
            config = (f"{prefix};mode={mode};pre={pre};post={post};prune={rate}%;rate=layer_average;lambda=0.7;"
                      f"layer=2;{ranking};input={context if use_student else 'backbone_full'};base_policy={policy};"
                      f"tail=8(extra);student={'VisA' if use_student else 'none'};gpu={gpu}")
            jobs.append(dict(job_id=f"{pre}_{post}_{rate}", method="hybrid_baseline", prune_rate=rate,
                             pre_stage=pre, post_stage=post, config=config, status="pending", reason=""))
    return jobs


def preflight(records, config, mode):
    vendor = str(PROJECT / "vendor" / "llava")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from llava.mm_utils import get_anyres_image_grid_shape
    from llava.model.language_model.attention_predictor_selection import native_anyres_counts

    merge = config["mm_patch_merge_type"]
    if config.get("num_hidden_layers") != 28 or merge not in {"spatial_unpad", "spatial_unpad_add_newl"}:
        raise ValueError("Use the existing 28-layer OneVision checkpoint with spatial_unpad packing")
    final_newline = merge.endswith("_add_newl")
    for size in sorted({tuple(r["image_size"]) for r in records}):
        width, height = grid = get_anyres_image_grid_shape(size, config["image_grid_pinpoints"], 384)
        original = sum(native_anyres_counts(27, grid, size, final_newline=final_newline))
        for rate in RATES:
            try:
                plan = hybrid_budget(original, rate, .7)
                pre_view_budget(dict(plan, raw_grid_height=height), width * height, final_newline=final_newline,
                                base_policy="joint" if mode == "mmadmode" else "anyres_first")
            except ValueError as error:
                raise ValueError(f"{mode} {rate}% cannot match the budget at image size {size}, grid {grid}: {error}") from error


def source_identity():
    return scope_identity(working_files(PROJECT), extra_roots=(Path(__file__).name,))


def setup(args, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory or --resume: {output}")
    is_mmad = args.mode == "mmadmode"
    gpu = args.gpu if args.gpu is not None else (3 if is_mmad else 5)
    data_root = Path(args.data_root or (DEFAULT_ROOT / ("MMAD" if is_mmad else "Traid_eval_data/mvtec"))).resolve()
    model = args.model_path.expanduser().resolve()
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if is_mmad:
        records, info = mm.load_binary(data_root, args.annotations, args.image_root, 42,
                                       exclude_sources=mm.EXCLUDED_SOURCES, mvtec_root=args.mvtec_root)
        preflight([dict(image_size=(1152, 1152))], config, args.mode)
    else:
        args.data_root, args.seed, args.prompt_version = data_root, 42, "v0"
        records, info = freeze(args), {}
        preflight(records, config, args.mode)
    checkpoint = student_identity(args.checkpoint, "visa")
    if is_mmad:
        mm.prepare_stretched_images(records, output, config)
        mm.describe(info)
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / ("mmad_binary.jsonl" if is_mmad else "mvtec_frozen.jsonl")
    frozen.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    spec = dict(mode=args.mode, gpu=gpu, model_path=str(model), model_config_sha256=common.digest(model / "config.json"),
                checkpoint=checkpoint, dataset_info=info, data_root=str(data_root), input_json=str(frozen),
                input_sha256=common.digest(frozen), sample_count=len(records), jobs=schedule(args.mode, gpu),
                source_identity=source_identity(), seed=42, warmup=10,
                definitions={"rates": "Layer-average pre-LLM + block-2 pruning. Tail 8 is extra.",
                             "visionzip": "VisionZip local top-k/Key merge; equal crop quotas within matched AnyRes total; common spatial packing.",
                             "fastv": "Block-2 last-prompt attention; physical hidden/KV deletion after block 2 using the mode's Base policy.",
                             "lat_mem": "TTFT ms and full-generation peak allocated GiB. Pruning compute included."})
    common.save_json(output / "run.json", spec)
    return spec


def load_run(output, mode):
    spec = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if spec["mode"] != mode or spec["jobs"] != schedule(mode, spec["gpu"]):
        raise ValueError("Saved mode or trial configuration differs")
    if spec["source_identity"] != source_identity():
        raise ValueError("A stage-ablation inference dependency changed; use the matching source to resume")
    for path, expected in ((spec["input_json"], spec["input_sha256"]),
                           (Path(spec["model_path"]) / "config.json", spec["model_config_sha256"])):
        if common.digest(path) != expected:
            raise ValueError(f"Saved input/model config changed: {path}")
    return spec


def evaluate(backend, method, samples, spec, job, output, total_memory):
    from llava_pruning.visa_comparison_methods import native_counts, observe_prefill, observed_budget

    is_mmad = spec["mode"] == "mmadmode"
    core = backend.model.get_model()
    path = output / "predictions.jsonl"
    records = mm.restore_records(path, samples) if is_mmad else common.restore_records(path, samples)
    if not is_mmad and records:
        # A complete final JSON object can survive without its trailing newline.
        with path.open("rb+") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
    resumed = len(records)
    rate = job["prune_rate"]

    def generate(index):
        sample = samples[index]
        prompt = sample.text if is_mmad else sample.source["prompt"]
        with observe_prefill(core) as lengths:
            result = backend.generate(sample, prompt, method, rate, "anyres_max_9",
                capture_visualization=False, capture_attention=False, random_seed=spec["seed"] + index,
                do_sample=False, include_pruning_time=True, benchmark_metrics=True)
        original = sum(native_counts(core, sample.source["image_size"])[1])
        stats = result["stats"]
        budget = observed_budget(job["method"], rate, stats, lengths, original, len(core.layers))
        budget.update({key: stats[key] for key in (*mm.HYBRID_COUNT_FIELDS, "stage2_llm_visual_tokens") if key in stats})
        row = dict(question_id=sample.sample_id, gt=sample.gt, answer=result["answer"], budget=budget,
                   **{key: result[key] for key in common.TIMING})
        if is_mmad:
            row["canonical_answer"], row["selected_option"] = mm.canonical_answer(sample.source, result["answer"])
        return row

    warmup = common.warmup_indices(len(samples), spec["warmup"]) if resumed < len(samples) else []
    context = method.generation_context(backend.model) if hasattr(method, "generation_context") else nullcontext()
    with context:
        for index in warmup:
            generate(index)
        started = time.perf_counter()
        with path.open("a", encoding="utf-8") as stream:
            for index in range(resumed, len(samples)):
                row = generate(index)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(row)
                if index == resumed or (index + 1) % 25 == 0 or index + 1 == len(samples):
                    elapsed = time.perf_counter() - started
                    eta = elapsed * (len(samples) - index - 1) / (index + 1 - resumed)
                    print(f"{job['job_id']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    if is_mmad:
        return mm.summarize_records(records, samples, job, total_memory)
    result = common.summarize_records(records, len(samples), job, total_memory)
    return extended_metrics(dict(result, nonbinary=0, mmad_score=None), .9118)


def worker(spec, job, output):
    report = dict(job_id=job["job_id"], input_sha256=spec["input_sha256"], status="failed", complete=False)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["gpu"]):
            raise ValueError("Worker must expose only its assigned GPU")
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.visa_comparison_methods import make_method

        if spec["mode"] == "mmadmode":
            samples = mm.load_frozen(spec["input_json"], exclude_sources=mm.EXCLUDED_SOURCES)
            for sample in samples:
                stat = sample.image.stat()
                if (stat.st_size, stat.st_mtime_ns) != (sample.source["image_bytes"], sample.source["image_mtime_ns"]):
                    raise ValueError(f"Frozen image changed: {sample.image}")
        else:
            samples = samples_for(spec)
        if len(samples) != spec["sample_count"]:
            raise ValueError("Frozen sample count differs")
        checkpoint = spec["checkpoint"]
        if job.get("post_stage") == "distill" and student_identity(checkpoint["path"], "visa") != checkpoint:
            raise ValueError("VisA student changed since setup")
        backend = LlavaBackend(spec["model_path"], "anyres_max_9", image_token_order="base_first")
        common.validate_single_gpu(backend.model, torch)
        core = backend.model.get_model()
        device = torch.cuda.get_device_properties(0)
        common.verify_runtime(output, backend.inference_config, device.name, device.total_memory)
        if job["prune_rate"]:
            from llava_pruning.stage_ablation import StageAblationMethod
            method = StageAblationMethod(core, Path(checkpoint["path"]), job, spec["mode"])
        else:
            method = make_method("llava_unpruned", core, 0, None)
        report.update(evaluate(backend, method, samples, spec, job, output, device.total_memory), status="success")
    except Exception as error:
        report["error"] = str(error)
        traceback.print_exc()
    finally:
        common.save_json(output / "metrics.json", report)
    return int(report["status"] != "success")


def write_reports(output, spec):
    is_mmad = spec["mode"] == "mmadmode"
    if is_mmad:
        results = mm.write_reports(output, spec)
    else:
        results = {}
        rows = []
        for job in spec["jobs"]:
            path = output / job["job_id"] / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if report and (report["job_id"] != job["job_id"] or report["input_sha256"] != spec["input_sha256"]):
                raise ValueError("Mismatched MVTec trial report")
            results[job["job_id"]] = report
            status = report.get("status", "pending")
            rows.append(mm.metric_row(report, job["config"], .9118) if report.get("complete") and status == "success"
                        else dict(config=f"{job['config']};status={status}"))
        common.save_csv(output / "summary.csv", rows, [k for k in mm.FIELDS if k not in {"mmad_score", "nonbinary"}])
    budgets = []
    for job in spec["jobs"]:
        result = results[job["job_id"]]
        budgets.append(dict(result.get("budget", {}), config=job["config"], status=result.get("status", "pending"),
                            reason=result.get("error", ""), target_average_prune_rate=job["prune_rate"]))
    common.save_csv(output / "budget_summary.csv", budgets, BUDGET_FIELDS)
    return results


def run(args):
    output = args.output_dir.expanduser().resolve()
    if args.worker:
        spec = load_run(output, args.mode)
        job = next(job for job in spec["jobs"] if job["job_id"] == args.worker)
        return worker(spec, job, output / job["job_id"])
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be a nonnegative physical GPU ID")
    spec = load_run(output, args.mode) if args.resume else setup(args, output)
    if args.resume and args.gpu is not None and args.gpu != spec["gpu"]:
        raise ValueError("Resume uses the originally assigned GPU")
    results = write_reports(output, spec)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec["gpu"]))
    failed = False
    for job in spec["jobs"]:
        job_id = job["job_id"]
        if results[job_id].get("status") == "success" and results[job_id].get("complete"):
            continue
        (output / job_id).mkdir(exist_ok=True)
        print(f"Starting {job_id} on GPU {spec['gpu']}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), args.mode, "--output-dir", str(output), "--worker", job_id]
        completed = subprocess.run(command, env=env, check=False)
        results = write_reports(output, spec)
        failed |= bool(completed.returncode or results[job_id].get("status") != "success")
    print(f"Summary | {output / 'summary.csv'}", flush=True)
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(run(parser().parse_args()))
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")
    except KeyboardInterrupt:
        sys.exit(130)
