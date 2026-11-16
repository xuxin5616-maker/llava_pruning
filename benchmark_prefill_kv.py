"""Compare pruning methods with the post-prefill K/V footprint on MVTec or MMAD."""

import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time
import traceback

import benchmark_mmad_binary as mm
import benchmark_visa_methods as common
from benchmark_fastv import DEFAULT_DATA_ROOTS, DEFAULT_MVTEC_JSON
from compare_distilled_visa100 import student_identity
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.mmad_source import scope_identity, working_files
from llava_pruning.pool_distill_parallel import stop_active
from llava_pruning.prefill_kv import KV_BYTES, KV_MIB, observe_post_prefill_kv
from visualize_mvtec_baseline import freeze, samples_for


PROJECT = Path(__file__).resolve().parent
RATES = (50, 70, 90)
DATA = Path("/home/yz/xxy/data/datasets")
FIELDS = (*mm.FIELDS[:8], KV_MIB, *mm.FIELDS[8:], "status", "reason")
SOURCE_FIELDS = ("source", "questions", "normal", "abnormal", *FIELDS)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("mvtecmode", "mmadmode", "both"))
    p.add_argument("--output-dir", type=Path, required=True)
    devices = p.add_mutually_exclusive_group()
    devices.add_argument("--gpu", type=int, help="Physical GPU. Defaults to 2 for MVTec and 3 for MMAD.")
    devices.add_argument("--gpus", type=int, nargs="+", help="Physical GPUs, one trial per GPU. In both mode, split the list in half.")
    p.add_argument("--model-path", type=Path, default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov"))
    p.add_argument("--checkpoint", type=Path, default=PROJECT / "output/attention_predictor_twig_train_01/best.pt")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--mmad-data-root", type=Path, help="Override the MMAD root in both mode")
    p.add_argument("--input-json", type=Path, default=Path(DEFAULT_MVTEC_JSON), help="MVTec question list")
    p.add_argument("--annotations", type=Path, help="MMAD annotation JSON")
    p.add_argument("--image-root", type=Path, help="MMAD image container")
    p.add_argument("--mvtec-root", type=Path, default=DATA / "MVTec-AD", help="External MVTec-AD images for MMAD")
    p.add_argument("--resume", action="store_true", help="Resume with the saved samples, methods and GPU")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    return p


def schedule(mode, gpus):
    is_mmad = mode == "mmadmode"
    if is_mmad:
        jobs = mm.schedule(RATES, [gpus[0]], base_guided_tail=8, joint_self_tails=(8,))
        jobs = [job for job in jobs if job.get("variant") != "base_guided"]
        for job in jobs:
            if job.pop("gpu", None) is not None:
                job["config"] = job["config"].rsplit(";gpu=", 1)[0]
        prefix = "dataset=MMAD;exclude=VisA;shot=0;"
        suffix = ";resize=stretch;canvas=1152;grid=3x3"
    else:
        jobs = common.schedule(RATES)
        prefix, suffix = "dataset=MVTec-AD;prompt=v0;", ""
        for job in jobs:
            job["config"] = prefix + job["config"].replace("tail=9(extra)", "tail=8(extra)")
    for job in jobs:
        if job["method"] == "hybrid_baseline":
            job["config"] += f";mode={mode}"
        job["config"] += suffix
    for rate in RATES:
        blocks = [2 if rate == 90 else 3, 7, 16]
        stages = "/".join(map(str, blocks))
        jobs.append(dict(job_id=f"sparsevlm_{rate:02d}", method="sparsevlm", prune_rate=rate,
                         after_blocks=blocks, status="pending", reason="",
                         config=f"{prefix}method=sparsevlm;prune={rate}%;rate=layer_average;layers={stages}{suffix}"))
    for index, job in enumerate(job for job in jobs if job["status"] == "pending"):
        job["gpu"] = gpus[index % len(gpus)]
        job["config"] += f";gpu={job['gpu']}"
    return jobs


def source_identity():
    return scope_identity(working_files(PROJECT), extra_roots=(Path(__file__).name,))


def setup(args, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory or --resume: {output}")
    is_mmad = args.mode == "mmadmode"
    gpus = args.gpus or [args.gpu if args.gpu is not None else (3 if is_mmad else 2)]
    root = Path(args.data_root or (DATA / "MMAD" if is_mmad else DEFAULT_DATA_ROOTS["mvtec"])).expanduser().resolve()
    model = args.model_path.expanduser().resolve()
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != 28:
        raise ValueError("Use the existing 28-block OneVision checkpoint for the matched schedules")
    checkpoint = student_identity(args.checkpoint, "visa")
    if is_mmad:
        rows, info = mm.load_binary(root, args.annotations, args.image_root, 42,
                                    exclude_sources=mm.EXCLUDED_SOURCES, mvtec_root=args.mvtec_root)
        mm.prepare_stretched_images(rows, output, config)
        mm.describe(info)
    else:
        args.data_root, args.seed, args.prompt_version = root, 42, "v0"
        rows, info = freeze(args), {}
    if any(row["gt"] not in (0, 1) for row in rows):
        raise ValueError("The comparison requires a binary label for each sample")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / ("mmad_binary.jsonl" if is_mmad else "mvtec_frozen.jsonl")
    frozen.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    spec = dict(mode=args.mode, gpus=gpus, model_path=str(model), checkpoint=checkpoint, data_root=str(root),
                model_config_sha256=common.digest(config_path), input_json=str(frozen), input_sha256=common.digest(frozen),
                sample_count=len(rows), dataset_info=info, source_identity=source_identity(), jobs=schedule(args.mode, gpus),
                seed=42, warmup=10, calibration_samples=12,
                definitions=dict(common.DEFINITIONS, post_prefill_kv=(
                    "Mean over measured samples of sum of actual decoder K/V elements times element size, divided by 2^20. "
                    "Snapshot after the first decoder forward finishes pruning, before processing any generated token. "
                    "Includes text and visual K/V, excludes allocator overhead, weights and temporary tensors."),
                    rate="Target LLM layer-average visual pruning. Hybrid tails are extra. FastV measures effective visibility.",
                    relative_accuracy="100 * ACC / the same dataset's measured unpruned ACC."))
    common.save_json(output / "run.json", spec)
    return spec


def load_run(output, mode):
    spec = json.loads((output / "run.json").read_text(encoding="utf-8"))
    current = source_identity()
    saved = spec["source_identity"]
    if "gpus" not in spec:
        # 8c08406 used a single GPU. Accept its exact inference dependencies
        # while adding GPU assignments to the in-memory manifest.
        old_files = dict(current["files"])
        old_files.pop("llava_pruning/pool_distill_parallel.py", None)
        old_files[Path(__file__).name] = "d296d3c3ceccff0f053cbb793fb786d24ceb8e0a87ab1b10390a20b6b673d813"
        if saved["policy"] != current["policy"] or saved["files"] != old_files:
            raise ValueError("An inference dependency changed; resume with the original source")
        spec["gpus"] = [spec["gpu"]]
        expected = schedule(mode, spec["gpus"])

        def without_gpu(job):
            config = ";".join(part for part in job["config"].split(";") if not part.startswith("gpu="))
            return {**{k: v for k, v in job.items() if k != "gpu"}, "config": config}

        if list(map(without_gpu, spec["jobs"])) != list(map(without_gpu, expected)):
            raise ValueError("Saved benchmark trial configuration differs")
        spec["jobs"] = expected
    elif saved != current:
        raise ValueError("An inference dependency changed; resume with the original source")
    if spec["mode"] != mode or spec["jobs"] != schedule(mode, spec["gpus"]):
        raise ValueError("Saved benchmark mode or trial configuration differs")
    for path, digest in ((spec["input_json"], spec["input_sha256"]),
                         (Path(spec["model_path"]) / "config.json", spec["model_config_sha256"])):
        if common.digest(path) != digest:
            raise ValueError(f"Frozen input or model config changed: {path}")
    return spec


def generate(backend, method, sample, rate, seed, *, measured):
    prompt = sample.source.get("prompt", sample.text)
    return backend.generate(sample, prompt, method, rate, "anyres_max_9", capture_visualization=False,
                            capture_attention=False, random_seed=seed, do_sample=False,
                            include_pruning_time=True, benchmark_metrics=measured)


def build_method(backend, samples, spec, job, output):
    core = backend.model.get_model()
    name, rate = job["method"], job["prune_rate"]
    if name == "hybrid_baseline":
        saved = spec["checkpoint"]
        if student_identity(saved["path"], "visa") != saved:
            raise ValueError("VisA student checkpoint changed")
        path = Path(saved["path"])
        if spec["mode"] == "mmadmode":
            return mm.make_trial_method(job, core, path)
        from llava_pruning.pool_distill import PoolDistillMethod
        from runMyPruningPoolSpatial import MethodSettings
        return PoolDistillMethod(core, path, rate=rate, share=.7, mode="base_full",
                                 settings=MethodSettings(prune_rate=0), pruning_audit="lean", tail_drop_layers=8,
                                 predictor_input="base_text", base_mapping="fast", base_policy="anyres_first")
    if name != "sparsevlm":
        from llava_pruning.visa_comparison_methods import make_method
        return make_method(name, core, rate, None)

    from benchmark_sparsevlm_mvtec import calibrate, check_budget
    from llava_pruning.sparsevlm import SparseVLMMethod
    blocks = job["after_blocks"]
    check_budget(blocks, len(core.layers), rate)
    calibration_path = output / "calibration.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        indices = common.warmup_indices(len(samples), spec["calibration_samples"])

        def measure(scale):
            method = SparseVLMMethod(core, scale, blocks)
            return mean(generate(backend, method, samples[i], rate, spec["seed"] + i,
                                 measured=False)["stats"]["actual_average_prune_rate"] for i in indices)

        calibration = calibrate(measure, rate)
        calibration["question_ids"] = [samples[i].sample_id for i in indices]
        common.save_json(calibration_path, calibration)
    return SparseVLMMethod(core, calibration["scale"], blocks)


def evaluate(backend, method, samples, spec, job, output, total_memory):
    from llava_pruning.visa_comparison_methods import native_counts, observe_prefill, observed_budget
    is_mmad = spec["mode"] == "mmadmode"
    core = backend.model.get_model()
    path = output / "predictions.jsonl"
    records = mm.restore_records(path, samples) if is_mmad else common.restore_records(path, samples)
    if any(type(row.get(KV_BYTES)) is not int or row[KV_BYTES] <= 0 for row in records):
        raise ValueError("Saved predictions do not have post-prefill K/V measurements")
    if records:
        with path.open("rb+") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
    start = len(records)
    rate = job["prune_rate"]

    def run_image(index):
        sample = samples[index]
        with observe_prefill(core) as lengths, observe_post_prefill_kv(core) as kv:
            result = generate(backend, method, sample, rate, spec["seed"] + index, measured=True)
        original = sum(native_counts(core, sample.source["image_size"])[1])
        budget = observed_budget(job["method"], rate, result["stats"], lengths, original, len(core.layers))
        row = dict(question_id=sample.sample_id, gt=sample.gt, answer=result["answer"], budget=budget,
                   **{key: result[key] for key in common.TIMING})
        row[KV_BYTES] = kv[0]
        if is_mmad:
            row["canonical_answer"], row["selected_option"] = mm.canonical_answer(sample.source, result["answer"])
        return row

    context = method.generation_context(backend.model) if hasattr(method, "generation_context") else nullcontext()
    with context:
        if start < len(samples):
            for index in common.warmup_indices(len(samples), spec["warmup"]):
                run_image(index)
        started = time.perf_counter()
        with path.open("a", encoding="utf-8") as stream:
            for index in range(start, len(samples)):
                row = run_image(index)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(row)
                if index == start or (index + 1) % 25 == 0 or index + 1 == len(samples):
                    elapsed = time.perf_counter() - started
                    eta = elapsed * (len(samples) - index - 1) / (index + 1 - start)
                    print(f"{job['job_id']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    if is_mmad:
        result = mm.summarize_records(records, samples, job, total_memory)
        groups = defaultdict(list)
        for row, sample in zip(records, samples):
            groups[sample.source["dataset"]].append(row[KV_BYTES])
        for source, values in groups.items():
            result["sources"][source][KV_MIB] = mean(values) / 2**20
    else:
        result = common.summarize_records(records, len(samples), job, total_memory)
        result = extended_metrics(dict(result, nonbinary=0, mmad_score=None), 1.)
    result[KV_MIB] = mean(row[KV_BYTES] for row in records) / 2**20
    return result


def worker(spec, job, output):
    report = dict(job_id=job["job_id"], input_sha256=spec["input_sha256"], status="failed", complete=False)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(job["gpu"]):
            raise ValueError("Expose only the worker's assigned GPU")
        import torch
        from llava_pruning.backend import LlavaBackend
        if spec["mode"] == "mmadmode":
            samples = mm.load_frozen(spec["input_json"], exclude_sources=mm.EXCLUDED_SOURCES)
            for sample in samples:
                stat = sample.image.stat()
                if (stat.st_size, stat.st_mtime_ns) != (sample.source["image_bytes"], sample.source["image_mtime_ns"]):
                    raise ValueError(f"Frozen MMAD image changed: {sample.image}")
        else:
            samples = samples_for(spec)
        if len(samples) != spec["sample_count"]:
            raise ValueError("Frozen sample count changed")
        torch.manual_seed(spec["seed"])
        backend = LlavaBackend(spec["model_path"], "anyres_max_9", image_token_order="base_first")
        common.validate_single_gpu(backend.model, torch)
        device = torch.cuda.get_device_properties(0)
        common.verify_runtime(output, backend.inference_config, device.name, device.total_memory)
        method = build_method(backend, samples, spec, job, output)
        report.update(evaluate(backend, method, samples, spec, job, output, device.total_memory), status="success")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        common.save_json(output / "metrics.json", report)
    return int(report["status"] != "success")


def write_reports(output, spec):
    results = {}
    for job in spec["jobs"]:
        folder = output / job["job_id"]
        path = folder / "metrics.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if result and (result["job_id"] != job["job_id"] or result["input_sha256"] != spec["input_sha256"]):
            raise ValueError("Mismatched saved comparison report")
        results[job["job_id"]] = result
    baseline = results["llava_unpruned_00"]
    if not baseline.get("complete"):
        baseline = {}
    rows, sources, budgets = [], [], []
    for job in spec["jobs"]:
        config = job["config"]
        result = results[job["job_id"]]
        status, reason = result.get("status", job["status"]), result.get("error", job["reason"])
        complete = result.get("complete") and status == "success"
        row = mm.metric_row(result, config, baseline.get("accuracy")) if complete else dict(config=config)
        if complete:
            row[KV_MIB] = result[KV_MIB]
        rows.append(dict(row, status=status, reason=reason))
        for source, questions in sorted(spec["dataset_info"].get("source_counts", {}).items()):
            value = result.get("sources", {}).get(source) if complete else None
            reference = baseline.get("sources", {}).get(source, {}).get("accuracy")
            row = mm.metric_row(value, config, reference) if value else dict(config=config)
            if value:
                row.update(normal=value["normal"], abnormal=value["abnormal"], **{KV_MIB: value[KV_MIB]})
            sources.append(dict(row, source=source, questions=questions, status=status, reason=reason))
        budgets.append(dict(result.get("budget", {}), config=config, status=status, reason=reason,
                            target_average_prune_rate=job["prune_rate"]))
    fields = FIELDS if spec["mode"] == "mmadmode" else tuple(key for key in FIELDS if key not in {"mmad_score", "nonbinary"})
    common.save_csv(output / "summary.csv", rows, fields)
    if sources:
        common.save_csv(output / "summary_by_source.csv", sources, SOURCE_FIELDS)
    common.save_csv(output / "budget_summary.csv", budgets, mm.BUDGET_FIELDS)
    return results


def run_trials(runs, *, process_factory=subprocess.Popen, poll_interval=.25):
    queues = defaultdict(deque)
    for output, spec in runs:
        results = write_reports(output, spec)
        for job in spec["jobs"]:
            old = results[job["job_id"]]
            if job["status"] == "pending" and not (old.get("complete") and old.get("status") == "success"):
                queues[job["gpu"]].append((output, spec, job))
    active, failures = {}, []
    try:
        while active or any(queues.values()):
            for gpu, queue in queues.items():
                if gpu in active or not queue:
                    continue
                output, spec, job = queue.popleft()
                job_id = job["job_id"]
                folder = output / job_id
                folder.mkdir(exist_ok=True)
                log_path = folder / "worker.log"
                command = [sys.executable, "-u", str(Path(__file__).resolve()), spec["mode"],
                           "--output-dir", str(output), "--worker", job_id]
                stream = log_path.open("a", encoding="utf-8")
                try:
                    process = process_factory(command, cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                              stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                              start_new_session=os.name == "posix")
                except BaseException:
                    stream.close()
                    raise
                active[gpu] = dict(process=process, stream=stream, output=output, spec=spec, job_id=job_id, log=log_path)
                print(f"Start GPU {gpu} | {spec['mode']}/{job_id} | {log_path}", flush=True)
            for gpu, trial in list(active.items()):
                process = trial["process"]
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                trial["stream"].close()
                del active[gpu]
                if code in (-2, 130):
                    raise KeyboardInterrupt
                output, spec, job_id = trial["output"], trial["spec"], trial["job_id"]
                report_path = output / job_id / "metrics.json"
                result = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
                if code or not (result.get("status") == "success" and result.get("complete")):
                    reason = result.get("error") or f"Worker exit={code}; see {trial['log']}"
                    common.save_json(report_path, dict(result, job_id=job_id, input_sha256=spec["input_sha256"],
                                                       status="failed", complete=False, error=reason))
                    failures.append(f"{spec['mode']}/{job_id}")
                write_reports(output, spec)
                print(f"Finish GPU {gpu} | {spec['mode']}/{job_id} | exit={code}", flush=True)
            if active:
                time.sleep(poll_interval)
    finally:
        stop_active(active)
    for output, _ in runs:
        print(f"Summary | {output / 'summary.csv'}", flush=True)
    if failures:
        print("Failed trials (see worker.log): " + ", ".join(failures), flush=True)
    return int(bool(failures))


def run(args):
    output = args.output_dir.expanduser().resolve()
    if args.worker:
        spec = load_run(output, args.mode)
        job = next(item for item in spec["jobs"] if item["job_id"] == args.worker and item["status"] == "pending")
        return worker(spec, job, output / job["job_id"])
    gpus = args.gpus or ([args.gpu] if args.gpu is not None else [])
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU IDs must be distinct and nonnegative")
    if args.mode == "both":
        if not args.resume and len(gpus) < 2:
            raise ValueError("Both datasets require at least two --gpus")
        split = len(gpus) // 2
        requests = [("mvtecmode", gpus[:split], output / "mvtecmode", args.data_root),
                    ("mmadmode", gpus[split:], output / "mmadmode", args.mmad_data_root)]
    else:
        requests = [(args.mode, gpus, output, args.data_root)]
    runs = []
    for mode, devices, folder, data_root in requests:
        child = argparse.Namespace(**vars(args))
        child.mode, child.gpu, child.gpus, child.data_root = mode, None, devices, data_root
        spec = load_run(folder, mode) if args.resume else setup(child, folder)
        if args.resume and devices and devices != spec["gpus"]:
            raise ValueError("Resume uses the originally assigned GPUs")
        runs.append((folder, spec))
    return run_trials(runs)


if __name__ == "__main__":
    try:
        sys.exit(run(parser().parse_args()))
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")
    except KeyboardInterrupt:
        sys.exit(130)
