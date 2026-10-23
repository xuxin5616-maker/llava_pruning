"""Rate sweep and reproducible result layout."""

import csv
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .backend import LlavaBackend, validate_image_token_order
from .data import load_samples
from .methods import load_method, resolve_method_config
from .metrics import Accuracy, METRIC_RULE
from .prompts import resolve_prompt
from .rates import format_prune_rate
from .selection import resolve_visualization_ids, visualization_key


BENCHMARK_SAMPLE_FIELDS = ("lat_seconds", "generation_seconds", "generated_tokens",
                           "peak_allocated_bytes", "peak_reserved_bytes")
BENCHMARK_SUMMARY_FIELDS = (
    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_std_ms",
    "mem_peak_allocated_gib", "mem_peak_reserved_gib", "mem_peak_allocated_pct",
    "device_total_gib", "generation_mean_ms", "mean_generated_tokens", "warmup_samples",
)
LAYER_AVERAGE_BUDGET_FIELDS = ("rate_semantics", "pruning_layer", "target_average_prune_rate",
                        "requested_layer_prune_rate", "actual_layer_prune_rate", "actual_average_prune_rate")


def _write_metrics(path, metrics):
    # The rule is shared by every rate and is already recorded in run.json.
    results = {key: value for key, value in metrics.items() if key != "rule"}
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run(*, model_path, input_json, data_root, prompt_version, method_name,
        method_config, roi_mode, save_prune_vis, save_attention_vis,
        output_dir, seed=None, no_sample=True, include_pruning_time=True,
        image_token_order="base_first", benchmark_metrics=False, warmup=10,
        visualize_samples=None):
    if type(include_pruning_time) is not bool:
        raise ValueError("include_pruning_time must be a boolean")
    if type(benchmark_metrics) is not bool:
        raise ValueError("benchmark_metrics must be a boolean")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if benchmark_metrics and (not no_sample or not include_pruning_time
                              or save_prune_vis or save_attention_vis):
        raise ValueError("--benchmark-metrics requires --no-sample, --include-pruning-time "
                         "and no sample visualization (use --no-plots)")
    validate_image_token_order(image_token_order, roi_mode)
    if roi_mode == "base_only" and method_config is None and method_name == "fastv":
        method_config = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
    method_config = resolve_method_config(method_name, method_config)
    method = load_method(method_name, method_config)
    if method.name in {"clse", "attention_predictor"} and any(rate != 0 for rate in method.rates):
        if roi_mode != "anyres_max_9":
            raise ValueError(f"{method.name} nonzero rates require --roi-mode anyres_max_9")
        if not no_sample:
            raise ValueError(f"{method.name} supports greedy decoding only; use --no-sample")
    if getattr(method, "budget_mode", "joint") == "base_anyres" and roi_mode != "anyres_max_9":
        raise ValueError("FastV base_anyres budget requires --roi-mode anyres_max_9")
    if roi_mode == "base_only":
        if tuple(method.rates) != (0,):
            raise ValueError("base_only is an unpruned baseline: prune_rates must be [0]. "
                             "Use --method fastv --method-config configs/baseline.json.")
        if save_prune_vis or save_attention_vis:
            raise ValueError("base_only is results-only; disable sample visualization")
    samples = load_samples(input_json, data_root)
    if not samples:
        raise ValueError("Input JSON has no samples")
    visualization_ids = resolve_visualization_ids(visualize_samples, samples)
    generated_seed = seed is None
    if generated_seed:
        seed = secrets.randbelow(2**31)
    if not 0 <= seed <= 2**32 - len(samples):
        raise ValueError("--seed must keep seed + sample index within NumPy's 32-bit range")
    # Validate all prompts before the expensive model load.
    prompts = [resolve_prompt(prompt_version, item.category, item.text) for item in samples]
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if benchmark_metrics:
        # Reuse the exact reporting/warmup protocol already used by VisionZip.
        # No heavy dependencies or CUDA checks are added to the legacy path.
        import torch
        from benchmark_fastv import (DEFINITIONS, measurement_summary,
                                     validate_single_gpu, warmup_indices)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("--benchmark-metrics requires exactly one visible CUDA GPU; "
                               "set CUDA_VISIBLE_DEVICES=0 (or another single GPU ID)")
    backend = LlavaBackend(model_path, roi_mode=roi_mode, image_token_order=image_token_order)
    resolved_budgets = {}
    if hasattr(method, "resolve_budgets"):
        resolved_budgets = {item["prune_rate"]: item for item in
                            method.resolve_budgets(len(backend.model.get_model().layers))}
    if benchmark_metrics:
        validate_single_gpu(backend.model, torch)
        total_memory = torch.cuda.get_device_properties(0).total_memory
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(method_config).read_text(encoding="utf-8"))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "input_json": str(Path(input_json).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "prompt_version": prompt_version,
        "method": method.name,
        "method_config": config,
        "roi_mode": roi_mode,
        "image_token_order": image_token_order,
        "save_prune_vis": save_prune_vis,
        "save_attention_vis": save_attention_vis,
        "seed": seed,
        "seed_origin": "generated" if generated_seed else "explicit",
        "do_sample": not no_sample,
        "include_pruning_time": include_pruning_time,
        "benchmark_metrics": benchmark_metrics,
        "samples": len(samples),
        "decoding": ("sampling, temperature=0.2, top_p=0.7, max_new_tokens<=512"
                     if not no_sample else "greedy, max_new_tokens<=512 (LLaVA context budget)"),
        "inference": getattr(backend, "inference_config", {}),
        "metric_rule": METRIC_RULE,
    }
    if visualization_ids is not None:
        metadata["visualize_samples"] = sorted(visualization_ids)
    if hasattr(method, "details"):
        metadata["method_details"] = method.details()
    if resolved_budgets:
        metadata["resolved_budgets"] = list(resolved_budgets.values())
    if method.name == "priortr":
        metadata["inference"] = dict(metadata["inference"],
                                     attention_source="independent_causal_qk_prior_and_task",
                                     score_dtype="native QK/aggregation and final scores; FP32 softmax/normalization/log")
    if method.name == "clse":
        metadata["inference"] = dict(metadata["inference"], attention_source="none; pure CLSE spectral evolution",
                                     score_dtype="FP32 scoring branch; native decoder dtype/backend unchanged")
    if method.name == "attention_predictor":
        metadata["inference"] = dict(metadata["inference"], attention_source="trained three-layer side Transformer",
                                     score_dtype="FP32 final QK/softmax; student precision in checkpoint; backbone untouched")
    if benchmark_metrics:
        metadata.update(
            warmup_requested=warmup,
            benchmark_definitions={key: DEFINITIONS[key] for key in
                                   ("lat", "memory", "memory_percent", "generation", "units")},
            benchmark_protocol="One model load; warm up separately before each rate, then evaluate "
                               "ALL input images. Warmup answers/timings excluded. No allocator cache "
                               "clearing between rates; reserved memory may reflect earlier rates. "
                               "Per-sample peaks reset immediately before full-answer generation.",
        )
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not include_pruning_time:
        print("Timing diagnostic: generation_seconds = generation_with_pruning_seconds - pruning_seconds. "
              "Pruning still runs; extra synchronization affects timing. This is not end-to-end speedup.",
              flush=True)
    summary_path = output / "summary.csv"
    summary_fields = ("prune_rate", "complete", "expected_samples", "evaluated_samples",
                      "labeled_samples", "correct", "incorrect", "unparsed", "accuracy",
                      "parsed_samples", "tp", "fp", "tn", "fn", "precision", "recall", "tnr")
    if benchmark_metrics:
        summary_fields += BENCHMARK_SUMMARY_FIELDS
    if resolved_budgets:
        summary_fields += LAYER_AVERAGE_BUDGET_FIELDS
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        csv.writer(summary_file).writerow(summary_fields)
    for rate in method.rates:
        rate_label = format_prune_rate(rate)
        rate_dir = output / f"prune_{rate_label}"
        rate_dir.mkdir()
        records_path = rate_dir / "predictions.jsonl"
        metrics_path = rate_dir / "metrics.json"
        accuracy = Accuracy(expected_samples=len(samples))
        measurements = []
        warmups_done = 0
        sample_budgets = []

        def current_metrics(complete):
            metrics = accuracy.result(rate, complete=complete)
            if benchmark_metrics:
                metrics.update(measurement_summary(measurements, total_memory))
                metrics["warmup_samples"] = warmups_done
            if resolved_budgets:
                metrics.update(resolved_budgets[rate])
                for field in ("actual_layer_prune_rate", "actual_average_prune_rate"):
                    values = [item[field] for item in sample_budgets if item.get(field) is not None]
                    metrics[field] = sum(values) / len(values) if values else None
            return metrics

        if benchmark_metrics:
            for index in warmup_indices(len(samples), warmup):
                backend.generate(
                    samples[index], prompts[index][0], method, rate, roi_mode,
                    capture_visualization=False, capture_attention=False,
                    random_seed=seed + index, do_sample=False,
                    include_pruning_time=True, benchmark_metrics=True,
                )
                warmups_done += 1
                print(f"rate={rate_label} warmup={warmups_done}/{min(warmup, len(samples))}", flush=True)
            print(f"rate={rate_label}: evaluating ALL {len(samples)} images with Lat/Mem measurement", flush=True)
        layer_path = rate_dir / "layer_tokens.csv"
        layer_fields = ("question_id", "method", "prune_rate", "layer", "image_tokens_in",
                        "image_tokens_out", "sequence_tokens_in", "sequence_tokens_out",
                        "removed_after_layer", "cumulative_prune_rate")
        with records_path.open("w", encoding="utf-8") as stream:
            for index, (sample, (prompt, _)) in enumerate(zip(samples, prompts)):
                visualize = (rate in method.visualize_rates and (save_prune_vis or save_attention_vis)
                             and (visualization_ids is None
                                  or visualization_key(sample.sample_id) in visualization_ids))
                result = backend.generate(
                    sample, prompt, method, rate, roi_mode,
                    capture_visualization=visualize,
                    capture_attention=visualize and save_attention_vis,
                    random_seed=seed + index,
                    do_sample=not no_sample,
                    include_pruning_time=include_pruning_time,
                    **({"benchmark_metrics": True} if benchmark_metrics else {}),
                )
                if visualize:
                    method.visualize(
                        result=result, vision_tower=backend.vision_tower,
                        output_dir=rate_dir / "visualizations",
                        sample_id=sample.sample_id,
                        ground_truth=sample.gt,
                        save_prune=save_prune_vis,
                        save_attention=save_attention_vis,
                    )
                record = {
                    "question_id": sample.sample_id,
                    "image": str(sample.image),
                    "origin_path": sample.origin_path,
                    "gt": sample.gt,
                    "answer": result["answer"],
                    "prune_rate": rate,
                    "method": method.name,
                    "generation_seconds": result["generation_seconds"],
                    "max_new_tokens": result.get("max_new_tokens"),
                }
                if not include_pruning_time:
                    record.update({key: result[key] for key in (
                        "generation_with_pruning_seconds", "pruning_seconds")})
                if benchmark_metrics:
                    measurement = {key: result[key] for key in BENCHMARK_SAMPLE_FIELDS}
                    record.update(measurement)
                if resolved_budgets:
                    budget = {key: result["stats"].get(key, resolved_budgets[rate].get(key))
                              for key in LAYER_AVERAGE_BUDGET_FIELDS}
                    record.update(budget)
                    sample_budgets.append(budget)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if result["stats"].get("budget_mode") == "base_anyres":
                    budget_path = rate_dir / "view_budgets.csv"
                    budget_fields = ("question_id", "fastv_layer", "prune_rate", "image_index", "name",
                                     "candidate_tokens", "kept_tokens", "masked_tokens", "actual_prune_rate")
                    header = not budget_path.exists()
                    with budget_path.open("a", encoding="utf-8", newline="") as budget_file:
                        writer = csv.DictWriter(budget_file, fieldnames=budget_fields)
                        if header:
                            writer.writeheader()
                        for batch in result["stats"]["batch"]:
                            for group in batch["groups"]:
                                writer.writerow({"question_id": sample.sample_id, "fastv_layer": method.layer,
                                                 "prune_rate": rate, **group})
                if result["stats"].get("layers"):
                    write_header = not layer_path.exists()
                    with layer_path.open("a", encoding="utf-8", newline="") as layer_file:
                        writer = csv.DictWriter(layer_file, fieldnames=layer_fields)
                        if write_header:
                            writer.writeheader()
                        for layer_row in result["stats"]["layers"]:
                            writer.writerow({"question_id": sample.sample_id, "method": method.name,
                                             "prune_rate": rate, **layer_row})
                accuracy.add(sample.gt, result["answer"])
                if benchmark_metrics:
                    measurements.append(measurement)
                _write_metrics(metrics_path, current_metrics(complete=False))
                progress = f"rate={rate_label} sample={index + 1}/{len(samples)} id={sample.sample_id}"
                if benchmark_metrics:
                    progress += (f" Lat={measurement['lat_seconds'] * 1000:.2f} ms "
                                 f"Mem={measurement['peak_allocated_bytes'] / 1024**3:.3f} GiB")
                print(progress, flush=True)
        metrics = current_metrics(complete=True)
        _write_metrics(metrics_path, metrics)
        with summary_path.open("a", encoding="utf-8", newline="") as summary_file:
            csv.writer(summary_file).writerow(metrics[field] for field in summary_fields)
        score = "N/A" if metrics["accuracy"] is None else f"{metrics['accuracy']:.2%}"
        print(f"rate={rate_label} accuracy={score} "
              f"({metrics['correct']}/{metrics['labeled_samples']}, "
              f"unparsed={metrics['unparsed']})", flush=True)
        detail = " ".join(f"{name}={metrics[name]:.2%}" if metrics[name] is not None else f"{name}=N/A"
                          for name in ("precision", "recall", "tnr"))
        print(f"rate={rate_label} {detail} (parsed labeled samples={metrics['parsed_samples']})", flush=True)
        if resolved_budgets:
            actual = metrics["actual_average_prune_rate"]
            actual_text = f"{actual:.4f}%" if actual is not None else "N/A"
            print(f"rate={rate_label} target-average={metrics['target_average_prune_rate']:.4f}% "
                  f"requested-layer-prune={metrics['requested_layer_prune_rate']:.6f}% "
                  f"actual-average={actual_text}", flush=True)
        if benchmark_metrics:
            print(f"rate={rate_label} Lat(mean)={metrics['lat_mean_ms']:.2f} ms "
                  f"Mem(peak allocated)={metrics['mem_peak_allocated_gib']:.3f} GiB "
                  f"Mem(peak reserved)={metrics['mem_peak_reserved_gib']:.3f} GiB", flush=True)
    return output
