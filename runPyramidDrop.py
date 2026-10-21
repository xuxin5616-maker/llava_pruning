"""PyramidDrop inference adaptation for this project's OneVision/Qwen2 backend."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
SCRIPT_VERSION = "1.1-onevision-pyramiddrop-lat-mem"
SOURCE_COMMIT = "6444f304aee4b7edcb022839fb9e8cd07859704a"
DEFAULT_MODEL = "/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/"
DEFAULT_INPUT = "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"
DEFAULT_DATA = "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/"
SUMMARY_FIELDS = (
    "setting", "prune_rate", "average_keep_rate", "rate_semantics", "keep_ratios", "complete", "expected_samples",
    "evaluated_samples", "labeled_samples", "correct", "incorrect", "unparsed", "accuracy",
    "parsed_samples", "tp", "fp", "tn", "fn", "precision", "recall", "tnr",
    "mean_generation_seconds", "max_peak_memory_allocated_mb", "mean_final_llm_visual_tokens",
    "mean_layer_average_visual_tokens", "mean_actual_average_keep_ratio",
)
BENCHMARK_FIELDS = (
    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_std_ms",
    "peak_allocated_bytes", "peak_reserved_bytes", "mem_peak_allocated_gib",
    "mem_peak_reserved_gib", "mem_peak_allocated_pct", "device_total_gib",
    "generation_mean_ms", "mean_generated_tokens", "warmup_samples",
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected True or False")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def resolve_plan(args, model_depth, *, validate_only=False):
    """Resolve cumulative fractions; never silently reinterpret a sweep's rate."""
    if type(model_depth) is not int or model_depth < 4:
        raise ValueError("PyramidDrop requires a model depth of at least four")
    layers = tuple(args.layers) if args.layers is not None else tuple(model_depth * i // 4 for i in (1, 2, 3))
    if (not layers or any(type(layer) is not int or not 1 <= layer < model_depth for layer in layers)
            or tuple(sorted(set(layers))) != layers):
        raise ValueError("layers must be increasing 1-based boundaries below the last decoder layer")
    rates = args.prune_rates if args.prune_rates is not None else (
        [args.prune_rate] if args.prune_rate is not None else None)
    average_rates = args.average_keep_rates
    if (average_rates is None and rates is None and args.keep_ratios is None
            and args.stage_keep_ratio is None and args.rate_semantics is None):
        average_rates = [100, 50, 70, 90]
    edges = (0, *layers, model_depth)
    widths = tuple(b - a for a, b in zip(edges, edges[1:]))

    def expected_average(ratios):
        return sum(width * ratio for width, ratio in zip(widths, (1.0, *ratios))) / model_depth

    if average_rates is not None:
        if rates is not None or args.keep_ratios is not None or args.stage_keep_ratio is not None or args.rate_semantics is not None:
            raise ValueError("average-keep-rates cannot be combined with prune-rates, keep-ratios, stage-keep-ratio or rate-semantics")
        if (not average_rates or len(set(average_rates)) != len(average_rates)
                or any(not math.isfinite(rate) or not 0 < rate <= 100 for rate in average_rates)):
            raise ValueError("average-keep-rates must be distinct finite percentages in (0,100]")
        if validate_only:
            # Depth-dependent reachability must wait for the actual checkpoint.
            return []
        plan = []
        for rate in average_rates:
            target = rate / 100
            lower_bound = widths[0] / model_depth
            if target <= lower_bound:
                raise ValueError(f"Average retention {rate}% is unreachable with these boundaries: the first full stage alone uses {lower_bound:.2%}; choose earlier layers")
            low, high = 0.0, 1.0
            for _ in range(80):
                retention = (low + high) / 2
                ratios = tuple(retention ** (i + 1) for i in range(len(layers)))
                if expected_average(ratios) < target:
                    low = retention
                else:
                    high = retention
            retention = 1.0 if rate == 100 else (low + high) / 2
            ratios = tuple(retention ** (i + 1) for i in range(len(layers)))
            rate_label = str(int(rate)) if float(rate).is_integer() else repr(float(rate))
            plan.append(dict(name="average_keep_" + rate_label.replace(".", "p"),
                             prune_rate=None, average_keep_rate=rate, layers=layers,
                             keep_ratios=ratios, rate_semantics="layer_average_retention",
                             expected_average_keep_ratio=expected_average(ratios), enabled=rate != 100))
        return plan
    if rates is not None:
        if args.keep_ratios is not None or args.stage_keep_ratio is not None:
            raise ValueError("prune rates cannot be combined with keep-ratios or stage-keep-ratio")
        if args.rate_semantics is None:
            raise ValueError("A prune-rate sweep requires explicit --rate-semantics final_cumulative or per_stage")
        if (not rates or len(set(rates)) != len(rates)
                or any(type(rate) is not int or not 0 <= rate < 100 for rate in rates)):
            raise ValueError("prune rates must be distinct integers from 0 to 99")
        plan = []
        for rate in rates:
            remaining = (100 - rate) / 100
            if args.rate_semantics == "final_cumulative":
                ratios = tuple(remaining ** ((i + 1) / len(layers)) for i in range(len(layers)))
                # Preserve the final requested fraction exactly; avoid power roundoff.
                ratios = (*ratios[:-1], remaining)
            else:
                ratios = tuple(remaining ** (i + 1) for i in range(len(layers)))
            plan.append(dict(name=f"prune_{rate:02d}", prune_rate=rate, average_keep_rate=None, layers=layers,
                             keep_ratios=ratios, rate_semantics=args.rate_semantics,
                             expected_average_keep_ratio=expected_average(ratios), enabled=rate != 0))
        return plan
    if args.rate_semantics is not None:
        raise ValueError("rate-semantics applies only with prune-rate or prune-rates")
    if args.keep_ratios is not None:
        if args.stage_keep_ratio is not None:
            raise ValueError("keep-ratios and stage-keep-ratio cannot be combined")
        ratios = tuple(args.keep_ratios)
        if (len(ratios) != len(layers) or any(not math.isfinite(r) or not 0 < r <= 1 for r in ratios)
                or any(a < b for a, b in zip(ratios, ratios[1:]))):
            raise ValueError("keep-ratios must match layers and be positive non-increasing cumulative fractions")
        name, semantics = "custom_schedule", "explicit_cumulative"
    else:
        retention = 0.5 if args.stage_keep_ratio is None else args.stage_keep_ratio
        if not math.isfinite(retention) or not 0 < retention <= 1:
            raise ValueError("stage-keep-ratio must be finite and in (0,1]")
        ratios = tuple(retention ** (i + 1) for i in range(len(layers)))
        name = "official_lambda_0p5" if retention == 0.5 else "stage_keep_" + str(retention).replace(".", "p")
        semantics = "per_stage_retention"
    return [dict(name=name, prune_rate=None, average_keep_rate=None, layers=layers, keep_ratios=ratios,
                 expected_average_keep_ratio=expected_average(ratios), rate_semantics=semantics, enabled=ratios[-1] < 1)]


def check_source():
    root = PROJECT / "third_party" / "pyramiddrop"
    source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    if source["commit"] != SOURCE_COMMIT:
        raise RuntimeError("PyramidDrop provenance commit differs from this entry point")
    actual = hashlib.sha256((root / "modeling_llama_pdrop.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != source["sha256_lf"]:
        raise RuntimeError("PyramidDrop reference snapshot differs from the pinned official source")
    return source


class PyramidDropMethod:
    name = "pyramiddrop"

    def __init__(self, setting):
        self.setting = setting

    def configure(self, core, prune_rate, *, capture_attention=False, capture_visualization=False):
        core.configure_pyramiddrop(enabled=self.setting["enabled"], layers=self.setting["layers"],
                                  keep_ratios=self.setting["keep_ratios"],
                                  capture_stages=capture_attention or capture_visualization)

    def stats(self, core, prune_rate):
        result = dict(core.get_pyramiddrop_stats())
        result.update(setting=self.setting["name"], rate_semantics=self.setting["rate_semantics"],
                      requested_prune_rate=self.setting["prune_rate"],
                      requested_average_keep_rate=self.setting["average_keep_rate"])
        return result

    def visualization_data(self, core, *, capture_attention=False):
        return {"stages": core.get_pyramiddrop_stages()}


def _cuda_devices(model):
    return sorted({parameter.device.index for parameter in model.parameters() if parameter.device.type == "cuda"})


def _reset_peak_memory(devices):
    import torch
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def _read_peak_memory(devices):
    import torch
    allocated = {str(device): torch.cuda.max_memory_allocated(device) / 1024**2 for device in devices}
    reserved = {str(device): torch.cuda.max_memory_reserved(device) / 1024**2 for device in devices}
    return dict(peak_memory_allocated_mb_by_device=allocated, peak_memory_reserved_mb_by_device=reserved,
                peak_memory_allocated_mb=sum(allocated.values()) if allocated else None)


def run_experiment(args, *, backend_factory=None):
    from llava_pruning.backend import LlavaBackend
    from llava_pruning.data import load_samples
    from llava_pruning.metrics import Accuracy, METRIC_RULE, OPTION
    from llava_pruning.prompts import resolve_prompt

    # Validate schedules without loading a checkpoint. Actual depth is checked again below.
    resolve_plan(args, max([4, args.model_depth, *(args.layers or [0])]) + (1 if args.layers else 0),
                 validate_only=True)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be a positive integer")
    if type(args.warmup) is not int or args.warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if args.benchmark_metrics and (args.save_prune_vis or args.save_attention_vis):
        raise ValueError("benchmark-metrics requires visualization disabled for comparable Lat/Mem")
    if args.benchmark_metrics:
        import torch
        from benchmark_fastv import DEFINITIONS, measurement_summary, validate_single_gpu, warmup_indices
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Benchmark requires exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES")
    source = check_source()
    samples = load_samples(args.input_json, args.data_root)
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("Input JSON has no selected samples")
    if not 0 <= args.seed <= 2**32 - len(samples):
        raise ValueError("seed + sample index must remain within NumPy's 32-bit range")
    prompts = [resolve_prompt(args.prompt_version, sample.category, sample.text)[0] for sample in samples]
    output = Path(args.output_dir or PROJECT / "outputs" / ("pyramiddrop_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    backend = (backend_factory or LlavaBackend)(args.model_path, roi_mode=args.roi_mode,
                                               image_token_order="base_first")
    core = backend.model.get_model()
    plan = resolve_plan(args, len(core.layers))
    total_memory = None
    benchmark_options = {}
    if args.benchmark_metrics:
        validate_single_gpu(backend.model, torch)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        benchmark_options["benchmark_metrics"] = True
    summary_fields = SUMMARY_FIELDS + (BENCHMARK_FIELDS if args.benchmark_metrics else ())
    devices = _cuda_devices(backend.model)
    inference = dict(getattr(backend, "inference_config", {}))
    inference.update(attention_source="next-layer norm/QK + RoPE + GQA; last full prompt token; softmax over all visible keys",
                     ranking_precision="native projection and QK dtype; FP32 softmax cast back before head mean")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_version": SCRIPT_VERSION,
        "method": "pyramiddrop", "source": source, "settings": plan,
        "model_depth": len(core.layers), "model_path": str(Path(args.model_path).expanduser().resolve()),
        "input_json": str(Path(args.input_json).resolve()), "data_root": str(Path(args.data_root).resolve()),
        "samples": len(samples), "limit": args.limit, "roi_mode": args.roi_mode,
        "image_token_order": "base_first", "prompt_version": args.prompt_version,
        "seed": args.seed, "do_sample": False, "warmup_samples_per_setting": min(args.warmup, len(samples)),
        "decoding": "shared backend: greedy, max_new_tokens<=512, DynamicCache, single image and batch 1",
        "budget": "floor(original packed visual span * cumulative ratio), minimum 1; all image_newline entries are candidates",
        "positions": "reindex contiguously after every drop; use each layer's actual cache length during decode",
        "save_prune_vis": args.save_prune_vis, "save_attention_vis": args.save_attention_vis,
        "benchmark_metrics": args.benchmark_metrics,
        "benchmark_definitions": ({key: DEFINITIONS[key] for key in
                                   ("lat", "memory", "memory_percent", "generation", "accuracy", "units")}
                                  if args.benchmark_metrics else None),
        "benchmark_execution": (
            "one model load; settings run sequentially in one process; per-setting evenly spaced warmups; "
            "allocator cache is not cleared between settings; memory reset at resident-input model entry"
            if args.benchmark_metrics else None),
        "inference": inference, "metric_rule": METRIC_RULE,
        "timing": "synchronized generate including vision forward, ranking, token deletion and LLM decode; excludes image preprocessing and file/plot output; no subtraction",
        "memory": (DEFINITIONS["memory"] if args.benchmark_metrics else
                   "per-sample PyTorch peak allocated/reserved in MiB, reset before CPU preprocessing/H2D; "
                   "sum of per-device peaks is not a simultaneous multi-device peak"),
        "comparison": "final cumulative and per-stage rates do not imply matched layer-average tokens or FLOPs across methods",
    })
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=summary_fields).writeheader()
    for setting in plan:
        method = PyramidDropMethod(setting)
        rate = setting["prune_rate"]
        folder = output / setting["name"]
        folder.mkdir()
        accuracy = Accuracy(expected_samples=len(samples))
        latencies, peaks, lengths, layer_means, actual_ratios = [], [], [], [], []
        measurements = []
        warmups = (warmup_indices(len(samples), args.warmup) if args.benchmark_metrics
                   else range(min(args.warmup, len(samples))))
        for index in warmups:
            backend.generate(samples[index], prompts[index], method, rate, args.roi_mode,
                             capture_visualization=False, capture_attention=False,
                             random_seed=args.seed + index, do_sample=False, include_pruning_time=True,
                             **benchmark_options)

        def save_metrics(complete):
            metrics = accuracy.result(rate, complete=complete)
            metrics.pop("rule", None)
            metrics.update(setting=setting["name"], average_keep_rate=setting["average_keep_rate"],
                           rate_semantics=setting["rate_semantics"],
                           keep_ratios=list(setting["keep_ratios"]),
                           mean_generation_seconds=sum(latencies) / len(latencies) if latencies else None,
                           max_peak_memory_allocated_mb=max(peaks) if peaks else None,
                           mean_final_llm_visual_tokens=sum(lengths) / len(lengths) if lengths else None,
                           mean_layer_average_visual_tokens=sum(layer_means) / len(layer_means) if layer_means else None,
                           mean_actual_average_keep_ratio=sum(actual_ratios) / len(actual_ratios) if actual_ratios else None)
            if args.benchmark_metrics:
                metrics.update({key: None for key in BENCHMARK_FIELDS})
                metrics.update(measurement_summary(measurements, total_memory), warmup_samples=len(warmups))
            write_json(folder / "metrics.json", metrics)
            return metrics

        save_metrics(False)
        fields = ("question_id", "image", "gt", "prediction", "answer", "setting", "prune_rate", "average_keep_rate",
                  "generation_seconds", "original_image_tokens", "final_llm_visual_tokens",
                  "layer_average_visual_tokens", "actual_average_keep_ratio", "peak_memory_allocated_mb")
        if args.benchmark_metrics:
            fields += ("lat_seconds", "generated_tokens", "peak_allocated_bytes", "peak_reserved_bytes")
        with (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream, \
                (folder / "classification_results.csv").open("w", encoding="utf-8-sig", newline="") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=fields)
            writer.writeheader()
            for index, (sample, prompt) in enumerate(zip(samples, prompts)):
                capture = setting["enabled"] and (args.save_prune_vis or args.save_attention_vis)
                if not args.benchmark_metrics:
                    _reset_peak_memory(devices)
                result = backend.generate(sample, prompt, method, rate, args.roi_mode,
                                          capture_visualization=capture,
                                          capture_attention=capture and args.save_attention_vis,
                                          random_seed=args.seed + index, do_sample=False, include_pruning_time=True,
                                          **benchmark_options)
                stats = result["stats"]
                rows = stats.get("layers", [])
                layer_average = sum(row["image_tokens_in"] for row in rows) / len(rows) if rows else None
                original_count = stats.get("original_image_tokens")
                actual_average_ratio = layer_average / original_count if layer_average is not None and original_count else None
                if args.benchmark_metrics:
                    memory = {key: result[key] for key in
                              ("lat_seconds", "generated_tokens", "peak_allocated_bytes", "peak_reserved_bytes")}
                    memory["peak_memory_allocated_mb"] = result["peak_allocated_bytes"] / 1024**2
                else:
                    memory = _read_peak_memory(devices)
                answer = result["answer"]
                match = OPTION.match(answer)
                prediction = (1 if match.group(1).upper() == "A" else 0) if match else None
                record = dict(question_id=sample.sample_id, image=str(sample.image), origin_path=sample.origin_path,
                              gt=sample.gt, answer=answer, prediction=prediction,
                              prediction_label={0: "Normal", 1: "Abnormal"}.get(prediction, "Unparsed"),
                              setting=setting["name"], prune_rate=rate, average_keep_rate=setting["average_keep_rate"], method=method.name,
                              generation_seconds=result["generation_seconds"], max_new_tokens=result.get("max_new_tokens"),
                              layer_average_visual_tokens=layer_average, actual_average_keep_ratio=actual_average_ratio,
                              pruning=stats, **memory)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                writer.writerow({key: record.get(key, stats.get(key)) for key in fields})
                csv_stream.flush()
                accuracy.add(sample.gt, answer)
                latencies.append(result["generation_seconds"])
                if args.benchmark_metrics:
                    measurements.append({key: result[key] for key in
                                         ("lat_seconds", "generation_seconds", "generated_tokens",
                                          "peak_allocated_bytes", "peak_reserved_bytes")})
                if memory["peak_memory_allocated_mb"] is not None:
                    peaks.append(memory["peak_memory_allocated_mb"])
                if stats.get("final_llm_visual_tokens") is not None:
                    lengths.append(stats["final_llm_visual_tokens"])
                if layer_average is not None:
                    layer_means.append(layer_average)
                if actual_average_ratio is not None:
                    actual_ratios.append(actual_average_ratio)
                save_metrics(False)
                if capture:
                    from llava_pruning.visualization import prediction_caption, save_vico_visualizations
                    save_vico_visualizations(result["image"], result["crop_metadata"][0], result["stages"],
                                             layer_stats=stats["layers"],
                                             base_grid=int(backend.vision_tower.num_patches_per_side),
                                             patch_size=int(backend.vision_tower.config.patch_size),
                                             output_dir=folder / "visualizations", sample_id=sample.sample_id,
                                             save_prune=args.save_prune_vis, save_attention=args.save_attention_vis,
                                             prediction_label=prediction_caption(sample.gt, answer), method_name="PyramidDrop")
                print(f"setting={setting['name']} sample={index + 1}/{len(samples)} id={sample.sample_id} "
                      f"Pred={record['prediction_label']} LLM_visual={stats.get('final_llm_visual_tokens')}", flush=True)
        metrics = save_metrics(True)
        with (output / "summary.csv").open("a", encoding="utf-8", newline="") as summary:
            row = {key: metrics[key] for key in summary_fields}
            row["keep_ratios"] = json.dumps(row["keep_ratios"])
            csv.DictWriter(summary, fieldnames=summary_fields).writerow(row)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--input-json", default=DEFAULT_INPUT)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--prompt-version", choices=("v0", "v1", "v2", "v3"), default="v0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", help="1-based completed decoder boundaries; default: equal quarters of actual model depth")
    rates = parser.add_mutually_exclusive_group()
    rates.add_argument("--prune-rate", type=int)
    rates.add_argument("--prune-rates", type=int, nargs="+")
    rates.add_argument("--average-keep-rates", type=float, nargs="+",
                       help="Layer-average visual token RETENTION percentages, e.g. 100 50 70 90; default sweep")
    parser.add_argument("--rate-semantics", choices=("final_cumulative", "per_stage"),
                        help="Required with prune-rate(s); final_cumulative=final removed fraction, per_stage=removed at each boundary")
    parser.add_argument("--stage-keep-ratio", type=float, help="Per-stage retention lambda, default .5; cannot combine with prune-rate(s)")
    parser.add_argument("--keep-ratios", type=float, nargs="+", help="Explicit cumulative fractions relative to original packed visual count")
    parser.add_argument("--roi-mode", choices=("anyres_max_9", "randomroi", "randompatch", "ex_base_copy", "anyres_only"), default="anyres_max_9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=0, help="Unrecorded warmup samples per setting; use the same value across compared methods")
    parser.add_argument("--benchmark-metrics", action="store_true",
                        help="Measure first-token Lat and full-answer peak Mem using the shared benchmark protocol; one visible GPU, no visualization")
    parser.add_argument("--save-prune-vis", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--save-attention-vis", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--plan-only", action="store_true", help="Print resolved schedules without model, images, or CUDA")
    parser.add_argument("--model-depth", type=int, default=28, help="Depth for plan-only preview; real runs always read the loaded model depth")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.plan_only:
            print(json.dumps(resolve_plan(args, args.model_depth), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(f"Saved PyramidDrop predictions and metrics: {run_experiment(args)}", flush=True)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
