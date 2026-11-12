"""Compare crop quotas and global AnyRes selection with MyPool alone on full MVTec."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time

import torch

import benchmark_visa_methods as common
import runMyPruningPoolSpatial as pool
import visualize_mvtec_baseline as baseline
from llava_pruning.metrics import Accuracy
from llava_pruning.pool_compact_visualization import render_compact
from llava_pruning.pool_distill import PoolDistillAdapter
from llava_pruning.pool_distill_visualization import capture_baseline_views
from llava_pruning.pool_lean import encode, pack_device_rows
from visualize_base_pruning import save_png


MODES = {"crop_quota": "MyPool crop quotas", "global_anyres": "MyPool global AnyRes"}
RATES = (50, 70)


def compress_global_anyres(adapter, features, scores, metric):
    """Apply VisionZip to Base and to a single crop-major AnyRes sequence."""
    view_size = features.shape[1]
    base_keep = adapter.view_plan["base_keep"]
    anyres_keep = adapter.view_plan["anyres_keep"]
    base_dominant = base_keep if base_keep == view_size else round(base_keep * pool.DOMINANT_RATIO)
    anyres_total = features.shape[0] * view_size - view_size
    anyres_dominant = anyres_keep if anyres_keep == anyres_total else round(anyres_keep * pool.DOMINANT_RATIO)
    base, (base_decision,) = pool.compress_tokens(
        features[:1], scores[:1], metric[:1], base_dominant, base_keep - base_dominant,
        audit=False, source_provenance=False,
    )
    anyres, (global_decision,) = pool.compress_tokens(
        features[1:].flatten(0, 1).unsqueeze(0), scores[1:].reshape(1, -1),
        metric[1:].flatten(0, 1).unsqueeze(0), anyres_dominant, anyres_keep - anyres_dominant,
        audit=False, source_provenance=False,
    )
    dominant = global_decision["dominant_indices"]
    contextual = global_decision["contextual_indices"]
    anchors = torch.cat((dominant, contextual))
    views, decisions = [base[0]], [base_decision]
    for crop in range(features.shape[0] - 1):
        start, end = crop * view_size, (crop + 1) * view_size
        views.append(anyres[0][(anchors >= start) & (anchors < end)])
        decisions.append(dict(
            dominant_indices=dominant[(dominant >= start) & (dominant < end)] - start,
            contextual_indices=contextual[(contextual >= start) & (contextual < end)] - start,
        ))
    keeps = [len(view) for view in views]
    if keeps[0] != base_keep or sum(keeps[1:]) != anyres_keep:
        raise RuntimeError("Global selection changed the Base or AnyRes budget")
    core = adapter.core
    projected = core.mm_projector(torch.cat(views))
    if projected.ndim != 2 or len(projected) != sum(keeps) or not bool(torch.isfinite(projected).all()):
        raise ValueError("Global AnyRes projector produced invalid features")
    packed, roles = pack_device_rows(projected.split(keeps), decisions, adapter.geometry.grid,
                                     core.image_newline, core.config.mm_patch_merge_type,
                                     prepared=adapter.prepared)
    return packed, decisions, roles


class PreLlmMethod:
    def __init__(self, core, mode, rate, *, temperature=pool.DEFAULT_ALLOCATION_TEMPERATURE):
        self.mode, self.rate = mode, rate
        settings = pool.MethodSettings(prune_rate=0, redundancy_layers=(26,), visionzip_layers=(26,),
                                       allocation_temperature=temperature, spatial_strength=0.)
        self.adapter = PreLlmAdapter(core, settings, rate=rate, share=0, mode="anyres_self", lean=True)
        self.adapter.compress_global = compress_global_anyres if mode == "global_anyres" else None

    def configure(self, core, rate, *, capture_attention, capture_visualization):
        if rate != self.rate or capture_attention or capture_visualization:
            raise ValueError("Use the pre-LLM-only diagnostic generation path")
        self.adapter.configure(self.adapter.settings.prune_rate, capture_visualization=False)

    def inference_context(self, model):
        return pool.visionzip_generation(model, self.adapter)

    def stats(self, core, rate):
        return self.adapter.plan


class PreLlmAdapter(PoolDistillAdapter):
    def encode_images(self, images):
        return encode(self, images, compress_global=self.compress_global)


def setup(args):
    output = args.output_dir.resolve()
    manifest = output / "run.json"
    if args.resume:
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        if spec.get("experiment") != "mypool_pre_llm_only":
            raise ValueError("Output directory does not contain this pre-LLM experiment")
        for path, digest in ((spec["input_json"], spec["input_sha256"]),
                             (Path(spec["model_path"]) / "config.json", spec["model_config_sha256"])):
            if common.digest(path) != digest:
                raise ValueError(f"Saved input or model configuration changed: {path}")
        return output, spec
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory or --resume: {output}")
    if args.gpu < 0 or args.panel_height < 1:
        raise ValueError("Expected a nonnegative GPU ID and a positive panel height")
    model = Path(args.model_path).resolve()
    config = model / "config.json"
    config_sha = common.digest(config)
    rows = baseline.freeze(args)
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "input_frozen.jsonl"
    frozen.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    spec = dict(experiment="mypool_pre_llm_only", model_path=str(model), model_config_sha256=config_sha,
                input_json=str(frozen), input_sha256=common.digest(frozen), data_root=str(Path(args.data_root).resolve()),
                sample_count=len(rows), gpu=args.gpu, seed=args.seed, panel_height=args.panel_height,
                rates=list(RATES), modes=list(MODES), prompt_version=args.prompt_version,
                budget="Layer-average percent of the native packed visual sequence; reserve structural newlines",
                global_anyres="Crop-major global dominant top-k and global contextual Key merge; Base has a separate budget")
    common.save_json(manifest, spec)
    return output, spec


def run_trial(backend, samples, spec, folder, mode, rate):
    folder.mkdir(parents=True, exist_ok=True)
    pictures = folder / "visualizations"
    pictures.mkdir(exist_ok=True)
    sample_indices = spec.get("sample_indices")
    rows = baseline.restored(folder, samples, mode, sample_indices=sample_indices)
    start = len(rows)
    started = time.perf_counter()
    temperature = spec.get("allocation_temperature", pool.DEFAULT_ALLOCATION_TEMPERATURE)
    method = PreLlmMethod(backend.model.get_model(), mode, rate, temperature=temperature)
    label = MODES[mode]
    if "allocation_temperature" in spec:
        label += f" | tau={temperature:g}"
    with (folder / "predictions.jsonl").open("a", encoding="utf-8") as stream:
        for index in range(start, len(samples)):
            sample = samples[index]
            source_index = index if sample_indices is None else sample_indices[index]
            with capture_baseline_views(method) as trace:
                result = backend.generate(
                    sample, sample.source["prompt"], method, rate, "anyres_max_9",
                    capture_visualization=False, capture_attention=False, random_seed=spec["seed"] + source_index,
                    do_sample=False, include_pruning_time=True, benchmark_metrics=False,
                )
            picture, details = render_compact(method, trace, backend.processor, result, sample, rate,
                                              spec["panel_height"], pre_llm_label=label)
            name = f"visualizations/sample_{source_index:09d}.png"
            with picture:
                save_png(picture, folder / name)
            row = dict(index=index, question_id=sample.sample_id, gt=sample.gt, answer=result["answer"],
                       output=name, selection_mode=mode, **details)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            if index == start or (index + 1) % 25 == 0 or index + 1 == len(samples):
                eta = (time.perf_counter() - started) * (len(samples) - index - 1) / (index + 1 - start)
                print(f"{mode} prune={rate}% {index + 1}/{len(samples)} ETA={eta:.1f}s", flush=True)
    accuracy = Accuracy(len(samples))
    for row in rows:
        accuracy.add(row["gt"], row["answer"])
    metrics = accuracy.result(rate, complete=True)
    metrics.update(selection_mode=mode, gpu=spec["gpu"],
                   actual_average_prune_rate=mean(100 * (1 - row["pre_llm_tokens"] / row["baseline_packed_tokens"])
                                                   for row in rows))
    common.save_json(folder / "metrics.json", metrics)
    return metrics


def main():
    defaults = baseline.parser().parse_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--input-json", default=defaults.input_json)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--prompt-version", default=defaults.prompt_version)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--panel-height", type=int, default=defaults.panel_height)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output, spec = setup(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    samples = baseline.samples_for(spec)
    from llava_pruning.backend import LlavaBackend
    backend = LlavaBackend(spec["model_path"], "anyres_max_9", image_token_order="base_first")
    common.validate_single_gpu(backend.model, torch)
    device = torch.cuda.get_device_properties(0)
    summary = []
    with torch.inference_mode():
        for mode in spec["modes"]:
            for rate in spec["rates"]:
                folder = output / mode / f"prune_{rate}"
                folder.mkdir(parents=True, exist_ok=True)
                common.verify_runtime(folder, backend.inference_config, device.name, device.total_memory)
                summary.append(run_trial(backend, samples, spec, folder, mode, rate))
                common.save_csv(output / "summary.csv", summary, tuple(summary[0]))
    print(f"Complete | {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
