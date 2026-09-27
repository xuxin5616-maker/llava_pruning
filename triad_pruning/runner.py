"""Rate sweep and reproducible result layout."""

import csv
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .backend import TriadBackend
from .data import load_samples
from .methods import load_method
from .metrics import Accuracy, METRIC_RULE
from .prompts import resolve_prompt


def run(*, model_path, input_json, data_root, prompt_version, method_name,
        method_config, roi_mode, save_prune_vis, save_attention_vis,
        output_dir, seed=None, no_sample=True):
    samples = load_samples(input_json, data_root)
    if not samples:
        raise ValueError("Input JSON has no samples")
    generated_seed = seed is None
    if generated_seed:
        seed = secrets.randbelow(2**31)
    if not 0 <= seed <= 2**32 - len(samples):
        raise ValueError("--seed must keep seed + sample index within NumPy's 32-bit range")
    method = load_method(method_name, method_config)
    # Validate all prompts before the expensive model load.
    prompts = [resolve_prompt(prompt_version, item.category, item.text) for item in samples]
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    backend = TriadBackend(model_path, roi_mode=roi_mode)
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
        "save_prune_vis": save_prune_vis,
        "save_attention_vis": save_attention_vis,
        "seed": seed,
        "seed_origin": "generated" if generated_seed else "explicit",
        "do_sample": not no_sample,
        "samples": len(samples),
        "decoding": ("sampling, temperature=0.2, top_p=0.7, max_new_tokens<=512"
                     if not no_sample else "greedy, max_new_tokens<=512 (Triad context budget)"),
        "inference": getattr(backend, "inference_config", {}),
        "metric_rule": METRIC_RULE,
    }
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output / "summary.csv"
    summary_fields = ("prune_rate", "complete", "expected_samples", "evaluated_samples",
                      "labeled_samples", "correct", "incorrect", "unparsed", "accuracy",
                      "parsed_samples", "tp", "fp", "tn", "fn", "precision", "recall", "tnr")
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        csv.writer(summary_file).writerow(summary_fields)
    for rate in method.rates:
        rate_dir = output / f"prune_{rate:02d}"
        rate_dir.mkdir()
        records_path = rate_dir / "predictions.jsonl"
        metrics_path = rate_dir / "metrics.json"
        accuracy = Accuracy(expected_samples=len(samples))
        with records_path.open("w", encoding="utf-8") as stream:
            for index, (sample, (prompt, prompt_source)) in enumerate(zip(samples, prompts)):
                visualize = rate in method.visualize_rates and (save_prune_vis or save_attention_vis)
                result = backend.generate(
                    sample, prompt, method, rate, roi_mode,
                    capture_visualization=visualize,
                    capture_attention=visualize and save_attention_vis,
                    random_seed=seed + index,
                    do_sample=not no_sample,
                )
                paths = None
                if visualize:
                    paths = method.visualize(
                        result=result, vision_tower=backend.vision_tower,
                        output_dir=rate_dir / "visualizations",
                        sample_id=sample.sample_id,
                        save_prune=save_prune_vis,
                        save_attention=save_attention_vis,
                    )
                record = {
                    "question_id": sample.sample_id,
                    "image": str(sample.image),
                    "origin_path": sample.origin_path,
                    "mask": str(sample.mask) if sample.mask else None,
                    "gt": sample.gt,
                    "prompt": prompt,
                    "prompt_source": prompt_source,
                    "answer": result["answer"],
                    "prune_rate": rate,
                    "roi_mode": roi_mode,
                    "roi_source": result["roi_source"],
                    "roi_boxes": result["roi_boxes"],
                    "generation_seconds": result["generation_seconds"],
                    "pruning_stats": result["stats"],
                    "input_token_ids": result.get("input_token_ids"),
                    "generated_token_ids": result.get("generated_token_ids"),
                    "max_new_tokens": result.get("max_new_tokens"),
                    "visualizations": paths,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                accuracy.add(sample.gt, result["answer"])
                metrics_path.write_text(
                    json.dumps(accuracy.result(rate, complete=False), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"rate={rate:02d} sample={index + 1}/{len(samples)} id={sample.sample_id}", flush=True)
        metrics = accuracy.result(rate, complete=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        with summary_path.open("a", encoding="utf-8", newline="") as summary_file:
            csv.writer(summary_file).writerow(metrics[field] for field in summary_fields)
        score = "N/A" if metrics["accuracy"] is None else f"{metrics['accuracy']:.2%}"
        print(f"rate={rate:02d} accuracy={score} "
              f"({metrics['correct']}/{metrics['labeled_samples']}, "
              f"unparsed={metrics['unparsed']})", flush=True)
        detail = " ".join(f"{name}={metrics[name]:.2%}" if metrics[name] is not None else f"{name}=N/A"
                          for name in ("precision", "recall", "tnr"))
        print(f"rate={rate:02d} {detail} (parsed labeled samples={metrics['parsed_samples']})", flush=True)
    return output
