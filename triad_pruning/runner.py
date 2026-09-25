"""Rate sweep and reproducible result layout."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .backend import TriadBackend
from .data import load_samples
from .methods import load_method
from .prompts import resolve_prompt


def run(*, model_path, input_json, data_root, prompt_version, method_name,
        method_config, roi_mode, save_prune_vis, save_attention_vis,
        output_dir, seed):
    samples = load_samples(input_json, data_root)
    if not samples:
        raise ValueError("Input JSON has no samples")
    method = load_method(method_name, method_config)
    # Validate all prompts before the expensive model load.
    prompts = [resolve_prompt(prompt_version, item.category, item.text) for item in samples]
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    backend = TriadBackend(model_path)
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
        "samples": len(samples),
        "decoding": "greedy, max_new_tokens=256",
    }
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for rate in method.rates:
        rate_dir = output / f"prune_{rate:02d}"
        rate_dir.mkdir()
        records_path = rate_dir / "predictions.jsonl"
        with records_path.open("w", encoding="utf-8") as stream:
            for index, (sample, (prompt, prompt_source)) in enumerate(zip(samples, prompts)):
                visualize = rate in method.visualize_rates and (save_prune_vis or save_attention_vis)
                result = backend.generate(
                    sample, prompt, method, rate, roi_mode,
                    capture_visualization=visualize,
                    capture_attention=visualize and save_attention_vis,
                    random_seed=seed + index,
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
                    "visualizations": paths,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"rate={rate:02d} sample={index + 1}/{len(samples)} id={sample.sample_id}", flush=True)
    return output
