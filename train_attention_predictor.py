"""Distill attention into independently copied Qwen2 blocks 3/4/5; no GT loss."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

from llava_pruning.predictor_data import load_visa_distillation, split_samples
from llava_pruning.prompts import GENERIC_PROMPT_V0
from llava_pruning.training_progress import TrainingETA, format_duration


def build_parser():
    parser = argparse.ArgumentParser(description="Frozen LLaVA H2 -> three-layer side attention, supervised by block22")
    parser.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    parser.add_argument("--data-root", default="/home/yz/xxy/data/datasets/VisA")
    parser.add_argument("--visa-split-csv", default=None, help="FULL CSV, not test-only list; defaults to data-root/split_csv/1cls.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("output/attention_predictor_copied_train_01"))
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Optional teacher H2 cache. Can consume hundreds of GB; omitted = recompute teacher each epoch")
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Batch size is 1; effective images per optimizer step")
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
                        help="Disable training recomputation of the first two copied blocks (uses more VRAM)")
    parser.add_argument("--student-precision", choices=("bfloat16", "float32"), default="bfloat16",
                        help="Affects ONLY student autocast; copied FlashAttention2 blocks require bfloat16. Backbone stays FP16")
    parser.add_argument("--teacher-layer", type=int, default=22, help="1-based teacher attention block; default22, student source fixed block2")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Print progress, phase ETA and total ETA every N images (first/last always printed)")
    return parser


def main():
    args = build_parser().parse_args()
    if args.epochs < 1 or args.gradient_accumulation < 1 or args.log_every < 1:
        raise ValueError("epochs, gradient-accumulation and log-every must be positive")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
            or not 0 <= args.seed < 2**32):
        raise ValueError("Invalid optimizer settings or seed")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output must be empty: {output}")
    samples, data_info = load_visa_distillation(args.data_root, args.visa_split_csv)
    train, validation = split_samples(samples, args.validation_fraction, args.seed)
    import numpy as np
    import torch
    from benchmark_fastv import validate_single_gpu
    from llava_pruning.backend import LlavaBackend
    from llava_pruning.predictor_training import (TeacherExamples, architecture_signature, backbone_file_identity,
                                                  epoch_pass, save_student, vendor_imports)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one visible GPU: CUDA_VISIBLE_DEVICES=5 python train_attention_predictor.py ...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    backend = LlavaBackend(args.model_path, roi_mode="anyres_max_9", image_token_order="base_first")
    backend.model.requires_grad_(False).eval()
    validate_single_gpu(backend.model, torch)
    core = backend.model.get_model()
    if not 2 < args.teacher_layer <= len(core.layers):
        raise ValueError("Teacher layer must be after block2 and within the model depth")
    vendor_imports()  # Add the project's pinned LLaVA to sys.path before vendor imports.
    device = core.embed_tokens.weight.device
    from llava.model.language_model.attention_predictor import CopiedAttentionPredictor, student_autocast
    if core.config._attn_implementation == "flash_attention_2" and args.student_precision != "bfloat16":
        raise ValueError("Copied FlashAttention2 blocks need --student-precision bfloat16; main model stays FP16")
    with student_autocast(device, args.student_precision):
        pass
    student = CopiedAttentionPredictor.from_backbone(core, gradient_checkpointing=args.gradient_checkpointing).to(device)
    architecture = student.config
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False)
    signature = architecture_signature(core)
    file_identity = backbone_file_identity(args.model_path, getattr(core.config, "mm_vision_tower", None))
    identity = dict(backbone_files=file_identity, backbone_signature=signature,
                    inference=backend.inference_config, image_token_order="base_first")
    examples = TeacherExamples(backend, teacher_layer=args.teacher_layer, cache_dir=args.cache_dir, identity=identity)
    output.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    for name, split in (("train", train), ("validation", validation)):
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for sample in split:
                stream.write(json.dumps(dict(question_id=sample.sample_id, image=sample.origin_path,
                                             category=sample.category, text=GENERIC_PROMPT_V0), ensure_ascii=False) + "\n")
        split_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(version=student.checkpoint_version, configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    source=data_info, training_samples=len(train), validation_samples=len(validation), split_sha256=split_hashes,
                    prompt=GENERIC_PROMPT_V0, architecture=asdict(architecture), backbone_signature=signature,
                    inference=backend.inference_config, objective="KL(teacher last-prompt mean-head spatial attention || side block3 attention)",
                    teacher_layer=args.teacher_layer, student_source_layer=2, backbone_frozen=True,
                    original_visa_test_is_training_source=True, final_test_dataset="independent MVTec",
                    cache_policy="optional H2 + target; keys include prompt/layers/backbone+image file identity and inference config",
                    initialization="independent pretrained copies of original Qwen2 decoder blocks 3, 4, 5; no LM head",
                    frozen_student_parts="block5 V/O/MLP/post-attention norm: downstream of the attention loss, never executed",
                    student_parameters=sum(p.numel() for p in student.parameters()),
                    student_trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad))
    (output / "train_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # Relative checkpoint makes this config portable when the training folder moves.
    method_config = dict(checkpoint="best.pt", layer=2, rate_semantics="layer_average",
                         prune_rates=[50, 70, 90], visualize_rates=[50, 70, 90])
    (output / "method.json").write_text(json.dumps(method_config, indent=2), encoding="utf-8")
    print(f"VisA ALL-CSV pool: {len(samples)} images -> train={len(train)}, validation={len(validation)}; no GT/masks", flush=True)
    print(f"Frozen teacher block{args.teacher_layer}; H2 -> copied native blocks 3/4/5, "
          f"width={architecture.input_size}, attention={architecture.attention_implementation}; "
          f"student={metadata['student_parameters']:,} parameters "
          f"(trainable={metadata['student_trainable_parameters']:,}); "
          f"gradient_checkpointing={args.gradient_checkpointing}", flush=True)
    print("Cache disabled: teacher is recomputed each epoch." if not args.cache_dir else
          f"Teacher cache: {args.cache_dir}. H2 is large; monitor disk space (potentially hundreds of GB).", flush=True)
    print("ETA: estimating after the first image; updates use recent train/validation speeds. "
          "Approximate, especially during cache warm-up; future checkpoint-save overhead is not estimated.", flush=True)
    fieldnames = ["epoch", "split", "samples", "seconds", "kl"]
    fieldnames += [f"{metric}_prune{rate}" for rate in (50, 70, 90) for metric in ("topk_overlap", "teacher_mass")]
    best = float("inf")
    eta = TrainingETA(args.epochs, {"train": len(train), "validation": len(validation)})
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            order = list(train)
            random.Random(args.seed + epoch).shuffle(order)
            for name, split, training in (("train", order, True), ("validation", validation, False)):
                def progress(index, total, sample_id, metrics):
                    estimate = eta.update(index)
                    if index == 1 or index % args.log_every == 0 or index == total:
                        approximate = " (provisional)" if estimate["provisional"] else ""
                        print(f"epoch={epoch}/{args.epochs} {name} {index}/{total} "
                              f"({estimate['phase_percent']:.1f}%, overall={estimate['overall_percent']:.1f}%) "
                              f"id={sample_id} loss={metrics['kl']:.6f}(KL) "
                              f"elapsed={format_duration(estimate['elapsed'])} "
                              f"phase_ETA={format_duration(estimate['phase_eta'])} "
                              f"total_ETA={format_duration(estimate['total_eta'])}{approximate}", flush=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                eta.begin(epoch, name)
                metrics = epoch_pass(student, examples, split, device=device, precision=args.student_precision,
                                     optimizer=optimizer if training else None, accumulation=args.gradient_accumulation,
                                     max_grad_norm=args.max_grad_norm, progress=progress)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                writer.writerow(dict(epoch=epoch, split=name, samples=len(split), seconds=seconds, **metrics))
                stream.flush()
                print(f"epoch={epoch} {name}: mean_loss={metrics['kl']:.6f}(KL), seconds={seconds:.2f}", flush=True)
                if not training:
                    options = dict(student_precision=args.student_precision, backbone_depth=len(core.layers),
                                   teacher_layer=args.teacher_layer, signature=signature, provenance=dict(
                                       source_sha256=data_info["source_sha256"], split_sha256=split_hashes,
                                       backbone_path=str(Path(args.model_path).resolve()), prompt=GENERIC_PROMPT_V0,
                                       image_token_order="base_first"), epoch=epoch, validation_kl=metrics["kl"])
                    save_student(output / "last.pt", student, **options)
                    if metrics["kl"] < best:
                        best = metrics["kl"]
                        save_student(output / "best.pt", student, **options)
                    (output / "training_status.json").write_text(json.dumps(dict(completed_epochs=epoch, best_validation_kl=best,
                                                                                complete=epoch == args.epochs), indent=2), encoding="utf-8")
    print(f"Done. Best validation loss={best:.6f}(KL); evaluate MVTec with --method attention_predictor --method-config {output / 'method.json'}", flush=True)


if __name__ == "__main__":
    main()
