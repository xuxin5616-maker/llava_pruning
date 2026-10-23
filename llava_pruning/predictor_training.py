"""Frozen teacher extraction and independent side-student optimization; no answer labels."""

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from .backend import VENDOR
from .prompts import GENERIC_PROMPT_V0


def vendor_imports():
    import sys
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from llava.model.language_model.attention_predictor import AttentionPredictor, PredictorConfig
    return AttentionPredictor, PredictorConfig


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".predictor-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def architecture_signature(core):
    from llava.model.language_model.attention_predictor import backbone_signature
    return backbone_signature(core.config)


@torch.no_grad()
def extract_teacher(core, packed, span, layout, *, teacher_layer=22):
    """Run native blocks 1..21; read only Q/K of block22, not its output.

    H2 is the RAW second-block output, not final RMSNorm or student features.
    Native decoder attention outputs are never requested and no KV is stored.
    """
    from llava.model.language_model.fastv_attention import last_prompt_attention
    if core.training or any(p.requires_grad for p in core.parameters()):
        raise ValueError("Teacher must be frozen and in eval mode")
    if type(teacher_layer) is not int or not 2 < teacher_layer <= len(core.layers):
        raise ValueError("Teacher layer must be after block2 and within actual model depth")
    if getattr(core.config, "use_sliding_window", False):
        raise ValueError("Teacher extraction currently requires non-sliding native attention")
    if packed.ndim != 3 or packed.shape[0] != 1:
        raise ValueError("Teacher extraction requires batch one")
    start, end = span
    if not 0 <= start < end < packed.shape[1] or end - start != layout["packed_tokens"]:
        raise ValueError("Teacher requires a complete, untruncated image and prompt suffix")
    relative = [i for group in layout["grids"] for i in group["indices"]]
    spatial = torch.tensor(relative, device=packed.device) + start
    ids = torch.arange(packed.shape[1], device=packed.device)[None]
    hidden, h2 = packed, None
    positions = ids[0]
    for index, layer in enumerate(core.layers):
        rotary = core.rotary_emb(hidden, ids)
        if index + 1 == teacher_layer:
            attention = last_prompt_attention(layer, hidden, rotary)[0, spatial]
            if not torch.isfinite(attention).all() or float(attention.sum()) <= 0:
                raise FloatingPointError("Invalid teacher visual attention distribution")
            return dict(hidden=h2.detach(), position_ids=ids.detach(), spatial_indices=spatial.detach(),
                        target=(attention / attention.sum()).detach())
        causal = core._pyramiddrop_causal_mask(hidden, 0, core.config._attn_implementation)
        hidden = layer(hidden, attention_mask=causal, position_ids=ids, past_key_value=None,
                       output_attentions=False, use_cache=False, cache_position=positions,
                       position_embeddings=rotary)[0]
        if index == 1:
            h2 = hidden.clone().detach()
    raise RuntimeError("Teacher attention was not captured")


@torch.no_grad()
def prepare_teacher_example(backend, sample, teacher_layer=22):
    from PIL import Image
    from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX)
    from llava.mm_utils import process_images, tokenizer_image_token
    core = backend.model.get_model()
    core.configure_clse(enabled=False, keep_ratio=1.)
    with Image.open(sample.image) as opened:
        image = opened.convert("RGB")
    pixels, _ = process_images([image], backend.processor, backend.model.config, return_pro_data=True)
    if isinstance(pixels, list):
        pixels = [item.to(backend.model.device, dtype=backend.model.dtype) for item in pixels]
    else:
        pixels = pixels.to(backend.model.device, dtype=backend.model.dtype)
    message = DEFAULT_IMAGE_TOKEN + "\n" + GENERIC_PROMPT_V0
    text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n")
    if getattr(backend.model.config, "mm_use_im_start_end", False):
        text = text.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    tokens = tokenizer_image_token(text, backend.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None].to(backend.model.device)
    previous = getattr(core, "_attention_teacher_capture", False)
    core._attention_teacher_capture = True
    try:
        packed = backend.model.prepare_inputs_labels_for_multimodal(
            tokens, None, None, None, None, pixels, image_sizes=[image.size])[4]
        if len(core._clse_spans) != 1 or len(core._clse_spans[0]) != 1 or core._clse_layout is None:
            raise ValueError("Missing actual teacher image geometry")
        return extract_teacher(core, packed, core._clse_spans[0][0], core._clse_layout, teacher_layer=teacher_layer)
    finally:
        core._attention_teacher_capture = previous


def backbone_file_identity(model_path, vision_path=None):
    """Cache identity uses resolved paths, sizes/mtimes, and config/tokenizer hashes.

    This is a cache invalidation fingerprint, NOT a cryptographic weight audit.
    No tens-of-GB weight read is introduced just to fingerprint a run.
    """
    result = []
    roots = [Path(model_path).resolve()]
    if vision_path and Path(str(vision_path)).is_dir():
        roots.append(Path(vision_path).resolve())
    for root in roots:
        files = []
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}:
                stat = path.stat()
                row = [path.name, stat.st_size, stat.st_mtime_ns]
                if path.suffix in {".json", ".txt", ".model"}:
                    row.append(hashlib.sha256(path.read_bytes()).hexdigest())
                files.append(row)
        result.append(dict(root=str(root), files=files))
    return result


class TeacherExamples:
    def __init__(self, backend, *, teacher_layer=22, cache_dir=None, identity=None):
        self.backend, self.teacher_layer = backend, teacher_layer
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.identity = identity
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, sample):
        key, path = None, None
        if self.cache_dir:
            stat = sample.image.stat()
            source = dict(version=1, scoring="last_prompt_mean_heads_spatial_l1", student_layer=2,
                          teacher_layer=self.teacher_layer, prompt=GENERIC_PROMPT_V0,
                          identity=self.identity, image=[str(sample.image.resolve()), stat.st_size, stat.st_mtime_ns])
            key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
            path = self.cache_dir / f"{key}.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("key") != key:
                    raise ValueError(f"Teacher cache identity mismatch: {path}")
                return payload["example"]
        example = prepare_teacher_example(self.backend, sample, self.teacher_layer)
        if path is not None:
            cpu = {key: value.detach().cpu() for key, value in example.items()}
            atomic_save(dict(key=key, example=cpu), path)
            return cpu
        return example


def distillation_loss(log_prob, target):
    target = target.detach().float()
    if (log_prob.shape != target.shape or log_prob.ndim != 1 or not torch.isfinite(log_prob).all()
            or not torch.isfinite(target).all() or bool((target < 0).any())
            or not torch.isclose(target.sum(), target.new_tensor(1.), atol=1e-5, rtol=1e-5)):
        raise ValueError("Distillation requires finite, normalized spatial distributions")
    return F.kl_div(log_prob.float(), target, reduction="sum")


def overlap_metrics(log_prob, target):
    result = {}
    for prune in (50, 70, 90):
        # Diagnostic spatial top-k, not the newline-reserved layer-average budget.
        k = max(1, math.ceil(len(target) * (100 - prune) / 100))
        chosen = log_prob.topk(k).indices
        teacher = target.topk(k).indices
        result[f"topk_overlap_prune{prune}"] = float(torch.isin(chosen, teacher).float().mean())
        result[f"teacher_mass_prune{prune}"] = float(target[chosen].sum())
    return result


def epoch_pass(student, examples, samples, *, device, precision, optimizer=None,
               accumulation=1, max_grad_norm=1., progress=None, scheduler=None):
    from llava.model.language_model.attention_predictor import student_autocast
    training = optimizer is not None
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("Gradient accumulation must be a positive integer")
    if scheduler is not None and (not training or scheduler.optimizer is not optimizer):
        raise ValueError("Training scheduler must belong to the supplied optimizer; no validation stepping")
    student.train(training)
    totals, pending, updates = {}, 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for index, sample in enumerate(samples):
        # Extract outside enable_grad: native teacher never builds a graph.
        example = {key: value.to(device) for key, value in examples(sample).items()}
        with torch.set_grad_enabled(training), student_autocast(device, precision):
            predicted = student(example["hidden"].detach(), example["position_ids"], example["spatial_indices"])
            loss = distillation_loss(predicted, example["target"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite distillation loss for {sample.sample_id}")
        metrics = dict(kl=float(loss.detach()), **overlap_metrics(predicted.detach(), example["target"]))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value
        # Snapshot the LR actually used by this accumulation group, not the
        # scheduler's NEXT LR after a possible optimizer update below.
        lr_used = optimizer.param_groups[0]["lr"] if training else None
        if training:
            (loss / accumulation).backward()
            pending += 1
            if pending == accumulation or index + 1 == len(samples):
                if pending < accumulation:
                    for parameter in student.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(accumulation / pending)
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite student gradient; optimizer step refused")
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                updates += 1
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        if progress is not None:
            progress(index + 1, len(samples), sample.sample_id, dict(metrics, lr_used=lr_used))
        del example, predicted, loss
    if not samples:
        raise ValueError("Training/validation split must not be empty")
    return dict({key: value / len(samples) for key, value in totals.items()}, optimizer_steps=updates)


def save_student(path, student, *, student_precision, backbone_depth, teacher_layer=22,
                 signature=None, provenance=None, epoch=0, validation_kl=None):
    atomic_save(dict(version=student.checkpoint_version, architecture=asdict(student.config),
                     state_dict={key: value.detach().cpu() for key, value in student.state_dict().items()},
                     student_layer=2, teacher_layer=teacher_layer, backbone_depth=backbone_depth,
                     student_precision=student_precision, backbone_signature=signature,
                     epoch=epoch, validation_kl=validation_kl, provenance=provenance or {}), path)
