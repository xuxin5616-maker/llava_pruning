"""Fixed LLaVA OneVision/Qwen2 backend; no model-family selection in the CLI."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Sample
from .methods import PruningMethod
from .roi import choose_roi, load_mask


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "llava"


def validate_image_token_order(order, roi_mode, merge_type=None):
    if order not in {"base_first", "anyres_first"}:
        raise ValueError(f"Unknown image token order: {order}")
    if order == "anyres_first":
        if roi_mode != "anyres_max_9":
            raise ValueError("anyres_first requires --roi-mode anyres_max_9")
        if merge_type is not None and merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"}:
            raise ValueError("anyres_first requires spatial_unpad or spatial_unpad_add_newl packing")


def configure_image_mode(config, roi_mode, image_token_order="base_first"):
    """Configure preprocessing/packing without changing model precision or attention."""
    if roi_mode not in {"randomroi", "randompatch", "anyres_max_9", "ex_base_copy"}:
        raise ValueError(f"Unknown ROI mode: {roi_mode}")
    validate_image_token_order(image_token_order, roi_mode)
    if roi_mode == "ex_base_copy":
        config.image_aspect_ratio = "ex_base_copy"
        # 'unpad' ensures the checkpoint's learned image_newline is loaded.
        # The dedicated packing branch does NOT unpad or pool the Base copies.
        config.mm_patch_merge_type = "spatial_unpad_ex_base_copy"
    elif roi_mode == "anyres_max_9":
        config.image_aspect_ratio = "anyres_max_9"
        # Preserve checkpoint packing, as in reference LLaVA.
    else:
        config.image_aspect_ratio = "randomroi"
        config.mm_patch_merge_type = "spatial_avgpool_auto_unpad_add_newl"
    validate_image_token_order(image_token_order, roi_mode, config.mm_patch_merge_type)
    config.image_token_order = image_token_order


def _generate_with_timing(model, generation_options, *, include_pruning_time=True):
    """Time one generate call; optional subtraction is a profiled diagnostic.

    Synchronized pruning sections cover explicit ranking/selection/masking/drop
    routines, not all downstream cache/position handling or Python dispatch.
    Extra synchronization changes scheduling, so adjusted time is not a claim
    about uninstrumented end-to-end speed. Default timing adds no inner syncs.
    """
    import torch
    if type(include_pruning_time) is not bool:
        raise ValueError("include_pruning_time must be a boolean")
    core = model.get_model()
    timer = None
    if not include_pruning_time:
        devices = {parameter.device for layer in getattr(core, "layers", ())
                   for parameter in layer.parameters()}
        if len(devices) > 1:
            raise ValueError("Exclude-pruning timing requires decoder weights on one device; "
                             "use one visible GPU without decoder offloading")
        from llava.model.language_model.pruning_timing import PruningTimer
        timer = PruningTimer()
    previous_timer = getattr(core, "_pruning_timer", None)
    core._pruning_timer = timer
    try:
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate(**generation_options)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
    finally:
        # A failed sample or a later default-mode run must not inherit a timer.
        core._pruning_timer = previous_timer
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("Invalid synchronized generation duration")
    timing = {"generation_seconds": elapsed}
    if timer is not None:
        pruning = timer.seconds
        if not math.isfinite(pruning) or not 0 <= pruning <= elapsed:
            raise RuntimeError("Measured pruning duration is inconsistent with generation duration")
        timing.update(generation_seconds=elapsed - pruning,
                      generation_with_pruning_seconds=elapsed, pruning_seconds=pruning)
    return generated, timing


class LlavaBackend:
    def __init__(self, model_path: str | Path, roi_mode: str = "randomroi",
                 image_token_order: str = "base_first"):
        validate_image_token_order(image_token_order, roi_mode)
        # Delay heavy imports so JSON/configuration checks run without a GPU stack.
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("LLaVA inference requires a CUDA GPU")
        # Match the current original LLaVA loader. Do not override TF32 flags
        # or silently fall back to another precision/attention implementation.
        from transformers import AutoTokenizer
        from llava.constants import (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN)
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Model checkpoint not found: {path}")
        config = LlavaQwenConfig.from_pretrained(path)
        configure_image_mode(config, roi_mode, image_token_order)
        # Like reference LLaVA's --overwrite_image_aspect_ratio, anyres leaves the
        # checkpoint's merge type intact (normally spatial_unpad).
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = LlavaQwenForCausalLM.from_pretrained(
            path, config=config, low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", torch_dtype=torch.float16,
            device_map="auto",
        )
        if getattr(config, "mm_use_im_patch_token", True):
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if getattr(config, "mm_use_im_start_end", False):
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.vision_tower = self.model.get_vision_tower()
        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model(device_map="auto")
        self.processor = self.vision_tower.image_processor
        self.inference_config = {
            "model_dtype": str(self.model.dtype),
            "vision_dtype": str(self.vision_tower.dtype),
            "attention": self.model.config._attn_implementation,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "image_aspect_ratio": self.model.config.image_aspect_ratio,
            "mm_patch_merge_type": self.model.config.mm_patch_merge_type,
            "image_token_order": image_token_order,
            "attention_source": "independent_qk_at_nonzero_rates_only",
            "score_dtype": "float32",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
        if roi_mode == "ex_base_copy":
            self.inference_config.update(base_view_count=3,
                                         visual_packing="base1 + base2 + base3 + one final newline")
        from importlib.metadata import version
        self.inference_config.update({
            "transformers": version("transformers"),
            "flash_attn": version("flash-attn"),
        })
        print(
            f"Inference config: dtype={self.model.dtype}, "
            f"vision_dtype={self.vision_tower.dtype}, "
            f"attention={self.model.config._attn_implementation}, "
            f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn_tf32={torch.backends.cudnn.allow_tf32}, "
            f"image_aspect_ratio={self.model.config.image_aspect_ratio}, "
            f"mm_patch_merge_type={self.model.config.mm_patch_merge_type}, "
            f"image_token_order={image_token_order}",
            flush=True,
        )

    def generate(self, sample: Sample, prompt: str, method: PruningMethod,
                 rate: int, roi_mode: str, *, capture_visualization: bool,
                 capture_attention: bool,
                 random_seed: int, do_sample: bool = False,
                 include_pruning_time: bool = True) -> dict:
        import torch
        from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        from llava.mm_utils import process_images, tokenizer_image_token

        method.configure(self.model.get_model(), rate, capture_attention=capture_attention,
                         capture_visualization=capture_visualization)
        if (capture_visualization and roi_mode == "anyres_max_9"
                and self.model.config.mm_patch_merge_type not in
                {"spatial_unpad", "spatial_unpad_add_newl"}):
            raise ValueError(
                "Anyres visualization requires spatial_unpad packing. The checkpoint's "
                "merge type is preserved to match reference; it is not silently overwritten."
            )
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        with Image.open(sample.image) as opened:
            image = opened.convert("RGB")
        roi_source, mask_path, bbox = choose_roi(sample, roi_mode)
        mask = None
        if mask_path is not None:
            mask = load_mask(mask_path)
        boxes_list = [bbox] if bbox is not None else None
        masks = [mask] if roi_source == "mask" else None
        pixels, crop_metadata = process_images(
            [image], self.processor, self.model.config,
            masks=masks, boxes_list=boxes_list, return_pro_data=True,
        )
        if isinstance(pixels, list):
            pixels = [item.to(self.model.device, dtype=self.model.dtype) for item in pixels]
        else:
            pixels = pixels.to(self.model.device, dtype=self.model.dtype)

        message = prompt if DEFAULT_IMAGE_TOKEN in prompt else f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        if len(prompt) > 4096:
            raise ValueError("reference LLaVA's single-image prompt limit is 4096 characters")
        if message.count(DEFAULT_IMAGE_TOKEN) != 1:
            raise ValueError("A sample prompt must contain exactly one <image> token")
        # Equivalent to the old qwen_1_5 single-turn conversation template.
        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n")
        if getattr(self.model.config, "mm_use_im_start_end", False):
            text = text.replace(DEFAULT_IMAGE_TOKEN,
                                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        tokens = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).to(self.model.device)

        # Match SimpleModelWorker's context budget, including its base-view
        # patch count approximation (not a newly chosen anyres token budget).
        max_new_tokens = min(
            512, getattr(self.model.config, "max_position_embeddings", 2048)
            - tokens.shape[-1] - self.vision_tower.num_patches,
        )
        if max_new_tokens < 1:
            raise ValueError("Prompt exceeds the original LLaVA context budget")
        generation_options = {
            "inputs": tokens, "images": pixels, "image_sizes": [image.size],
            "do_sample": do_sample,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        generated, timing = _generate_with_timing(
            self.model, generation_options, include_pruning_time=include_pruning_time)
        answer = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        core = self.model.get_model()
        result = {
            "answer": answer,
            **timing,
            "roi_source": roi_source,
            "roi_boxes": crop_metadata[0].get("roi_boxes", []),
            "stats": method.stats(core, rate),
            "max_new_tokens": max_new_tokens,
        }
        if capture_visualization:
            result.update(method.visualization_data(core, capture_attention=capture_attention))
            result["image"] = image
            result["crop_metadata"] = crop_metadata
        return result
