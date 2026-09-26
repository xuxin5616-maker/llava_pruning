"""Fixed Triad OneVision/Qwen2 backend; no model-family selection in the CLI."""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Sample
from .methods import PruningMethod
from .roi import choose_roi, load_mask


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "llava"


class TriadBackend:
    def __init__(self, model_path: str | Path, roi_mode: str = "randomroi"):
        # Delay heavy imports so JSON/configuration checks run without a GPU stack.
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Triad inference requires a CUDA GPU")
        from transformers import AutoTokenizer
        from llava.constants import (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN)
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Model checkpoint not found: {path}")
        if roi_mode not in {"randomroi", "randompatch", "anyres_max_9"}:
            raise ValueError(f"Unknown ROI mode: {roi_mode}")
        config = LlavaQwenConfig.from_pretrained(path)
        config.image_aspect_ratio = "anyres_max_9" if roi_mode == "anyres_max_9" else "randomroi"
        config.mm_patch_merge_type = (
            "spatial_unpad" if roi_mode == "anyres_max_9"
            else "spatial_avgpool_auto_unpad_add_newl"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = LlavaQwenForCausalLM.from_pretrained(
            path, config=config, low_cpu_mem_usage=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
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

    def generate(self, sample: Sample, prompt: str, method: PruningMethod,
                 rate: int, roi_mode: str, *, capture_visualization: bool,
                 capture_attention: bool,
                 random_seed: int) -> dict:
        import torch
        from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        from llava.mm_utils import process_images, tokenizer_image_token

        method.configure(self.model.get_model(), rate, capture_attention=capture_attention)
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

        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = self.model.generate(
                inputs=tokens, images=pixels, image_sizes=[image.size],
                do_sample=False, max_new_tokens=256, use_cache=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        answer = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        core = self.model.get_model()
        result = {
            "answer": answer,
            "generation_seconds": elapsed,
            "roi_source": roi_source,
            "roi_boxes": crop_metadata[0].get("roi_boxes", []),
            "stats": method.stats(core, rate),
        }
        if capture_attention:
            result["attentions"] = method.image_attentions(core)[0]
        if capture_visualization:
            result["masks"] = method.image_masks(core)[0]
            result["image"] = image
            result["crop_metadata"] = crop_metadata
        return result
