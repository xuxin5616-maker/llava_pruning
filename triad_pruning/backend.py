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
        # Match the current original Triad loader. Do not override TF32 flags
        # or silently fall back to another precision/attention implementation.
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
        if roi_mode != "anyres_max_9":
            config.mm_patch_merge_type = "spatial_avgpool_auto_unpad_add_newl"
        # Like Triad's --overwrite_image_aspect_ratio, anyres leaves the
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
            "attention_source": "independent_qk_at_nonzero_rates_only",
            "score_dtype": "float32",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
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
            f"mm_patch_merge_type={self.model.config.mm_patch_merge_type}",
            flush=True,
        )

    def generate(self, sample: Sample, prompt: str, method: PruningMethod,
                 rate: int, roi_mode: str, *, capture_visualization: bool,
                 capture_attention: bool,
                 random_seed: int, do_sample: bool = False) -> dict:
        import torch
        from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        from llava.mm_utils import process_images, tokenizer_image_token

        method.configure(self.model.get_model(), rate, capture_attention=capture_attention)
        if (capture_visualization and roi_mode == "anyres_max_9"
                and self.model.config.mm_patch_merge_type not in
                {"spatial_unpad", "spatial_unpad_add_newl"}):
            raise ValueError(
                "Anyres visualization requires spatial_unpad packing. The checkpoint's "
                "merge type is preserved to match Triad; it is not silently overwritten."
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
            raise ValueError("Triad's single-image prompt limit is 4096 characters")
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
            raise ValueError("Prompt exceeds the original Triad context budget")
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            generation_options = {
                "do_sample": do_sample,
                "temperature": 0.2,
                "top_p": 0.7,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
            }
            generated = self.model.generate(
                inputs=tokens, images=pixels, image_sizes=[image.size],
                **generation_options,
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
            "input_token_ids": tokens[0].detach().cpu().tolist(),
            "generated_token_ids": generated[0].detach().cpu().tolist(),
            "max_new_tokens": max_new_tokens,
        }
        if capture_attention:
            result["attentions"] = method.image_attentions(core)[0]
        if capture_visualization:
            result["masks"] = method.image_masks(core)[0]
            result["image"] = image
            result["crop_metadata"] = crop_metadata
        return result
