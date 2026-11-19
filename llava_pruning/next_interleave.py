"""The fixed single-image protocol for LLaVA-NeXT-Interleave Qwen 7B."""

import json
from pathlib import Path


MODEL_NAME = "llava-next-interleave-qwen-7b"
DEFAULT_MODEL_PATH = Path("/home/yz/xxy/data/checkpoints") / MODEL_NAME
VISION_REPO = "google/siglip-so400m-patch14-384"
INPUT_SIZE = (1152, 1152)
VISUAL_GRID = (2, 2)
PYRAMID_BLOCKS = (8, 16, 24)
NATIVE_PINPOINTS = {(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)}


def validate_config(config):
    expected = {
        "model_type": "llava_qwen",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "mm_hidden_size": 1152,
        "mm_patch_merge_type": "spatial_unpad",
        "mm_vision_select_layer": -2,
        "mm_vision_select_feature": "patch",
        "image_aspect_ratio": "anyres",
    }
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError("NeXT-Interleave Qwen 7B config mismatch: " + ", ".join(mismatches))
    if Path(config.get("mm_vision_tower", "")).name != VISION_REPO.split("/")[-1]:
        raise ValueError("NeXT-Interleave requires its SigLIP SO400M patch14/384 tower")
    if {tuple(pair) for pair in config.get("image_grid_pinpoints", [])} != NATIVE_PINPOINTS:
        raise ValueError("NeXT-Interleave requires the native grid choices")


def verify_model_download(model_path):
    model_path = Path(model_path).expanduser().resolve()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("NeXT checkpoint index lists no weight shards")
    for name in sorted(shards | {"tokenizer.json", "tokenizer_config.json"}):
        path = model_path / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Finish the NeXT checkpoint download: {path}")
    return model_path


def vision_tower_path(model_path, override=None):
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"SigLIP checkpoint directory not found: {path}")
        return str(path)
    sibling = Path(model_path).expanduser().resolve().parent / VISION_REPO.split("/")[-1]
    return str(sibling) if sibling.is_dir() else None


def verify_student(checkpoint, model_path, vision_path):
    manifest = checkpoint.get("training_manifest")
    if manifest is None:
        raise ValueError("NeXT-Interleave needs a VisA student with its train_run.json")
    run = json.loads(Path(manifest).read_text(encoding="utf-8"))
    settings, inference = run["configuration"], run["inference"]
    native_config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if (settings.get("dataset") != "visa"
            or Path(settings["model_path"]).expanduser().resolve() != Path(model_path).resolve()
            or inference.get("model_profile") != MODEL_NAME
            or inference.get("image_resize") != list(INPUT_SIZE)
            or inference.get("native_anyres_grid") != list(VISUAL_GRID)
            or inference.get("vision_tower") != (vision_path or native_config["mm_vision_tower"])):
        raise ValueError("Distill the student on this NeXT checkpoint with the same SigLIP and 1152px inputs")


def make_backend(model_path, vision_path=None):
    import torch

    from .backend import LlavaBackend

    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    # Both native AnyRes and the shared packing path select the same 2x2 grid
    # here. The cap in anyres_max_9 never downsamples its 54x54 token lattice.
    backend = LlavaBackend(model_path, "anyres_max_9", image_token_order="base_first",
                           image_resize=INPUT_SIZE, vision_tower_path=vision_path, model_dtype=torch.bfloat16)
    from llava.mm_utils import get_anyres_image_grid_shape
    tower = backend.vision_tower
    grid = get_anyres_image_grid_shape(INPUT_SIZE, backend.model.config.image_grid_pinpoints, tower.image_size)
    if grid != VISUAL_GRID or tower.image_size != 384 or tower.num_patches != 729:
        raise ValueError(f"Expected a 384px Base plus a 2x2 AnyRes grid, got {grid} with {tower.image_size}px views")
    backend.inference_config.update(model_profile=MODEL_NAME, native_anyres_grid=list(grid), encoder_views=5)
    return backend
