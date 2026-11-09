"""Remove one complete AnyRes crop before the LLM using MyPool crop scores."""

from contextlib import contextmanager
import math
from types import SimpleNamespace


DROP_MODES = ("drop_most_important", "drop_least_important")


def remove_crop(packed, grid, side, image_size, crop_index, final_newline):
    """Filter native-packed tokens whose receptive field intersects the crop."""
    import torch
    from torch.nn import functional as F
    from llava.model.llava_arch import unpad_image

    width, height = grid
    row, col = divmod(crop_index, width)
    removed = torch.zeros((1, height * side, width * side), dtype=torch.float32, device=packed.device)
    removed[:, row * side:(row + 1) * side, col * side:(col + 1) * side] = 1
    removed = unpad_image(removed, image_size)
    rows, cols = removed.shape[-2:]
    scale = math.sqrt(rows * cols / (9 * side ** 2))
    if scale > 1.1:
        removed = F.interpolate(
            removed[None], size=(int(rows // scale), int(cols // scale)),
            mode="bilinear", align_corners=False,
        )[0]
    spatial_keep = removed[0] == 0
    row_keep = spatial_keep.any(dim=1, keepdim=True)
    anyres_keep = torch.cat((spatial_keep, row_keep), dim=1).flatten()
    base_keep = torch.ones(side ** 2, dtype=torch.bool, device=packed.device)
    pieces = [base_keep, anyres_keep]
    if final_newline:
        pieces.append(row_keep.any().reshape(1))
    keep = torch.cat(pieces)
    if keep.numel() != len(packed):
        raise RuntimeError("Crop mask does not match native packed visual tokens")
    return packed[keep]


class CropAblation:
    def __init__(self, core, mode):
        if mode not in DROP_MODES:
            raise ValueError(f"Unknown crop ablation: {mode}")
        self.core = core
        self.mode = mode
        self.details = {}

    def configure(self, core, rate, **capture):
        if core is not self.core or rate != 0 or any(capture.values()):
            raise ValueError("Crop ablation uses whole-crop removal without token pruning or visualization")
        config = core.config
        if (config.image_aspect_ratio != "anyres_max_9" or
                config.mm_patch_merge_type not in {"spatial_unpad", "spatial_unpad_add_newl"} or
                getattr(config, "image_token_order", "base_first") != "base_first"):
            raise ValueError("Crop ablation requires native Base-first AnyRes max9 packing")
        core.configure_fastv(enabled=False)
        self.details = {}

    def prepare_inputs(self, image_size, tokens):
        from .visa_comparison_methods import native_counts

        self.image_size = tuple(image_size)
        self.grid, counts = native_counts(self.core, self.image_size)
        self.original = sum(counts)

    def encode_images(self, images):
        from runMyPruningPoolSpatial import pooled_crop_redundancy

        tower = self.core.get_vision_tower()
        if len(tower.vision_tower.vision_model.encoder.layers) != 26:
            raise ValueError("MyPool scoring requires the active 26-block OneVision SigLIP tower")
        # SigLipVisionTower returns block 26's output, the same tensor MyPool scores.
        features = tower(images)
        redundancy = pooled_crop_redundancy(features, self.grid, reduction="mean", audit=False)["redundancy"]
        # argmin/argmax select the first row-major crop on ties.
        index = redundancy.argmin() if self.mode == "drop_most_important" else redundancy.argmax()
        self.crop_index = int(index.item())
        self.redundancy = redundancy
        return self.core.mm_projector(features)

    def pack(self, image_features, image_sizes, modalities):
        import torch
        from runMyPruningPoolSpatial import TokenDecisions, pack_full_views

        if len(image_features) != 1 or tuple(image_sizes[0]) != self.image_size or modalities != ["image"]:
            raise ValueError("Crop ablation expects one unchanged image")
        projected = image_features[0]
        views, count, _ = projected.shape
        side = self.core.get_vision_tower().num_patches_per_side
        decisions = TokenDecisions(count, count, torch.arange(count, device=projected.device).expand(views, -1))
        merge_type = self.core.config.mm_patch_merge_type
        packed, _ = pack_full_views(
            projected, decisions, self.grid, self.image_size, self.core.image_newline, merge_type,
        )
        if len(packed) != self.original:
            raise RuntimeError("Crop ablation packing differs from the native reference length")
        kept = remove_crop(packed, self.grid, side, self.image_size, self.crop_index,
                           merge_type == "spatial_unpad_add_newl")
        self.details["input_image_tokens"] = len(kept)
        return [kept]

    def stats(self, core, rate):
        if "input_image_tokens" not in self.details:
            raise RuntimeError("Crop selection did not execute")
        return dict(self.details, crop_index=self.crop_index, crop_grid=list(self.grid),
                    crop_redundancies=self.redundancy.cpu().tolist())

    @contextmanager
    def inference_context(self, model):
        from runVisionZip import visionzip_generation

        # Reuse only its single-image embedding insertion and cache handling.
        adapter = SimpleNamespace(encode_images=self.encode_images, pack=self.pack, stats={})
        with visionzip_generation(model, adapter):
            yield
