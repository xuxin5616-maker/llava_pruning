"""Method registry; parameters belong to each method's own config file."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class PruningMethod(Protocol):
    name: str
    rates: tuple[int, ...]
    visualize_rates: frozenset[int]

    def configure(self, core, prune_rate: int, *, capture_attention: bool) -> None: ...

    def stats(self, core) -> dict: ...

    def image_masks(self, core) -> list: ...

    def image_attentions(self, core) -> list: ...

    def visualize(self, *, result: dict, vision_tower, output_dir: Path,
                  sample_id: str, save_prune: bool, save_attention: bool) -> list: ...


@dataclass(frozen=True)
class FastVMethod:
    layer: int
    rates: tuple[int, ...]
    visualize_rates: frozenset[int]
    min_tokens: int
    preserve_image_newline: bool
    name: str = "fastv"

    @classmethod
    def from_config(cls, path: str | Path) -> "FastVMethod":
        with Path(path).open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        rates = tuple(config["prune_rates"])
        visualize = frozenset(config.get("visualize_rates", []))
        if (not rates or len(set(rates)) != len(rates) or
                any(type(rate) is not int or not 0 < rate < 100 for rate in rates)):
            raise ValueError("FastV prune_rates must be distinct integers from 1 to 99")
        if not visualize.issubset(rates):
            raise ValueError("FastV visualize_rates must be a subset of prune_rates")
        layer = config["layer"]
        min_tokens = config.get("min_tokens", 1)
        if type(layer) is not int or layer < 1 or type(min_tokens) is not int or min_tokens < 1:
            raise ValueError("FastV layer and min_tokens must be positive integers")
        return cls(layer, rates, visualize, min_tokens,
                   bool(config.get("preserve_image_newline", True)))

    def configure(self, core, prune_rate: int, *, capture_attention: bool) -> None:
        if prune_rate not in self.rates:
            raise ValueError(f"Prune rate {prune_rate} is not configured")
        core.configure_fastv(
            enabled=True,
            layer=self.layer,
            keep_ratio=1.0 - prune_rate / 100.0,
            min_tokens=self.min_tokens,
            preserve_image_newline=self.preserve_image_newline,
            capture_attention=capture_attention,
        )

    def stats(self, core) -> dict:
        return core.get_fastv_stats()

    def image_masks(self, core) -> list:
        return core.get_fastv_image_masks()

    def image_attentions(self, core) -> list:
        return core.get_fastv_image_attentions()

    def visualize(self, *, result: dict, vision_tower, output_dir: Path,
                  sample_id: str, save_prune: bool, save_attention: bool) -> list:
        from .visualization import save_fastv_visualizations

        return save_fastv_visualizations(
            [result["image"]], result["crop_metadata"], result["masks"],
            image_attentions=result.get("attentions"),
            base_grid=int(vision_tower.num_patches_per_side),
            patch_size=int(vision_tower.config.patch_size),
            output_dir=output_dir, sample_id=sample_id,
            fastv_layer=result["stats"]["fastv_layer"],
            keep_ratio=result["stats"]["keep_ratio"],
            save_prune=save_prune, save_attention=save_attention,
        )


METHODS = {"fastv": FastVMethod.from_config}


def load_method(name: str, config_path: str | Path) -> PruningMethod:
    try:
        return METHODS[name](config_path)
    except KeyError as error:
        raise ValueError(f"Unknown pruning method {name!r}; available: {', '.join(METHODS)}") from error
