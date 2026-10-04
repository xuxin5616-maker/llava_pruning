"""Method registry; parameters belong to each method's own config file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class PruningMethod(Protocol):
    name: str
    rates: tuple[int, ...]
    visualize_rates: frozenset[int]

    def configure(self, core, prune_rate: int, *, capture_attention: bool,
                  capture_visualization: bool = False) -> None: ...

    def visualization_data(self, core, *, capture_attention: bool) -> dict: ...

    def stats(self, core, prune_rate: int) -> dict: ...

    def visualize(self, *, result: dict, vision_tower, output_dir: Path,
                  sample_id: str, save_prune: bool, save_attention: bool,
                  ground_truth: int | None = None) -> list: ...


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
                any(type(rate) is not int or not 0 <= rate < 100 for rate in rates)):
            raise ValueError("FastV prune_rates must be distinct integers from 0 to 99")
        if not visualize.issubset(rates):
            raise ValueError("FastV visualize_rates must be a subset of prune_rates")
        if 0 in visualize:
            raise ValueError("Zero-rate baseline has no FastV pruning/attention visualization")
        layer = config["layer"]
        min_tokens = config.get("min_tokens", 1)
        if type(layer) is not int or layer < 1 or type(min_tokens) is not int or min_tokens < 1:
            raise ValueError("FastV layer and min_tokens must be positive integers")
        return cls(layer, rates, visualize, min_tokens,
                   bool(config.get("preserve_image_newline", True)))

    def configure(self, core, prune_rate: int, *, capture_attention: bool,
                  capture_visualization: bool = False) -> None:
        if prune_rate not in self.rates:
            raise ValueError(f"Prune rate {prune_rate} is not configured")
        core.configure_fastv(
            enabled=prune_rate != 0,
            layer=self.layer,
            keep_ratio=1.0 - prune_rate / 100.0,
            min_tokens=self.min_tokens,
            preserve_image_newline=self.preserve_image_newline,
            capture_attention=capture_attention and prune_rate != 0,
        )

    def stats(self, core, prune_rate: int) -> dict:
        if prune_rate == 0:
            return {"mode": "disabled_baseline", "keep_ratio": 1.0,
                    "fastv_layer": None, "batch": []}
        return core.get_fastv_stats()

    def image_masks(self, core) -> list:
        return core.get_fastv_image_masks()

    def image_attentions(self, core) -> list:
        return core.get_fastv_image_attentions()

    def visualization_data(self, core, *, capture_attention: bool) -> dict:
        result = {"masks": self.image_masks(core)[0]}
        if capture_attention:
            result["attentions"] = self.image_attentions(core)[0]
        return result

    def visualize(self, *, result: dict, vision_tower, output_dir: Path,
                  sample_id: str, save_prune: bool, save_attention: bool,
                  ground_truth: int | None = None) -> list:
        from .visualization import prediction_caption, save_fastv_visualizations

        return save_fastv_visualizations(
            [result["image"]], result["crop_metadata"], result["masks"],
            image_attentions=result.get("attentions"),
            base_grid=int(vision_tower.num_patches_per_side),
            patch_size=int(vision_tower.config.patch_size),
            output_dir=output_dir, sample_id=sample_id,
            fastv_layer=result["stats"]["fastv_layer"],
            keep_ratio=result["stats"]["keep_ratio"],
            save_prune=save_prune, save_attention=save_attention,
            prediction_label=prediction_caption(ground_truth, result.get("answer")),
        )


@dataclass(frozen=True)
class ViCoMethod:
    layers: tuple[int, ...]
    rates: tuple[int, ...]
    visualize_rates: frozenset[int]
    min_tokens: int = 1
    name: str = "vico"

    @classmethod
    def from_config(cls, path: str | Path) -> "ViCoMethod":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {"layers", "prune_rates", "visualize_rates", "min_tokens", "rate_semantics"}
        if set(config) - allowed:
            raise ValueError(f"Unknown ViCo config keys: {sorted(set(config) - allowed)}; use configs/vico.json")
        if config.get("rate_semantics", "final_cumulative") != "final_cumulative":
            raise ValueError("ViCo prune_rates represent final_cumulative percentages")
        layers = tuple(config["layers"])
        rates = tuple(config["prune_rates"])
        visualize = frozenset(config.get("visualize_rates", []))
        minimum = config.get("min_tokens", 1)
        if (not layers or any(type(n) is not int or n < 1 for n in layers)
                or tuple(sorted(set(layers))) != layers):
            raise ValueError("ViCo layers must be distinct increasing positive integers")
        if (not rates or any(type(r) is not int or not 0 <= r < 100 for r in rates)
                or len(set(rates)) != len(rates)):
            raise ValueError("ViCo prune_rates must be distinct integers from 0 to 99")
        if (any(type(r) is not int for r in visualize) or not visualize.issubset(rates)
                or 0 in visualize):
            raise ValueError("ViCo visualize_rates must be a nonzero subset of prune_rates")
        if type(minimum) is not int or minimum < 1:
            raise ValueError("ViCo min_tokens must be a positive integer")
        return cls(layers, rates, visualize, minimum)

    def keep_ratios(self, rate: int) -> tuple[float, ...]:
        if rate not in self.rates:
            raise ValueError(f"Prune rate {rate} is not configured")
        final = (100 - rate) / 100.0
        count = len(self.layers)
        return tuple(final ** (stage / count) if stage < count else final
                     for stage in range(1, count + 1))

    def configure(self, core, prune_rate: int, *, capture_attention: bool,
                  capture_visualization: bool = False) -> None:
        core.configure_vico(enabled=prune_rate != 0, layers=self.layers,
                            keep_ratios=self.keep_ratios(prune_rate), min_tokens=self.min_tokens,
                            capture_attention=capture_attention and prune_rate != 0,
                            capture_visualization=capture_visualization and prune_rate != 0)

    def stats(self, core, prune_rate: int) -> dict:
        return core.get_vico_stats()

    def visualization_data(self, core, *, capture_attention: bool) -> dict:
        return {"stages": core.get_vico_stages()}

    def visualize(self, *, result: dict, vision_tower, output_dir: Path,
                  sample_id: str, save_prune: bool, save_attention: bool,
                  ground_truth: int | None = None) -> list:
        from .visualization import prediction_caption, save_vico_visualizations
        return save_vico_visualizations(
            result["image"], result["crop_metadata"][0], result["stages"],
            layer_stats=result["stats"]["layers"],
            base_grid=int(vision_tower.num_patches_per_side),
            patch_size=int(vision_tower.config.patch_size),
            output_dir=output_dir, sample_id=sample_id,
            save_prune=save_prune, save_attention=save_attention,
            prediction_label=prediction_caption(ground_truth, result.get("answer")),
        )


METHODS = {"fastv": FastVMethod.from_config, "vico": ViCoMethod.from_config}


def resolve_method_config(name: str, config_path=None) -> Path:
    if name not in METHODS:
        raise ValueError(f"Unknown pruning method {name!r}; available: {', '.join(METHODS)}")
    return Path(config_path) if config_path is not None else Path(__file__).resolve().parents[1] / "configs" / f"{name}.json"


def load_method(name: str, config_path: str | Path | None = None) -> PruningMethod:
    path = resolve_method_config(name, config_path)
    return METHODS[name](path)
