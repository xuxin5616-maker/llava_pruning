"""Public pure-CLSE adapter: joint Base/AnyRes, one drop after LLM block 1."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .rates import is_prune_rate


SOURCE_COMMIT = "cfc9710abe4a1bd5480e521f3acd86ded4855ff3"


def source_details():
    root = Path(__file__).resolve().parents[1] / "third_party/clse"
    source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    if source["commit"] != SOURCE_COMMIT:
        raise RuntimeError("CLSE source commit mismatch")
    for name, expected in source["sha256_lf"].items():
        actual = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"CLSE reference hash mismatch: {name}")
    return source


@dataclass(frozen=True)
class CLSEMethod:
    rates: tuple
    visualize_rates: frozenset
    cutoff_ratio: float = .1
    temperature: float = .1
    epsilon: float = 1e-6
    channel_chunk: int = 128
    layer: int = 1
    rate_semantics: str = "layer_average"
    name: str = "clse"

    @classmethod
    def from_config(cls, path):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {"layer", "score_type", "rate_semantics", "prune_rates", "visualize_rates",
                   "cutoff_ratio", "temperature", "epsilon", "channel_chunk"}
        if set(config) - allowed:
            raise ValueError(f"Unknown CLSE config keys: {sorted(set(config) - allowed)}")
        if type(config.get("layer", 1)) is not int or config.get("layer", 1) != 1:
            raise ValueError("This CLSE adapter compares input embeddings to block 1 output; layer must be 1")
        if config.get("score_type", "clse") != "clse":
            raise ValueError("This adapter supports pure score_type=clse only; no attention fallback")
        if config.get("rate_semantics", "layer_average") != "layer_average":
            raise ValueError("CLSE rates must use layer_average semantics")
        rates = tuple(config["prune_rates"])
        visual = tuple(config.get("visualize_rates", []))
        if not rates or any(not is_prune_rate(r) for r in rates) or len(set(rates)) != len(rates):
            raise ValueError("CLSE rates must be distinct percentages in [0,100)")
        if any(not is_prune_rate(r) for r in visual) or 0 in visual or not set(visual).issubset(rates):
            raise ValueError("CLSE visualize_rates must be a nonzero subset of prune_rates")
        kwargs = {key: config.get(key, default) for key, default in
                  (("cutoff_ratio", .1), ("temperature", .1), ("epsilon", 1e-6), ("channel_chunk", 128))}
        for key in ("cutoff_ratio", "temperature", "epsilon"):
            value = kwargs[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"CLSE {key} must be positive and finite")
        if type(kwargs["channel_chunk"]) is not int or kwargs["channel_chunk"] < 1:
            raise ValueError("CLSE channel_chunk must be a positive integer")
        source_details()
        return cls(rates, frozenset(visual), **kwargs)

    def resolve_budget(self, rate, model_depth):
        if rate not in self.rates:
            raise ValueError("CLSE rate is not configured")
        if type(model_depth) is not int or model_depth < 2:
            raise ValueError("CLSE needs at least two decoder blocks")
        boundary_rate = rate * model_depth / (model_depth - 1)
        if boundary_rate >= 100:
            raise ValueError(f"CLSE average pruning target {rate}% is unreachable at model depth {model_depth}")
        return dict(prune_rate=rate, rate_semantics="layer_average", pruning_layer=1,
                    target_average_prune_rate=rate, requested_layer_prune_rate=boundary_rate)

    def resolve_budgets(self, model_depth):
        return [self.resolve_budget(rate, model_depth) for rate in self.rates]

    def configure(self, core, prune_rate, *, capture_attention=False, capture_visualization=False):
        budget = self.resolve_budget(prune_rate, len(core.layers))
        core.configure_clse(enabled=prune_rate != 0, keep_ratio=(100 - budget["requested_layer_prune_rate"]) / 100,
                            cutoff_ratio=self.cutoff_ratio, temperature=self.temperature, epsilon=self.epsilon,
                            channel_chunk=self.channel_chunk, capture_attention=capture_attention and prune_rate != 0,
                            capture_visualization=capture_visualization and prune_rate != 0)

    def stats(self, core, prune_rate):
        stats = dict(core.get_clse_stats())
        stats.update(self.resolve_budget(prune_rate, len(core.layers)))
        original, final, rows = stats.get("original_image_tokens"), stats.get("final_llm_visual_tokens"), stats.get("layers", [])
        stats["actual_layer_prune_rate"] = 100 * (1 - final / original) if original and final is not None else None
        stats["actual_average_prune_rate"] = (
            100 * (1 - sum(row["image_tokens_in"] for row in rows) / (len(rows) * original))
            if original and len(rows) == len(core.layers) else None)
        return stats

    def visualization_data(self, core, *, capture_attention=False):
        return {"stages": core.get_clse_stages()}

    def details(self):
        return dict(source=source_details(), adaptation="OneVision/Qwen2; native decoder precision/backend",
                    layer_semantics="compare input embeddings to block 1 output; prune once AFTER block 1",
                    score_type="clse", attention_source="none; pure spectral scores, no attention requests",
                    score_dtype="FP32 Gaussian high-pass FFT and evolution calibration; decoder untouched",
                    budget_scope="joint Base + AnyRes spatial top-k; all structural newlines protected",
                    spectral_geometry="separate Base and actual packed AnyRes FFT; concatenate energies for joint mean/std calibration",
                    calibration=f"upstream clipped intensity -> sample mean/std -> sigmoid(standardized/{self.temperature}); no L2 normalization",
                    rate_semantics="layer_average", budget_rounding="floor total packed target, then subtract protected newlines; infeasible samples raise",
                    average_definition="100*(1-sum(layer input visual tokens)/(depth*original packed visual tokens)); sample means in summary",
                    position_policy="preserve original IDs and compact ALL existing KV caches",
                    display="JET pure CLSE scores; attention_overlay.png is a historical filename, not an attention probability")

    def visualize(self, *, result, vision_tower, output_dir, sample_id, save_prune, save_attention, ground_truth=None):
        from .visualization import prediction_caption, save_vico_visualizations
        return save_vico_visualizations(
            result["image"], result["crop_metadata"][0], result["stages"], layer_stats=result["stats"]["layers"],
            base_grid=int(vision_tower.num_patches_per_side), patch_size=int(vision_tower.config.patch_size),
            output_dir=output_dir, sample_id=sample_id, save_prune=save_prune, save_attention=save_attention,
            prediction_label=prediction_caption(ground_truth, result.get("answer")), method_name="CLSE",
            score_caption="Pure CLSE spectral evolution score (input -> block 1); JET, NOT attention or defect probability.")
