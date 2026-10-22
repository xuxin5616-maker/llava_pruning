"""Public method adapter for the pinned PriorTR algorithm."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .rates import is_prune_rate


SOURCE_COMMIT = "e442a4c14af1b333067a6b8310c9111cad922033"


def source_details():
    root = Path(__file__).resolve().parents[1] / "third_party/priortr"
    source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    if source["commit"] != SOURCE_COMMIT:
        raise RuntimeError("PriorTR source commit mismatch")
    for name, expected in source["sha256_lf"].items():
        actual = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"PriorTR reference hash mismatch: {name}")
    return source


@dataclass(frozen=True)
class PriorTRMethod:
    layer: int
    rates: tuple
    visualize_rates: frozenset
    query_aggregation: str = "last"
    head_aggregation: str = "mean"
    epsilon: float = 1e-10
    name: str = "priortr"
    rate_semantics: str = "final_cumulative"

    @classmethod
    def from_config(cls, path):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {"layer", "prune_rates", "visualize_rates", "query_aggregation", "head_aggregation", "epsilon", "rate_semantics"}
        if set(config) - allowed:
            raise ValueError(f"Unknown PriorTR config keys: {sorted(set(config) - allowed)}")
        layer = config.get("layer", 2)
        rates = tuple(config["prune_rates"])
        visual = tuple(config.get("visualize_rates", []))
        if type(layer) is not int or layer < 1:
            raise ValueError("PriorTR layer must be a positive integer")
        if not rates or any(not is_prune_rate(rate) for rate in rates) or len(set(rates)) != len(rates):
            raise ValueError("PriorTR rates must be distinct percentages in [0,100)")
        if (any(not is_prune_rate(rate) for rate in visual) or 0 in visual
                or not set(visual).issubset(rates)):
            raise ValueError("PriorTR visualize_rates must be a nonzero subset of prune_rates")
        query, head, epsilon = config.get("query_aggregation", "last"), config.get("head_aggregation", "mean"), config.get("epsilon", 1e-10)
        if query not in {"last", "question"} or head not in {"mean", "max"}:
            raise ValueError("Invalid PriorTR aggregation")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("PriorTR epsilon must be positive and finite")
        semantics = config.get("rate_semantics", "final_cumulative")
        if semantics not in {"final_cumulative", "layer_average"}:
            raise ValueError("PriorTR rate_semantics must be final_cumulative or layer_average")
        source_details()
        return cls(layer, rates, frozenset(visual), query, head, epsilon, rate_semantics=semantics)

    def resolve_budget(self, rate, model_depth):
        if rate not in self.rates:
            raise ValueError("PriorTR rate is not configured")
        if type(model_depth) is not int or not 1 <= self.layer < model_depth:
            raise ValueError("PriorTR layer must be below the actual model depth")
        boundary_rate = rate * model_depth / (model_depth - self.layer) if self.rate_semantics == "layer_average" else rate
        if boundary_rate >= 100:
            maximum = 100 * (model_depth - self.layer) / model_depth
            raise ValueError(f"Average pruning target {rate}% is unreachable after layer {self.layer} "
                             f"of {model_depth}; target must be below {maximum:.6f}%")
        average_target = rate if self.rate_semantics == "layer_average" else rate * (model_depth - self.layer) / model_depth
        return {"prune_rate": rate, "rate_semantics": self.rate_semantics, "pruning_layer": self.layer,
                "target_average_prune_rate": average_target, "requested_layer_prune_rate": boundary_rate}

    def resolve_budgets(self, model_depth):
        return [self.resolve_budget(rate, model_depth) for rate in self.rates]

    def configure(self, core, prune_rate, *, capture_attention=False, capture_visualization=False):
        budget = self.resolve_budget(prune_rate, len(core.layers))
        core.configure_priortr(enabled=prune_rate != 0, layer=self.layer,
                               keep_ratio=(100 - budget["requested_layer_prune_rate"]) / 100,
                               query_aggregation=self.query_aggregation, head_aggregation=self.head_aggregation,
                               epsilon=self.epsilon, capture_attention=capture_attention and prune_rate != 0,
                               capture_visualization=capture_visualization and prune_rate != 0)

    def validate_prompt(self, message, rate):
        if rate != 0 and not message.startswith("<image>\n"):
            raise ValueError("PriorTR requires <image> followed by newline BEFORE the question; tokens are not inserted or reordered")

    def validate_token_boundary(self, tokens, tokenizer, image_token_index, rate, use_im_end=False):
        if rate == 0:
            return
        positions = (tokens[0] == image_token_index).nonzero(as_tuple=False).flatten()
        separator = "<im_end>" if use_im_end else "\n"
        expected = tokenizer.encode(separator, add_special_tokens=False)
        if (len(positions) != 1 or not expected or int(positions[0]) + 1 >= tokens.shape[-1]
                or int(tokens[0, int(positions[0]) + 1]) != expected[0]):
            raise ValueError("PriorTR probe must be an existing instruction-independent separator token after <image>")

    def stats(self, core, prune_rate):
        stats = dict(core.get_priortr_stats())
        stats.update(self.resolve_budget(prune_rate, len(core.layers)))
        original = stats.get("original_image_tokens")
        rows = stats.get("layers", [])
        final = stats.get("final_llm_visual_tokens")
        stats["actual_layer_prune_rate"] = 100 * (1 - final / original) if original and final is not None else None
        stats["actual_average_prune_rate"] = (
            100 * (1 - sum(row["image_tokens_in"] for row in rows) / (len(rows) * original))
            if original and len(rows) == len(core.layers) else None)
        return stats

    def visualization_data(self, core, *, capture_attention=False):
        return {"stages": core.get_priortr_stages()}

    def details(self):
        return {"source": source_details(), "adaptation": "OneVision/Qwen2 independent QK; native decoder precision/backend",
                "layer_semantics": "1-based block whose attention is scored; prune AFTER this block",
                "score_dtype": "native QK; FP32 softmax then native aggregation; FP32 corrected math then native score",
                "position_policy": "preserve original IDs and compact ALL existing KV caches",
                "budget_scope": "all packed image tokens jointly, INCLUDING structural image_newline",
                "budget_rounding": "floor; minimum 1 safety extension",
                "rate_semantics": self.rate_semantics,
                "average_definition": "100*(1-sum(layer input visual tokens)/(model_depth*original packed visual tokens)); sample means in summary",
                "prior_probe": "reuse first text token immediately after actual packed image span; no insertion",
                "display": "signed corrected score mapped by per-sample spatial min-max; not attention probability"}

    def visualize(self, *, result, vision_tower, output_dir, sample_id, save_prune, save_attention, ground_truth=None):
        from .visualization import prediction_caption, save_vico_visualizations
        return save_vico_visualizations(
            result["image"], result["crop_metadata"][0], result["stages"],
            layer_stats=result["stats"]["layers"], base_grid=int(vision_tower.num_patches_per_side),
            patch_size=int(vision_tower.config.patch_size), output_dir=output_dir, sample_id=sample_id,
            save_prune=save_prune, save_attention=save_attention,
            prediction_label=prediction_caption(ground_truth, result.get("answer")), method_name="PriorTR",
            score_caption="PriorTR corrected score P*log(P/Q); signed spatial min-max colors, NOT attention probability.")
