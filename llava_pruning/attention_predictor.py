"""Public learned-attention pruning adapter; training is a separate entry."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .rates import is_prune_rate
from .priortr import PriorTRMethod
from .clse import CLSEMethod


@dataclass(frozen=True)
class AttentionPredictorMethod:
    checkpoint: Path
    rates: tuple
    visualize_rates: frozenset
    layer: int = 2
    rate_semantics: str = "layer_average"
    name: str = "attention_predictor"
    selection_mode: str = "base_guided"

    @classmethod
    def from_config(cls, path):
        path = Path(path).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        allowed = {"checkpoint", "layer", "rate_semantics", "prune_rates", "visualize_rates", "selection_mode"}
        if set(config) - allowed:
            raise ValueError(f"Unknown attention_predictor keys: {sorted(set(config) - allowed)}")
        if type(config.get("layer", 2)) is not int or config.get("layer", 2) != 2:
            raise ValueError("attention_predictor prunes after completed LLM block 2")
        if config.get("rate_semantics", "layer_average") != "layer_average":
            raise ValueError("attention_predictor requires layer_average rates")
        selection_mode = config.get("selection_mode", "base_guided")
        if selection_mode not in {"base_guided", "joint"}:
            raise ValueError("attention_predictor selection_mode must be base_guided or joint")
        rates, visual = tuple(config["prune_rates"]), tuple(config.get("visualize_rates", []))
        if not rates or any(not is_prune_rate(r) for r in rates) or len(set(rates)) != len(rates):
            raise ValueError("Predictor rates must be distinct percentages in [0,100)")
        if any(not is_prune_rate(r) for r in visual) or 0 in visual or not set(visual).issubset(rates):
            raise ValueError("Predictor visualize_rates must be a nonzero subset of prune_rates")
        checkpoint = Path(config["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        checkpoint = checkpoint.resolve()
        if any(r != 0 for r in rates) and not checkpoint.is_file():
            raise FileNotFoundError(f"Predictor checkpoint not found: {checkpoint}. Run train_attention_predictor.py first; "
                                    "then use its output method.json.")
        return cls(checkpoint, rates, frozenset(visual), selection_mode=selection_mode)

    resolve_budget = PriorTRMethod.resolve_budget
    resolve_budgets = PriorTRMethod.resolve_budgets
    stats = CLSEMethod.stats
    visualization_data = CLSEMethod.visualization_data

    def configure(self, core, prune_rate, *, capture_attention=False, capture_visualization=False):
        budget = self.resolve_budget(prune_rate, len(core.layers))
        core.configure_attention_predictor(enabled=prune_rate != 0,
                                           keep_ratio=(100 - budget["requested_layer_prune_rate"]) / 100,
                                           checkpoint=self.checkpoint, capture_attention=capture_attention,
                                           capture_visualization=capture_visualization, selection_mode=self.selection_mode)

    def details(self):
        digest = None
        if self.checkpoint.is_file():
            sha = hashlib.sha256()
            with self.checkpoint.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
            digest = sha.hexdigest()
        return dict(checkpoint=str(self.checkpoint), checkpoint_sha256=digest,
                    algorithm="KL-trained three-layer side attention; not full MAP/TwigVLM reproduction",
                    student="architecture comes from checkpoint: v2 independently pretrained native Qwen2 blocks 3/4/5 (no LM head); v1 legacy small branch",
                    teacher="fixed checkpoint teacher block (default22); last prompt query -> visual keys, mean heads; independent QK",
                    gradient="frozen backbone; student trained separately with KL(teacher||student); no GT",
                    selection_mode=self.selection_mode,
                    ranking_source=("Base prediction only; bilinear probability interpolation (align_corners=False) to actual unpadded/max9 AnyRes grid"
                                    if self.selection_mode == "base_guided" else "own predicted attention for Base and AnyRes"),
                    budget_scope=("separate Base/whole AnyRes with same continuous spatial retention; preserve packed target after newline reservation"
                                  if self.selection_mode == "base_guided" else "joint Base+AnyRes spatial top-k"),
                    budget_rounding="floor total packed target; protect newlines; proportional largest remainder for base_guided",
                    checkpoint_compatibility="selection policy is inference-only; existing v1/v2 weights do not need retraining",
                    position_policy="preserve original IDs, physically compact all existing KV caches",
                    inference="native backbone FP16/FlashAttention2; student precision stored in checkpoint",
                    rate_semantics="layer_average", pruning_layer=2)

    def visualize(self, *, result, vision_tower, output_dir, sample_id, save_prune, save_attention, ground_truth=None):
        from .visualization import prediction_caption, save_vico_visualizations
        return save_vico_visualizations(
            result["image"], result["crop_metadata"][0], result["stages"], layer_stats=result["stats"]["layers"],
            base_grid=int(vision_tower.num_patches_per_side), patch_size=int(vision_tower.config.patch_size),
            output_dir=output_dir, sample_id=sample_id, save_prune=save_prune, save_attention=save_attention,
            prediction_label=prediction_caption(ground_truth, result.get("answer")), method_name="AttentionPredictor",
            score_caption=("Base predicted attention; AnyRes is mapped from Base, NOT its own attention or defect probability."
                           if self.selection_mode == "base_guided" else
                           "Side Transformer block3 predicted teacher attention; NOT defect probability."))
