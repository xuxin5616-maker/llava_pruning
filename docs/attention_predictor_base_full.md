# Base full + Base-guided whole-AnyRes pruning

Current policy: `selection_mode=base_full`. Keep **every Base token**, map the
distilled Base attention probabilities onto the actual complete AnyRes grid,
and select ONE global AnyRes top-k. No crop quota, crop mean softmax,
`crop_temperature`, spatial prior, or own AnyRes ranking is used in this mode.

For the six-job own-AnyRes vs Base-mapped comparison, with PNGs only at70,
use [compare_attention_sources.py](attention_source_comparison.md).
The extra `selection_mode=anyres_self` has the SAME Base-full budgets but ranks
AnyRes by its OWN distilled prediction; it is not the old joint budget mode.

## What changes and what stays unchanged

- The side Transformer still receives the full H2 (Base, AnyRes, and text).
  Only Base scoring positions are requested. Conditional Base probabilities
  are FP32 bilinearly mapped (`align_corners=False`) to the actual post-unpad /
  post-max9 AnyRes shape. Equal scores are resolved by original packed order.
- All Base positions and structural newlines are protected. Chosen tokens are
  taken from ORIGINAL backbone H2, not the side branch's hidden states.
- Decoder FP16 + FlashAttention2, three copied layers, KL objective, and
  checkpoint format do not change. Existing `best.pt` files work without
  retraining. Original position IDs and physical KV-cache compaction remain.
- Pre-LLM pruning is still **not connected**. The existing provenance contract
  is reserved for a later adapter; a sparse pre-LLM sequence cannot simply be
  treated as a new unpruned baseline or reshaped into an assumed dense grid.
- Explicit previous modes remain available: `base_crop` (Base crop budgets /
  own AnyRes ranking), `base_interpolated` (equal-rate groups / mapped ranking),
  and `joint`. Missing mode or legacy `base_guided` resolves to `base_full`.
  **An existing file explicitly containing `base_crop` must be changed or
  replaced by the new config; it will not silently change algorithms.**

## Rate definition and feasibility

Let `D` be the decoder depth, `L=2` the completed pruning boundary, `N` the
ORIGINAL packed visual count (Base + spatial AnyRes + newlines), `B` the Base
count, `P` protected newlines, and `A=N-B-P`. For requested whole-image
layer-average removal `r` percent:

```text
boundary removal = r * D / (D-L)
K_total = floor(N * (1 - boundary removal/100))
K_base = B
K_anyres = K_total - B - P
```

This is the same total budget as before, **not** removal relative to AnyRes
alone. All deletion is transferred to AnyRes. `K_anyres=0` is permitted at
the exact feasible boundary; negative quotas raise, never clamp or filter.
Actual average removal is reported from layer input counts, including integer
rounding. Side-branch compute is measured in Lat/Mem, not added as fake decoder
layers to the token-average denominator.

For the usual 28-layer backbone, the maximum possible whole-image average is
`100*(26/28)*A/N`. Example: Base729 + one AnyRes729 +27 newlines gives a maximum
of about45.58%. Such an image cannot satisfy a50% or70% whole-image average.
Base729 +9 tiles6561 +81 newlines can satisfy30/50/70, but not90.
`anyres_max_9` is a resolution budget, not a promise of nine tiles per image.

Before warmups or evaluation, the runner scans **image sizes only**, reproduces
native crop selection/unpad/max9 counts, and checks ALL image/rate pairs. It
reports incompatible IDs, sizes, and maximum average removal, without creating
partial run outputs or skipping examples. Model loading still precedes this
preflight because the actual vision config is needed. If it fails, lower the
requested rates explicitly; changing the denominator or dataset is not an
automatic fallback.

## Config and server command

`configs/attention_predictor_base_full.json` points to the user's existing
`output/attention_predictor_twig_train_01/best.pt` and requests30/50/70.
Checkpoint paths are relative to the JSON location, not the shell directory.
For a different training folder, update its `checkpoint` field. Alternatively
change the training folder's existing `method.json` to:

```json
{
  "checkpoint": "best.pt",
  "layer": 2,
  "rate_semantics": "layer_average",
  "selection_mode": "base_full",
  "prune_rates": [30, 50, 70],
  "visualize_rates": [70]
}
```

The example below asks for metrics AND sample visualizations, assuming all
image/rate pairs pass the strict feasibility check:

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --roi-mode anyres_max_9 \
  --method attention_predictor \
  --method-config configs/attention_predictor_base_full.json \
  --no-sample --benchmark-metrics --warmup 10 \
  --save-prune-vis --save-attention-vis \
  --seed 42 \
  --output-dir output/attention_predictor_base_full_30_50_70_01
```

For metrics only, remove both `--save-*` arguments and add `--no-plots`.
Use `--visualize-samples` to restrict saved IDs without reducing evaluation.
Use a new/empty output directory. No100-image limit is added.

## Lat/Mem and optional images in the same run

Use `--benchmark-metrics` explicitly. The backbone loads once and warms up
separately per rate. All input samples, not only warmups or pictured IDs,
contribute to classification and measurements.

- **Lat**: synchronized resident inputs entering `model.generate` until the
  first selected token has been appended and GPU work synchronized. Includes
  vision/projector/packing, full H2 prefill, side scoring, mapping, top-k,
  cache compaction and required model work. Excludes model loading, disk I/O,
  CPU preprocessing, H2D preparation and saving images. This remains TTFT,
  not full-answer generation duration.
- **Mem**: per-call PyTorch peak allocated/reserved during FULL answer
  generation, including resident weights, side branch, inputs, and temporary
  activations. Reset after input preparation; excludes loading/preprocessing
  peaks. GiB and allocated/device-total percent are reported. This is not
  nvidia-smi whole-board memory or GPU compute utilization.
- If images are also requested for `attention_predictor`, **ALL rates are
  measured first with visualization capture disabled**. Only afterward are
  selected samples regenerated with the same seed and parameters to capture
  masks/maps. PNG export is outside measured calls. This extra pass increases
  wall time but does not enter Lat/Mem, and cannot inflate later measured
  reserved peaks with visualization allocations.
- The extra pass's full answer must match the measured answer. Mismatch raises
  before misleading images are saved; already complete metrics are preserved.
  `visualization_verification.csv` records these comparisons. It is not a
  second accuracy evaluation and is not averaged into metrics.

## Results

`summary.csv`, per-rate `metrics.json` and `predictions.jsonl` retain ACC/PRE/
Recall/TNR, Lat (`lat_mean_ms`, p50/p95/std), peak Mem (`mem_peak_allocated_gib`,
`mem_peak_reserved_gib`, allocated percent), full generation timing, and the
requested/actual whole-image layer-average budget. Per-rate `view_budgets.csv`
shows Base deletion0 and actual AnyRes deletion; `layer_tokens.csv` audits
physical layer lengths. `common_spatial_keep_ratio` is blank because these
two groups no longer share a rate. There is no `crop_budgets.csv` in this mode.

Visualizations remain `comparison.png` and `attention_overlay.png`. Base is
untouched in pruning masks; AnyRes attention is explicitly labeled Base-mapped,
not its own attention or anomaly probability. PNGs may project Base visibility
onto the combined original image, so inspect the AnyRes panels for its removal.

Local CPU/mock tests cover budgets, infeasibility, native counts, Base-only score
requests, image orders/newlines, zero identity, cache/position behavior, passive
first-token measurement, peak reporting, and separation of clean measurement
from visualization. They do not establish real7B GPU speedup or dataset accuracy.
