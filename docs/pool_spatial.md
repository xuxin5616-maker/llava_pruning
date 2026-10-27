# PoolSpatial probability-mixture variant

`runMyPruningPoolSpatial.py` is the supplied standalone script
`1.23-pool-spatial-probability-mix`, integrated beside `vendor/` and
`llava_pruning/`. The method code is unchanged from the supplied file. It uses
the project's vendored LLaVA model and existing `llava_pruning.prompts` module;
no external VisionZip checkout, new model checkpoint, or dependency upgrade is
needed. Keep the existing Torch 2.1.2 / Transformers 4.46.1 server environment.

This is a separate entry point, not a new `run.py --method` choice. The old
`runMyPruningPoolSpatialBaseFull.py`, `run.py`, predictor, and their defaults
are unchanged.

## Method and defaults

- Use annotation-free AnyRes max9, Base first, and the checkpoint's
  `spatial_unpad` packing. The crop grid depends on the original image; max9
  does not mean that every image has nine crops.
- At each selected SigLIP feature block, L2-normalize every token, pool Base
  bins and AnyRes crops independently, normalize each pooled vector, and compare
  it with its own tokens by cosine similarity. Default `mean` takes the token
  mean within each branch, then adds the Base and AnyRes branch scores to give
  crop redundancy `R`. Multiple feature layers are scored independently and
  their scalar crop scores are averaged.
- Allocate AnyRes quotas from `p_R = softmax(-R/T)`,
  `p_S = P/sum(P)`, and `p = (1-lambda)*p_R + lambda*p_S`. `P` is the existing
  global flat-top raised-cosine spatial prior, pooled into Base bins.
  There is no second softmax after mixing. Reserve at least 4 tokens per crop,
  cap each at 729, redistribute capped excess, and preserve the exact aggregate
  integer budget. An empty prior falls back to redundancy allocation.
- `--spatial-mix-weight` and `--spatial-strength` are aliases for the same
  probability fraction `lambda` in `[0,1]`, default **0 (disabled)**. This is
  **not** BaseFull's additive score-level delta. Set it explicitly to enable
  the new fusion; lambda zero uses the original allocator unchanged.
- Default feature and VisionZip attention blocks are both **26**, with
  `T=0.05` and pruning rate 70. Layer numbers are one-based **SigLIP blocks**,
  not LLM blocks. `--redundancy-layers` and `--visionzip-layers` accept other
  increasing distinct indices in 1..26. Merge Key/features remain block26.
- Native per-view VisionZip received attention selects dominant tokens. Block26
  Key similarity merges other tokens into contextual anchors. Redundancy and
  the spatial prior change crop budgets, not within-crop attention rankings.
- Base and aggregate AnyRes prune independently at the requested raw-content
  rate: `ceil(729*(1-r))` and `ceil(M*729*(1-r))`. Individual AnyRes crops may
  retain different counts. For nine crops at 70%, Base retains 219 and AnyRes
  retains 1969 total. **BaseFull instead keeps Base729 and AnyRes1459** at the
  same total content budget.
- The main model stays FP16 + FlashAttention2 with greedy full-answer
  generation. Redundancy-only feature copies use FP32. Nonzero packing restores
  anchors to spatial order and protects a newline after every raw AnyRes row;
  it does not unpad or max9-resize sparse holes. Position IDs are consecutive in
  the compressed sequence. The reported final packed reduction can differ from
  the requested raw-content rate.
- At 0%, the method still runs scoring and all-token selection, passes the
  original full tensor/layout to the projector, and uses native full-grid
  packing. It does not substitute a separate baseline model path. Paired exact
  GPU verification is available with `--prune-rate 0 --verify-zero`; CPU tests
  do not establish checkpoint-level answer equivalence.

## Server command

This example explicitly enables probability mixing with lambda 0.02, uses the
new script's layer26 defaults, evaluates all input records, and saves no images:

```bash
CUDA_VISIBLE_DEVICES=5 python runMyPruningPoolSpatial.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --prune-rate 70 \
  --redundancy-layers 26 \
  --visionzip-layers 26 \
  --allocation-temperature 0.05 \
  --spatial-mix-weight 0.02 \
  --save-prune-vis False \
  --output-dir output/pool_spatial_probability_70_01
```

Use a new/empty output directory. To use block21 like the previous BaseFull
experiment, change **both** layer arguments to 21; budget/fusion policies still
differ from BaseFull. To use the example in the supplied file header, use
pruning30, both layers21, lambda0.02, and temperature0.1 explicitly.

To enable existing pruning visualizations, set `--save-prune-vis True`. By
default, only GT-abnormal samples save images; `--vis-abnormal-only False`
includes all samples. Classification always processes all records. Enabling
visualization also saves score/allocation diagnostics as in the supplied code.

## Outputs and timing limits

The standalone runner saves:

- `run.json`: resolved configuration, version, model/packing details.
- `summary.csv`: ACC, precision, recall, TNR, counts, and method settings.
- `prune_70/metrics.json`, `predictions.jsonl`, and
  `classification_results.csv`: aggregate metrics and per-image results.
- `prune_70/crop_budgets.csv`: redundancy, probabilities, quotas, and actual
  per-crop retained counts.
- `prune_70/geometry_audit.csv`: padding, unpadding, and max9 geometry audit.

`generation_seconds` is synchronized **full-answer generation time**, including
pruning and excluding CPU file loading/preprocessing. The supplied standalone
runner does **not** currently export first-token Lat or peak Mem. Do not treat
this duration as TTFT or reuse `run.py`/`benchmark_fastv.py`'s Lat/Mem claims for
it. No benchmark semantics were added or changed during this integration.

ACC counts unparsed labeled responses as incorrect; precision/recall/TNR use
parsed A/B labeled responses only and report unparsed counts separately.

## CPU validation

```bash
python -m unittest discover -s tests -p test_pool_spatial.py -v
python runMyPruningPoolSpatial.py --self-test --spatial-mix-weight 0.02 \
  --output-dir output/pool_spatial_self_test_01
```

These checks use synthetic features/mock modules, no model downloads or GPU.
Real model accuracy, runtime, and zero-rate GPU equivalence must be validated
on the server.
