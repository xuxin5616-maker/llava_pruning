# MyPool + attention distillation

`run_pool_distill.py` combines the supplied `runMyPruningPoolSpatial.py` with
the existing block2 side predictor. This is a new results-only entry; the
standalone MyPool and `run.py --method attention_predictor` remain available.
No training or checkpoint conversion is performed.

Default sweep: whole-image LLM-layer-average removal30/50/70%, lambda
0/.25/.5/.75/1, both `anyres_self` and `base_full`. ALL input records run.
MyPool compresses BOTH Base and AnyRes; stage2 protects the remaining Base.

## Command

Sync the full updated project to the server, then run in its root directory:

```bash
python run_pool_distill.py \
  --gpu 5 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --prune-rates 30 50 70 \
  --lambdas 0 0.25 0.5 0.75 1 \
  --selection-modes anyres_self base_full \
  --baseline-acc 91.18 \
  --output-dir output/pool_distill_30_50_70_01
```

No edits to training `method.json` are needed. The checkpoint argument is the
existing best.pt, not a new training directory. All30 configurations execute
as isolated sequential child processes on this one GPU; each loads the model
once. Their allocators and side-branch residency are independent. Lambda0 is
intentionally measured for both sources as a duplicate endpoint control.

MyPool defaults are the CURRENT supplied script's defaults: redundancy layer26,
VisionZip attention layer26, T=.05, spatial prior mixing0. Use
`--redundancy-layers 21 --visionzip-layers 21 --allocation-temperature 0.1
--spatial-mix-weight 0.02` only if that is the MyPool setting you want to test.
The spatial weight is independent of the new stage-allocation lambda.
Backbone FP16/FlashAttention2, greedy generation, original prompt templates,
and existing side precision remain unchanged.

## Exact meaning of lambda

Let r be the desired average fraction, D the LLM depth, k=2, N0 the ORIGINAL
native packed image count (Base + AnyRes after native unpad/max9 + newlines),
N1 the count entering LLM, N2 the count after block2. Then:

```text
r_actual = 1 - [k*N1 + (D-k)*N2] / (D*N0)
         = (1-N1/N0) + (D-k)/D * (N1-N2)/N0
target pre contribution    = (1-lambda)*r
target distill contribution= lambda*r
N1 = floor(N0 * [1-(1-lambda)*r])
N2 = floor([D*N0*(1-r)-k*N1]/(D-k))
```

N2 is bounded by N1 for sub-token rounding; lambda0 explicitly sets N2=N1.
Budgets are integer and differ from the continuous target by less than2/N0
in fraction (200/N0 percentage points). Target and realized values are both
exported. The denominator never becomes the stage1 remaining count.

For D28, r30%, ignoring integer rounding:

| lambda | pre contribution | distill contribution | stage2 deletes from remaining |
|---|---:|---:|---:|
| 0 |30pp|0pp|0%|
| .25 |22.5pp|7.5pp|10.42%|
| .5 |15pp|15pp|19.00%|
| .75 |7.5pp|22.5pp|26.20%|
| 1 |0pp|30pp|32.31%|

The original MyPool CLI rate refers to RAW encoder content, not the native
packed count. The composed method consequently adapts its integer budget:
reserve MyPool's raw row newlines, divide remaining content slots between
Base and aggregate AnyRes in proportion to their raw capacities using exact
Hamilton rounding, then use MyPool's unchanged redundancy/probability/capacity
allocator to distribute the AnyRes total between crops. Dominant selection,
54/64 dominant split, contextual merging, projector and anchor-row packing
are reused. Lambda0 therefore means MyPool-only under THIS common packed
budget, not a claim of identity to standalone `--prune-rate 30`'s raw budget.

Lambda0 never loads or evaluates the side predictor. Lambda1 never executes
MyPool scoring/merging: it uses the existing dense predictor path and native
packing. Targets incompatible with protected Base/newlines/crop minima fail
explicitly; no image is skipped and no target is silently relaxed.

## Sparse scoring and source provenance

For each retained token, track view, raw anchor, all merged source indices,
and actual PRE-projector coefficients. A contextual feature is
`anchor + mean(other assigned features)`: anchor coefficient1; others1/m.
These are not assumed to sum to1. Normalize them only for scalar averaging.
The projector can be nonlinear; source-based scalar transport is an explicit
scoring rule, not a claim that projected/LLM features combine linearly.

Both scoring branches consume the entire COMPRESSED H2 and its actual IDs.
`anyres_self` ranks remaining AnyRes by its own side-predicted attention.
`base_full` requests the surviving Base probabilities, backprojects their
mass to raw27x27 Base sources with normalized merge coefficients, samples
the resulting map bilinearly at raw AnyRes source centers using actual
resize/padding geometry (padding outside image content scores zero), then
takes each AnyRes token's normalized source-coefficient-weighted mean.
This uses all merged source locations, not just the contextual anchor.
All remaining AnyRes compete in ONE top-k. Base and newlines are protected.

Stage1 uses MyPool's consecutive compressed-sequence position IDs. Stage2
keeps those IDs and physically compresses existing KV caches; it does not
invent unpruned sequence IDs for merged tokens. The original full-grid
predictor selector is never handed an invented dense sparse layout.

The existing predictor was trained on DENSE H2. Compression/merging changes
its input distribution. The code intentionally reuses that checkpoint for
this experiment; accuracy/attention agreement after MyPool are not guaranteed
by the earlier dense-input training results. Fine-tuning on sparse examples
would be a separate training change.

## Outputs and timing

```text
output/pool_distill_30_50_70_01/
  comparison.json                       # commands, status, budget definitions
  summary.csv                           #30 rows, three rates x two sources x five lambdas
  prune_30_anyres_self_lambda_0.5/
    run.json                            #configuration / inference / timing
    predictions.jsonl                   #ID, GT, answer, Lat and memory
    budgets.csv                         #per-image N0/N1/N2, realized rates
    metrics.json
    summary.csv
  ...
```

This sweep saves NO images or feature/attention arrays. It reports
ACC/PRE/Recall/TNR/BACC/F1/relative ACC, unparsed counts, Lat mean/p50/p95/std in ms, peak allocated
and reserved GiB, allocated/device-total percentage, full generation duration,
and per-image actual contributions. Runtime progress prints elapsed and ETA.
No input count limit is imposed. Default10 warmups per trial are excluded.

### Classification additions and the unpruned reference

Both child and root `summary.csv`, and each `metrics.json`, now include:

| Column | Definition | Units |
|---|---|---|
| `f1` | `2*TP/(2*TP+FP+FN)`, defect positive class | fraction0..1 |
| `bacc` | `(Recall+TNR)/2` | fraction0..1 |
| `baseline_accuracy` | supplied unpruned reference ACC | fraction0..1 |
| `relative_accuracy_pct` | `100*accuracy/baseline_accuracy` | percent |

These use the SAME shared calculation as `ablate_attention_predictor.py`.
F1/BACC use parsed labeled A/B answers; unparsed labeled answers remain errors
in ACC and are reported separately. Undefined denominators stay null/empty.
Relative ACC can exceed100%; it is accuracy retention, not a Lat/Mem/token ratio.

The earlier additions existed only in the ablation entry; the hybrid entry
had not called them. Default `--baseline-acc 91.18` reuses that entry's supplied
reference, NOT a new measured zero-pruning run. If your corresponding original
model/dataset/prompt baseline differs, pass its ACC in percent. The reference
and its source are recorded in root `comparison.json` and each `run.json`.

Existing hybrid outputs can receive the missing columns without inference:

```bash
python run_pool_distill.py \
  --report-only output/pool_distill_30_01 \
  --baseline-acc 91.18
```

This reads exact confusion counts and ACC from completed trials' `metrics.json`,
updates both levels of CSV/JSON reports, and preserves predictions, budgets,
timings and failed/running statuses. No model/data paths or GPU are needed.
Incomplete trials are excluded, never marked successful. A moved/copied root
is supported; original absolute child paths are not followed outside the root.

50/70 targets remain subject to the existing feasibility checks. For example,
with one729-token AnyRes crop, Base729 and27 protected newlines, lambda1 has
maximum whole-image layer-average removal about45.58% at block2 of28. The code
stops with an error for an impossible trial; it does not prune protected Base,
lower the requested rate, or skip test images to produce a misleading result.

Lat uses the shared synchronized resident-input->first-selected-token observer.
It includes vision/projector/packing, MyPool scoring/allocation/merging and
the side predictor/physical pruning. Loading, CPU preprocessing, H2D transfer,
and file saving are excluded. Mem is peak PyTorch allocated/reserved over full
answer generation, including resident backbone/side weights, not NVML or GPU
compute utilization. Native side loading occurs before measured generation.

Tests cover exact budgets, real synthetic MyPool merging/provenance,
source-weighted scoring, actual tiny-model autoregressive generation with
sparse pruning/KV handling, lambda endpoints, and isolated launcher outputs.
CUDA/7B accuracy and resource results still require running on the server.
