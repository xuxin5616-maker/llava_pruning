# llava_pruning — LLaVA visual-token pruning

A small, single-image inference scaffold for LLaVA OneVision/Qwen2 checkpoints. Choose **FastV** (`--method fastv`, the default) or **ViCo / PyramidDrop** (`--method vico`). It loads the model once and evaluates the configured rates sequentially on the same GPU. Both methods use the original decoder forward at rate 0. Project and Python package names are now `llava_pruning`; import and redraw commands use this new name.

## ViCo quick start

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --roi-mode anyres_max_9 \
  --method vico \
  --method-config configs/vico.json \
  --save-prune-vis --save-attention-vis \
  --no-sample \
  --output-dir output/vico_anyres_01
```

`--method-config` may be omitted: it selects `configs/<method>.json` automatically. To run FastV, change `--method vico` to `--method fastv` and the config to `configs/fastv.json` (or omit it). Do not pass the FastV configuration to ViCo. Choose an empty/new output directory for each experiment. All records in the input JSON are evaluated; no 100-image limit is imposed by `run.py`.

ViCo defaults to pruning **after layers 8, 16 and 24**. The sweep's 0/10/.../90 values mean **final cumulative pruning percentages**, distributed geometrically over the three boundaries. For a final 90% target, stage retention is approximately 46.42%, 21.54%, 10% of the original packed image span. The author example `[0.5,0.25,0.125]` instead corresponds to a final 87.5% pruning rate. The adapter physically shortens hidden-state sequences and uses stage-specific KV-cache lengths, while retaining FP16 + FlashAttention2 and independently computed ranking scores. It is an inference adaptation, not a claim of reproducing the author's accuracy/speed numbers.

When enabled, sample visualizations are saved for 10/30/50/70/90 only. Each sample still has only `comparison.png` and `attention_overlay.png`, now with one labelled row per ViCo pruning boundary. Previously removed tokens are gray in later attention rows, not assigned fabricated scores. All 28 layers' counts (including unchanged layers and rate 0) are stored in `prune_XX/layer_tokens.csv`, without duplicating those rows in prediction JSON. `run.py` still draws ACC/PRE/Recall/TNR curves with a 50%–100% y-axis after the sweep. See [the ViCo adapter specification](docs/vico.md) for token-count scope, rounding, position conventions, limitations and tests.

For a baseline-only ViCo run, copy `configs/vico.json`, set `prune_rates` to `[0]` and `visualize_rates` to `[]`, and pass that file. `configs/baseline.json` is a **FastV** configuration.

## Inputs

The CLI accepts a JSON list (`.json`) or one object per line (`.jsonl`). For example:

```json
{"question_id":"000000108","image":"000000108.png","text":"Is there any defect in this image? If yes, say 'yes', otherwise say 'no'. Then describe it.","gt":1,"origin_path":"screw/test/thread_top/005.png","mask":"musc/000000108.png","musc_scores":0.6233050227165222}
```

`--data-root` is the root for relative paths. A bare `image` filename is searched first at `<data-root>/<filename>` and then at `<data-root>/imgs/<filename>`; `mask` is resolved at `<data-root>/<mask>` and may be a grayscale image, `.npy`, or `.npz` with an `anomaly_map` array. `origin_path` determines the MVTec category (`screw` here). `musc_scores` and other extra fields are ignored, not treated as pruning scores. IDs stay strings, preserving leading zeroes. `bbox` may be supplied later as `[[x_min,y_min,x_max,y_max], ...]`, following the legacy crop helper's inclusive coordinates; when both mask and bbox are present, mask takes precedence in this scaffold.

For known MVTec categories, `--prompt-version v0|v1|v2|v3` selects the original LLaVA MVTec templates; the record's `text` is a fallback only for unknown categories. This deliberately matches the old `config:vN` experiment path, and means the example's `text` is **not** the model prompt for `screw`. The selected prompt version is recorded once in `run.json`; resolved prompts are not repeated in prediction JSON.

## Run

Inference now matches the restored local LLaVA loader: **FP16 (`torch.float16`) + FlashAttention2**, greedy decoding, and at most **512 new tokens** using LLaVA's context-budget calculation. TF32 settings are left at the environment's values, as in LLaVA. Vision-tower loading follows the original loader (including its lazy-loading behavior); actual model/vision dtypes, attention implementation and library versions are recorded in `run.json`. There is no silent SDPA/BF16 fallback.

Use the **same environment as the original LLaVA**, including its `flash-attn` build, Torch, Transformers, CUDA and image libraries. `requirements.txt` pins the shared Python dependencies but does not install the platform-specific FlashAttention extension. Run on one visible GPU for the initial comparison:

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov \
  --input-json /path/to/questions.jsonl \
  --data-root /path/to/dataset \
  --prompt-version v0 \
  --method fastv \
  --method-config configs/fastv.json \
  --roi-mode randomroi \
  --save-prune-vis \
  --save-attention-vis \
  --output-dir outputs/experiment_01
```

`--roi-mode randomroi` uses `mask`, then `bbox`, then random crops if neither exists. `--roi-mode randompatch` ignores both annotations and always chooses random crops. Both modes use the LLaVA `randomroi` image packing; they differ only in crop selection. `--roi-mode anyres_max_9` ignores masks/boxes and, like original LLaVA's `--overwrite_image_aspect_ratio`, changes the aspect-ratio setting **without overwriting the checkpoint's merge type**. For the pure-anyres checkpoint discussed here this is `spatial_unpad`; it is not the `anyres_max_9_randomroi` hybrid. The ROI mode and seed are recorded once in `run.json`; per-sample crop coordinates are no longer exported.

### Feature-score visualization only (`visualize_scores.py`)

This independent script does **not prune tokens, run LLM generation, or change
`run.py`**. It keeps the checkpoint's original AnyRes tile preprocessing
and observes the **1-based SigLIP block outputs 7, 14, 21, 26**, plus the actual
MLP projector output. The existing loader removes the last of the checkpoint's
27 SigLIP blocks, so block 26 is the actual projector input; its hidden state is
taken **before** SigLIP's final `post_layernorm`, just like the existing tower.
Models with a different active block count are rejected instead of relabelled.

```bash
CUDA_VISIBLE_DEVICES=5 python visualize_scores.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --output-dir outputs/feature_scores_01
```

Default: **all input images**. Add `--limit 1` for an optional one-image smoke
test; there is no fixed 100-image cap. Do not pass `--method`, `--method-config`,
`--roi-mode`, prompt, sampling, or pruning-rate options to this script. Masks,
bboxes and question text are unused and need not exist. Use a new/empty output
directory and exactly one visible GPU. It reuses the current full checkpoint
loader (including the original dtype/FlashAttention configuration), so decoder
weights are still loaded and the same environment/VRAM capacity is required,
but **only `encode_images()` is executed**, never the LLM decoder.

At each stage, **first L2-normalize every token along its channel dimension**:
`u = x / ||x||_2`. Then compute the view means and scores:

- Global score: `-cos(u, mean(all unit tokens of the reference view))`.
- Local score: `-cos(u, mean(all unit tokens of that same tile))`.

**Global reference now defaults to `--global-reference center-crop`.** Keep the
center 75% of the original width and height (56.25% area before pixel rounding),
resize that crop back to the original size with bicubic interpolation, then use
the usual square Base resize and SigLIP preprocessing. Crop dimensions are
`max(1, floor(0.75 * dimension))`; centered offsets round down. For 1024x1024,
the crop is `(128,128,896,896)` (768x768), then enlarged to 1024x1024.
Only the reference view is replaced: AnyRes tiles, geometry, Local scores and
layer averaging continue to use the **full original image**. Each stage uses
its own newly encoded reference mean; this is not a crop of old features.
Reference and tiles share one encoder batch; no extra encoder pass is needed.
Use `--global-reference base` for the previous full-image reference. This option
does not affect `run.py` or `runVFlowOpt.py`. Figures identify the cropped
reference, and each sample's metadata records the exact crop/resize geometry.
Old score-only caches cannot produce the new reference scores: rerun the encoder
into a new output directory. Offline redraw retains the cache's reference mode.

This is normalization **before averaging**, not just cosine normalization of
the already-averaged vector. Token magnitude no longer weights its direction
in the reference mean. Zero vectors cannot become unit vectors: they stay zero
in the mean and have undefined (NaN) cosine scores; a zero reference mean is
also undefined. This applies independently to every captured stage, including
the projector output. Only the detached scoring path is changed, not the
encoder/projector forward outputs passed to subsequent layers.

New caches and JSON metadata record
`feature_normalization: l2_per_token_before_mean`. Old `scores.npz` contains
scores, not full token vectors, so it **cannot be converted** to this scoring
rule offline: rerun `visualize_scores.py` into a new output directory, then
point `visualize_patch_means.py` at those new results. Existing caches remain
readable but retain their old scoring semantics.
`runVFlowOpt.py` is a separate local feature-residual probe (see below); it does
not reuse these normalized Global/Local cosine scores.

Means use **all encoded tokens**, including padding-context tokens; padding is
excluded only from the displayed image. Both means are computed in that stage's
own feature space, including projected Base/tile features for the final stage.
Scores use detached FP32 values without modifying the original forward tensors.
Higher values mean greater dissimilarity, **not an anomaly probability or LLM
attention**. This is the explicitly requested direct reference-to-tile comparison,
not the paper's interpolated Base score map and not a complete GlobalCom2 method.

Each `000001_<question_id>/` contains:

- `scores_overview.png`: five rows (the five stages); columns show the reference view,
  original image + numbered tile boundaries, global-score overlay, local-score
  overlay. GT is labelled; no prediction is invented.
- `scores_tiles_01.png` (and further numbered pages if needed): five rows,
  global/local overlays side by side for each actual tile; four tiles per page.
- `scores.npz`: small FP32 `global_scores` / `local_scores` arrays shaped
  `[5 stages, number of AnyRes tiles, tokens per tile]`, plus `stages`. Both tile
  and token order are row-major. No full features/attention matrices are saved.
- `metadata.json`: geometry, feature shapes, shared color limits and filenames.

The output root's `run.json` records scoring/display definitions and loader
configuration. These observations are **before AnyRes unpadding, `max_9` feature
downsampling and newline insertion**: they depict actual encoder tiles, not a
post-packing LLM image sequence. `anyres_max_9` does not mean the encoder always
receives exactly nine tiles.

All panels/pages of one image share one raw-score color scale (`--color-scale
sample`, the default), with no per-layer/per-tile min-max normalization. For
the same limits across different images use `--color-scale fixed` (`[-1, 1]`).
JET runs from blue/low through cyan, green and yellow to red/high; overlay opacity is 0.70. Geometric
padding, undefined cosine and pixels outside patch support are gray, not zero.
For patch14/384, the 27x27 grid covers 378x378 pixels: the remaining six-pixel
bottom/right strip of **each tile** is not falsely assigned a score. Token
scores are expanded using nearest patch support, not presented as independently
measured per-pixel scores. Original images are never modified.

### Offline AnyRes tile-mean visualization (`visualize_patch_means.py`)

**One-command alternative:** add `--display-mode patch-means` to
`visualize_scores.py` to generate the same tile/layer-mean figure immediately
after each sample is encoded. No separate offline run is needed:

```bash
CUDA_VISIBLE_DEVICES=5 python visualize_scores.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --display-mode patch-means \
  --global-reference center-crop \
  --output-dir outputs/feature_scores_center75_means_01
```

This mode writes only `patch_means.png` per sample (top: layers 7/14/21/26;
bottom: original, mean 7+14, mean 7+14+21, mean 7+14+21+26), plus
`patch_means.csv`, the unchanged-format full `scores.npz`, and `metadata.json`.
Local and projector scores remain in the cache but are excluded from the
figure and averages. The exact offline selection/aggregation/renderer is
reused; there is no extra encoder pass or LLM generation. Per-token L2 before
view means remains enabled.

Without `--display-mode`, `tokens` retains the original PNGs and defaults to
`--color-scale sample`. `patch-means` defaults to `--color-scale fixed` (0..1),
like the standalone script. Explicit `--color-scale sample` or `fixed` overrides
either default. These are display options, not changes to feature extraction.
Use a new/empty output directory. Keep `visualize_patch_means.py` alongside the
entry script because the new mode imports its CPU helpers.

Reuse the previous feature-score results **without loading or running any model**.
This standalone file needs only NumPy, Pillow and Matplotlib: no Torch,
Transformers, FlashAttention, CUDA, model checkpoint or input-question JSON.
It computes an arithmetic mean of the saved **Global token scores** in each
AnyRes tile at SigLIP layers 7/14/21/26. **Local and projector are excluded**.
Three extra rows average the **raw tile means** of the first 2, 3 and 4 selected
layers. All seven score panels are then scaled with **`(raw_score + 1) / 2`**, mapping
the theoretical cosine-score range **[-1, 1] to [0, 1]**. This is the same fixed
linear transform for every image/layer/tile, not per-layer min-max normalization.
It does not average hidden features or recompute cosine similarity.

To check the **actual uploaded script**, run
`python visualize_patch_means.py --version`: it must print `4.2-global-horizontal`.
`--help` must list `{fixed,sample}`. The former `layer` normalization mode was
removed. Runs also print/save the script version
and selected color scale, so a previous copy can be distinguished immediately.

```bash
python visualize_patch_means.py \
  --results-dir outputs/feature_scores_02 \
  --color-scale fixed \
  --output-dir outputs/feature_scores_02_patch_means_horizontal
```

Alternatively, edit `RESULTS_DIR` at the top of the file and run
`python visualize_patch_means.py`. If `OUTPUT_DIR` is empty, output defaults to
a sibling directory named `<results directory name>_patch_means`. All available
sample folders are processed, or `--results-dir` can name one sample folder.
Use a new/empty output directory outside the source result tree. Existing
score contents/images/metadata are never overwritten. The only source change is
restoring a score archive's filename by removing an appended `.log` (see below).

Each sample needs its original `metadata.json` and `scores.npz`. A filename with
an appended `.log` is **automatically renamed after the archive is validated**:
`scores.npz.log` becomes `scores.npz` in the same sample folder. No data is
rewritten, and each rename is printed in the terminal. Discovery accepts
`.npz`, `.npz.log`, `.npy` and `.npy.log`; the reader checks binary content, not
the extension. The contents must still be the original archive containing
`global_scores` and `stages`; `local_scores` is no longer required or read.
An arbitrary single NPY array or a
text log is not that archive and is rejected rather than guessing its layout.
Both the original five-stage archive (with projector) and a four-stage archive
containing only SigLIP 7/14/21/26 are accepted. Layers are matched by their saved
names rather than blindly slicing the first entries. Projector is excluded from
all calculations and output rows. Neither Local nor projector is deleted from
the cache; this change only affects the offline redraw script.
Two candidate archives in one sample folder are rejected as ambiguous. An
existing destination file is never intentionally replaced; a name conflict
stops processing. A normal `.npz` filename is left unchanged on subsequent runs.

Original images are read from the saved paths for the overlay. If the images
have moved, keep the old result-root `run.json` and add `--data-root /new/data/root`
(or edit `DATA_ROOT` at the top); the original relative directory structure is
preserved, including `imgs/`. No vision encoder preprocessing is rerun.

Each new sample folder contains:

- `patch_means.png`: a compact **two-row, four-column** layout (2400 x 1350 px).
  The top row shows **SigLIP 7 / 14 / 21 / 26**; the bottom row shows **Original /
  mean of first 2 / mean of first 3 / mean of first 4**. The three mean panels are
  side by side, not additional rows. Original appears once, with tile boundaries
  but **no patch numbers**. Each heatmap retains tile IDs and **scaled mean scores
  (0..1)**, with one JET color per visible tile at 70% opacity over the image.
  All heatmaps share a horizontal colorbar.
  These are tile-level
  aggregates, not new per-pixel measurements; unobserved six-pixel token borders
  receive the same **tile-level** color, not invented individual token scores.
- `patch_means.csv`: raw `global_mean`, scaled `global_score_01`, row/column, visible flag,
  `source_layers`, `layer_count`, and valid/total token counts. For composite rows,
  these counts sum token observations across the selected layers; they are **not
  weights** for averaging layer scores. No Local columns are saved. The raw column
  remains unchanged for audit; `global_score_01` matches the displayed values.
- `metadata.json`: source-file paths, image, geometry, display mode/color limits,
  layer-group definitions, version, `figure_layout`, and explicit `score_scaling` formula/ranges.

The root `run.json` records aggregation/display settings and completion. Means
include all finite saved token scores in each tile, **including padding-context
tokens**. NaN is excluded and counts are reported; entirely undefined means are
gray and blank in CSV, not zero. Infinite/out-of-range values fail validation.
Image padding is cropped out of the overview; entirely invisible tiles remain
in CSV but do not set the optional sample-wide color limits.

Score/CSV order remains unchanged ("first" refers to the selected layers, not blocks 1..4):

1. SigLIP 7
2. SigLIP 14
3. SigLIP 21
4. SigLIP 26
5. `mean_first_2`: `(score_7 + score_14) / 2`
6. `mean_first_3`: `(score_7 + score_14 + score_21) / 3`
7. `mean_first_4`: `(score_7 + score_14 + score_21 + score_26) / 4`

Each `score` above is a **raw per-tile mean**, before the fixed linear transform.
Layers have equal weight. If any constituent layer has no valid tile mean, that
composite tile is undefined (gray / blank CSV), not silently averaged over fewer
layers. Partial NaNs within a layer still use the original finite-token mean.

Default `--color-scale fixed` uses the scaled **[0, 1]** color range for all panels
and all images: raw -1 -> 0, raw 0 -> 0.5, raw 1 -> 1. The same raw score has the
same heatmap color. This relabels the previous [-1, 1] display; it does **not**
increase color contrast or independently stretch each layer's observed range.
These scores are not probabilities. Raw overshoot up to 1e-6 (roundoff tolerance)
is clipped to the endpoints only after averaging/scaling; larger errors fail.

Optional `--color-scale sample` sets shared color limits to the finite, visible
scaled Global means across all seven score panels **of one image**. It can improve contrast,
but colors are then not directly comparable across different images. Both
modes show scaled tile labels/colorbar units and preserve both raw and scaled
CSV columns. The optional palette-range adjustment does not further transform scores.
Averaging can dilute a small hotspot; the original token scores do not change.

### Local SigLIP feature residuals (`runVFlowOpt.py`)

Version `3.1-integrated-feature-residual` integrates the supplied local-residual
script. This is a custom diagnostic, **not the original VFlowOpt method**, not
attention, entropy, a pruning decision or an anomaly probability.

For each **1-based SigLIP block 7 / 14 / 21 / 26**, use its output **before the
final post-layernorm**. Base and each AnyRes crop retain their native **27 x 27**
token grid (384-pixel input, 14-pixel patch embedding). For each center token,
compute the mean of valid neighboring **raw feature vectors**, excluding the
center, in a **3 x 3 / 5 x 5 / 7 x 7** window:

- Default `--metric l2`: `||center - neighbor_mean||_2`.
- Optional `--metric cosine`: `1 - cosine(center, neighbor_mean)`.
- **No feature L2 normalization before averaging**, no global-mean reference,
  no Softmax. This does not modify `visualize_scores.py` or its Global score.
- AnyRes neighborhoods cross crop boundaries in the stitched logical token
  grid; Base uses its own separate neighborhood grid. Only fully content-covered
  patches are valid centers/neighbors. Edges use available neighbors; no
  reflection/zero padding. No-neighbor centers have NaN scores; zero-norm cosine
  is NaN, while zero-vector L2 can be defined.
- Logical crop adjacency bridges the six unencoded edge pixels. These pixels,
  padding, partial-padding patches and undefined scores are gray in the plots.
  Independently encoded crops have different contexts and positional resets;
  residuals at crop boundaries are not necessarily defects.

Fresh dataset input (all records by default; no implicit 100-image limit):

```bash
CUDA_VISIBLE_DEVICES=5 python runVFlowOpt.py \\
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \\
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \\
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \\
  --metric l2 \\
  --color-scale raw \\
  --output-dir outputs/feature_residual_l2_01
```

Alternatively replace `--input-json ...` with
`--results-dir outputs/feature_scores_02`. This reuses image paths and geometry,
**not cached scalar scores**: SigLIP must run again because the old NPZ does not
contain the needed feature vectors. Keep `--data-root` if images were moved.
The checkpoint may also be taken from the saved `run.json` or the script's
`MODEL_PATH`. `RESULTS_DIR`, `OUTPUT_DIR`, and `DATA_ROOT` can be edited at the
script's top. Use a new/empty output directory separate from the source results.
`--limit N` is optional. `--version` checks which script is installed.

Only SigLIP embeddings and encoder execute; **no projector, LLM forward,
generation, training or pruning**. The existing loader still loads the LLM
weights and needs GPU memory for them; its model precision and attention mode
are unchanged. A single visible GPU is required. Scoring uses CPU float64
accumulation in channel chunks and saves float32 scalar scores; GPU feature
outputs are read, never replaced. This probe is not a speed benchmark.

Each sample produces **four PNGs**:
`siglip_7_residual_l2_raw.png`, `siglip_14_residual_l2_raw.png`,
`siglip_21_residual_l2_raw.png`, `siglip_26_residual_l2_raw.png` (suffixes follow
the selected metric/scale). Each figure has a Base row and an AnyRes row, each
showing its reference image and the three window sizes. JET overlays retain
alpha 0.70. GT is shown if available; there is no LLM prediction.

`--color-scale raw` displays actual residual values. `minmax` optionally maps
scores to [0,1], **jointly across Base, all AnyRes crops and all three windows
within one image/layer**, never separately per crop; constant maps become 0.5.
Layers/images use separate colorbar ranges, so equal colors across different
figures do not imply equal raw values. All six heatmaps within one figure use
the same range, including Base versus AnyRes.

Numeric output: `residuals.npz` (both raw metrics, validity masks, neighbor
counts, layers and windows), `crop_summary.csv`, sample `metadata.json`, root
`run.json`. Base scores are [4 stages, 3 windows, 729 tokens]; tile scores
are [4 stages, 3 windows, crops, 729 tokens]. Raw scores are preserved even when
min-max display is chosen; full feature vectors are not retained.

Deploy `runVFlowOpt.py`, `feature_residual_math.py`, and
`feature_residual_capture.py` together, inside the existing project (the
current preprocessing/geometry helpers are still required). The pre-integration
pixel-entropy script is preserved as `run_pixel_entropy.py`; run its `--help`
for its legacy CPU-only options. Old `--entropy-mode` / Softmax options do not
apply to the new feature-residual entry point.

### Experimental mode: anyres without Base (`ex` branch)

Use `--roi-mode anyres_only` to remove the global Base view while retaining the **same high-resolution tile preprocessing and packing rules as `anyres_max_9`**. Existing modes are unchanged. Base is neither resized/preprocessed nor passed through the vision encoder; every encoded view is an anyres tile, including the single-tile case. Masks/bboxes do not select crops in this mode (normal JSON/path validation still applies).

The checkpoint's grid pinpoints, tile ordering, unpadding, `max_9` downsampling rule and structural row-newlines are retained. `max_9` is not a requirement to always input exactly nine tiles. Supported checkpoint merge types are `spatial_unpad` and `spatial_unpad_add_newl`; the latter retains its extra final newline. Other merge types are rejected rather than silently replaced. The resulting visual sequence is the original **anyres block only**, with no Base prefix. With fixed encoded tile features, it matches the original `base_first` packing after removing the Base block. The sequence is shorter, so subsequent text position IDs shift normally; this is not a position-preserving mask of the Base tokens.

Both FastV and ViCo rank/prune the remaining packed anyres span. Existing newline handling, FP16, FlashAttention2, independent attention scoring, prompts and generation settings are unchanged. Rate 0 disables pruning but still has **no Base**, so it is a different input baseline from `anyres_max_9`. Configured pruning percentages now refer to the remaining sequence, not to the former Base+anyres total. `run.json` records `roi_mode: anyres_only`, `base_view_count: 0` and the packing description. Keep the default `--image-token-order base_first` (omit the option); `anyres_first` is not meaningful without Base and is rejected.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --roi-mode anyres_only \
  --method fastv \
  --method-config configs/fastv.json \
  --no-sample \
  --save-prune-vis --save-attention-vis \
  --output-dir outputs/anyres_only_fastv_01
```

For ViCo, change both `--method vico` and `--method-config configs/vico.json`, using a new output directory. Rate sweeps (default 0--90), selected visualization rates, timing flags and whole-input evaluation remain unchanged; there is no 100-sample restriction. `ex.py` is not changed.

`comparison.png` has just **Original / High-resolution (anyres only)**, with actual spatial-token and newline counts. There is no invented Base panel or duplicate Combined panel. `attention_overlay.png` contains only the high-resolution source and its attention overlay, not a Base column. ViCo still stacks pruning stages with gray for previously removed tokens; both PNGs keep the `GT: ... | Pred: ...` banner. Original source images in the figures are for reference, not extra model inputs. Saved JSON remains compact and lists only the actual anyres view.

### Experimental mode: three Base copies (`ex` branch)

Use `--roi-mode ex_base_copy` at the same CLI level as `anyres_max_9`. The input is **three total views: the original Base plus two identical Base copies**, not an original plus three extra copies. Each is the full-image Base preprocessing used by anyres (square resize, then the unchanged vision processor). There is no anyres grid selection, ROI crop, mask/bbox-guided selection, or pixel-level panorama. Input JSON/path validation is unchanged.

All three image tensors are passed through the vision encoder/projector as separate views in one batch. Their full, row-major token sequences are concatenated:

```text
[Base 1 tokens] [Base 2 tokens] [Base 3 tokens] [one final image_newline]
```

No pooling, unpadding, row-newline insertion or resizing of the feature grids is applied. With the 384px / patch14 SigLip tower, each view has 27 x 27 = 729 spatial tokens: **2187 spatial tokens + 1 structural token = 2188 image tokens**. The last structural token retains FastV's existing optional tail-token protection; it is not a fourth view. The dedicated merge name `spatial_unpad_ex_base_copy` includes `unpad` solely to load the existing learned newline parameter, not to remove patches. `run.json` records this packing and the three-view count.

This mode uses the default `--image-token-order base_first`; do not combine it with `anyres_first`, which remains restricted to actual anyres inputs. Each copy occupies different LLM sequence positions, so identical pixels do **not** guarantee identical attention or pruning decisions. Ranking is over the combined visual sequence, not a separate quota for each copy. FP16, FlashAttention2, independent ranking attention, prompts, generation, timing flags and method configs are unchanged. At rate 0 the decoder still bypasses pruning, but the three-copy input is intentionally different from anyres, so its answers need not match the anyres baseline.

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --roi-mode ex_base_copy \
  --method vico \
  --method-config configs/vico.json \
  --no-sample \
  --save-prune-vis --save-attention-vis \
  --output-dir output/ex_base_copy_vico_01
```

For FastV, use `--method fastv --method-config configs/fastv.json` and a different output directory. Both methods run all records at the configured rates (default 0--90); visualization rates remain 10/30/50/70/90. No `ex.py` scheduling or 100-sample limit is added. Existing modes/defaults and both method config files are unchanged.

With visualization enabled, `comparison.png` shows **Original / Base 1 / Base 2 / Base 3 / Combined**, including separate spatial-token removal counts. Combined black means all covering copies removed that area, not the overall token pruning percentage. `attention_overlay.png` shows the three Base sources and their separate attention overlays, using a shared scale within each stage. ViCo stacks the configured pruning stages in each PNG; already-removed tokens are gray in later attention rows. Both files retain the `GT: ... | Pred: ...` header. Exactly the same two PNG types and compact JSON/count outputs are saved; no extra per-copy image files are created.

The Base-copy masks and overlays leave uncovered convolution margins unchanged: at 384px / patch14, the 27 x 27 anchors cover 378 x 378 processed pixels, not the last six rows/columns. Blackouts depict spatial anchors, not exact removal of pixel information from contextualized features.

For a view-order comparison, add `--image-token-order anyres_first` to an existing `--roi-mode anyres_max_9` command. The default, `--image-token-order base_first`, keeps the original order. Both `run.py` and `ex.py` accept the option; it applies at **all pruning rates, including 0%**. Swapping requires `spatial_unpad` or `spatial_unpad_add_newl`. With `spatial_unpad`, the sequence becomes `[anyres except its last newline] [base] [the original last newline]`; with `spatial_unpad_add_newl`, it becomes `[anyres including row newlines] [base] [the extra final newline]`. Keep `preserve_image_newline: true` if using the default FastV protection. Image pixels, token counts, normal position-ID generation and pruning rules are unchanged. The selected order is saved in `run.json` and visualization `decisions.json`; plot offsets and row-newline records follow the order automatically. Use separate output directories for the two runs. Returning to the original order only requires omitting the option or choosing `base_first`.

Decoding is now **greedy by default**, matching the user's current LLaVA `do_sample=False`. Existing `--no-sample` commands remain valid; use `--sample` only to opt back into sampling (temperature 0.2, top-p 0.7). Unless `--seed` is specified, each run generates a new seed; the actual seed and decoding mode are recorded in `run.json`. In `randompatch` mode the seed controls crop selection; greedy decoding does not disable random crops.

### Include or exclude pruning in the reported time

`--include-pruning-time` is the default: `generation_seconds` is the synchronized wall time of `model.generate()`, including vision encoding, LLM generation and pruning. Omit both timing flags to keep this behavior.

Add **`--exclude-pruning-time`** to subtract separately measured pruning blocks from the reported value. Pruning still runs, with unchanged decisions, precision and attention backend. All decoder weights must be on one device (use one visible GPU, without decoder offloading). `run.json` records `include_pruning_time: false`; each prediction adds only two scalar timing fields:

```text
generation_seconds = generation_with_pruning_seconds - pruning_seconds
```

`generation_with_pruning_seconds` is the full generation time of that **instrumented** call, and `pruning_seconds` is the sum of synchronized wall-time intervals around independent scoring, top-k selection and mask construction/application (FastV), or scoring, selection and sequence gathering (ViCo). Rate 0 has no pruning blocks and subtracts zero. Normal downstream position/cache handling and decoder attention kernels remain included; this is not an estimate of every possible cost caused by pruning.

**Exclusion is a profiling diagnostic, not actual end-to-end speedup.** Synchronizing each measured block changes CPU/GPU overlap and adds overhead, especially with FastV masks on many layers/decode steps. Do not equate the adjusted value to an uninstrumented run's latency. Report default include-mode time for real speed comparisons. Keep `--save-prune-vis` and `--save-attention-vis` off for timing experiments; capture/bookkeeping within a measured pruning routine is part of that routine's interval. Both modes still exclude checkpoint loading, input image preprocessing/tokenization, output decoding, drawing and file writes. `ex.py` remains unchanged and uses include mode.

## Independent attention scoring and baseline check

At nonzero pruning rates the main decoder **stays on FlashAttention2 in every layer**. FastV does not request `output_attentions=True` and does not read returned Transformer attention matrices. A read-only side calculation uses the ranking layer's input normalization and Q/K projections, RoPE and GQA head mapping to recompute only the last valid prompt query against all prompt keys. Projection/RoPE follow the model dtype; QK, softmax and head averaging use FP32 for score stability. These scores do not change the decoder's hidden states or KV cache. This extra computation is not claimed to be bitwise identical to the old eager FP16 attention scores.

At **0%**, the side calculation and pruning mask are bypassed entirely and the decoder directly calls the original `Qwen2Model.forward`. Generation also preserves original LLaVA's mask/position-ID handling and wrapper behavior. Anyres preprocessing is unchanged except for collecting visualization metadata. Prediction JSON keeps the full decoded answer, but no longer exports input/generated token ID arrays.

First run only the baseline (replace paths with your own):

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov \
  --input-json /path/to/questions.jsonl \
  --data-root /path/to/dataset \
  --prompt-version v0 --roi-mode anyres_max_9 \
  --method-config configs/baseline.json --no-sample \
  --output-dir output/anyres_fp16_baseline

python compare_results.py \
  --baseline /path/to/original_llava_answers.json \
  --candidate output/anyres_fp16_baseline/prune_00/predictions.jsonl \
  --output output/anyres_fp16_baseline/comparison.json
```

The comparison aligns `question_id`/`id`, compares complete trimmed `answer`/`conversations -> gpt -> value` strings, reports missing/extra IDs and every mismatch, and exits nonzero on differences. `identical: true` is evidence about those two runs, **not a guarantee across environments or checkpoints**. Compare against a fresh run of the restored LLaVA with the identical checkpoint, `config:v0`, anyres mode and greedy decoding. In particular, verify both scripts open the same image files (`imgs/<name>` versus a same-named file at the dataset root). Do not use accuracy alone: LLaVA's `mean` is a category average and its answer parser differs from this scaffold's metric parser.

For the full 0--90% sweep, omit `--method-config configs/baseline.json` and add the visualization switches as needed. Rate 0 still saves no visualizations.

Run `python -m unittest discover -s tests -v` in the pinned environment (Python 3.10+). The small CUDA/FlashAttention integration test skips without a CUDA GPU/FlashAttention; CPU tests cover independent scores, masks, cache continuation and the zero-rate forward path. Optional direct comparisons with the downloaded LLaVA source also test anyres pixels, packed embeddings and wrapper equivalence; set `LLAVA_REFERENCE_DIR` to that repository root if needed. No test downloads model weights. Do not copy the local `.venv` test environment to the server.

`configs/fastv.json` owns FastV-only settings: `layer`, `prune_rates`, `visualize_rates`, `min_tokens`, and newline handling. By default it sweeps 0, 10, 20, ..., 90%; `prune_00` disables FastV completely. Only 10, 30, 50, 70, 90% generate images, when a visualization switch is enabled. `--save-prune-vis` and `--save-attention-vis` are independent. Visualizations include original/crop-level FastV decisions and a 60/40 JET attention overlay from the ranking layer. The folder layout is `outputs/experiment_01/prune_10/{predictions.jsonl,metrics.json,visualizations/sample_<id>/image_0/...}`. Output directories must be empty to prevent accidental overwrite; choose a new directory for each experiment.

Each image directory now saves only `comparison.png` (`--save-prune-vis`) and `attention_overlay.png` (`--save-attention-vis`). When both switches are enabled there are exactly two PNGs, for anyres, randomroi and randompatch alike. Originals, individual crops, blackouts and binary masks are no longer saved separately. `decisions.json`, predictions and accuracy statistics are retained; previously generated files are not deleted.

Both PNGs now have a short top label such as `GT: Normal | Pred: Abnormal`. GT comes from the input `gt` (0=Normal, 1=Abnormal); Pred uses the same A/B parser as the metrics (A=Abnormal, B=Normal). Missing GT is `Unknown`, and unparseable answers are `Unparsed`. For ViCo this label is the final answer at the current pruning rate, not a separate prediction for each layer. No new flags are needed; only newly generated images are labelled, and existing images are not modified.

For `anyres_max_9`, both FastV and ViCo comparison rows show **Original / Global / High-resolution (anyres) / Combined**. Global and high-resolution panels use their own cumulative spatial-token masks and report their own pruned counts and percentages. The combined panel turns black only where no covering view kept a token: its black area is **not** the overall pruning rate. A footer reports total, spatial and structural-newline token removals separately; newlines are not drawn as pixels. These counts are also saved in `decisions.json`. ViCo repeats the four panels at each pruning boundary. Attention output and ROI-mode layouts are unchanged; existing saved figures are not automatically redrawn.

JSON output is compact by default, with no extra switch:

- `run.json`: creation time and shared run/model/method/decoding settings, seed, input paths and metric definitions.
- `predictions.jsonl`: `question_id`, `image`, `origin_path`, `gt`, `answer`, `prune_rate`, `method`, `generation_seconds` and the effective `max_new_tokens`. Exclude-pruning timing adds only `generation_with_pruning_seconds` and `pruning_seconds`. `generation_seconds` follows the selected timing mode, excludes drawing/file writes, and is not total run wall time.
- `metrics.json`: progress, sample counts and evaluation results; the metric rule is recorded once in `run.json`.
- `decisions.json`: method/stage settings and short scalar token-count summaries only. No attention arrays, patch masks, coordinate/index lists, or duplicate full-layer records are saved.

Images are still drawn from full data in memory before those arrays are discarded. ViCo per-layer details remain in `layer_tokens.csv`. Compact JSON alone cannot reconstruct individual patch masks or attention maps offline. Existing outputs are not modified or deleted; these changes apply to new runs only. `ex.py` and its resource reports are unchanged.

To convert an existing `summary.csv` to Excel without inference, edit `SUMMARY_CSV` at the top of the standalone `summary_to_excel.py` and run `python summary_to_excel.py`, or pass its path:

```bash
python summary_to_excel.py /path/to/summary.csv
```

Only `openpyxl` is required (`python -m pip install "openpyxl>=3.1,<4"` if missing). The script keeps the original columns, row order, incomplete rows and blank values. ACC/PRE/Recall/TNR display as percentages without changing their stored fractions. Output is `summary.xlsx` beside the CSV; repeat exports use `summary_2.xlsx`, etc. An optional `--output /path/to/new.xlsx` selects a new file and refuses overwrite. It does not invent missing timing columns or read predictions to calculate additional results.

Each completed rate prints image-level accuracy and appends a row to `summary.csv`. Each `prune_XX/metrics.json` also tracks partial progress while running (`complete: false` until the rate finishes). Accuracy uses the first answer option only: `A` means defect (`gt=1`), `B` means no defect (`gt=0`); unparsed labeled answers count as incorrect. Records without `gt` are excluded. This is binary classification accuracy, not segmentation accuracy or AUROC. The `accuracy` field is a fraction, e.g. `0.9` means 90%.

`run.py` now automatically saves **one `metrics_vs_pruning.png`** in the output directory after all configured rates finish. It plots ACC, PRE (Precision), Recall and TNR on one set of axes, with pruning percentage on the horizontal axis and a fixed **50%--100%** vertical range. The paper-style figure uses a white background, serif type, blue/red marked solid/dashed lines and a light dotted grid. Values below 50% are outside the view, not raised to 50%; the figure warns about these points and their exact values remain in the data files. No Excel, GPU profiling, sample-count limit or `ex.py` scheduling is added to this workflow. Existing sample-visualization flags remain independent.

The existing `summary.csv` and each `metrics.json` now also record `parsed_samples`, `tp`, `fp`, `tn`, `fn`, `precision`, `recall` and `tnr`. Defect (`gt=1`, answer A) is the positive class. **ACC still uses all labeled samples and counts unparsed answers as incorrect; PRE/Recall/TNR use only parsed, labeled A/B answers.** On that parsed subset, PRE = TP/(TP+FP), Recall = TP/(TP+FN), TNR = TN/(TN+FP). Unparsed counts are retained separately. Zero denominators produce JSON `null` / empty CSV cells and gaps in the chart, never invented zeros. All stored metrics are fractions in [0,1]; the chart converts them to percentages. Scores are pooled over samples, not category-macro averages.

In an existing working server environment, install only the added plotting dependency (do not reinstall Torch/FlashAttention):

```bash
python -m pip install "matplotlib>=3.7,<4"
```

Continue using the same `run.py` command and your chosen input JSON; no new flags are needed. Matplotlib is checked before model loading and uses a headless backend. To redraw from a **new-format** saved `summary.csv` without inference:

```bash
python -m llava_pruning.metric_plot /path/to/run/output
```

Older accuracy-only summaries lack PRE/Recall/TNR and cannot be plotted by this command without recomputing those statistics from the predictions. Plot export failure leaves evaluation results intact.

The common CLI intentionally has no `--layer`: another pruning method may have no layer parameter or different parameters. Add a method implementation in `llava_pruning/methods.py`, register it in `METHODS`, and give it its own config file. Keep model-family-specific code in `llava_pruning/backend.py` or add a separate backend when the supported checkpoint family changes.

## 0--90% benchmark, plots and Excel with `ex.py`

`python ex.py` runs **0, 10, 20, ..., 90%**. By default, GPU 4 runs 0/20/40/60/80% and GPU 5 runs 10/30/50/70/90%, with **at most one worker per GPU**. Each rate starts a fresh process/model copy, so allocator peaks are reset independently and each total time includes model loading. Defaults match the earlier command: checkpoint `/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/`, MVTec `question_musc.jsonl` and data root under `/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/`, prompt v0, anyres_max_9 and greedy decoding. **All jobs disable sample visualization completely:** no attention-overlay/pruning images, no visualization attention/mask capture, and no `visualizations/` directory. The scores required for pruning itself still run at nonzero rates. It reuses the other settings in `configs/fastv.json` without changing that file; normal `run.py` visualization switches remain available. The final benchmark curves are separate from sample visualizations.

```bash
python -m pip install -r requirements-report.txt
python ex.py --output-dir output/ex_0_90_100
# Optional: --model-path /path/to/checkpoint --input-json /path/to/questions.jsonl
#           --data-root /path/to/dataset --seed 42

# Better controlled timing: all ten rates sequentially on the same GPU.
python ex.py --gpus 5 --output-dir output/ex_0_90_gpu5_100
```

All jobs process only the **first 100 input records in file order**, with no shuffling (or all available records if there are fewer than 100). The launcher saves their shared subset as `input_first_100.json` in the experiment directory; the source JSON/JSONL is not changed. No inference is run on later records. Accuracy and generation time cover only this subset; total time and memory peaks still include model loading. The job specs and benchmark summary record the source path, subset path, selected count, shared seed and GPU assignment. This limit applies only to `ex.py`; normal `run.py` is unchanged. Reporting dependencies are checked before GPU workers launch; install only `requirements-report.txt` into an already-working server environment, not a replacement Torch/FlashAttention stack.

Do not prefix this command with a single-GPU restriction; the launcher sets each child's `CUDA_VISIBLE_DEVICES` before Torch is imported. Use `--gpus 4 5` (default) or `--gpus 5` to select physical GPU IDs. Without `--output-dir`, it creates a timestamped `output/ex_YYYYMMDD_HHMMSS` directory. Explicit output directories must be new/empty. Watch `gpu<N>_prune_<RR>.log` there; model outputs are in `gpu<N>_prune_<RR>/prune_<RR>/`. The terminal reports the start and completion of every rate; detailed progress stays in its log.

After all rates finish, the experiment directory contains:

- `benchmark_curves.png` (300 dpi) and `benchmark_curves.pdf` (vector): four horizontal panels for accuracy, total/generation time, allocated/reserved peak memory, and mean per-image generation time. White background, serif type, blue circles/red squares, solid/dashed lines and light dotted grids follow the supplied paper-figure style. Labels are English for portable server rendering. No smoothing or invented data points.
- `benchmark.xlsx`: numeric `Metrics` table (one row per rate), `Settings`, `Definitions` (units and measurement scope), and an embedded `Figure`. Includes accuracy, counts, timings, memory, mean generation milliseconds/image, generation-only images/s, GPU identity and error fields. Missing values remain blank; failure/partial results remain explicitly marked, with gaps in curves rather than zeros.
- `benchmark.json`: raw resource measurements and shared settings; `benchmark.csv`: flat numeric table including derived metrics. Logs, predictions and per-rate accuracy metrics are also retained.

The reports contain `total_seconds` (process launch to exit, including model loading, inference and JSON/log saving), `generation_seconds_sum` (sum of existing synchronized per-image generate timings), and peak PyTorch `allocated`/`reserved` VRAM in GiB. For inference timing, prefer `generation_seconds_sum`: it excludes loading, image preprocessing and result-file writing, but includes the pruning-score computation and the first cold call (no warmup exclusion). `mean_generation_ms` is this sum divided by evaluated samples, times 1000; it is **not per-token latency or time to first token**. Peaks are reset before model loading and cover the whole run, not just the final sample. These exclude CUDA contexts, other processes and memory allocated outside PyTorch; they are **not whole-board nvidia-smi usage or GPU utilization percentages**. `experiment_wall_seconds` is the complete scheduled sweep's elapsed time, not the sum of all workers' durations. Per-job timings exclude waiting in the GPU queue, and all measured timings exclude final report export. Failed jobs retain logs and a resource report where possible, with nonzero exit status rather than fabricated zero memory values; the remaining rates still run after a worker failure.

To re-export reports from existing measurements (including an older two-rate run), without loading the model or rerunning inference:

```bash
python ex.py --report-only output/ex_0_90_100
```

This replaces only the derived `benchmark.csv`, plot files and workbook in that directory. It preserves `benchmark.json`, logs and predictions, and does not invent rates absent from an old run. If export fails, raw JSON/CSV remain available and the launcher prints this recovery command.

Nonzero rates still include independent Q/K scoring overhead. The current implementation masks attention and retains the full hidden-state/KV-cache sequence; it does not guarantee a speedup even with sample visualization disabled. The two GPUs may also differ or share CPU/disk bottlenecks; these timings alone do not establish a pruning speedup. For more controlled comparisons, use the single-GPU option and repeat measurements. Run this on the inference server with the selected GPUs available; the launcher does not stop other users' jobs. Stop an old benchmark before restarting with the updated script and a new output directory; editing this file cannot change already-running workers. When updating the server, copy `ex.py`, `benchmark_report.py` and `requirements-report.txt` together.

## Research cautions

- This FastV implementation masks visual keys in decoder attention. It does **not** shorten the token sequence or prove wall-clock acceleration from token removal. `generation_seconds` is diagnostic, not a claimed speedup metric.
- FlashAttention's 2D mask unpads discarded query positions as well during prefill; text/retained queries still attend to the retained key set. Cached sequence coordinates are not compacted. Independent score computation adds overhead, and non-finite FP16 Q/K raises an error rather than silently changing precision.
- `randomroi` with ground-truth masks or boxes is **annotation-guided inference**. It can leak test labels and must not be reported as a label-free evaluation. `randompatch` is the annotation-free alternative in this scaffold.
- The saved pruning map projects token decisions onto spatial anchors; it is not a pixel-exact receptive-field map. Attention overlay uses ranking-layer prompt-to-image attention, not an anomaly segmentation prediction.
- For pure anyres, the high-resolution view is projected back onto the source image approximately after unpadding/downsampling; its row-newline tokens have no pixel region and are reported separately in `decisions.json`.
- This scaffold has not been validated end-to-end against the checkpoint in this Windows workspace because the checkpoint and CUDA runtime are not present here. Validate on the target GPU before reporting results.

The `vendor/llava` subset is adapted from the existing Triad/LLaVA code and retains its Apache-2.0 license. Before public release, confirm licensing for the remaining Triad-derived prompt code, checkpoint, and dataset, and choose a license for new scaffold files. Renaming the project does not change these sources or authorship. The new ViCo adapter follows the [PyramidDrop authors' method and implementation](https://github.com/Cooperx521/PyramidDrop); see [the adaptation notes](docs/vico.md).
