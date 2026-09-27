# Triad Pruning (research scaffold)

A small, single-image inference scaffold for pruning experiments on the existing Triad OneVision/Qwen2 checkpoint. This first version implements FastV only. It runs a rate sweep in one process, loading the model once; rates are evaluated sequentially on the same GPU. Rate 0 is a real no-FastV baseline.

## Inputs

The CLI accepts a JSON list (`.json`) or one object per line (`.jsonl`). For example:

```json
{"question_id":"000000108","image":"000000108.png","text":"Is there any defect in this image? If yes, say 'yes', otherwise say 'no'. Then describe it.","gt":1,"origin_path":"screw/test/thread_top/005.png","mask":"musc/000000108.png","musc_scores":0.6233050227165222}
```

`--data-root` is the root for relative paths. A bare `image` filename is searched first at `<data-root>/<filename>` and then at `<data-root>/imgs/<filename>`; `mask` is resolved at `<data-root>/<mask>` and may be a grayscale image, `.npy`, or `.npz` with an `anomaly_map` array. `origin_path` determines the MVTec category (`screw` here). `musc_scores` and other extra fields are ignored, not treated as pruning scores. IDs stay strings, preserving leading zeroes. `bbox` may be supplied later as `[[x_min,y_min,x_max,y_max], ...]`, following the legacy crop helper's inclusive coordinates; when both mask and bbox are present, mask takes precedence in this scaffold.

For known MVTec categories, `--prompt-version v0|v1|v2|v3` selects the original Triad MVTec templates; the record's `text` is a fallback only for unknown categories. This deliberately matches the old `config:vN` experiment path, and means the example's `text` is **not** the model prompt for `screw`. The resolved prompt is stored with each prediction.

## Run

Inference now matches the restored local Triad loader: **FP16 (`torch.float16`) + FlashAttention2**, greedy decoding, and at most **512 new tokens** using Triad's context-budget calculation. TF32 settings are left at the environment's values, as in Triad. Vision-tower loading follows the original loader (including its lazy-loading behavior); actual model/vision dtypes, attention implementation and library versions are recorded in `run.json`. There is no silent SDPA/BF16 fallback.

Use the **same environment as the original Triad**, including its `flash-attn` build, Torch, Transformers, CUDA and image libraries. `requirements.txt` pins the shared Python dependencies but does not install the platform-specific FlashAttention extension. Run on one visible GPU for the initial comparison:

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/Triad_ov \
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

`--roi-mode randomroi` uses `mask`, then `bbox`, then random crops if neither exists. `--roi-mode randompatch` ignores both annotations and always chooses random crops. Both modes use the Triad `randomroi` image packing; they differ only in crop selection. `--roi-mode anyres_max_9` ignores masks/boxes and, like original Triad's `--overwrite_image_aspect_ratio`, changes the aspect-ratio setting **without overwriting the checkpoint's merge type**. For the pure-anyres checkpoint discussed here this is `spatial_unpad`; it is not the `anyres_max_9_randomroi` hybrid. The actual ROI source and boxes are recorded per prediction.

Decoding is now **greedy by default**, matching the user's current Triad `do_sample=False`. Existing `--no-sample` commands remain valid; use `--sample` only to opt back into sampling (temperature 0.2, top-p 0.7). Unless `--seed` is specified, each run generates a new seed; the actual seed and decoding mode are recorded in `run.json`. In `randompatch` mode the seed controls crop selection; greedy decoding does not disable random crops.

## Independent attention scoring and baseline check

At nonzero pruning rates the main decoder **stays on FlashAttention2 in every layer**. FastV does not request `output_attentions=True` and does not read returned Transformer attention matrices. A read-only side calculation uses the ranking layer's input normalization and Q/K projections, RoPE and GQA head mapping to recompute only the last valid prompt query against all prompt keys. Projection/RoPE follow the model dtype; QK, softmax and head averaging use FP32 for score stability. These scores do not change the decoder's hidden states or KV cache. This extra computation is not claimed to be bitwise identical to the old eager FP16 attention scores.

At **0%**, the side calculation and pruning mask are bypassed entirely and the decoder directly calls the original `Qwen2Model.forward`. Generation also preserves original Triad's mask/position-ID handling and wrapper behavior. Anyres preprocessing is unchanged except for collecting visualization metadata. Input and generated token IDs are saved in `predictions.jsonl` for diagnosis.

First run only the baseline (replace paths with your own):

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/Triad_ov \
  --input-json /path/to/questions.jsonl \
  --data-root /path/to/dataset \
  --prompt-version v0 --roi-mode anyres_max_9 \
  --method-config configs/baseline.json --no-sample \
  --output-dir output/anyres_fp16_baseline

python compare_results.py \
  --baseline /path/to/original_triad_answers.json \
  --candidate output/anyres_fp16_baseline/prune_00/predictions.jsonl \
  --output output/anyres_fp16_baseline/comparison.json
```

The comparison aligns `question_id`/`id`, compares complete trimmed `answer`/`conversations -> gpt -> value` strings, reports missing/extra IDs and every mismatch, and exits nonzero on differences. `identical: true` is evidence about those two runs, **not a guarantee across environments or checkpoints**. Compare against a fresh run of the restored Triad with the identical checkpoint, `config:v0`, anyres mode and greedy decoding. In particular, verify both scripts open the same image files (`imgs/<name>` versus a same-named file at the dataset root). Do not use accuracy alone: Triad's `mean` is a category average and its answer parser differs from this scaffold's metric parser.

For the full 0--90% sweep, omit `--method-config configs/baseline.json` and add the visualization switches as needed. Rate 0 still saves no visualizations.

Run `python -m unittest discover -s tests -v` in the pinned environment (Python 3.10+). The small CUDA/FlashAttention integration test skips without a CUDA GPU/FlashAttention; CPU tests cover independent scores, masks, cache continuation and the zero-rate forward path. Optional direct comparisons with the downloaded Triad source also test anyres pixels, packed embeddings and wrapper equivalence; set `TRIAD_REFERENCE_DIR` to that repository root if needed. No test downloads model weights. Do not copy the local `.venv` test environment to the server.

`configs/fastv.json` owns FastV-only settings: `layer`, `prune_rates`, `visualize_rates`, `min_tokens`, and newline handling. By default it sweeps 0, 10, 20, ..., 90%; `prune_00` disables FastV completely. Only 10, 30, 50, 70, 90% generate images, when a visualization switch is enabled. `--save-prune-vis` and `--save-attention-vis` are independent. Visualizations include original/crop-level FastV decisions and a 60/40 JET attention overlay from the ranking layer. The folder layout is `outputs/experiment_01/prune_10/{predictions.jsonl,metrics.json,visualizations/sample_<id>/image_0/...}`. Output directories must be empty to prevent accidental overwrite; choose a new directory for each experiment.

Each image directory now saves only `comparison.png` (`--save-prune-vis`) and `attention_overlay.png` (`--save-attention-vis`). When both switches are enabled there are exactly two PNGs, for anyres, randomroi and randompatch alike. Originals, individual crops, blackouts and binary masks are no longer saved separately. `decisions.json`, predictions and accuracy statistics are retained; previously generated files are not deleted.

Each completed rate prints image-level accuracy and appends a row to `summary.csv`. Each `prune_XX/metrics.json` also tracks partial progress while running (`complete: false` until the rate finishes). Accuracy uses the first answer option only: `A` means defect (`gt=1`), `B` means no defect (`gt=0`); unparsed labeled answers count as incorrect. Records without `gt` are excluded. This is binary classification accuracy, not segmentation accuracy or AUROC. The `accuracy` field is a fraction, e.g. `0.9` means 90%.

The common CLI intentionally has no `--layer`: another pruning method may have no layer parameter or different parameters. Add a method implementation in `triad_pruning/methods.py`, register it in `METHODS`, and give it its own config file. Keep model-family-specific code in `triad_pruning/backend.py` or add a separate backend when the supported checkpoint family changes.

## Research cautions

- This FastV implementation masks visual keys in decoder attention. It does **not** shorten the token sequence or prove wall-clock acceleration from token removal. `generation_seconds` is diagnostic, not a claimed speedup metric.
- FlashAttention's 2D mask unpads discarded query positions as well during prefill; text/retained queries still attend to the retained key set. Cached sequence coordinates are not compacted. Independent score computation adds overhead, and non-finite FP16 Q/K raises an error rather than silently changing precision.
- `randomroi` with ground-truth masks or boxes is **annotation-guided inference**. It can leak test labels and must not be reported as a label-free evaluation. `randompatch` is the annotation-free alternative in this scaffold.
- The saved pruning map projects token decisions onto spatial anchors; it is not a pixel-exact receptive-field map. Attention overlay uses ranking-layer prompt-to-image attention, not an anomaly segmentation prediction.
- For pure anyres, the high-resolution view is projected back onto the source image approximately after unpadding/downsampling; its row-newline tokens have no pixel region and are reported separately in `decisions.json`.
- This scaffold has not been validated end-to-end against the checkpoint in this Windows workspace because the checkpoint and CUDA runtime are not present here. Validate on the target GPU before reporting results.

The `vendor/llava` subset is adapted from the existing Triad/LLaVA code and retains its Apache-2.0 license. Before public release, confirm licensing for the remaining Triad-derived prompt code, checkpoint, and dataset, and choose a license for new scaffold files.
