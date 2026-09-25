# Triad Pruning (research scaffold)

A small, single-image inference scaffold for pruning experiments on the existing Triad OneVision/Qwen2 checkpoint. This first version implements FastV only. It runs a rate sweep in one process, loading the model once; rates are evaluated sequentially on the same GPU.

## Inputs

The CLI accepts a JSON list (`.json`) or one object per line (`.jsonl`). For example:

```json
{"question_id":"000000108","image":"000000108.png","text":"Is there any defect in this image? If yes, say 'yes', otherwise say 'no'. Then describe it.","gt":1,"origin_path":"screw/test/thread_top/005.png","mask":"musc/000000108.png","musc_scores":0.6233050227165222}
```

`--data-root` is the root for relative paths. A bare `image` filename is searched first at `<data-root>/<filename>` and then at `<data-root>/imgs/<filename>`; `mask` is resolved at `<data-root>/<mask>` and may be a grayscale image, `.npy`, or `.npz` with an `anomaly_map` array. `origin_path` determines the MVTec category (`screw` here). `musc_scores` and other extra fields are ignored, not treated as pruning scores. IDs stay strings, preserving leading zeroes. `bbox` may be supplied later as `[[x_min,y_min,x_max,y_max], ...]`, following the legacy crop helper's inclusive coordinates; when both mask and bbox are present, mask takes precedence in this scaffold.

For known MVTec categories, `--prompt-version v0|v1|v2|v3` selects the original Triad MVTec templates; the record's `text` is a fallback only for unknown categories. This deliberately matches the old `config:vN` experiment path, and means the example's `text` is **not** the model prompt for `screw`. The resolved prompt is stored with each prediction.

## Run

Install the pinned environment from `requirements.txt` with a compatible NVIDIA/CUDA setup, then run:

```bash
python run.py \
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

`--roi-mode randomroi` uses `mask`, then `bbox`, then random crops if neither exists. `--roi-mode randompatch` ignores both annotations and always chooses random crops. The actual ROI source and boxes are recorded per prediction. Both modes use the Triad `randomroi` image packing; they differ only in crop selection.

`configs/fastv.json` owns FastV-only settings: `layer`, `prune_rates`, `visualize_rates`, `min_tokens`, and newline handling. By default it sweeps 10, 20, ..., 90%; only 10, 30, 50, 70, 90% generate images, when a visualization switch is enabled. `--save-prune-vis` and `--save-attention-vis` are independent. Visualizations include original/crop-level FastV decisions and a 60/40 JET attention overlay from the ranking layer. The folder layout is `outputs/experiment_01/prune_10/{predictions.jsonl,visualizations/sample_<id>/image_0/...}`. Output directories must be empty to prevent accidental overwrite; choose a new directory for each experiment.

The common CLI intentionally has no `--layer`: another pruning method may have no layer parameter or different parameters. Add a method implementation in `triad_pruning/methods.py`, register it in `METHODS`, and give it its own config file. Keep model-family-specific code in `triad_pruning/backend.py` or add a separate backend when the supported checkpoint family changes.

## Research cautions

- This FastV implementation masks visual keys in decoder attention. It does **not** shorten the token sequence or prove wall-clock acceleration from token removal. `generation_seconds` is diagnostic, not a claimed speedup metric.
- `randomroi` with ground-truth masks or boxes is **annotation-guided inference**. It can leak test labels and must not be reported as a label-free evaluation. `randompatch` is the annotation-free alternative in this scaffold.
- The saved pruning map projects token decisions onto spatial anchors; it is not a pixel-exact receptive-field map. Attention overlay uses ranking-layer prompt-to-image attention, not an anomaly segmentation prediction.
- This scaffold has not been validated end-to-end against the checkpoint in this Windows workspace because the checkpoint and CUDA runtime are not present here. Validate on the target GPU before reporting results.

The `vendor/llava` subset is adapted from the existing Triad/LLaVA code and retains its Apache-2.0 license. Before public release, confirm licensing for the remaining Triad-derived prompt code, checkpoint, and dataset, and choose a license for new scaffold files.
