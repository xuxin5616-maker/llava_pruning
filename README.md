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

When enabled, sample visualizations are saved for 10/30/50/70/90 only. Each sample still has only `comparison.png` and `attention_overlay.png`, now with one labelled row per ViCo pruning boundary. Previously removed tokens are gray in later attention rows, not assigned fabricated scores. All 28 layers' counts (including unchanged layers and rate 0) are stored in `prune_XX/layer_tokens.csv` and prediction metadata. `run.py` still draws ACC/PRE/Recall/TNR curves with a 50%–100% y-axis after the sweep. See [the ViCo adapter specification](docs/vico.md) for token-count scope, rounding, position conventions, limitations and tests.

For a baseline-only ViCo run, copy `configs/vico.json`, set `prune_rates` to `[0]` and `visualize_rates` to `[]`, and pass that file. `configs/baseline.json` is a **FastV** configuration.

## Inputs

The CLI accepts a JSON list (`.json`) or one object per line (`.jsonl`). For example:

```json
{"question_id":"000000108","image":"000000108.png","text":"Is there any defect in this image? If yes, say 'yes', otherwise say 'no'. Then describe it.","gt":1,"origin_path":"screw/test/thread_top/005.png","mask":"musc/000000108.png","musc_scores":0.6233050227165222}
```

`--data-root` is the root for relative paths. A bare `image` filename is searched first at `<data-root>/<filename>` and then at `<data-root>/imgs/<filename>`; `mask` is resolved at `<data-root>/<mask>` and may be a grayscale image, `.npy`, or `.npz` with an `anomaly_map` array. `origin_path` determines the MVTec category (`screw` here). `musc_scores` and other extra fields are ignored, not treated as pruning scores. IDs stay strings, preserving leading zeroes. `bbox` may be supplied later as `[[x_min,y_min,x_max,y_max], ...]`, following the legacy crop helper's inclusive coordinates; when both mask and bbox are present, mask takes precedence in this scaffold.

For known MVTec categories, `--prompt-version v0|v1|v2|v3` selects the original LLaVA MVTec templates; the record's `text` is a fallback only for unknown categories. This deliberately matches the old `config:vN` experiment path, and means the example's `text` is **not** the model prompt for `screw`. The resolved prompt is stored with each prediction.

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

`--roi-mode randomroi` uses `mask`, then `bbox`, then random crops if neither exists. `--roi-mode randompatch` ignores both annotations and always chooses random crops. Both modes use the LLaVA `randomroi` image packing; they differ only in crop selection. `--roi-mode anyres_max_9` ignores masks/boxes and, like original LLaVA's `--overwrite_image_aspect_ratio`, changes the aspect-ratio setting **without overwriting the checkpoint's merge type**. For the pure-anyres checkpoint discussed here this is `spatial_unpad`; it is not the `anyres_max_9_randomroi` hybrid. The actual ROI source and boxes are recorded per prediction.

Decoding is now **greedy by default**, matching the user's current LLaVA `do_sample=False`. Existing `--no-sample` commands remain valid; use `--sample` only to opt back into sampling (temperature 0.2, top-p 0.7). Unless `--seed` is specified, each run generates a new seed; the actual seed and decoding mode are recorded in `run.json`. In `randompatch` mode the seed controls crop selection; greedy decoding does not disable random crops.

## Independent attention scoring and baseline check

At nonzero pruning rates the main decoder **stays on FlashAttention2 in every layer**. FastV does not request `output_attentions=True` and does not read returned Transformer attention matrices. A read-only side calculation uses the ranking layer's input normalization and Q/K projections, RoPE and GQA head mapping to recompute only the last valid prompt query against all prompt keys. Projection/RoPE follow the model dtype; QK, softmax and head averaging use FP32 for score stability. These scores do not change the decoder's hidden states or KV cache. This extra computation is not claimed to be bitwise identical to the old eager FP16 attention scores.

At **0%**, the side calculation and pruning mask are bypassed entirely and the decoder directly calls the original `Qwen2Model.forward`. Generation also preserves original LLaVA's mask/position-ID handling and wrapper behavior. Anyres preprocessing is unchanged except for collecting visualization metadata. Input and generated token IDs are saved in `predictions.jsonl` for diagnosis.

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
