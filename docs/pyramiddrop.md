# PyramidDrop：OneVision/Qwen2 复现说明

论文：[PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction](https://arxiv.org/abs/2410.17247)。[官方代码](https://github.com/Cooperx521/PyramidDrop)，固定提交 `6444f304aee4b7edcb022839fb9e8cd07859704a`。CVPR 2025 正式版本题目为 [Conical Visual Concentration for Efficient Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.html)。

## 方法与实现范围

PyramidDrop 在浅层保留全部视觉 token，随后分阶段删除与当前指令相关性较低的 token。每次剪枝使用完整提示模板最后一个 token 的 query 与所有可见 token 的 key 计算注意力，跨 head 平均后对视觉项排序。保留 top-k 并恢复原顺序，文本全部保留。

本次根据官方源码实现推理适配。官方 `pdrop_rank_drop` 实际使用**下一层**的 input layernorm、Q/K 投影和 RoPE，不是刚完成层返回的 attention，也不是视觉编码器 attention。softmax 对所有可见 key 计算，随后才抽取视觉位置；不能只对图像 key 做 softmax。Q/K 矩阵乘法沿用模型精度，softmax 转 FP32 后按官方代码回转原精度，再跨 head 平均。

官方示例的 32 层模型在第 8、16、24 层完成后剪枝，下一层输入分别保留原始视觉 token 的 50%、25%、12.5%。本入口默认按实际模型深度四等分：28 层 OneVision/Qwen2 对应 7、14、21。阶段边界也可显式指定。沿用原项目 SigLIP、图像预处理、projector、完整 dense AnyRes 拼接，进入语言模型后才剪枝。所有 Base/AnyRes patch 与 `image_newline` 都按真实打包顺序参与一个视觉候选池；不设置视图独立预算或额外 newline 保护。

核心会**实际缩短** hidden-state 序列，保留浅层完整 KV，后续层写入较短 KV。剪枝后位置从 0 重新连续编号；decode 使用各层实际 cache 长度，不再排序。Qwen2 的 GQA key 先按 query head 分组重复。新增核心独立于旧 `--method vico` 配置；旧入口仍按原配置运行。

支持：eval、单样本、不带 padding、greedy、DynamicCache、Transformers 4.46.1。真实实验入口沿用 FP16 + FlashAttention2，不静默更换精度或 attention backend。训练、beam、sampling、static/quantized/sliding cache 等会被明确拒绝。代码中 eager/SDPA 用于本地小模型数值验证；FlashAttention2 和真实权重的服务器验证需要实际运行。

## 本项目表格的百分比定义

当前默认实验是 **LLM各层平均视觉token剪枝率50%、70%**，明确使用
`--average-prune-rates 50 70`。70%表示平均删除70%、平均保留30%，不是最终累计删除70%。
原有`--average-keep-rates`平均保留率接口及显式`final_cumulative`接口继续兼容，含义保持原样。

设原始视觉段长度为 N0，阶段 s 的层数为 Ls，则平均保留率为：

`R = sum(Ls * Ns) / (L * N0)`。

四阶段等长时，以每阶段相对上一阶段保留 λ 构造金字塔：

`R = (1 + λ + λ² + λ³) / 4`。

平均剪枝率为`P = 100 * (1-R)`。入口按目标`R=1-P/100`反解λ，保留每阶段相同相对保留系数的
金字塔策略；不等长边界使用实际层数加权。28层模型在第7、14、21层后剪枝，新实验的阶段累计保留率为：

| 目标平均剪枝率 | 第7层后 / N0 | 第14层后 / N0 | 第21层后 / N0 |
|---|---:|---:|---:|
| 50% | 54.3689% | 29.5598% | 16.0713% |
| 70% | 16.7320% | 2.7996% | 0.4684% |

平均剪枝70%对应最终累计删除约99.5316%，因此与此前最终累计70%的实验不是相同预算。
使用原来的模型精度、attention模式、独立Q/K评分和剪枝层号；仅改变预算目标与对应的阶段保留数量。
所有packed视觉token（Base、AnyRes和newline）都计入原始分母，与现有PyramidDrop处理一致。

下面的表格仅解释旧的显式`--average-keep-rates`接口，这里的数值是**平均保留率**：

| 各层平均保留率 | 第一次剪后 / N0 | 第二次剪后 / N0 | 第三次剪后 / N0 |
|---|---:|---:|---:|
| 50% | 54.3689% | 29.5598% | 16.0713% |
| 70% | 76.5485% | 58.5967% | 44.8549% |
| 90% | 93.0136% | 86.5153% | 80.4710% |
| 论文 λ=0.5 | 50% | 25% | 12.5% |

论文默认 λ=0.5 的平均保留率是46.875%，平均剪枝率为53.125%，需单独报告。
每阶段数量仍遵循`floor(N0 * ratio)`，极短视觉段至少保留一个token；整数取整及最小数量保护会影响实际平均值。
逐图和汇总同时输出目标、连续求解值和实际值。固定前7/28层完整保留时，平均剪枝目标必须小于75%，
因此这组层号无法运行平均剪枝90%；代码明确报错，不降低目标或跳过样本。

平均视觉 token 一致不代表 FLOPs、端到端时间或显存一致。最终比较使用同样输入、分辨率、提示词、生成长度上限、硬件、预热和测量方式，并报告实测结果。

## 服务器运行

在原项目环境、项目根目录执行。默认权重和MVTec路径沿用项目已有设置。
新实验：GPU5、全量数据、平均剪枝50%和70%、不保存图片，统计首token Lat和完整生成峰值Mem：

```bash
CUDA_VISIBLE_DEVICES=5 python runPyramidDrop.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --roi-mode anyres_max_9 \
  --average-prune-rates 50 70 \
  --benchmark-metrics \
  --warmup 10 \
  --output-dir output/pyramiddrop_avg_prune_50_70_gpu5_01
```

一次加载模型，按50%、70%顺序运行；固定greedy解码。输出目录必须为空或不存在。无需修改JSON配置或重新训练。
只检查预算、无需模型或GPU：

```bash
python runPyramidDrop.py --plan-only --model-depth 28 --average-prune-rates 50 70
```

`--average-prune-rates 0`运行原始不剪枝路径。旧平均保留率接口的小样本验证仍可显式运行：

```bash
CUDA_VISIBLE_DEVICES=0 python runPyramidDrop.py \
  --average-keep-rates 100 50 70 90 \
  --limit 2 \
  --save-prune-vis --save-attention-vis \
  --output-dir outputs/pyramiddrop_smoke
```

正式 MVTec AD：

```bash
CUDA_VISIBLE_DEVICES=0 python runPyramidDrop.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 --roi-mode anyres_max_9 \
  --average-keep-rates 100 50 70 90 \
  --output-dir outputs/pyramiddrop_mvtec
```

VisA 使用同一命令替换 `--input-json`、`--data-root`、`--output-dir`。沿用已有 `load_samples` 和 `resolve_prompt`：MVTec 已知类别使用原 v0/v1/v2/v3 模板，其他类别使用记录的 `text`；输入路径相对于 data-root，标签 A=异常、B=正常。VisA 的 text 必须与既有 A/B 评估口径一致，否则先修正实验提示词，不能把解析失败当作模型性能变化。

论文默认配置单独运行：

```bash
CUDA_VISIBLE_DEVICES=0 python runPyramidDrop.py \
  --stage-keep-ratio 0.5 \
  --output-dir outputs/pyramiddrop_official
```

预览无需模型或 CUDA：

```bash
python runPyramidDrop.py --plan-only --model-depth 28 --average-keep-rates 50 70 90
```

`--prune-rates 0 50 70 90 --rate-semantics final_cumulative` 表示最终累计删除比例，便于沿用之前实验的预算口径；`--rate-semantics per_stage` 表示每次删除比例。`--keep-ratios .5 .25 .125 --layers 7 14 21` 则可显式指定相对原始视觉段的各阶段保留率。不同口径不能混用同一个百分比表头。

需要与当前 FastV/VisionZip 的首 token Lat、Mem 统计一致时，使用：

```bash
CUDA_VISIBLE_DEVICES=5 python runPyramidDrop.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 --roi-mode anyres_max_9 \
  --prune-rates 0 50 70 90 --rate-semantics final_cumulative \
  --benchmark-metrics --warmup 10 \
  --output-dir output/pyramiddrop_final_lat_mem_01
```

默认关闭采样、关闭可视化，输入全部运行，无固定100张限制。上面的兼容命令显式选择了
`final_cumulative`，因此其中0/50/70/90表示最终累计剪枝率；它们不是新默认实验的平均剪枝目标。
该兼容模式的中间阶段按剩余比例的1/3、2/3、1次幂分配，整数取floor。
层号由实际模型深度确定，28层时为7/14/21，可用`--layers`覆盖。

## 输出与测量

默认一次加载权重，依次运行50%和70%平均剪枝率。旧平均保留率实验必须显式使用`--average-keep-rates`。

- `run.json`：官方来源和哈希、模型层数、求解后的阶段比例、平均预算目标、环境、提示词、数据路径、计时和指标定义。
- `summary.csv`：ACC、PRE、Recall、TNR、未解析数量、平均生成秒数、峰值allocated显存、最终视觉token数、全层平均视觉token数与实际平均保留率。
  新实验的`prune_rate`和`target_average_prune_rate`是目标平均剪枝率，`rate_semantics=layer_average`；
  `expected_average_prune_rate`是连续求解值，`actual_average_prune_rate`由实际每层token数量统计；`pruning_layers`记录剪枝边界。
- 加 `--benchmark-metrics` 后，另保存 `lat_mean_ms`、`lat_p50_ms`、`lat_p95_ms`、`lat_std_ms`、`mem_peak_allocated_gib`、`mem_peak_reserved_gib`、`mem_peak_allocated_pct`、生成 token 数与预热数；单样本记录包含首 token 秒数和显存字节数。
- `average_prune_50/`和`average_prune_70/`：新实验目录；其中`predictions.jsonl`保存每图答案、每层token数、
  目标/实际平均剪枝率、时间和显存。旧平均保留率仍使用`average_keep_XX/`，最终累计接口仍使用`prune_XX/`。
- 每个设置目录的 `metrics.json`、`classification_results.csv`：中断时保留 `complete=false`，全部完成后才置为 true。
- 可选 `visualizations/sample_*/image_0/comparison.png`、`attention_overlay.png`、`decisions.json`：每个剪枝边界一行，全部映射回原始图像；较早已删除 token 在后续注意力图中明确标为无当前评分。

计时与原 shared backend 一致：同步 CUDA 的完整 `generate`，含视觉编码、剪枝与生成，排除图像读取预处理和保存绘图。默认不预热，以保持现有实验习惯；可加 `--warmup 1`，比较方法应使用相同设置。显存是每个样本期间 PyTorch peak allocated/reserved，单位 MiB，包含常驻权重；不等于设备全部进程总占用。正式比较先用单张可见 GPU，避免 offload。

上述默认完整生成时间不称为 Lat。`--benchmark-metrics` 才启用与 `run.py` 相同的首 token 观察器：输入驻留并同步后，从进入 `model.generate` 开始，到首个生成 token 选出、追加并同步结束，含视觉编码、projector、打包、prefill 和剪枝。Mem 在同一模型入口重置峰值，在完整回答生成期间测量，排除预处理和 H2D 峰值。默认非 benchmark 路径的显存重置在预处理之前，二者口径不同。benchmark 要求单张可见 GPU、权重全部驻留、关闭可视化，按每个设置独立均匀抽样预热；预热不计入统计。依次运行设置共用一个模型和 allocator 缓存，reserved 不能当作每种方法独立进程的干净峰值。

ACC 将未解析回答计为错误；PRE/Recall/TNR 沿用既有实现，只对成功解析的有标签样本计算，并报告 unparsed。这是保留原项目评估规则，不是新增论文指标。

## 复现状态

本地验证使用随机初始化的小 Qwen2/GQA 模型与合成输入，检查算法、KV cache、连续生成、基线回退和运行输出。它证明实现接线与数值行为，不能证明工业异常检测质量或真实 GPU 加速。真实 LLaVA-OV 权重、FlashAttention2、MVTec AD 和 VisA 的最终结果由服务器运行后补充；当前未填写截图中的任何实验数字。
