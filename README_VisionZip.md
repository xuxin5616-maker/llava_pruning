# 独立 VisionZip 实验入口

`runVisionZip.py` 基于用户提供的单文件脚本整合。没有改动 `run.py`、
FastV/ViCo、vendor 模型代码或现有配置。
复用 `llava_pruning` 的模型加载、数据读取、提示词、分类指标和首 token 计时，
以及 `benchmark_fastv.py` 的预热选样、显存检查和测量汇总函数。
因此请在**当前完整工程**中运行，不要把它当作脱离本项目的独立软件包。

## 运行

### 3.5：逐张记录实际网格、padding 与 unpad

当前版本 `3.5-padding-grid-audit` 保留下面 3.4 的 newline 修复，额外记录
`prune_XX/geometry_audit.csv`（每张图一行，不依赖可视化开关）。同样的信息也在
`predictions.jsonl` 的 `pruning.geometry_audit` 中。

- `original_width/height`、`anyres_grid_width/height`：模型收到的原图尺寸和实际
  AnyRes 网格；`anyres_tiles` 不包含 Base。`run.json` 另存实际 `image_grid_pinpoints`。
- `padding_left/top/right/bottom`：预处理按比例缩放后，应补的四边像素数；
  `padding_required` 表示至少一边需要补边。
- `padding_call_observed`：是否实际执行了对应原图、目标尺寸的现有补边函数；
  `padding_output_size_matches`：该函数的输出尺寸是否符合目标（未执行为 null）。
  `padding_missing_or_mismatch=True` 标记需要补边但未观测到该调用，或输出尺寸不符。
  这是调用与尺寸检查，不是通过黑色像素判断补边，也不重新处理图片。
- `baseline_unpad_tokens`：原版在完整特征网格上会移除的位置数，不代表稀疏合并
  后实际还剩多少 padding token。`unpad_executed` 记录本次是否执行特征去补边；
  `unpad_required_but_skipped=True` 标记原版会移除位置，但本次非零率跳过了这一步。

**padding 与 unpad 不是一回事**：当前非零率仍正常执行图片补边，只是跳过特征
去补边。这次只记录，不更改它们、特征内容、剪枝公式、精度或 FlashAttention2。
所有新增审计运算与 CSV 写入都在模型 generate 计时之外。

`anyres_max_9` 不固定为 3×3；网格由原图尺寸和 checkpoint 候选分辨率选择。
例如在此前 384 的 1～6 倍候选配置下，原图 384×384 选 1×1，1024×1024
选 3×3。每个编码器图块都是 384×384，不能据此判断原图或总网格。

### 3.4：恢复非零率的 AnyRes 行分隔 token

从 `3.4-sparse-row-newlines` 起，默认在 AnyRes 主要 token 的每个
**原始全局空间行**末尾插入 checkpoint 的 learned `image_newline`：
横向相邻 tile 共用一行末尾的分隔，空行也保留分隔，不是每 27 个保留 token
插入一次。Base 不额外逐行插入；contextual token 仍按原来的顺序放在空间
主要 token 序列之后。若配置为 `spatial_unpad_add_newl`，还在整个序列末尾
追加该配置要求的一个 newline。当前用户的 `spatial_unpad` 没有这个额外末尾项。

只改分隔 token：选择预算、合并公式、主要/上下文 token 的相对顺序、
projector 输入布局、FP16、FlashAttention2、Lat/Mem 计量均保持不变。
0% 的完整网格处理与严格校验逻辑不变；非零率仍未恢复 unpad/max9 特征缩放。
这是本地适配改动，不应称为上游原封不动的稀疏拼接。

如果**磁盘原图**为 384×384，且候选分辨率包含用户配置中的 384×384，
就会选择 1×1 AnyRes 网格；无几何补边，也不触发 max9 缩放。
此时 Base 和 AnyRes 各有 729 个空间 token；0% 共 1485 个序列 token，
1% 为 `722 + 722 + 27 = 1471`，恢复前为 1444。
每个编码器视图都是 384×384，不代表磁盘原图也是这个尺寸。
max9 是原版对拼接并 unpad 后的高分辨率特征网格进行条件双线性缩放，
目标量级是 9×729 个空间 token（不含 Base/分隔），并非强制切成 9 个图块。

沿用原路径，重新跑 1%，关闭图片并保留 ACC/PRE/Recall/TNR、Lat/Mem：

```bash
CUDA_VISIBLE_DEVICES=5 python runVisionZip.py \
  --prune-rate 1 \
  --save-prune-vis False \
  --output-dir output/visionzip_01_newline_01
```

新增 `--no-sparse-row-newlines` 可复现 3.3 的非零率无分隔拼接，方便只对比
这一项变化；它不影响 0% 的原版 newline 处理。实验输出目录须为新/空目录。
`run.json` 记录 `sparse_row_newlines`；逐样本 JSON/CSV 记录
`newline_tokens_after`、最终 LLM token 数和实际减少比例。

保持服务器原来的 Python 环境，无需为了此脚本升级 Torch/Transformers。
使用 `CUDA_VISIBLE_DEVICES` 只暴露一张卡，模型需完全放得下，不支持计量时 CPU/disk offload。
一次调用只跑一个剪枝率，处理全部输入图片，没有 100 张上限。

默认路径与现有实验相同；MVTec、70%、保存原脚本的可视化：

```bash
CUDA_VISIBLE_DEVICES=5 python runVisionZip.py \
  --prune-rate 70 \
  --output-dir output/visionzip_70_01
```

对比 Lat/Mem 时关闭可视化（仍保存分类结果和统计）：

```bash
CUDA_VISIBLE_DEVICES=5 python runVisionZip.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --prune-rate 70 \
  --save-prune-vis False \
  --output-dir output/visionzip_70_benchmark_01
```

VisA：优先读取 `split_csv/1cls.csv` 的 `split=test`，同时包含正常与异常；
提示词为去掉类别名称的 v0，保留 `A. Yes / B. No`：

```bash
CUDA_VISIBLE_DEVICES=5 python runVisionZip.py \
  --dataset visa \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv \
  --prune-rate 70 \
  --save-prune-vis False \
  --output-dir output/visionzip_visa_70_01
```

`--visa-split-csv` 可以省略以自动检测；不要同时给 VisA 指定 `--input-json`。
CSV 不完整或缺少正常测试样本会明确报错，不会偷偷从训练集补正常图片。

0% 对照：改为 `--prune-rate 0`，同时换一个新输出目录。
**0% 仍执行 VisionZip 的 attention 捕获、评分、选择、projector 和拼接，不再绕过方法。**
完整预算对应全部 729 个 token 均为 dominant、contextual 为 0。选取函数照常执行，
输出保持原 token 数值和空间顺序，无需合并。版本 `3.2-zero-layout` 在全部 token
原序保留时直接复用原特征张量，保留其 stride、连续性和存储偏移，不再通过
`masked_select`/`cat` 重建 projector 输入。这样避免仅因内存布局变化而改变
FP16 线性层的计算分支；不是绕回参考模型，也没有放宽逐元素相等检查。
完整网格的拼接保留去 padding、
max9 特征缩放、每行分隔 token 和可选的末尾分隔 token，与模型的原输入契约一致。
选择与合并算法不变；3.4 默认给非零率稀疏序列补回行分隔，详见上方说明。
50%、90% 同理分别传入 `--prune-rate 50`、`90`。
只支持整数 0～99，默认值已修正为 **70**。

版本 `3.3-all-rate-layout` 同时对非零剪枝生效：在评分、选择和合并完成之后，
将压缩特征按编码器原来的**内存轴排列**重新存储，再送入 projector。只复制数值，
不改变 token 的逻辑顺序、合并公式、FP16 或 FlashAttention2，也不补回完整长度。
例如编码器 `[B,N,C]` 使用 `[B,C,N]` 存储时，压缩后保持 `[B,C,K]` 存储；
步长随保留长度 K 改变。原输入已经连续时不需要额外拷贝。
0% 仍直接复用原张量，严格一致性检查不变。

这是轻量布局修复，不保证非零剪枝的 projector 输出与未剪枝逐元素一致，
因为输入尺寸不同仍可能改变底层计算。布局复制发生在生成计时内，计入 Lat/Mem。
`run.json` 记录布局策略，逐样本 `predictions.jsonl` 的 `pruning` 下记录
`encoder_feature_stride` 和 `projector_input_stride`，便于在服务器核对实际排列。
原有非零率稀疏 AnyRes 拼接规则不变，因此 0%→1% 的指标差异不能只归因于布局。

### 在服务器上实际核对 0% 是否与原版一致

```bash
CUDA_VISIBLE_DEVICES=5 python runVisionZip.py \
  --prune-rate 0 \
  --verify-zero \
  --output-dir output/visionzip_zero_check_01
```

默认评估完整 MVTec 输入；可指定 `--input-json`/`--data-root`，或使用 `--dataset visa`。
此模式是**独立诊断模式，不测 Lat/Mem、不画图**，每张图运行两次：

1. 候选：完整执行本脚本的 VisionZip 0% 路径。
2. 参考：同一模型权重、同一输入、同一提示词、同一精度和配置下的未剪枝路径。

逐项检查输入像素、prompt token IDs、选择后的视觉特征、projector 输出、
拼接后的 LLM 输入、position IDs、attention mask、完整生成 token IDs 及答案。
张量必须 shape/dtype/逐元素值均相同，不仅比较最终 A/B 或总准确率。
参考路径仅用于检查，不会代替候选结果，也没有失败后回退。

- `zero_equivalence.json`：只有 `status="passed"`、`complete=true` 且检查数量等于
  输入数量，才说明这批服务器实验全部通过。
- `zero_equivalence.jsonl`：逐样本检查结果；首次不一致会记录对应阶段并停止，
  总报告标为 failed，不会假装验证通过。每个阶段另附 CPU 快照前的
  `reference_layout`、`candidate_layout`（stride、连续性、存储偏移、设备）和
  `layout_equal`，用于定位“输入数值相同但计算结果不同”的问题；布局字段仅用于
  诊断，是否通过仍严格检查 shape/dtype/逐元素值和生成结果。终端会列出不一致阶段。
- CPU 快照和额外推理会增加耗时，所以不产生 `summary.csv` 性能统计；
  去掉 `--verify-zero` 可正常统计 VisionZip 0% 的指标。
- 0% 仍有评分、选择等开销，所以即使答案完全一致，Lat/Mem **不必等于原版**。
  评估方法开销时不要把这个耗时当作纯原版 LLaVA 的耗时。

### 比较布局修复后的 0% 与 1%

当前脚本 `python runVisionZip.py --version` 应显示 `3.5-padding-grid-audit`。
3.3 的布局修复继续保留；可加 `--no-sparse-row-newlines` 单独复现旧拼接对照。
先按上面的方式运行 0% 严格校验（输出目录换成新的，例如
`output/visionzip_layout_zero_check_01`）。验证模式没有分类汇总 CSV。
如果要对比 ACC/PRE/Recall/TNR 和 Lat/Mem，再顺序运行以下两个正常实验：

```bash
for rate in 0 1; do
  CUDA_VISIBLE_DEVICES=5 python runVisionZip.py \
    --prune-rate "$rate" \
    --save-prune-vis False \
    --output-dir "output/visionzip_layout_rate_${rate}_01" || break
done
```

沿用默认 MVTec 完整输入及提示词，不限制 100 张；自定义数据时两个实验需使用
相同的模型、输入 JSON、data-root、prompt-version 和 seed。
两份 `summary.csv` 对比总体指标，`prune_00/` 与 `prune_01/` 下的
`classification_results.csv` 按 `question_id` 对齐后可比较 A/B 翻转及 token 数。
1% 不能带 `--verify-zero`，该开关只用于未压缩路径的严格等价检查。

共同设置固定为 AnyRes `anyres_max_9`、`base_first`、FP16、LLM FlashAttention2、
greedy decoding（`do_sample=False`），完整生成答案，不强行只生成一个 token。
默认预热 10 张（小数据集取实际张数），然后**重新从头评估全部样本**。
预热不进入任何分类或时间/显存统计，`--warmup` 可调整为其他正整数。
输出目录必须不存在或为空，不覆盖旧结果。

## 结果及计量口径

```text
output/visionzip_70_01/
  run.json
  summary.csv
  prune_70/
    metrics.json
    predictions.jsonl
    classification_results.csv
    visualizations/                 # --save-prune-vis True，0% 也可保存
      sample_<ID>/image_0/
        comparison.png
        view_00_global.png
        view_01_tile_01.png ...      # 每个真实编码视图
        tiles_01.png ...            # 每页最多 4 个 AnyRes tile
        decisions.json
```

- `summary.csv` / `metrics.json`：ACC、PRE、Recall、TNR、未解析数、Lat 均值/P50/P95、
  PyTorch 峰值 allocated/reserved 显存、allocated 占显卡总容量的百分比等。
  四个分类指标是 0～1 小数，Lat 汇总是毫秒，显存汇总是 GiB。
- `predictions.jsonl` / `classification_results.csv`：逐样本答案、GT、预测、Lat（秒）、
  生成时间、峰值显存（字节）和实际 token 数；不塞入逐 token 详细数组。
- ACC 把未解析回答算错；PRE/Recall/TNR 只用可解析 A/B。零分母为 null。
- Lat：输入已在 GPU 上，从进入 `model.generate` 到第一个 token 选出并同步。
  包括视觉编码、projector、VisionZip 排序/聚合/拼接和 LLM prefill；
  不含加载模型、读取图片、CPU 预处理、输入传 GPU、保存图片/JSON。
- Mem：完整答案生成期间的 PyTorch 峰值，包含已驻留权重与输入。
  不等于 `nvidia-smi` 整卡占用，也不是 GPU 计算利用率；reserved 包含缓存。
- 预热和评估使用同一被动计时路径，观测器不会改变 logits、答案长度或提前停止。
  `run_wall_seconds` 是整个运行阶段的墙钟时间，包含加载、预热和保存，不要当作 Lat。
- 画图/CPU 导出发生在生成计时之后；开启可视化仍需额外保留 GPU 决策数组，
  会影响计量开销。与 `benchmark_fastv.py` 对比时使用 `--save-prune-vis False`。
  关闭时，拼接直接使用 GPU 索引，不复制详细决策到 CPU。
- 中途失败保留已完成样本，并将统计标为 `status=failed, complete=false`。

## 算法边界

这是 **VisionZip 的 SigLIP/OneVision 适配**，不是宣称作者已提供的完整 OneVision 版本。
依据原脚本的官方提交 `8f86b55c6f000eb033e6912538af2dd7dcb30502`：

- 当前 vendor 的 SigLIP 有 26 个活跃 block；使用最后一个活跃 block 已计算好的
  attention 概率，对 head 和 query 求均值，给无 CLS 的 token 排序。
  不读取 LLM attention，不改 LLM 的计算精度或 attention 实现。
- 每个 Base/AnyRes 视图独立保留 `ceil(729 × (1-rate/100))` 个输出 token，
  有压缩时按 54:10 分成 dominant/contextual。70% 对应 185+34=219 个/视图；
  完整保留时为 729+0，选择结果是保持空间顺序的恒等变换。
- 保留原脚本的 K 跨头平均后 L2 归一化、等间隔 anchor、
  `anchor hidden + mean(assigned OTHER hidden)` 聚合和稀疏 AnyRes 排序；
  未把聚合偷偷改为包含 anchor 的均值。
- 非零率使用稀疏拼接；3.4 默认按原始空间行恢复分隔 token，
  但仍不执行 unpad 和 max9 特征缩放。contextual token 仍保留在末尾。
  因此 `--prune-rate 70` **不等于最终 LLM 视觉序列恰好减少 70%**。
  每张图单独记录原版 packed 数量、最终数量和实际减少比例。
  很低的名义剪枝率甚至可能让最终序列比原版更长。
- 可视化是 dominant、contextual anchor、merged source 三种状态的透明度图，
  不是 attention 热图，也不是将原图对应像素彻底删除；合并源仍贡献信息。

参考：
[官方代码固定版本](https://github.com/JIA-Lab-research/VisionZip/tree/8f86b55c6f000eb033e6912538af2dd7dcb30502)，
[论文无 CLS 算法](https://arxiv.org/html/2412.04467v1#A2)。
原脚本的版权声明和 Apache-2.0 许可证保留在文件内。

## 本地验证

```bash
python runVisionZip.py --self-test --output-dir output/visionzip_synthetic_01
python -m unittest discover -s tests -p test_visionzip.py -v
```

`--self-test` 只生成明确标记为 synthetic 的几何/透明度示意图，不加载模型、不需要 GPU。
单元测试使用 CPU 小张量、模拟后端和随机初始化的小 Qwen；不代表验证了服务器
7B checkpoint 的准确率、CUDA 兼容性或真实 Lat/Mem。
0% 已测试：不调用参考 prefill 的情况下执行选择与拼接；不同长宽比、padding、
max9 缩放、分隔 token 配置和 dtype 的拼接与参考一致；真实 SigLIP attention 模块
搭配非连续残差特征、MLP projector 和小 Qwen 的完整配对检查；FP16/FP32 特征的
原存储及非零偏移保留。FP16 CUDA/7B 完整数据仍需上面的 `--verify-zero` 实测。
非零率还覆盖：FP16/FP32 六种内存轴排列的无损重排；1/50/70/90/99% 的压缩数值
和决策不变；1% 使用非连续 projector 输入的小 Qwen 完整生成。
