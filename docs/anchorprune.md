# AnchorPrune：独立 OneVision／SigLIP 对照

使用 `run.py --method anchorprune`。这是独立剪枝方法，在所有 LLM block 之前一次性选择图像
token；不使用 MyPool、蒸馏 checkpoint、λ 或额外 tail 删除。输入数据、v0 prompt、贪心解码、
模型权重、FP16 和 FlashAttention2 均复用现有后端。

## 来源和实现边界

- 论文：Kyuan Oh、Bumsoo Kim，
  [AnchorPrune v5](https://arxiv.org/abs/2607.07033v5)，§3、附录 A.2～A.4。
- 官方代码：[MULTI-cau/AnchorPrune](https://github.com/MULTI-cau/AnchorPrune)，
  提交 `0e0098b591133433f787e870383ca7ad406e2486`。
- 原样保存的核心选择器：`third_party/anchorprune/selection.py`，附完整 Apache-2.0 许可证，
  `SOURCE.json` 记录源提交和文件 SHA256；没有安装另一套 Transformers 或替换模型实现。

官方公开适配器包含 CLIP/LLaVA-1.5 和 Qwen2.5-VL，没有 OneVision/SigLIP 的直接复现配置。
本实现沿用论文 non-CLIP 分支及多视图流程，以下内容属于明确的 OneVision 适配：

1. 原生 SigLIP 前向和 projector 生成完整特征；原生 AnyRes 执行 unpad、条件 max9 缩放、newline
   和视图拼接。随后从**已经打包的投影后特征**中选择，因此未剪枝分母和实际 LLM 输入一致。
2. 将问题正文单独 tokenize，使用本模型的 input embedding。投影后的图像 token 和每个问题
   token 各自 L2 归一化，图像 token 的锚点优先级为其与所有问题 token 的**最大余弦相似度**。
   这是论文公式 (9)，不使用 CLIP 分支的负余弦，也不使用蒸馏预测 attention。
3. 旁路捕获 SigLIP 原生 Q/K 投影结果。默认 active block 26，在独立 FP32 分块计算中执行
   非因果 QK、按 key softmax，再对 heads 和 queries 取平均，得到 received-attention 重要性。
   原生视觉/LLM attention 前向、计算精度和返回值均不替换；不请求 LLM attention 张量。
4. received-attention 使用与原生特征相同的 unpad 和双线性 max9 处理。Base 和每个 AnyRes
   crop 作为视觉单元；max9 后跨边界 token 的单元归属采用中心最近邻（nearest-exact），连续
   重要性仍用双线性插值。没有跨 token 的特征合并。
5. 在每个视觉单元中用官方 `adaptive_relevance_anchor` 建立锚点。单元预算按其在实际打包
   候选中的数量比例分配，用最大余数凑整保持总量；再用官方 `importance_weighted_expansion`
   在 **Base＋全部 AnyRes** 上全局扩展。crop 最终保留量可不同，Base 不额外受保护。
6. 所有选中的图像 token 恢复原始打包顺序，再与文本拼接。LLM 位置编号按照压缩后的序列
   连续生成；这发生在进入 LLM 之前，没有中途修改已有 KV 缓存的过程。

参数 `tau=0.2`、累计 `patience=3`、单元锚点上限 `floor(K_unit/2)` 来自论文。论文的固定
`K_min/K_unit=5/32` 预算表被扩展到动态 OneVision 预算：默认最小锚点数为
`ceil(K_unit*5/32)`，不超过该单元锚点上限；极小单元按官方行为至少保留一个有效锚点。
这些预算扩展与 OneVision 评分适配会写入 `run.json`，不应将结果称为论文原始 backbone 的直接复现。

## 平均剪枝率

沿用现有定义：对每个样本，原生未剪枝打包图像段长度为 `N0`，包括 Base、unpad/max9 后的
AnyRes 和全部图像 newline。若 LLM 有 `D` 层，第 `l` 层输入的图像 token 数为 `N_l`，则

```text
r_average = 1 - sum(N_l) / (D * N0)
```

AnchorPrune 在第一个 LLM block 前剪枝，每层接收相同的 `N_keep`，因此

```text
N_keep = floor(N0 * (1 - requested_rate / 100))
K_spatial = N_keep - protected_native_newlines
r_actual = 1 - N_keep / N0
```

newline 全部保留，但占用目标保留预算，不在分母中偷偷排除。若预算不足以保留 newline 和
至少一个空间 token，会在运行前检查中报错，不自动放宽目标或跳过样本。实际比例因整数取整
会有少量误差，每张图以及每档汇总均记录 `actual_average_prune_rate`。每层第一次 prefill 的
实际序列长度也被核对，确保所报告比例对应真实的 LLM 输入。这是 token 数平均比例，不是 FLOPs 比例。

0% 使用原生图像准备和生成路径，不安装 Q/K 捕获和剪枝包装，不计算相关性或选择集合。

## 一键运行：单卡5

默认 `configs/anchorprune.json` 运行 1%、50%、70%、90%。按需要修改其中的
`prune_rates`（百分数，50 表示 50%，0.5 表示 0.5%）。

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --roi-mode anyres_max_9 \
  --method anchorprune \
  --method-config configs/anchorprune.json \
  --no-sample \
  --benchmark-metrics \
  --summary-format compact \
  --baseline-acc 91.18 \
  --warmup 10 \
  --seed 42 \
  --no-plots \
  --output-dir output/anchorprune_average_gpu5_01
```

一个模型顺序跑各剪枝率；每档单独预热，然后计算 JSON 中全部样本。没有100张限制。本适配器
首版只导出结果，不生成样本可视化。首次运行使用新的输出目录；普通 `run.py` 不提供断点续跑。

## 输出、指标和时间

顶层 `summary.csv`：

```text
config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct
```

ACC、BACC、F1 为百分数，relative_accuracy_pct 为 `100*ACC_fraction/0.9118`。
参考值可用 `--baseline-acc` 修改。完整的 PRE/Recall/TNR、平均/分位数 Lat、峰值 allocated/
reserved Mem、目标/实际平均剪枝率均保留在各档 `prune_XX/metrics.json`；该 JSON 的分类指标
仍为 0～1 小数。逐图答案和 Lat/Mem 在 `predictions.jsonl`，配置和来源在 `run.json`。

Lat 为准备好的驻留输入进入 `model.generate` 到首个生成 token 的同步时间。包含视觉编码、
projector、原生打包、独立 attention、相关性、锚点选取、上下文扩展和 LLM prefill；不扣除
剪枝计算。模型加载、文件读取、CPU tokenize/图像预处理、主输入 H2D、预热和保存不计入。
问题 embedding 查表及该评分分支的动态数据传输位于计时范围内。

Mem 为完整回答期间 PyTorch 的峰值 allocated GiB，包含驻留权重和输入，也包括本方法的临时
评分张量；不是 GPU 利用率。保留的图片 token 越少不代表实际 Lat 必然越低，官方 greedy
上下文扩展也有计算开销，应以保存的实测值为准。

本地验证包括独立 QK 与完整 softmax 对照、多视图锚点和全局扩展、原生 unpad/max9/两种视图
顺序对齐、0% 逐项 KV/生成一致、小型真实 LLM 的物理剪枝与后续生成，以及共用 CSV 输出。
7B checkpoint 在 CUDA/FlashAttention2 上的准确率和速度仍需在服务器上运行上述命令测量。
