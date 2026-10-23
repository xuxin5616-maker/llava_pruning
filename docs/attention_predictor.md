# 第2层旁路预测第22层 attention

这是本项目的可训练实验方法，**不是 MAP 的复现**。沿用 attention 蒸馏思路，但按照本实验选择：
固定第22个 LLM block 作为教师，读取第2个 block 的原始输出，以三层轻量 Transformer 的
第三层 attention 评分，在第2层后对原始特征做物理剪枝。主模型权重、FP16、FlashAttention2 均不改。

## 先训练：VisA 全部 CSV 图片，内部训练/验证

```bash
CUDA_VISIBLE_DEVICES=5 python train_attention_predictor.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --epochs 5 \
  --output-dir output/attention_predictor_train_01
```

只使用一个可见 GPU。默认宽度512、8 heads、FFN2048、dropout0、AdamW lr=1e-4、weight_decay=.01、
每张图一个 microbatch，累计4张再更新，梯度裁剪1。以上是可调的项目起始值，**不是论文公开超参**。
用 `--hidden-size`、`--num-heads`、`--intermediate-size`、`--learning-rate`、`--gradient-accumulation`
等修改；`--help` 可查看。原始模型始终 FP16，只有学生使用默认 BF16 autocast（FP32参数和优化器状态）；
最后一行 attention 的 QK、softmax、KL 为 FP32。无 BF16 GPU 时显式选 `--student-precision float32`，
可能更慢、占用更多显存，不会自动改主模型的 dtype/backend。

CSV 至少包含 `object,image`，`image` 为相对 `--data-root` 的真实图片路径。
使用本地**完整** CSV 所有行（原来的 train 和 test 合并为训练数据源），不读取 label、gt、mask。
没有完整 CSV、路径缺失、重复路径或 symlink 别名都会报错，不回退到现有 test-only 读取器或目录猜测。
按类别、seed42 新划分90% train / 10% validation，至少各留一张；固定名单和哈希保存。
**这些 VisA 图片不能再作为独立/零样本测试集，最终评估使用独立 MVTec。**

所有图片使用现有 `GENERIC_PROMPT_V0`：

```text
Are there any defects in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly.
```

## 教师与学生到底计算什么

- 学生输入：完整 multimodal sequence 的 **block2 原始输出 H2**，包含系统、视觉、提示词等 token；
  不是最终 RMSNorm 后特征，不把生成答案或 GT 喂给学生。
- 教师：原生 decoder block1..21 正常前向，用 block22 的输入经自身 input_layernorm、Q/K、RoPE
  独立计算最后一个 prompt query 的 attention。因为目标是 block22 内部的 attention，**不必计算
  block22 的输出**；没有运行答案生成，也没有保留 teacher KV 或请求 native attention 输出。
- 教师按原 FastV 口径：先对所有 prompt keys softmax，再 mean heads，取空间视觉 tokens，
  排除结构 newline 后 L1 归一化为目标分布。Base 和 AnyRes 联合，不各自归一化。
- 学生：LayerNorm＋降维投影，前两层为完整小型 pre-norm causal Transformer（QKV/O、残差、FFN）；
  第三层只执行自身的 norm、Q/K、RoPE、最后 query 的 attention。它的 V/O/FFN 位于目标 attention
  之后，不会影响该分数，故不实例化、不浪费训练。**不是额外 MLP 评分头，也不是三层原尺寸 Qwen2。**
- 旁路使用原始序列 position IDs 和 causal mask，教师和学生 heads 数可以不同，都是先 mean heads。
  损失为 `KL(teacher || student)`，在所有空间视觉 token 上计算，只反传到旁路。
- 前两层 side attention 用 PyTorch SDPA，第三层仅一行 QK，避免主动构造完整 attention 矩阵。
  主干仍使用原有 FlashAttention2。

Layer2/22 都指 **1-based 完成的 block / block 内部 attention**。本版本输入位置固定2；教师默认22，
可用 `--teacher-layer` 做独立教师层实验，实际值保存在 checkpoint。

## 提取成本与可选缓存

默认不缓存 H2，每一轮会重新做教师前向，第一张和每20张都有进度。这不是卡住；原模型加载、
视觉编码、21层完整视觉前向都会花时间。训练无须生成完整答案。

默认首张、每20张、每阶段最后一张会立即打印进度，包含 `elapsed`（训练循环已用时间）、
`phase_ETA`（当前训练/验证阶段预计剩余）、`total_ETA`（所有剩余 epoch 的训练＋验证预计剩余）。
用 `--log-every 1` 可每张打印。估计使用最近样本速度，并区分训练/验证；未见过的阶段暂借当前
速度，标注 provisional。缓存首次提取和随后命中速度不同，ETA 会动态修正；不是准确承诺，
不计尚未知的未来 checkpoint 保存开销。日志不增加额外 CUDA 同步，不改训练或 Lat/Mem 协议。
每条进度还显示当前 `loss=...(KL)`；阶段结束显示 `mean_loss`，训练结束显示最佳验证 loss。
这些均为原有 KL 蒸馏损失，没有新增损失项或改变优化目标；history.csv 中仍使用 `kl` 列。

有充足磁盘时可以加：

```bash
--cache-dir output/attention_teacher_cache
```

缓存包含 H2、原始位置、空间索引和教师目标（无 GT）。**H2 很大**：单图大致为
`prompt_tokens * backbone_hidden_size * 2 bytes`（默认 FP16），上万张可能数百 GB。
这里只在明确指定参数时创建；不会默认把所有特征落盘。首次提取仍慢，后续 epoch 命中缓存后省去教师前向。
缓存 key 包含提示词、层号、推理配置、主/视觉模型文件身份、图片大小/mtime，避免串用旧参数结果；
这是大小/mtime＋小配置哈希的缓存失效机制，不是全量权重/图片内容的密码学认证。

## 输出

- `train_run.json`：配置、架构、教师层、推理模式、来源和划分哈希，不导出完整 attention 数组。
- `train.jsonl`、`validation.jsonl`：固定图片名单，没有 GT/mask 字段。
- `history.csv`：每轮训练/验证 KL、耗时、Top-k 重合率、保留的教师 attention 总质量。
  这里 Top-k 指空间 token 移除50/70/90的拟合诊断，**不是最终含 newline 的层平均预算**。
- `best.pt`：以内部 validation KL 选择的最佳旁路；`last.pt`：最后一轮旁路。
  包含架构、精度、教师层、主模型结构签名和来源，不保存主模型权重。
- `method.json`：相对路径指向同目录 `best.pt`，可直接交给 `run.py`。
- `training_status.json`：已完成 epoch、最佳 validation KL 和是否全部完成。

不使用 MVTec 测试准确率挑选 checkpoint。当前不实现优化器恢复；中断后可保留 best/last 做评估，
新训练请使用空输出目录。指定相同 cache-dir 时可复用通过身份校验的教师特征。

## 再评估：MVTec 平均剪枝率50/70/90

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 --roi-mode anyres_max_9 \
  --method attention_predictor \
  --method-config output/attention_predictor_train_01/method.json \
  --no-sample --no-plots --benchmark-metrics --warmup 10 --seed 42 \
  --output-dir output/attention_predictor_mvtec_01
```

评估用与训练一致的 backbone 权重，不是把旁路用于另一个同尺寸但不同权重的模型。
读取 checkpoint 时校验版本、宽度、深度和实际 AnyRes packing/order；结构签名校验不能证明权重内容相同。
非零率仅支持 batch1、单张完整 AnyRes 图、无序列 padding、spatial_unpad(_add_newl)、greedy DynamicCache。

在第2层后，旁路对完整 H2 评分；依据第三层 attention 联合选 Base＋AnyRes token，
**从原始 H2 选择，而非从旁路输出选择特征**，继续运行原 LLM block3..D。
原始 position IDs 保留，所有已写入 KV 缓存物理压缩，新 token 位置从原始 prompt 长度继续。
newline 不参与排名但始终保留并计入层平均成本；不足以容纳结构 token 的预算明确报错。

设原 packed 视觉数 N，剪后保留 K，深度 D：

```text
平均剪枝率 = 100 * (1 - (2*N + (D-2)*K)/(D*N))
单次目标 = 平均目标 * D/(D-2)
```

28层下平均50/70/90对应单次53.8462/75.3846/96.9231%。用 floor 整数预算，输出另报实际单次/平均比例，
并提供 `layer_tokens.csv` 独立核查。旁路的额外计算不算在这28层 token 平均数里，因此
**平均剪枝率不等于真实 FLOPs 或加速比例**。

ACC/PRE/Recall/TNR、Lat、峰值 Mem 和预算字段沿用公共 `summary.csv` / `metrics.json` / 精简 predictions。
Lat 包含视觉、projector、packing、主干 prefill、旁路评分与剪枝至首个 token；Mem 包含旁路常驻参数
及其临时激活，统计完整答案生成。不会把旁路计算排除后宣称真实加速。
旁路权重每个模型只加载一次并跨样本/剪枝率复用；0%不加载/运行旁路，走原生主干路径。

需要图片时，用新输出目录，将 `--no-plots --benchmark-metrics --warmup 10` 替换成
`--save-prune-vis --save-attention-vis`；可用 `--visualize-samples` 限制保存 ID，不减少评估样本。
仍只有 `comparison.png`、`attention_overlay.png` 两类 PNG，图上带 GT/Pred；分数图为预测 attention，
不是缺陷概率。原 CLSE/FastV/PriorTR 配置不变。

## 本地验证范围

覆盖可微分第三层真实 QK、全部学生参数梯度、冻结教师无梯度、H2及教师层位置、真实 eager attention
oracle、KL 小样本拟合、尾部梯度累计、真实预处理和打包、不生成教师答案、checkpoint/缓存身份、
VisA全行无GT读取与内部划分、真实 generate/原始位置/KV压缩、0%身份、跨模式清理、完整训练 CLI
与共享评估统计接线。可选 native CUDA FP16 + FlashAttention2 / BF16 学生训练测试需要服务器：

```bash
python -m unittest discover -s tests -p 'test_*predictor*.py' -q
```

CPU 小模型验证不代表 VisA 已训练或 MVTec 准确率/真实 GPU 加速已复现。
