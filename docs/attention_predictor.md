# 第2层旁路预测第22层 attention

这是本项目的可训练实验方法，**不是 MAP 或完整 TwigVLM 的复现**。结构初始化参考
[TwigVLM 原论文 §3.1](https://arxiv.org/html/2503.14075v2#S3.SS1)：K=2、T=3 时，
从当前主模型**第3、4、5个 decoder block**复制三层独立旁路，保留原尺寸、预训练权重和原生
RMSNorm/GQA/RoPE/FFN。主干与旁路不共享可训练参数；不是从第20～22层初始化。
仍固定第22个 LLM block 作为 attention 教师，读取第2个 block 的原始输出，以旁路第三层
attention 评分，在第2层后对原始特征做物理剪枝。不使用 LM head、答案监督、FinalWipe 或
推测解码。主模型权重、FP16、FlashAttention2 均不改。

## 先训练：VisA 全部 CSV 图片，内部训练/验证

```bash
CUDA_VISIBLE_DEVICES=5 python train_attention_predictor.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --epochs 1 \
  --output-dir output/attention_predictor_copied_train_01
```

只使用一个可见 GPU。宽度、Q/KV heads、FFN、激活函数、RMSNorm epsilon、RoPE 和 attention 后端
全部继承当前主模型。你的 Qwen2-7B 配置对应宽度3584、28个Q heads/4个KV heads、FFN18944；
不再接受 `--hidden-size`、`--num-heads`、`--intermediate-size`、`--dropout`，以免误把原尺寸模型改小。
优化器默认对齐 **原版 TwigVLM ICCV2025 附录A、表5**（不是2026年的 TwigVLM++ 扩展版）：

| 参数 | 新默认值 |
| --- | --- |
| epochs | 1 |
| optimizer | AdamW，betas=(0.9, 0.98) |
| 峰值 learning rate | 5e-5 |
| weight decay | 0 |
| 有效 batch size | 128，单卡 microbatch=1 × gradient_accumulation=128 |
| warmup | 总优化器更新步数前3%，线性预热，步数向上取整 |
| decay | warmup之后 cosine 衰减至0 |

损失仍是 **attention KL**，数据仍是 VisA，不引入论文的答案预测训练。只对齐优化参数，不能把它
称作完整 TWIG 复现，也不能保证1轮是此数据与目标下的最优轮数。梯度裁剪保持本项目原有的1；
优化器只接收有效路径的可训练参数，foreach=False 以降低临时显存。
可用 `--epochs`、`--learning-rate`、`--weight-decay`、`--adam-beta1`、`--adam-beta2`、
`--warmup-ratio`、`--gradient-accumulation` 修改，`--help` 可查看。命令行显式参数会覆盖默认值，
所以旧命令中的 `--epochs 5` 需要删掉或改成1。

学习率使用项目已安装 Transformers 的 `get_cosine_schedule_with_warmup`，不升级依赖。
步数为 `epochs * ceil(train_images / gradient_accumulation)`；仅在每次 `optimizer.step()` 后
推进一次 scheduler，不按图片推进，不在验证时推进，跨epoch不重置。最后不足128张的组按实际
张数取平均、更新一次，不丢样本或强行补齐。梯度累计不会同时存128张的计算图。
warmup从0学习率开始，因此首组日志 `lr=0` 是调度定义，不是卡住；以后逐步上升。极小测试集
如果 warmup 覆盖了全部更新步数会在加载主模型前报错；可显式减小累计张数或设
`--warmup-ratio 0` 做 smoke test，不会默默跳过训练。

原始模型始终 FP16，只有学生使用默认 BF16 autocast（独立 FP32参数和优化器状态）；最后一行
attention 的 QK、softmax、KL 为 FP32。学生前两层使用复制的原生 FlashAttention2，默认启用
非重入梯度检查点（仅训练时重算这两层以节省激活显存）；用 `--no-gradient-checkpointing` 关闭，
占用会更高。主模型不启用训练/梯度检查点，也不改变推理计算。
**原尺寸旁路比旧512维版更占显存，best/last checkpoint 也明显更大**；40GB显卡能否容纳仍需按实际
AnyRes序列长度实测，不能凭CPU测试保证。非 BF16 显卡或指定 float32 时不会静默更换后端；复制的
FlashAttention2 学生明确要求 BF16 autocast。float32 选项仅用于已有非Flash后端的调试/CPU测试。

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
- 学生：没有额外输入 LayerNorm 或降维投影，直接输入完整 H2。旁路前两层完整执行复制的原生
  block3、block4；第三层保留完整 block5 副本，但只执行自身 input RMSNorm、Q/K、RoPE 和最后
  query 的 attention。第三层 V/O/FFN/post-attention norm 位于该 attention 之后，不影响这个
  损失，故**冻结、不执行，但仍随完整层副本保存**。其余旁路参数可训练；不复制答案预测头。
- 旁路使用原始序列 position IDs 和 causal mask；新版复制原生 Q/KV heads 数，教师和学生都先 mean Q heads。
  损失为 `KL(teacher || student)`，在所有空间视觉 token 上计算，只反传到旁路。
- 前两层使用与主模型相同的原生 decoder/attention 类，不请求 attention 输出，不创建 KV。
  第三层独立计算可微分的一行 QK，处理原生 GQA 与 Q/K bias，避免请求或构造完整 attention 矩阵；
  未使用教师评分函数的 no_grad 装饰器。主干仍使用原有 FlashAttention2。

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
每条进度还显示当前 `loss=...(KL)`、`lr`（该图所在累计组实际使用的学习率，验证时为 --）以及
`step=已完成优化器更新/计划总更新数`；阶段结束显示 `mean_loss`，训练结束显示最佳验证 loss。
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

- `train_run.json`：配置、架构、教师层、推理模式、来源和划分哈希，以及优化器betas、峰值LR、
  有效batch、cosine/warmup计划与准确更新步数，不导出完整 attention 数组。
- `train.jsonl`、`validation.jsonl`：固定图片名单，没有 GT/mask 字段。
- `history.csv`：每轮训练/验证 KL、耗时、Top-k 重合率、保留的教师 attention 总质量。
  这里 Top-k 指空间 token 移除50/70/90的拟合诊断，**不是最终含 newline 的层平均预算**。
  训练诊断仍比较全部空间token的原始旁路输出，不是下述 Base-guided 推理映射后的选取结果。
  另含 `optimizer_steps`（该阶段实际更新数；验证为0）、`global_step`（累计更新数）、
  `lr_next`（该阶段结束后下一次更新将使用的学习率，不是该阶段平均LR；最终为0）。
- `best.pt`：以内部 validation KL 选择的最佳旁路；`last.pt`：最后一轮旁路。
  包含架构、精度、教师层、主模型结构签名和来源，不保存主模型权重。
- `method.json`：相对路径指向同目录 `best.pt`，可直接交给 `run.py`。
- `training_status.json`：已完成 epoch、最佳 validation KL、实际/计划优化器更新步数和是否全部完成。

不使用 MVTec 测试准确率挑选 checkpoint。当前不实现优化器恢复；中断后可保留 best/last 做评估，
新训练请使用空输出目录。指定相同 cache-dir 时可复用通过身份校验的教师特征。

新训练写 **version=2** checkpoint，包含完整三层副本及原生 Qwen2 架构/后端和初始化层号。
旧512维 **version=1** checkpoint 仍可用于旧方法评估，不会被伪装成新版；新版必须重新训练，
不支持将旧权重直接转换或恢复优化器。教师 H2/attention 的口径没有变，身份一致的旧教师缓存可复用。

## 再评估：MVTec 平均剪枝率50/70/90

**当前默认仅修改推理选取，无需重新训练。** 已有 v1/v2 checkpoint 和训练目录下的
`method.json` 可以继续使用；旧 JSON 没有 `selection_mode` 时默认使用新策略 `base_guided`。
这不涉及 `runMyPruningPoolSpatialBaseFull.py`，训练 KL、教师层及权重格式不变。

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 --roi-mode anyres_max_9 \
  --method attention_predictor \
  --method-config output/attention_predictor_copied_train_01/method.json \
  --no-sample --no-plots --benchmark-metrics --warmup 10 --seed 42 \
  --output-dir output/attention_predictor_base_guided_mvtec_01
```

评估用与训练一致的 backbone 权重，不是把旁路用于另一个同尺寸但不同权重的模型。
读取 checkpoint 时校验版本、宽度、深度和实际 AnyRes packing/order；结构签名校验不能证明权重内容相同。
非零率仅支持 batch1、单张完整 AnyRes 图、无序列 padding、spatial_unpad(_add_newl)、greedy DynamicCache。

在第2层后，旁路输入仍是**完整 H2（Base、AnyRes和文本都在）**，不是只输入Base：

1. 仅取第三层对 **Base 空间 token** 的预测attention，归一化为Base图上的分布。
2. 在FP32中将 Base **概率图** 双线性插值到实际打包的整个AnyRes网格，
   `align_corners=False`。使用模型真正unpad/max9缩放后的行列数，不假定原始crop一定3×3，
   不对log分数直接插值，也不读取AnyRes自身预测分数参与排名。
3. Base、**整个AnyRes** 各自独立Top-k，采用相同连续空间保留比例；AnyRes没有逐crop固定配额，
   同一AnyRes大图内的不同crop保留数可以不同。分数相同时按原始行优先空间次序决定。
4. **从原始 H2 选择，而非从旁路输出选择特征**，继续运行原 LLM block3..D。

原始 position IDs 保留，所有已写入 KV 缓存物理压缩，新 token 位置从原始 prompt 长度继续。
newline 不参与排名但始终保留并计入层平均成本。为不改变原有层平均总预算，先计算packed总名额
`T=floor(N*keep_ratio)`，预留P个newline后，以统一的 `q=(T-P)/(B+A)` 分给Base的B个和AnyRes的A个
空间token：两组连续名额分别是qB、qA，最大余数法取整且总数严格为T-P（余数相同时Base优先）。
两组整数误差各小于一个token，不按预测attention质量跨组抢占名额。不足以为两组各保留至少一个
空间token时明确报错，不通过削减newline或放宽目标兜底。

评估每个非零率额外写 `prune_XX/view_budgets.csv`：每图两行Base/AnyRes，包含各自空间token的
原始/保留/删除数量、实际剪枝率、共同空间保留比例、保护newline数。`run.json` 记录选取策略；
预测JSON仍只保留原有精简字段，不塞入分数数组。

如需复现旧的“各用自己的预测分数＋联合Top-k”，在同一 `method.json` 显式加
`"selection_mode": "joint"`；新策略为 `"selection_mode": "base_guided"`。不修改checkpoint本身，
可用同一权重对照两种推理策略，但请使用不同的输出目录。

设原 packed 视觉数 N，剪后保留 K，深度 D：

```text
平均剪枝率 = 100 * (1 - (2*N + (D-2)*K)/(D*N))
单次目标 = 平均目标 * D/(D-2)
```

28层下平均50/70/90对应单次53.8462/75.3846/96.9231%。用 floor 整数预算，输出另报实际单次/平均比例，
并提供 `layer_tokens.csv` 独立核查。注意上述单次比例包含受保护newline；两组**空间**token的
共同实际剪枝率还会因newline预留而略高，应以 `view_budgets.csv` 为准。
旁路的额外计算不算在这28层 token 平均数里，因此
**平均剪枝率不等于真实 FLOPs 或加速比例**。

ACC/PRE/Recall/TNR、Lat、峰值 Mem 和预算字段沿用公共 `summary.csv` / `metrics.json` / 精简 predictions。
Lat 包含视觉、projector、packing、主干 prefill、旁路评分与剪枝至首个 token；Mem 包含旁路常驻参数
及其临时激活，统计完整答案生成。不会把旁路计算排除后宣称真实加速。
旁路权重每个模型只加载一次并跨样本/剪枝率复用；0%不加载/运行旁路，走原生主干路径。

需要图片时，用新输出目录，将 `--no-plots --benchmark-metrics --warmup 10` 替换成
`--save-prune-vis --save-attention-vis`；可用 `--visualize-samples` 限制保存 ID，不减少评估样本。
仍只有 `comparison.png`、`attention_overlay.png` 两类 PNG，图上带 GT/Pred；Base为预测attention，
AnyRes热图为Base映射分数，图片说明明确标注，不冒充AnyRes自身attention或缺陷概率。
可视化统一缩放分数只用于显示，不影响组内排序。原 CLSE/FastV/PriorTR 配置不变。

## 本地验证范围

覆盖原生第3/4/5层权重精确复制且无参数别名、完整结构/GQA/RoPE/预训练初始化、第三层真实可微QK、
有效学生路径梯度/无效尾部冻结、梯度检查点前向反向等价、冻结教师无梯度、H2及教师层位置、真实 eager attention
oracle、KL 小样本拟合、尾部梯度累计、真实预处理和打包、不生成教师答案、checkpoint/缓存身份、
VisA全行无GT读取与内部划分、真实 generate/原始位置/KV压缩、0%身份、跨模式清理、完整训练 CLI
与共享评估统计接线，以及Base-only排名输入、概率插值数值oracle、实际矩形/AnyRes-first/末尾newline
几何、两组同率精确总预算、组间质量不抢预算、旧joint选项与分组CSV。可选 native CUDA FP16 +
FlashAttention2 / BF16 学生训练测试需要服务器：

```bash
python -m unittest discover -s tests -p 'test_*predictor*.py' -q
```

CPU 小模型验证不代表 VisA 已训练或 MVTec 准确率/真实 GPU 加速已复现。
