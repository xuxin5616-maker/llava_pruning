# PriorTR：OneVision/Qwen2 适配

论文：[Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction](https://arxiv.org/abs/2606.24156)，ECCV 2026。
作者代码：[CodeChildCZJ/PriorTR](https://github.com/CodeChildCZJ/PriorTR)，固定提交
`e442a4c14af1b333067a6b8310c9111cad922033`。参考源码、哈希与 Apache-2.0
许可证在 `third_party/priortr/`，运行时不导入这些原版 LLaMA 文件。

## 接入方式与默认值

使用现有入口 `run.py --method priortr --method-config configs/priortr.json`。
模型仍是已有的 OneVision/Qwen2，FP16 + FlashAttention2；不升级依赖、不换权重，
不读取 Transformer 返回的 attention，也不将其改为 eager。评分在目标块的输入
hidden states 上独立重算该块的 layernorm、Q/K、GQA 和 RoPE；主块正常前向后执行剪枝。

`layer: 2` 是 **1-based 的已完成块编号**：计算第 2 块的 attention，完成该块后裁剪，
第 3 块接收短序列。它不等同于现有 FastV 的 layer 参数索引约定。
当前默认 `rate_semantics: "layer_average"`，遍历平均剪枝率 50%、70%、90%。
这些数值是跨 LLM 各层的平均删除比例，不是保留率，也不是该层瞬间删除的比例。

设实际模型共 D 层，在第 L 层完成后一次删除 r，则平均剪枝率
`r_avg = r*(D-L)/D`。代码读取真实深度并换算 `r = r_avg*D/(D-L)`，
而不是把 28 层写死。28 层、L=2 时对应的实际请求删除率为：

| 目标平均剪枝率 | 第 2 层完成后请求删除率 |
|---|---:|
| 50% | 53.84615384615385% |
| 70% | 75.38461538461539% |
| 90% | 96.92307692307692% |

不可达预算会在开始推理前明确报错。例如 L=3、D=28 时，即使全删，平均上限也只有
89.2857%，所以无法请求平均 90%。旧配置省略 `rate_semantics` 时仍表示
`final_cumulative`；显式设为该值可沿用原先按最终删除率配置的口径。

| 项目 | 论文附录 A.3 | 固定版本 LLaVA 高分辨率分支 | 本适配默认 |
|---|---|---|---|
| 任务 query | 图像后文本平均 | last | last，可选 question |
| epsilon | 1e-6 | 1e-10 | 1e-10，可配置 |
| 剪枝块 | L=2 | prune_layer=3，1-based | layer=2，可配置 |
| attention 获取 | attention matrix | 剪枝块请求完整 attention | 独立计算必要的因果 Q/K 行 |

这是 OneVision/Qwen2 的推理适配，不声称与作者 LLaMA 权重实验逐位一致。
保持原 decoder 精度/后端不变，也意味着主前向与作者可能使用的 eager 内核数值不同。
独立投影的矩阵形状也可能改变低精度舍入。评分的投影/QK 沿用模型 dtype，softmax
转 FP32 后回转原 dtype，再聚合；P/Q 归一化与 log 用 FP32，最终评分回转原 dtype。

## Q、P 与 token 范围

实际多模态打包给出视觉段 `[start,end)`；分隔 token 是位置 `end` 的**第一个文本 token**。
复用现有 `<image>\n问题` 中的分隔符，不额外添加 token。它与视觉段内的学习型
`image_newline` 是两种不同的 token。若用户自定义 prompt 把问题放在图像前，非零率会拒绝，
避免所谓“先验”已读到问题。0% 不修改、不重排原 prompt。

Q 取分隔符 query 到视觉段的 attention；P 默认取最后一个 prompt token 到视觉段的
attention。`query_aggregation: "question"` 改为从分隔符到 prompt 末尾的 query 平均，
按当前固定作者代码 **包含分隔符及模板尾部 token**。每行先在全部因果可见 key 上
做 softmax，再取视觉段。跨 head 聚合后分别对 P、Q 做 L1 归一化，计算
`S = P * log((P + epsilon) / (Q + epsilon))`，排序保留 top-k 并恢复空间顺序。

默认 `head_aggregation: "mean"`，可选 `max`。`question` 查询分块计算，每块最多 32
个 query，不分配完整的 T×T attention。超出有效范围或 native QK 出现 NaN/Inf 会报错，
不会偷偷修改主模型精度。评分不写入 decoder 输出或 KV。

默认对 **整个已打包视觉段共同排序**：Base、AnyRes 与结构型 image_newline 一起参与。
这沿用作者代码的候选范围，与上一轮 FastV 的 Base/AnyRes 分组独立 70% 实验不同。
不要把两种预算口径当成同一设置。保留数量为 `floor(N*(100-r)/100)`，其中 r 是换算后
在剪枝层请求的删除百分比。极小预算至少留一个 token，统计中记录这一安全扩展；
实际整数比例会略偏离请求值。平均 token 比例不等同于 FLOPs 或 Lat。

## 物理裁剪、位置与缓存

裁剪 hidden states，并裁掉 **所有已计算层** 的相应 KV 项（包括浅层缓存）。
保留原始位置编号，例如 `[0,1,5,9,10,...]`，不重排为连续编号。
decode 的逻辑位置从原 prompt 长度继续增长；物理 cache 长度另行管理。
Qwen2 原有 rotary embedding 支持这些显式位置，不替换 RoPE 模块。
这与旧 ViCo 的重新连续编号、保留浅层完整缓存的行为不同。

首版支持：eval、batch=1、一个图像段、无 prompt padding、greedy、DynamicCache、
单个剪枝边界、decoder 在单个设备。拒绝训练 labels、DPO、beam、sampling、静态/滑动/
量化 cache、外部缓存续接、sliding-window attention。核心测试在 CPU eager/SDPA 上验证；
服务器仍使用 FP16/FlashAttention2。0% 直接走已有原生 decoder，无评分或剪枝计算。

## 全量数据 + ACC/PRE/Recall/TNR/Lat/Mem

```bash
CUDA_VISIBLE_DEVICES=5 python run.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 --roi-mode anyres_max_9 \
  --method priortr --method-config configs/priortr.json \
  --no-sample --no-plots --benchmark-metrics --warmup 10 --seed 42 \
  --output-dir output/priortr_avg50_70_90_layer2_01
```

遍历配置中的平均 50/70/90，全量输入、一次加载模型、各率独立预热。同一进程运行各率，
allocator 缓存不会在不同率之间清空。Lat 为输入已驻留并同步后进入 generate，直到首
token 生成并同步，包含 vision/projector/打包/剪枝/prefill。Mem 为完整回答生成期间的
PyTorch 峰值 allocated/reserved，包含权重与输入；排除预处理/H2D，非整卡占用或算力利用率。
与当前公共 benchmark 完全复用同一观察器和聚合函数。

根目录 `summary.csv` 及各 `prune_XX/metrics.json` 保存分类和资源指标；
`predictions.jsonl` 保持精简，保存答案、时间与 benchmark 数值；
`layer_tokens.csv` 保存每个 decoder 层的 token 数，`run.json` 保存来源、配置和统计定义。

输出目录分别为 `prune_50`、`prune_70`、`prune_90`，目录名表示目标平均剪枝率。
`summary.csv`、`metrics.json` 和每张图答案记录均包含：

- `rate_semantics`：当前为 `layer_average`。
- `pruning_layer`：2。
- `prune_rate`、`target_average_prune_rate`：目标平均剪枝百分比。
- `requested_layer_prune_rate`：根据真实模型深度换算的单次删除百分比。
- `actual_layer_prune_rate`：依据整数保留数量计算的实际单次删除百分比。
- `actual_average_prune_rate`：`100*(1-sum(N_layer_in)/(D*N_original))`，按实际各层输入视觉 token 数计算。

summary/metrics 中后两项是所有已评估样本的等权平均；不是忽略取整的理论值，也不是
将不同长度样本的 token 混合加权。`layer_tokens.csv` 保留逐层数量，可独立复算。
要增加 0% 基线，将 `prune_rates` 设为 `[0,50,70,90]` 即可；基线仍不做评分和裁剪。

## 可视化

同一条命令去掉 `--no-plots --benchmark-metrics --warmup 10`，加
`--save-prune-vis --save-attention-vis`。当前默认 `visualize_rates: []`，不导出样本图。
若需要，设为 `[50,70,90]`，这些是目标平均剪枝率；图片中阶段的删除计数表示当层的
实际单次删除率。0% 无剪枝图。
也可加 `--visualize-samples sample_000000000 sample_000000270 ...` 只导出指定 ID；
仍评估全量输入。保存两种 PNG：

- `comparison.png`：原图、Base、AnyRes 和覆盖叠加，实际剪枝率标注与最终 GT/Pred。
- `attention_overlay.png`：**PriorTR 校正分数 S** 的 JET 叠加。S 可为负，显示用每张图
  的空间分数 min-max，不能按旧 attention 图将负值截为 0。图中明确注明它不是概率。

颜色映射仅影响显示，top-k 使用原始 S，不做 min-max。只另存精简的 `decisions.json`
计数，不导出完整 Q/P/S 数组。公共入口仍会生成最终 ACC/PRE/Recall/TNR 统计曲线。
要测公平 Lat/Mem，请用关闭样本可视化的 benchmark 命令。

## 验证范围

测试将独立 Q/K 得到的分数与固定作者策略在真实小 Qwen2 eager attention 上对照，
覆盖 last/question、mean/max、FP32/BF16。验证先验因果隔离、各层 KV 实际缩短、原始
position IDs、逐步 decode 与独立重算 oracle、一致的 0% 路径、重复 generate、
方法切换、公共 CLI 的分类/Lat/Mem 汇总和负分数可视化。
可选 CUDA/FlashAttention2 测试需要服务器执行：

```bash
python -m unittest discover -s tests -p 'test_priortr*.py' -q
```

小模型/模拟计时测试验证实现与统计接线，不代表真实 MVTec/VisA 准确率或 A100 加速已复现。
