# ViCo / PyramidDrop：Qwen2 推理适配说明

## 来源与范围

- CVPR 2025：[Conical Visual Concentration for Efficient Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.html)。
- 作者仓库：[Cooperx521/PyramidDrop](https://github.com/Cooperx521/PyramidDrop)。
- 对照实现：作者 `llava/model/modeling_llama_pdrop.py` 的 `pdrop_forward`、`pdrop_rank_drop`，以及 README 的层号/保留比例约定。
- 本工程是该算法的 **Qwen2 / Transformers 4.46.1 推理适配**，不是作者原始 LLaMA 实验的原样复现，也未声称复现论文准确率或加速比。

## 方法参数

`--method vico` 自动选 `configs/vico.json`，默认如下：

```json
{
  "layers": [8, 16, 24],
  "rate_semantics": "final_cumulative",
  "prune_rates": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
  "visualize_rates": [10, 30, 50, 70, 90],
  "min_tokens": 1
}
```

`layers` 是 **1 起始的已完成层号**，在第 8、16、24 层之后执行筛选。28 层模型的四阶段长度是 8/8/8/4，不是等长阶段。层号必须严格递增且小于总层数；不允许在最后一层之后再剪。

`prune_rates` 是 **最终累计剪枝百分比**，不是每个阶段都剪该百分比。设最终剪枝率为 R（0～99 的整数），阶段剪枝次数为 S，第 j 次剪枝后目标保留比例为：

`keep[j] = (1 - R/100) ** (j/S)`，其中 `j=1,...,S`。

三个边界时，每次都保留当前数量约 `(1 - R/100)**(1/3)`。最终剪 90% 时，各阶段约保留原始数量的 46.42%、21.54%、10%。实际目标数量按原始图像 token 数乘累计保留比例向上取整，并至少保留 `min_tokens` 个；不会恢复前面已删的 token。因此实际剪枝率可能稍小，尤其在微型测试序列中。

作者原代码同样需要设置保留比例，示例为 `[0.5,0.25,0.125]`，最终累计剪掉 87.5%。作者使用整数截断；本适配使用向上取整和最少保留限制。当前公共扫描接口仍限定整数百分比，不能把 90% 结果标成作者的 87.5% 设置。

## Attention 与实际计算

1. 非零剪枝时，主干保持已有 FP16 + FlashAttention2，不请求 `output_attentions=True`，不替换 decoder attention 类。
2. 排序使用独立旁路：下一层的 input norm、Q/K 投影、RoPE、GQA 头展开，最后一个有效 prompt token 的 Query 对当前全部 Keys 打分，softmax 后平均多头，再截取视觉 token 分数。这遵循作者代码在边界后使用下一层投影的层号约定。
3. 投影与 RoPE 用模型 dtype；旁路 QK、softmax、多头平均用 FP32。它不是将主干变成 FP32，也不保证排序数值与作者原 FP16 实现逐位一致。
4. 按分数选 top-k，再按原序恢复排列。拼接“图像前文本 + 保留视觉 token + 图像后文本”，真实缩短序列，不只是改 attention mask。
5. 同作者策略，每次裁剪后连续重编号 position IDs。各层 KV cache 保留自己阶段的长度；后续生成按该层缓存长度计算位置。单 token decode 不重新排序/剪枝。
6. token 总数指多模态拼接后的完整图像 span，包含 anyres 行分隔 newline。ViCo 对它们一起排序，newline 没有可画的像素位置。本工程旧 FastV 的末尾 newline 保护逻辑保持不变，二者不能仅按名义百分比视为完全相同的预算。

0% 会禁用 ViCo，自始至终直接调用原 `Qwen2Model.forward`，不打分、不缩短序列、不重编号位置。逐层计数仅来自预处理得到的 span 元数据，不给 baseline 加 attention hook。相同输出仍需相同权重、输入、prompt、环境和解码设置，不能凭此保证跨机器逐位一致。

## 保存内容

- 所有扫描率，包括 0%：原有 predictions、metrics、summary 与 ACC/PRE/Recall/TNR 曲线保留。
- 每个 `prune_XX/layer_tokens.csv`：每个样本每层一行，包含图像/总序列输入输出数量、本层后删除数量和累计实际剪枝率。`image_tokens_in` 是该层实际处理的数量；`image_tokens_out` 是其后筛选完成的数量。
- `predictions.jsonl` 的 `pruning_stats`：同样记录阶段和所有层的统计。
- 可视化仍由两个开关控制，默认只对 10/30/50/70/90 生成；0% 不生成样本图。
- 每张样本最终只有两类 PNG：`comparison.png` 与 `attention_overlay.png`，按实际剪枝阶段排成三行，不重复画未剪枝的层。关闭某个开关就不保存对应 PNG。
- comparison 显示当前阶段之后的累计保留/删除情况；attention 是当前阶段删除之前的排序分数，前面已经删掉的 token 显示灰色，不把缺失分数伪装成低 attention。每阶段独立归一化，共享该阶段各 view 的色标；不能用不同阶段颜色直接比较原始分值。
- 索引始终映射回初始完整图像网格，支持现有 randomroi/randompatch/anyres 布局。`decisions.json` 保存阶段 mask、分数、有效分数位置和坐标信息。

## 限制与验证

当前限定单张图、batch size 1、无 padding、普通单序列采样/greedy decode、DynamicCache、decoder 在同一设备；不支持训练、beam expansion、滑动窗口或 static/quantized cache。现有 `run.py` 满足这些条件。请用 `CUDA_VISIBLE_DEVICES=5` 等只暴露一张 GPU；不会为了兼容而悄悄切换精度或 attention 后端。

本地 CPU 测试覆盖 GQA 旁路打分、原 forward 的 0% 等价、28 层实际裁剪数量、不同长度缓存的多步续写、真实 HF `generate`、方法切换/重复样本重置、anyres/ROI 阶段图、CSV/JSON 保存以及旧 FastV 回归。CUDA 测试保留给服务器执行：

```bash
CUDA_VISIBLE_DEVICES=5 python -m unittest discover -s tests -p 'test_vico*.py' -v
```

GPU 测试使用随机初始化的小模型，无需下载权重。通过后还应使用真实 checkpoint 做短数据集检查，再跑全量。这里没有引入 `ex.py` 的 100 张限制、显存采样或计时实验；`ex.py` 仍是 FastV 专用。
