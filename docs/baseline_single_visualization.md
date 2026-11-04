# 基准模式：只可视化 ID48、平均剪枝率50%

入口 `visualize_pool_distill.py` 默认执行已确认的基准模式：MyPool前置＋第2层后蒸馏，
`base_full`、`base_text`、λ=0.7、`fast`、`lean`、`anyres_first`、tail=9。
模型、蒸馏权重和数据默认路径与之前一致；固定贪心生成，seed=42。

```bash
python visualize_pool_distill.py \
  --gpu 5 \
  --sample-id 48 \
  --prune-rate 50 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --output-dir output/baseline_vis_48_prune50_01
```

使用默认服务器路径时可简写：

```bash
python visualize_pool_distill.py --gpu 5
```

## 已有旧版拼接图：直接拆分，不重跑模型

如果此前三张总览图已保存在`output/baseline_vis_48_prune50_01`，更新代码后执行：

```bash
python visualize_pool_distill.py \
  --split-existing \
  --output-dir output/baseline_vis_48_prune50_01
```

只使用CPU，从旧版未缩放的总览图中无损提取每个视图；不加载Torch、模型、checkpoint或数据集，
不重新推理。新图直接放在**同一个输出文件夹**，与原三张总览图并列。旧PNG、`run.json`和
`view_counts.csv`保持原样，另存`individual_views.json`记录导出文件列表。

## 选图与执行

- `48`、`000000048`、`sample_000000048`按数值ID匹配，保留JSON中的原始question_id。
  必须唯一命中；如果同一文件同时出现`48`和`000000048`等歧义记录，会报错。
- 只读取/运行命中记录。没有全量评估、预热或额外生成调用；输入清单保存为`input_selected.json`。
  随机种子仍使用`42＋原清单中的样本索引`，与完整基准运行一致。
- Base和AnyRes图直接观察本次真正送入视觉编码器的CPU预处理结果，不重新裁一套展示图片。
  原生AnyRes给出几块就显示几块，不强制3×3，不改变checkpoint的网格候选或输入图像。
- 快照仅保存原方法实际返回的蒸馏选择索引；`fast`映射和基准预算不变。前置源索引导出及PNG
  绘制在生成之后完成，未启用旧的模型可视化标志，也不切换到reference映射。

## 每个视图单独保存

| 文件名 | 内容 |
|---|---|
| `input_base.png`、`input_anyres_01.png`等 | 真正的编码器输入，每张只包含一个Base或crop |
| `pre_llm_base.png`、`pre_llm_anyres_01.png`等 | MyPool之后、进入LLM之前的独立token位置 |
| `post_distill_base.png`、`post_distill_anyres_01.png`等 | 第2层蒸馏剪枝之后、tail删除之前的累计保留位置 |

每张PNG只包含一个视图，保留原始像素尺寸，不缩放或拼接，顶部有简短阶段名、GT、最终模型
判断（Normal/Abnormal）和保留token数。每个阶段的文件名以前缀区分，AnyRes按实际网格的
行优先顺序编号；所有文件直接放在同一输出目录，不再新增阶段子文件夹。

如果有1张Base和9张AnyRes，3个阶段共保存**30张单独PNG**。新推理默认只生成单图；
旧结果用`--split-existing`导出时，原先3张总览仍保留在旁边。

剪掉的位置填充RGB **(255,255,255)**。按用户选择，**已合并但不再独立保留的源位置也填白**；
contextual token保留时只显示其实际anchor位置。因此这些图表示token位置，不表示全部信息
来源的覆盖范围。后置图包含前置阶段已经填白的区域，不会把它们恢复。

27×27 token对应实际14×14卷积patch支持，共378×378像素；384×384输入末端未编码的6像素
边缘保留原始像素，不将其误标为已剪token。newline不是图像中的像素方块，不绘制；它计入
实际打包预算，图中单独标注。原始输入图的计数是编码器空间token总量，不包含虚构的newline。

另存`view_counts.csv`，按Base/各crop列出原始729、前置保留、蒸馏后保留、各阶段删除数。
`run.json`只记录配置、ID、GT/答案、实际网格、相关预算和输出说明，不保存庞大的attention数组。

## 剪枝率与计时

50%仍是原两阶段LLM平均剪枝率目标，λ=0.7；tail=9仍属于额外消融。最终LLM最后9层已无图像
token，但后置PNG明确展示**tail之前**的蒸馏决策，因此不是一张全白的最终状态图。

这次是单样本诊断可视化，不输出数据集准确率或可用于对比的Lat/Mem。基准的全量性能测量继续
使用`run_pool_distill.py`。本地已用小型28层模型检查开启/关闭快照的生成token完全一致，实际
7B权重和ID48图像需要在服务器执行本命令。
