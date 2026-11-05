# 全量VisA：AnchorPrune与PyramidDrop的结果和可视化

```bash
python run_anchor_pyramiddrop_vis.py \
  --gpu 5 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --prune-rates 50 70 90 \
  --output-dir output/visa_anchor_pyramiddrop_full_vis_01
```

严格使用官方CSV的所有test行，没有100张限制，不使用train/val、不读取标注mask。
标准官方CSV是2162张，运行时打印实际数量。没有学生权重参数，两种方法直接使用LLaVA主模型。
统一无类别v0 A/B、greedy、原生FP16/FlashAttention2、AnyRes max9、Base在前。

剪枝率均为LLM各层平均率：AnchorPrune在LLM前剪50/70/90%；
PyramidDrop固定第7/14/21层后剪，求解各阶段比例以达到50/70%平均目标。
PyramidDrop90%不可达，表中保留该行、指标为空，不启动模型、不伪造图片。
无剪枝参考只跑一次。指标共6个有效实验，测量完成后才额外推理绘图；
Lat/Mem测量中没有快照hook或PNG保存。各方法独立进程串行使用同一GPU，warmup10不计入统计。

## 图片

默认保存到 `<output-dir>/visualizations/<method>/prune_<rate>/sample_<index>/`。
每个Base/AnyRes块独立PNG，顶部标GT和本次模型判断；`sample_index.csv`对应实际图片路径。

- AnchorPrune：`input_base.png`、`input_anyres_01.png`等，以及`pre_llm_base.png`、`pre_llm_anyres_01.png`等。
- PyramidDrop：输入图，以及`after_layer_07_*`、`after_layer_14_*`、`after_layer_21_*`，均为累计剪枝状态。

白色RGB255表示剪掉的位置。原生unpad已排除的区域标灰，避免把预处理裁去的padding算作方法剪枝。
384输入中未被27×27、patch14卷积编码的末端6像素边缘保持原样。

**AnyRes native packing先unpad，必要时max9双线性降采样，因此LLM token不总与原始crop格子一一对应。**
显示时把native packed二值mask按最近邻映射回真实输入crop；未发生max9缩放时空间格子对应精确，
发生缩放时只是空间投影，不代表所有信息来源。各图标题计数标为projected cells，不能与packed token数混同。
每个样本的`run.json`保存真实packed保留数、newline保留数、原始/处理后网格及是否max9 resize；
`view_counts.csv`列的是各原始视图的投影格子数。PyramidDrop可能剪掉newline，newline不画成像素块。

剪枝算法、预算、排序、合并、模型dtype与attention后端均未改；AnchorPrune直接读取实际selected索引，
PyramidDrop只开启现有诊断快照。重放与原测量的token预算不一致会报错；答案不同则分别记录，不覆盖旧回答。

## 指标和恢复

`summary.csv`沿用`config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct`，
相对ACC使用本次全量VisA不剪枝参考。`budget_summary.csv`保留目标/实际平均率等预算审计。
各组`metrics.json`包含PRE/Recall/TNR、未解析数、Lat分位数、allocated/reserved显存。

测量中断后同版本代码继续：

```bash
python run_anchor_pyramiddrop_vis.py --resume \
  --output-dir output/visa_anchor_pyramiddrop_full_vis_01
```

测量已经完整完成，只补图或恢复中断的绘图：

```bash
python run_anchor_pyramiddrop_vis.py --visualize-only \
  --output-dir output/visa_anchor_pyramiddrop_full_vis_01
```

补图保留原CSV、回答、Lat和Mem。可用`--vis-output-dir`指定另一个图片目录。
所有样本、所有可达剪枝率都会保存：若每图1个Base＋9个AnyRes，则每图两方法所有配置共140张PNG；
2162张约302680张PNG，实际数量随原生crop数变化。绘图会额外占用推理和保存时间。
本地CPU测试用于接口/坐标/数值一致性检查，正式模型、速度和准确率需要服务器运行。
