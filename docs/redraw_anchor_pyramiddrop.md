# 离线重画AnchorPrune与PyramidDrop可视化

`redraw_anchor_pyramiddrop_vis.py`只需要Pillow与Python标准库，不加载模型、Torch或GPU，不重新推理。
直接读取旧实验`visualizations/`中的PNG和每个样本的`run.json`，去掉旧标题，保留原有白色剪枝和灰色unpad区域。

```bash
cd /home/yz/xxy/code/pruning/llava_pruning &&
git fetch origin ex &&
git show origin/ex:redraw_anchor_pyramiddrop_vis.py > /tmp/redraw_anchor_pyramiddrop_vis.py &&
python /tmp/redraw_anchor_pyramiddrop_vis.py \
  --results-dir /home/yz/xxy/code/pruning/llava_pruning/output/visa_anchor_pyramiddrop_full_vis_01 \
  --output-dir /home/yz/xxy/code/pruning/llava_pruning/output/visa_anchor_pyramiddrop_compact_01
```

以上方式只取出新脚本到`/tmp`运行，工程中的代码文件保持原样；适合其它实验仍在运行时使用。
也可以修改脚本顶部的`RESULTS_DIR`、`OUTPUT_DIR`，直接运行脚本。

- 每个方法、每个剪枝率、每个样本只保存一张PNG。
- 从左到右：原始AnyRes拼接图、最终剪枝后的Base、最终剪枝后的AnyRes拼接图。
- 原始拼接图读取`input_anyres_*.png`；两组AnyRes使用相同网格和显示大小，便于直接对照。
- AnchorPrune的剪枝状态读取`pre_llm_*`，PyramidDrop只取`after_layer_21_*`；不额外输出第7/14层的图。
- 顶部保留方法、平均剪枝率、ID、GT和对应旧图片的模型预测。
- 实际AnyRes网格沿用旧图，不固定成3×3，也不改变已经运行的剪枝决策。
- 三个面板默认等高768像素，最近邻缩放，避免对白色格子边界混色。
  `--panel-height 0`保留AnyRes拼接后的原生高度；`--panel-height 1152`可加大显示尺寸。
- 新图在`<output>/anchorprune/prune_50/sample_000000000.png`等位置。
- `redraw_index.csv`记录图片位置及失败原因；进度打印已完成数量、耗时和ETA。
- 旧PNG、summary.csv、Lat/Mem和预测记录都保持原样。可以在同一新输出目录重新运行以更新重画图片。

需要旧版可视化PNG已经生成。只有summary.csv或回答JSONL，没有逐视图PNG时，无法离线恢复剪枝位置。
缺失的图或最终阶段会单独报错并写入索引，其余样本继续；有失败时脚本最终退出码为1。
原图发生过max9特征缩放的样本，仍保留旧可视化的空间投影含义，不把白色面积解释为精确packed token比例。
