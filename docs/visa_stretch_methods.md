# VisA：六种对比方法统一拉伸至1152方图

独立入口 `benchmark_visa_stretch.py`，不运行MyPool+蒸馏，不需要学生权重、λ或tail。
复用原 `benchmark_visa_methods.py` 的原有worker、六种算法、层平均预算、Lat/Mem和报表逻辑。
这是新实验，旧原分辨率或补边结果不会被复用或覆盖。

```bash
python benchmark_visa_stretch.py \
  --gpus 3 4 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --prune-rates 50 70 90 \
  --output-dir output/visa_stretch_methods_50_70_90_01
```

## 范围与预处理

- 完整官方test名单，标准划分2162张，不取100张子集；不读取训练图片或mask。
- 全部图片直接BICUBIC拉伸为1152×1152，不保留宽高比、不补边，另存无损PNG缓存。
- 所有实验共用同一输入名单、像素、提示词、顺序与seed；Base来自同一方图，AnyRes固定9块384×384。
- 保留原模型候选网格和FP16/FlashAttention2配置，仅验证方图自然选中3×3。
- 不剪枝Base+AnyRes基准重新测量一次。随后FastV、VisionZip、PyramidDrop、AnchorPrune、CLSE、PriorTR各跑50/70/90平均剪枝率。
- PyramidDrop固定7/14/21层，90%平均剪枝不可达，只保留明确标记、空指标行。
- 因此CSV共19行，实际运行18个实验；不包含我们的方法，不训练任何权重。

## GPU分配

GPU3：不剪枝、FastV、PyramidDrop、CLSE；GPU4：VisionZip、AnchorPrune、PriorTR。
每张卡9个有效任务顺序运行，两张卡之间并行；每个方法的不同剪枝率使用同一张卡。
每个任务独立进程加载模型并预热10张，不把预热计入指标。
`--gpus 5` 可改为同一张卡串行，适合更严格地对比Lat；每行config记录物理GPU编号。

## 预算和统计

平均剪枝率仍参照原生未剪枝packed图像token：Base+AnyRes+结构换行，共28层平均。
标准spatial_unpad下，拉伸后的3×3输入为729+6561+81=7371个图像token。
FastV保持mask版本，有效剪枝与物理长度分开记录；其它算法不改原选择、合并与位置处理。

输出沿用：

- `summary.csv`：config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct。
- Ratio使用**本轮新测的拉伸不剪枝ACC**，不用69.29%、70.12%或91.18%常数。
- `budget_summary.csv`：目标与实际层平均率、有效／物理率、token数量及不可达原因。
- `run.json`：预处理、原图／缓存／输入／源码指纹、GPU和方法设置。
- `visa_test_stretch.jsonl`、`prepared/images/`：冻结输入与预处理缓存，不是可视化。
- 每任务的日志、metrics.json、predictions.jsonl：回答、逐样本时间/显存/预算与完整统计。

Lat为驻留输入进入generate到首token，包含在线vision/projector/评分/合并/剪枝和prefill。
CPU拉伸/保存PNG、读图、CPU预处理、H2D、模型加载、预热和结果保存不计入Lat。
Mem为完整回答生成的PyTorch peak allocated GiB，另存reserved和占设备容量比例。
没有可视化图；主终端会直接打印失败任务的错误和日志位置，不把部分成功称为全部完成。

中断后沿用原目录：

```bash
python benchmark_visa_stretch.py --resume --output-dir output/visa_stretch_methods_50_70_90_01
```

恢复检查原图、CSV、缓存、模型配置和源码，跳过已完成任务并继续已保存样本。
本地测试覆盖输入、调度、报表和恢复；真实7B GPU准确率／Lat／Mem需要在服务器运行。
