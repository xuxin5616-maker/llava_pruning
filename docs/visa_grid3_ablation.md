# VisA 固定 3×3：输入视图与 tail 对照

独立入口 `benchmark_visa_grid3.py`，不修改原评测、训练和模型默认配置。
默认使用完整官方 VisA test，预热10张不计指标；没有100张限制。

## 运行

`--previous-output` 填此前包含 VisA `summary.csv`、`run.json` 的目录，读取同一学生权重。
也可直接用 `--checkpoint /path/to/last.pt` 替代该参数；权重必须未使用 VisA test 训练。

```bash
python benchmark_visa_grid3.py \
  --gpus 3 4 \
  --previous-output output/mvtec_train_visa_two_gpu_01/visa_all_methods \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --prune-rates 70 --distill-share 0.7 --tails 0 9 \
  --output-dir output/visa_grid3_prune70_tail0_9_01
```

五组：Base-only不剪枝、AnyRes-only不剪枝、Base+AnyRes不剪枝、
MyPool+蒸馏70%/λ0.7/tail0、相同方法tail9。
三组不剪枝在GPU3顺序跑；两组剪枝在GPU4顺序跑，两张卡之间并行。
同一类比较用同一张卡；CSV的config记录GPU编号。单卡对照可用 `--gpus 5`。
剪枝仍为base_full、base_text、fast mapping、lean、anyres_first，greedy，无可视化。
tail9仍为两阶段预算之外的额外删除，不与tail0强行凑成相同总实际剪枝率。

## 图像和计时

1. 仅从官方CSV的test行读取图片；原始文件不改。
2. 每张RGB图保持比例、BICUBIC缩放，居中补黑边至1152×1152，另存无损PNG。
3. 所有组共用这份PNG；强制候选画布仅为1152×1152，得到3×3裁剪。
4. Base-only的Base也来自同一张补边图；这不是原图直接拉伸得到的旧Base。
5. 补边在所有组中作为方形图的一部分保留，不再按未补边的原始宽高去掉。
6. 运行时检查视觉编码器实际输入分别为1、9、10个384×384视图。
   原生spatial_unpad的视觉token分别为729、6642、7371（含原生换行）；
   如果checkpoint使用spatial_unpad_add_newl，按原配置统计额外换行。
7. 预处理、PNG写入、文件读取、CPU处理、输入搬运、模型加载和预热不计入Lat。
   Lat为驻留输入进入generate至第一个token，包含全部在线剪枝；Mem为完整回答生成峰值allocated GiB。
8. 保持原生FP16、FlashAttention2和现有预测旁路精度，不重新训练学生。

## 输出和恢复

- `summary.csv`：沿用11列config、tp/fp/tn/fn、acc、Lat、mem、bacc、f1、relative_accuracy_pct。
- Ratio分母使用**本轮补边后的Base+AnyRes不剪枝ACC**，不使用旧69.29%或MVTec91.18%。
- `budget_summary.csv`：实际层平均预算、两阶段值、tail贡献与token计数。
- `visa_test_grid3.jsonl`：同一测试名单、原图尺寸、缩放尺寸与补边位置。
- `prepared/images/`：模型输入缓存，属于预处理数据，不是可视化图片。
- 每组目录：逐样本回答、时间/显存/预算、指标和精度环境；根目录每组有独立日志。
- 终端打印预处理与逐样本ETA、GPU任务进度。中断后已写入的样本保留。

```bash
python benchmark_visa_grid3.py --resume --output-dir output/visa_grid3_prune70_tail0_9_01
```

恢复使用原配置和GPU，检查源码、原始输入、缓存、模型配置、checkpoint和精度环境。
本地CPU测试覆盖像素一致性、真实视觉打包、预算/统计、恢复和两卡调度；真实7B CUDA结果需在服务器测量。
