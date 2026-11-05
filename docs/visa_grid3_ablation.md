# VisA 固定 3×3：补边／拉伸、输入视图与 tail 对照

独立入口 `benchmark_visa_grid3.py`，不修改原评测、训练和模型默认配置。
默认使用完整官方 VisA test，预热10张不计指标；没有100张限制。
默认同时运行 `letterbox`（保持比例补边）和 `stretch`（不保持比例直接缩放），各5组，共10组。

## 运行

`--previous-output` 填此前包含 VisA `summary.csv`、`run.json` 的目录，读取同一学生权重。
也可直接用 `--checkpoint /path/to/last.pt` 替代该参数；权重必须未使用 VisA test 训练。

```bash
python benchmark_visa_grid3.py \
  --gpus 3 4 \
  --resize-modes letterbox stretch \
  --previous-output output/mvtec_train_visa_two_gpu_01/visa_all_methods \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --prune-rates 70 --distill-share 0.7 --tails 0 9 \
  --output-dir output/visa_grid3_letterbox_stretch_70_tail0_9_01
```

每种缩放各五组：Base-only不剪枝、AnyRes-only不剪枝、Base+AnyRes不剪枝、
MyPool+蒸馏70%/λ0.7/tail0、相同方法tail9。
GPU3顺序跑补边的5组，GPU4顺序跑拉伸的5组，两张卡之间并行。
同一缩放方式内的基准和剪枝组使用同一张卡；CSV的config记录resize和GPU编号。
单卡对照可用 `--gpus 5`，此时两套依次运行。只运行其中一套可用 `--resize-modes letterbox` 或 `--resize-modes stretch`。
剪枝仍为base_full、base_text、fast mapping、lean、anyres_first，greedy，无可视化。
tail9仍为两阶段预算之外的额外删除，不与tail0强行凑成相同总实际剪枝率。

## 图像和计时

1. 仅从官方CSV的test行读取图片；原始文件不改。
2. 补边组：每张RGB图保持比例、BICUBIC缩放，居中补黑边至1152×1152，另存无损PNG。
   拉伸组：同一原图直接以BICUBIC缩放至1152×1152，不补边、不保留宽高比，另存无损PNG。
3. 同一缩放组的5个实验共用同一份PNG；保留checkpoint原候选网格列表。
   1152×1152方图在原候选中自然选择3×3，运行时检查实际选择；不改模型配置或学生兼容性检查。
4. 每组的Base也从该组的1152方图生成；两种缩放使用同一名单、顺序、GT、提示词和checkpoint。
5. 补边作为补边组方图的一部分保留，不再按未补边的原始宽高去掉；拉伸组没有新增补边。
6. 运行时检查视觉编码器实际输入分别为1、9、10个384×384视图。
   原生spatial_unpad的视觉token分别为729、6642、7371（含原生换行）；
   如果checkpoint使用spatial_unpad_add_newl，按原配置统计额外换行。
7. 预处理、PNG写入、文件读取、CPU处理、输入搬运、模型加载和预热不计入Lat。
   Lat为驻留输入进入generate至第一个token，包含全部在线剪枝；Mem为完整回答生成峰值allocated GiB。
8. 保持原生FP16、FlashAttention2和现有预测旁路精度，不重新训练学生。

## 输出和恢复

- 根目录 `summary.csv`：合并两套，共10行，沿用11列config、tp/fp/tn/fn、acc、Lat、mem、bacc、f1、relative_accuracy_pct。
- `letterbox/summary.csv`、`stretch/summary.csv`：每套分别5行。
- Ratio分母使用**各自缩放方式下的Base+AnyRes不剪枝ACC**；两套各自的基准Ratio都是100%。
  不共用补边／拉伸分母，不使用旧69.29%或MVTec91.18%。
- 根目录与两个子目录均有 `budget_summary.csv`：实际层平均预算、两阶段值、tail贡献与token计数。
- 每套的 `visa_test_grid3.jsonl`：同一测试名单、原图尺寸、缩放尺寸与补边位置。
- 每套的 `prepared/images/`：模型输入缓存，属于预处理数据，不是可视化图片。
- 每套的任务目录：逐样本回答、时间/显存/预算、指标和精度环境，旁边有任务日志。
- 根目录 `letterbox.log`、`stretch.log`：分别查看预处理和GPU任务调度进度。
- 终端打印预处理与逐样本ETA、GPU任务进度。中断后已写入的样本保留。

```bash
python benchmark_visa_grid3.py --resume --output-dir output/visa_grid3_letterbox_stretch_70_tail0_9_01
```

恢复使用原配置和GPU，检查源码、原始输入、缓存、模型配置、checkpoint和精度环境。
双模式版本使用新目录运行；旧版只有补边的实验保留，不与新版结果混合恢复。
本地CPU测试覆盖像素一致性、真实视觉打包、预算/统计、恢复和两卡调度；真实7B CUDA结果需在服务器测量。

## 修复旧双模式版本的学生配置签名失败

v2.0版本曾将运行时 `image_grid_pinpoints` 改成只有1152方图，导致与训练权重中记录的
原候选列表不同，报 `Predictor backbone architecture/AnyRes packing/order does not match training`。
修复版保持原配置，仍自然选中3×3；不跳过签名检查，不改权重或推理精度。

若两套的三组不剪枝均成功，剪枝任务因上述同一错误失败，更新代码后使用**原输出目录**：

```bash
python benchmark_visa_grid3.py \
  --repair-grid-config \
  --output-dir output/visa_grid3_letterbox_stretch_70_tail0_9_01
```

该参数完成专项校验后自动resume：保留6组已完成的回答、Lat、Mem等文件，
只启动4组失败的剪枝任务。使用原保存GPU3/4、权重、输入和参数，无需重新预处理或训练。
原生候选和旧强制候选对现有1152方图均选中3×3，完成的不剪枝结果可以保留。

恢复只允许已知v2.0脚本差异，其余源码、权重、原图、测试名单、PNG缓存都需匹配。
有其他错误或剪枝任务已有非空测量记录时拒绝迁移，不自动混合不同版本结果。
旧配置保存在 `run_before_grid_config_fix.json`；失败任务的旧metrics/inference移动到
`before_grid_config_fix/`，成功任务的原始指标和回答不改。
后续中断再用普通 `--resume` 即可。
