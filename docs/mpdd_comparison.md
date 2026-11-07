# MPDD：六种对比方法＋不剪枝参考＋Base引导tail9/tail0

入口 `benchmark_mpdd.py`。只读取原始 MPDD 的六类 `test/`，不使用训练图、mask、正常参考图。
沿用 FP16 / FlashAttention2、greedy、batch1、原生 AnyRes max9；原图不额外缩放或生成缓存PNG。
所有方法使用相同通用v0提示词，去掉类别名称：

```text
Are there any defects in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly.
```

## 在服务器运行

先检查已下载的数据，不需要模型、学生权重或GPU，也不会生成文件：

```bash
python benchmark_mpdd.py \
  --data-root /home/yz/xxy/data/datasets/MPDD \
  --check-data
```

`--data-root` 必须直接包含 `bracket_black/`、`bracket_brown/`、`bracket_white/`、
`connector/`、`metal_plate/`、`tubes/`。
每类按 `test/good/`（也兼容 `test/normal/`）读取正常图，其余缺陷目录读取异常图。
缺少类别、缺少正常/异常测试图、无标签图片或目录混乱会明确报错，不会跳过后继续。
打印实际总数、各类别正常/异常数和图片尺寸分布，不把测试集限制为100张。
即使某些数据源常见为1024×1024，也以本地图片实测为准；AnyRes网格仍由原模型候选分辨率决定。

一键运行确认的全部配置，使用GPU3、4，每张卡同时最多一个任务：

```bash
python benchmark_mpdd.py \
  --gpus 3 4 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/MPDD \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --prune-rates 50 70 90 \
  --output-dir output/mpdd_methods_tail9_tail0_50_70_90_01
```

单卡可改为 `--gpus 5`。同一方法的三个剪枝率固定在同一GPU；config列保留真实GPU分配。
每个任务新进程加载模型，预热10张，预热不计入统计。每25张打印进度、耗时和ETA，父进程每30秒报告进度。
不自动保存样本图、注意力图或指标曲线。

## 包含哪些实验

| 方法 | 配置 | 目标平均剪枝率 |
|---|---|---|
| LLaVA不剪枝 | Base＋AnyRes原生输入 | 0% |
| FastV | 第2层后mask，物理序列长度不变 | 50/70/90% |
| VisionZip | LLM前现有SigLIP适配及合并/packing | 50/70/90% |
| PyramidDrop | 固定第7/14/21层 | 50/70%；90%不可达，留空并解释 |
| AnchorPrune | 现有LLM前实现 | 50/70/90% |
| CLSE | 第1层后，纯clse评分 | 50/70/90% |
| PriorTR | 第2层后 | 50/70/90% |
| 我们的方法tail9 | MyPool＋蒸馏，λ=.7、base_full＋base_text＋fast＋anyres_first | 50/70/90%，另加tail9删除 |
| 我们的方法tail0 | 同上；只关闭末尾额外删除 | 50/70/90% |

学生沿用已有VisA训练权重，不在MPDD重新训练。两组Base引导都保留原两阶段预算；
tail9在最后9层之前额外删除剩余image token，tail0让第2层后保留的视觉token继续经过其余所有层。
tail9的实际总平均剪枝率因此高于标称两阶段目标，预算表单独报告这部分贡献。
原生PyramidDrop固定方案前7/28层全保留，90%平均剪枝率无法达到，不能用最终累计90%冒充。
总表25行，实际启动24个任务；不把不可达项当成执行失败。

## 输出和指标

- `summary.csv`：`config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct`。
- `budget_summary.csv`：各方法的目标/实际平均剪枝率、物理/有效剪枝、两阶段率、额外tail贡献和实际token计数。
- `run.json`：模型、学生来源与指纹、数据统计、任务配置、GPU分配和统计口径。
- `mpdd_test.jsonl`：本次测试图片、标签、提示词、尺寸、文件大小/修改时间；不包含mask或训练样本。
- 每组目录：`metrics.json`、`predictions.jsonl`、`inference.json`；每组日志为入口目录的同名 `.log`。

ACC/BACC/F1/相对ACC为百分数。相对ACC分母是本次实测MPDD不剪枝ACC，不固定为MVTec的91.18%。
GT=1为缺陷，GT=0为正常。ACC中未解析回答算错；PRE/Recall/TNR/BACC/F1按可解析A/B计算，
未解析数量及完整精确率/召回率/TNR等保留在 `metrics.json`。

Lat为GPU驻留输入进入`generate`到首token完成的平均毫秒数，包括视觉编码、projector、packing、
在线MyPool评分/选择/合并、蒸馏、映射、剪枝和LLM prefill。
CPU读图/预处理、H2D、静态输入准备、权重加载、预热和保存不计入Lat。
mem为完整回答生成期间的PyTorch peak allocated GiB，包含模型/学生权重与输入；不是GPU计算利用率。
多卡Lat来自其实际分配卡；设备环境、reserved峰值和分位数等另存每组结果。

中断或某组失败后，修复对应日志中的问题，再恢复原目录：

```bash
python benchmark_mpdd.py --resume \
  --output-dir output/mpdd_methods_tail9_tail0_50_70_90_01
```

恢复使用保存的参数和GPU，跳过完整成功组，失败组从已有逐样本记录继续。
使用当前工作区源码，不复制/冻结源码，也不因源码变化阻止后续任务；每个worker记录启动时的源码指纹。
模型配置、测试清单、图片尺寸/大小/修改时间、学生权重和推理环境仍检查一致性。
图片未逐文件计算内容哈希；若改变算法并希望与旧结果分开比较，应指定新输出目录。

本地CPU检查验证数据读取、任务调度/恢复、原实现复用及小模型的tail行为；正式GPU指标由服务器运行产生。
