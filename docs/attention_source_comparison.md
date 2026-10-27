# AnyRes 自身评分 vs Base 映射评分

`compare_attention_sources.py` 一条命令运行六组对照，使用相同蒸馏checkpoint：

| 模式 | AnyRes排名来源 | Base | 整体LLM层平均剪枝率 | 样本图片 |
| --- | --- | --- | --- | --- |
| anyres_self | 旁路预测的AnyRes自身attention | 全保留 | 30/50/70 | 仅70 |
| base_full | 旁路预测的Base attention，映射到整个AnyRes | 全保留 | 30/50/70 | 仅70 |

两种都是三层蒸馏旁路的预测评分，不是原生第22层教师attention。输入仍是完整H2。
anyres_self请求联合视觉评分，但只在AnyRes中排序；Base分数不会抢占预算。
base_full仅取Base评分，在FP32中双线性映射到实际unpad/max9后的AnyRes网格。
两者的保护Base/newline、原始位置编号、物理KV压缩和整体层平均预算完全相同，
不逐crop分配，不合并token，不改变主干FP16/FlashAttention2或KL损失，无须重训。

## 完整命令

在服务器工程根目录运行，使用新/空输出目录：

```bash
python compare_attention_sources.py \
  --gpu 5 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --output-dir output/attention_sources_30_50_70_01
```

默认值就是上述GPU、模型/数据和checkpoint路径；默认warmup10、seed42、greedy、AnyRes max9。
脚本自动设置子进程CUDA_VISIBLE_DEVICES，`--gpu`为物理GPU编号。若训练目录不同，
只需修改`--checkpoint`。两份运行配置自动生成，不必修改训练目录里的method.json。

两种模式在同一GPU上**依次执行，不并发**，共两个独立子进程，各自加载模型一次并跑三个率。
加载不计入Lat；两组不继承对方的PyTorch显存分配器缓存。终端直接显示子进程进度和报错。
第一组失败会停止，不继续运行后伪装成完整比较。GPU外部负载与运行顺序仍可能影响实测时间。

评估全部输入记录，无100张限制。只保存指定样本的70%图片时追加：

```bash
--visualize-samples sample_000000000 sample_000000270
```

这不会减少指标的评估样本。只要指标、不保存图片时追加`--no-visualizations`。

## 输出

```text
output/attention_sources_30_50_70_01/
  comparison.json            # 总配置、实际命令、完成/失败状态
  summary.csv                # 六行：两种来源 × 三个率，含分类指标、Lat、Mem
  configs/                   # 自动生成的两份配置，visualize_rates仅70
  anyres_self/
    summary.csv
    prune_30/                # 结果、指标与预算，无PNG
    prune_50/                # 结果、指标与预算，无PNG
    prune_70/visualizations/ # comparison.png、attention_overlay.png
  base_full/
    summary.csv
    prune_30/
    prune_50/
    prune_70/visualizations/
```

根summary增加selection_mode/attention_source，区分相同剪枝率的两种方法。
指标单位沿用公共输出：ACC/PRE/Recall/TNR为0..1，Lat为ms，Mem为GiB/百分比。
各组保留既有predictions、metrics、view_budgets、layer_tokens等文件。
启动器传`--no-metric-plot`，不额外生成指标曲线，但不禁用70%的样本图片。
旧输出不会被删除；本次只在新目录生成70%的图片。

## Lat / Mem 和图片

Lat定义不变：驻留输入进入model.generate到首token选出并同步，包含视觉、projector、prefill、
旁路、评分映射/排序/剪枝，排除加载、CPU预处理、输入搬运和保存。
Mem为完整答案生成期间PyTorch峰值allocated/reserved，含常驻主模型、旁路与临时激活；
不是显卡算力利用率，也不是nvidia-smi整卡使用量。

每个子进程先完成30/50/70的全部干净测量，不捕获图片；随后仅对70需要保存的样本
额外生成一次，核对答案后保存。额外前向和保存只增加总墙钟耗时，不混入Lat/Mem。
每组图片阶段在该组所有测量之后，且子进程退出才启动下一组，不把图片缓存带入另一组。

**两种模式都执行相同的严格预算预检。** Base全保留时，crop较少的图片可能无法达到
50或70的整体层平均目标；此时报告样本并停止，不删样本、不降低目标、不换AnyRes-only口径。
参见[Base-full预算限制](attention_predictor_base_full.md)。

CPU/mock测试覆盖评分来源改变选择、预算一致、Base/newline保护、六组命令、汇总、
仅70图片与失败处理。真实checkpoint的准确率、Lat和Mem需要在服务器实测。
