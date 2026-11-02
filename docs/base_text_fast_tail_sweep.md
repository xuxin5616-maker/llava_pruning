# Base＋Text／fast：tail=0～15

入口`run_pool_tail_sweep.py`，仅运行Base预测分数映射后剪AnyRes（`base_full`），旁路输入
`base_text`，Base映射实现`fast`。原有MyPool前置阶段仍压缩Base和AnyRes；第2层后的蒸馏阶段
保留剩余Base，使用Base映射分数剪AnyRes。末尾删除则删除全部剩余图片段（包含newline）。

## 运行

```bash
python run_pool_tail_sweep.py \
  --gpus 7 6 5 4 3 2 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --prune-rates 0.3 0.5 0.7 \
  --lambdas 0.6 0.7 0.8 0.9 \
  --tail-start 0 \
  --tail-end 15 \
  --output-dir output/base_text_fast_tail_0_15_01
```

本入口将全部小数比例0.3/0.5/0.7转换为30/50/70%，也接受全百分数30/50/70。
不接受同一列表混用0.3和50。其它旧脚本的`--prune-rates`单位仍为百分数，没有改变。

3种剪枝率 × 4种λ × 16种tail = **192组**。只跑fast，没有优化前对照、AnyRes自身评分或
full旁路输入。使用JSON中全部图片，不保存可视化，沿用现有checkpoint，不重新训练。

## tail含义及预算

对28层模型，tail=N（N>0）表示：进入第`29-N`层之前，一次性删除当前全部剩余图片token。
例如tail=1只在第28层无图，tail=15在第14～28层无图；tail=0不额外删除。
文本位置编号保留原值，早期层仍保留剩余图像KV，末尾N层只建立和使用文本KV。

tail是额外消融，30/50/70仍对应原有两阶段的层平均剪枝目标，最终包含末尾删除的实际平均
剪枝率更高。逐图预算保留两阶段实际值、tail额外贡献及最终实际值，统计公式不变。
只要tail不越过第2层评分边界就可以；本28层模型最多26，本次只运行0～15。

## 调度、输出及续跑

同一剪枝率和λ的0～15全部固定在同一张GPU上，最多6个模型并发，每卡一个；12个组合在6卡
上循环分配。本命令每卡32个独立配置。每个配置独立加载模型，加载和warmup不计入Lat。
主模型精度和attention模式沿用原配置，旁路精度沿用checkpoint，实际剪枝处理仍计入Lat。

顶层`summary.csv`输出十一列：

```text
config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct
```

示例：`prune=30%;lambda=0.6;base_full;input=base_text;tail=15;mapping=fast`。
ACC、BACC、F1均为百分比；`relative_accuracy_pct`为ACC相对原版91.18%的百分比
（可用`--baseline-acc`修改参考值）。Lat为平均首token毫秒，Mem为完整生成期间峰值allocated GiB。
日志在`logs/`，各子目录保留逐图答案、时间/显存、配置、预算和断点记录。
父进程每30秒打印进度、elapsed和ETA；逐图进度在对应任务日志中。

```bash
python run_pool_tail_sweep.py \
  --resume \
  --output-dir output/base_text_fast_tail_0_15_01
```

恢复原参数和GPU安排，验证并复用已完成结果，未完成任务按逐图断点继续。续跑不允许更换tail范围、
GPU安排或其它实验参数，避免将不同设置混在一起。第一次运行请使用新的输出目录。
