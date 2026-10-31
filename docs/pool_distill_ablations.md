# MyPool＋蒸馏旁路消融实验

入口：`run_pool_distill_ablations.py`。沿用已有checkpoint，不重新训练；全量读取指定JSON，
不限制100张，不保存可视化。主模型仍使用原有FP16、FlashAttention2，旁路精度由checkpoint决定。

## 一条命令运行63组

在服务器项目根目录运行：

```bash
python run_pool_distill_ablations.py \
  --gpus 5 6 7 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --prune-rates 30 50 70 \
  --lambdas 0.7 0.8 0.9 \
  --pruning-audit lean \
  --output-dir output/pool_distill_ablations_01
```

每张GPU同时运行一个独立模型进程。单卡可将`--gpus 5 6 7`换成`--gpu 5`。
四个子实验组依次执行，每组内部在指定GPU间调度任务：

| 子目录 | 旁路输入 | 评分 | 末尾处理 | 组数 |
|---|---|---|---|---:|
| tail_2 | 完整H2 | anyres_self、base_full | 最后2层无图像token | 18 |
| tail_3 | 完整H2 | anyres_self、base_full | 最后3层无图像token | 18 |
| tail_4 | 完整H2 | anyres_self、base_full | 最后4层无图像token | 18 |
| base_text | H2中的Base＋Text | 仅base_full | 不额外删除末尾图像 | 9 |

`3种剪枝率 × 3种λ × (3种末尾层数 × 2种评分 + 1种Base＋Text) = 63`。
组间分别运行，不把Base＋Text与末尾删除交叉组合。

## 末尾删除的实施位置

原有两阶段预算保持不变，完成第2层后仍按原方法预测评分并剪枝。对于28层模型：

| 末尾层数 | 一次性删除的边界 | 不再接收图像token的层 |
|---:|---|---|
| 2 | 第26层结束后、第27层开始前 | 27、28 |
| 3 | 第25层结束后、第26层开始前 | 26、27、28 |
| 4 | 第24层结束后、第25层开始前 | 25、26、27、28 |

删除剩余完整图片段，包含Base、AnyRes及图片内部newline。Text全部保留，其RoPE位置编号保持原值。
最后几层直接处理缩短后的文本序列，不运行图像query。

前面各层的图像KV保留，后续生成token在这些层仍可访问图像；最后几层从prefill开始只建立
文本KV，后续解码也只访问文本KV。不同层的缓存长度独立处理。

30/50/70仍表示原有两阶段目标。末尾删除额外降低各层平均token数量：

```text
原有层token总和 = 2*N1 + (D-2)*N2
消融层token总和 = 2*N1 + (D-2-tail)*N2
额外平均剪枝百分点 = 100*tail*N2/(D*N0)
```

逐图`budgets.csv`中，`two_stage_average_prune_rate`保留原两阶段实际值，
`actual_tail_contribution`记录额外贡献，`actual_average_prune_rate`包含额外删除。
`final_llm_tokens`为原第二阶段目标，`stage2_llm_visual_tokens`为该阶段观察值；
`final_llm_visual_tokens`为最后一层的实际图像数量，末尾删除实验中为0。

## Base＋Text旁路

主干输入保持不变：MyPool输出 → LLM前2层 → H2。仅在调用旁路时，选取图片段中的剩余Base
空间token及图片段外的全部Text，排除AnyRes与图片newline。按原顺序拼接，保留所选行原本的
位置编号；旁路最后Query仍是同一个末尾提示token。

使用现有旁路checkpoint预测Base评分，按实际来源合并权重回投并映射到AnyRes，只剪AnyRes，
保留剩余Base。映射、合并来源及Top-k规则沿用现有实现。

这是推理上下文消融，不是重新训练，也不表示H2中的Base/Text不含AnyRes信息：这些特征已在
主干前2层融合了完整多模态输入。

## 输出和计时

顶层及各子实验`summary.csv`只有八列：

```text
config,tp,fp,tn,fn,acc,Lat,mem
```

配置示例：`prune=30%;lambda=0.7;base_full;input=full;tail=2`。
`acc`是百分比；`Lat`是平均首token延迟（毫秒）；`mem`是完整回答生成期间PyTorch峰值allocated
显存（GiB），包含驻留模型及输入，不代表GPU计算利用率。

计时边界与现有实验相同：准备好的输入进入`model.generate`至第一个生成token。
实际MyPool、旁路、映射/选取、末尾删除和模型计算均计时。固定几何预处理和未使用的诊断统计
沿用`lean`策略；不从Lat中减去任何测得的计算耗时。

CSV仅精简展示；原始`metrics.json`、配置、逐图预测及预算日志继续保留，供核对和续跑。
原`run_pool_distill.py`默认仍输出旧格式；单独需要精简可加`--summary-format compact`。
也支持独立指定`--tail-drop-layers 2`或`--predictor-input base_text --selection-modes base_full`。

## 中断续跑

```bash
python run_pool_distill_ablations.py \
  --resume \
  --output-dir output/pool_distill_ablations_01
```

恢复原配置及GPU安排，各子实验复用已完成配置和逐图进度。不同旁路输入、末尾删除设置不会
混用结果；旧的普通实验目录不能当作本套件目录续跑。
