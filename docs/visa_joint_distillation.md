# VisA：Base和AnyRes联合剪枝

`benchmark_visa_joint.py`：直接BICUBIC拉伸1152×1152、固定3×3，全量官方test，不保存可视化。
默认GPU5/6，平均剪枝率50/70/90，λ=.7/.9/.99，只跑tail=0，共9组剪枝，另测1组不剪枝基准。
沿用相同MVTec训练权重、FP16/FlashAttention2、lean和原MyPool阶段。评分为anyres_self，旁路输入为full。

## 唯一的算法改动

- LLM前MyPool不变，仍压缩Base和AnyRes。
- 第2层后取消Base保护和分组配额，把所有剩余Base/AnyRes空间token合并统一Top-k。
- 旁路输入完整的Base+AnyRes+Text，由同一学生预测器为所有空间token评分；Base和AnyRes都使用自己的预测attention，不做Base映射，也不对两组分别归一化。
- 沿用原蒸馏checkpoint，无需重新训练；这里的attention来自蒸馏预测器。
- 保留结构换行，保留token按原顺序送入后续LLM，位置/cache处理不改。
- 两阶段总预算和λ含义不变；不保证两组同率，也不保证某组一定保留token。
- tail=0，不执行末尾删除。实际Base/AnyRes保留数在第2层后统计。
- 保留原protect/anyres_first策略，通用入口可用 `--base-policy joint` 选择新策略。
- joint要求0<λ<1，组保留数由实际评分决定；缺少预算的旧记录不能仅按剪枝率重建组计数。

## 与GPU3/4的六方法实验同时运行

为了保持正在运行的源码指纹不变，原目录只fetch，在旁边创建独立工作目录；不要在运行中的目录切换源码。
以下 `--previous-output` 指向之前成功读到学生权重的stretch结果目录，自动取同一checkpoint。

```bash
cd /home/yz/xxy/code/pruning/llava_pruning
git fetch origin ex
git worktree add --detach ../llava_pruning_joint_self origin/ex
cd ../llava_pruning_joint_self

python benchmark_visa_joint.py \
  --gpus 5 6 \
  --previous-output /home/yz/xxy/code/pruning/llava_pruning/output/visa_grid3_letterbox_stretch_70_tail0_9_01/stretch \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --prune-rates 50 70 90 \
  --lambdas 0.7 0.9 0.99 \
  --tails 0 \
  --output-dir output/visa_joint_self_50_70_90_lambda07_09_099_tail0_01
```

也可用 `--checkpoint /absolute/path/to/last.pt` 替代previous-output。
剪枝率/λ组合轮流分给GPU5/6，各卡串行，两卡之间并行。

## 输出

- summary.csv保持11列config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct，config注明anyres_self、input=full、joint、tail=0。
- Ratio分母使用这轮重新测量的拉伸不剪枝ACC；Lat包含所有在线剪枝，Mem含学生权重。
- budget_summary.csv增加pre_base_tokens、pre_anyres_tokens、stage2_base_tokens、stage2_anyres_tokens、stage2_base_policy。
  每个样本的实际整数计数同时保存在predictions.jsonl的budget中，汇总为样本均值。
- 预处理缓存、回答/时间/显存/预算、日志和恢复行为沿用现有评测框架。
- 不重新训练、不读取GT/mask参与剪枝，不复用旧保护Base策略的准确率或时间。

中断后在该独立目录运行：

```bash
python benchmark_visa_joint.py --resume --output-dir output/visa_joint_self_50_70_90_lambda07_09_099_tail0_01
```
