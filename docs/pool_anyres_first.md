# 蒸馏阶段：优先剪AnyRes，预算不足再剪Base

`run_pool_distill.py`新增`--base-policy anyres_first`。默认`protect`仍执行原有规则。
这只改变两阶段混合方法中第2层后的选取；LLM前MyPool的分配、合并、模型精度、attention后端、
已有checkpoint和平均剪枝率分母均保持原样。新策略要求`0 < lambda < 1`，不改纯蒸馏的lambda=1路径。

设第2层后总保留预算为K、剩余Base数为B、newline数为P：

- 若`K-P >= B`，保留全部Base，用现有分数选取`K-P-B`个AnyRes，与旧方法的选取相同。
- 若`K-P < B`，删除全部AnyRes，用同一次旁路预测的Base分数保留最高的`K-P`个Base。
- 先计算分数，再删token；保留原有位置编号及KV剪枝方式，分数相同时保持原顺序。
- 不降低目标、不跳过样本；结构newline加至少一个空间token、MyPool前置crop最小数量仍需满足预算。

Base+Text旁路仍读取第2层的全部剩余Base与Text，不读取AnyRes行；这些H2特征已包含前两层的多模态交互。
使用Base映射模式时仍沿用现有的来源权重、反投影和插值。Base需要进一步压缩时，按Base自身的预测分数排序。
实际评分、映射、选取和模型计算继续计入Lat。

## 单GPU5：90%，lambda=0.7/0.9，tail=9

```bash
python run_pool_distill.py \
  --gpu 5 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --prune-rates 90 \
  --lambdas 0.7 0.9 \
  --selection-modes base_full \
  --predictor-input base_text \
  --base-mapping fast \
  --base-policy anyres_first \
  --tail-drop-layers 9 \
  --pruning-audit lean \
  --summary-format compact \
  --baseline-acc 91.18 \
  --output-dir output/base_text_anyres_first_90_tail9_gpu5_01
```

本入口`--prune-rates`使用百分数，因此90%写`90`，不能写`0.9`。
两组顺序运行，全量数据，不保存可视化。模型加载、warmup和保存不计入Lat。
28层模型在进入第20层前额外删除全部剩余图像token；90%仍表示原有两阶段目标，tail贡献额外计入实际平均剪枝率。

例如3×3布局、原始packed长度7371：lambda=0.7最终保留298个Base、81个newline；
lambda=0.9保留196个Base、81个newline，两组AnyRes都全部删除。其它图片按实际布局计算。

CSV仍为11列：

```text
config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct
```

`config`增加`;base_policy=anyres_first`；BACC/F1与ACC都用百分数，参考ACC默认91.18%。
每图预算新增`stage2_base_tokens`和`stage2_base_pruned_tokens`，区分蒸馏阶段与tail删除之后的数量。
`stage2_anyres_tokens`不会出现负数。尾部删除后的`final_base_tokens`和`final_anyres_tokens`均为0。

首次运行请使用新输出目录；旧结果缺少新策略参数时视为`protect`，禁止混入`anyres_first`续跑。
同一新实验中断后：

```bash
python run_pool_distill.py --resume --gpu 5 \
  --output-dir output/base_text_anyres_first_90_tail9_gpu5_01
```

本地验证使用真实小模型生成、实际MyPool合并和缓存；7B模型的ACC/Lat/Mem需在服务器执行上述命令测量。
