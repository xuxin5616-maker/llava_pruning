# 两份蒸馏权重对比：固定100张VisA test

入口 `compare_distilled_visa100.py` 不重新训练，复用现有混合基准方法和 Lat/Mem 统计实现。
默认路径为上次双卡训练生成的MVTec学生和原VisA TWIG学生：

```bash
python compare_distilled_visa100.py \
  --gpu 5 \
  --mvtec-checkpoint output/mvtec_train_visa_two_gpu_01/student/last.pt \
  --visa-checkpoint output/attention_predictor_twig_train_01/best.pt \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --prune-rates 50 70 90 \
  --output-dir output/distilled_mvtec_vs_visa_100_01
```

如果MVTec权重来自之前单独的训练入口，显式改为
`--mvtec-checkpoint output/attention_predictor_mvtec_train_01/last.pt`。
如果实际VisA训练文件夹是 `attention_predictor_copied_train_01`，也通过参数显式覆盖，
不会自动搜索其他旧权重、猜测目录或换用另一份checkpoint。

## 样本与实验

只从官方CSV的 `split=test` 取样。seed42，按类别×正常/异常分层：每个非空组先取1张，
剩余名额按各组剩余容量比例分配，整数名额使用最大余数法。每组内部固定随机抽样，
最终按ID排序。默认精确100张，覆盖所有12类及各类正常/异常，不直接取排序后的前100张。
列表写入 `visa_test_100.jsonl`，两个权重的所有剪枝率、以及不剪枝参考均复用同一个文件和顺序。
`--samples` 可显式更改数量，数量不足会报错，不悄悄改成更小的实验。

统一无类别名称的v0 A/B、greedy、anyres_max_9。两个权重各跑平均目标50/70/90%，共6组；
额外共享1组同批100张的不剪枝参考，用于相对ACC，总共7个实验。
每组新进程串行使用同一个GPU5；默认warmup10不计入100张测评统计。
只替换蒸馏权重：MyPool前置＋第2层蒸馏，λ=.7，base_full，base_text，fast，lean，anyres_first，
tail9仍作为两阶段预算之外的额外删除。不运行其他对照方法，不保存图片。

## VisA训练来源说明

原VisA学生的训练源合并了官方VisA train/test，再内部划训练/验证；因此这份权重在100张test上的
结果是诊断对照，不能称作独立测试性能。此入口按用户明确请求允许这份权重，记录来源标记；
原七方法独立评估入口 `benchmark_visa_methods.py` 的防重叠检查保持不变。
`run.json` 会保存两份权重及训练清单指纹、官方test来源、固定样本构成和训练重叠审计。
如果权重目录中有 `train.jsonl` / `validation.jsonl`，按图片路径核对选中100张中有多少张出现在这些
列表里；若没有名单，重叠数为null，不声称已经验证不重叠。MVTec这一组用于跨数据集VisA评估。

## 输出及恢复

- `summary.csv` 保持 `config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct`。
  config用 `student_trained_on=mvtec/visa` 区分权重；ACC/BACC/F1为百分数，Lat为平均首token延迟ms，
  mem为完整生成期间的peak allocated GiB；相对ACC使用本次100张的共享不剪枝参考，不使用MVTec91.18%。
- `budget_summary.csv` 保留实际平均率、两阶段值、额外tail贡献及token数。
- `mvtec/`、`visa/`、`reference/` 保存各实验的完整指标、逐图回答、时间/显存及预算记录。
  PRE/Recall/TNR和未解析数保存在每组 `metrics.json`。
- 根目录各 `.log` 保存加载、warmup、逐25张进度和ETA；终端每30秒输出已保存数量。

中断后使用相同代码恢复：

```bash
python compare_distilled_visa100.py --resume \
  --output-dir output/distilled_mvtec_vs_visa_100_01
```

恢复使用保存的权重、样本、GPU和参数，跳过完整实验、保留部分实验的已保存样本。
权重、输入、源码或运行精度/库/GPU身份发生变化时拒绝混用。CPU测试不能代替服务器GPU验证。
