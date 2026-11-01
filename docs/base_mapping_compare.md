# Base＋Text：优化前后完整模型对比

`compare_pool_mapping.py`只比较`base_full`，旁路输入固定为`base_text`，末尾不额外删图
（`tail=0`）。使用真实模型和输入JSON中的全部图片，统计ACC、首token Lat及峰值Mem，不保存图片。
没有训练、合成数据替换或100张限制。

```bash
python compare_pool_mapping.py \
  --gpus 7 6 5 4 3 2 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --prune-rates 30 70 \
  --lambdas 0.7 0.8 0.9 \
  --output-dir output/base_text_mapping_compare_01
```

共12组：2种剪枝率 × 3种λ × 原版reference/优化版fast。原版与优化版均使用lean辅助路径，
只改变Base映射实现；预算、旁路输入、checkpoint、随机种子、warmup、解码及Lat/Mem口径相同。
两个版本都计入实际剪枝工作，不扣时间。该脚本不会把合成映射速度当成模型Lat。

初始任务分配固定为：

| 物理GPU | 剪枝率 | λ | 执行顺序 |
|---:|---:|---:|---|
| 7 | 30% | 0.7 | reference → fast |
| 6 | 30% | 0.8 | reference → fast |
| 5 | 30% | 0.9 | reference → fast |
| 4 | 70% | 0.7 | reference → fast |
| 3 | 70% | 0.8 | reference → fast |
| 2 | 70% | 0.9 | reference → fast |

每卡同时只有一个模型进程，每对实验始终使用同一张物理GPU。各GPU独立推进，不使用DDP。
若改变剪枝率/λ/GPU数量，则按配置顺序轮流分配GPU，每对仍固定在同一卡。
多GPU实验共享服务器CPU和系统资源，因此速度结果仍会受其它任务负载影响。

顶层`summary.csv`只含8列，成对排列：

```text
config,tp,fp,tn,fn,acc,Lat,mem
```

配置末尾明确标注`mapping=reference`或`mapping=fast`。ACC为百分比，Lat为平均首token毫秒，
Mem为完整回答期间PyTorch峰值allocated GiB。结果包括驻留模型与输入，Mem不是计算利用率。
逐任务日志在顶层`logs/`，完整配置、metrics、逐图预测和预算保存在各任务子目录。

每对完成后按question_id核对全部答案；顶层`comparison.json`的`pair_checks`记录原始文本差异数、
解析后的A/B差异数、ACC变化及Lat比值。即使ACC相同，也能发现抵消后的逐图预测变化。
发生预测差异不会掩盖或覆盖结果，需结合该记录评估是否采用快路径。

中断后：

```bash
python compare_pool_mapping.py \
  --resume \
  --output-dir output/base_text_mapping_compare_01
```

自动恢复原配置/GPU分配，复用已完成任务并继续逐图进度。为保持配对，续跑时不能更换GPU安排
或其它实验参数。第一次运行使用新目录，勿用旧普通实验目录续跑本对比。
