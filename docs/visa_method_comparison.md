# VisA 官方测试集：七种方法的一键对照

入口为 `benchmark_visa_methods.py`。不改旧入口和原有配置文件，不保存可视化。

## 运行

在服务器项目目录运行。不提供学生权重时，自动跳过基准模式，运行其它六种方法及不剪枝参考：

```bash
python benchmark_visa_methods.py \
  --gpu 5 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --prune-rates 50 70 90 \
  --output-dir output/visa_methods_avg_50_70_90_01
```

默认 seed42、warmup10、greedy、batch1、原生 FP16/FlashAttention2、Base 在前、AnyRes max9。
每组单独启动进程，串行使用 GPU5。不提供 `--checkpoint` 时共运行18个有效实验；
基准模式的50/70/90%三行标为 `status=skipped`，指标为空，原因记录为未提供MVTec学生权重。
PyramidDrop90%额外保留一行“不可达”记录，指标为空。

准备好新的 **MVTec训练权重** 后，在上述命令中加上
`--checkpoint output/attention_predictor_mvtec_train_01/last.pt`，并使用新的输出目录，即可连同基准模式一起运行，共21个有效实验。
对应的无验证全图训练入口见 [MVTec蒸馏训练说明](mvtec_distillation_for_visa.md)。
不提供权重才跳过；显式提供了错误路径或已知使用VisA test训练的权重时仍会报错。

中断后保留同目录，执行：

```bash
python benchmark_visa_methods.py --resume --output-dir output/visa_methods_avg_50_70_90_01
```

恢复使用保存的参数和 GPU，跳过已完成实验；未完成实验重新加载模型并预热，保留已保存的逐样本结果。
没有权重启动的实验，恢复时仍跳过基准模式；不能在 `--resume` 时添加权重改变原实验任务。
源码、输入、split CSV、模型配置、学生权重发生变化时拒绝恢复，应使用新目录。预热不计入指标。
本地只能完成 CPU 小模型测试，正式的 CUDA/FP16/Flash 数值、速度和显存需要在服务器验证。

## 数据与提示词

严格读取 `split_csv/1cls.csv` 中 `split=test` 的行，使用 CSV 路径和标签，不按缺陷文件夹猜测正常图。
标准官方划分覆盖12类，测试共2162张（正常962、异常1200）；实际数量以运行时 CSV 为准并记录。
运行前冻结一份 `visa_test.jsonl`，所有实验用相同顺序和相同输入。

统一使用去掉类别名称的 v0：

```text
Are there any defects in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly.
```

GT 仅用于结果统计，不传入模型评分/剪枝。旧 VisA 蒸馏训练使用过 test；若权重旁的
`train_run.json` 明确记录 `original_visa_test_is_training_source=true`，入口拒绝运行。
没有训练清单的权重只能记录用户提供的训练来源，不能声称独立核验了训练集。

## 平均剪枝率及方法差异

统一参照每张图原生未剪枝的 packed token 数 N0（Base＋AnyRes＋结构换行 token），
定义 `r_avg = 1 - sum_l(N_l)/(28*N0)`，N_l 为进入第 l 个 LLM block 的图像 token 数。
统计时记录每个 LLM block 首次前向的实际物理长度，检查方法自己的计数；不使用生成阶段的单 token 解码长度。

| 方法 | 位置与预算 | 说明 |
|---|---|---|
| FastV | 第2层后；保留数 floor((28N0(1-r)-2N0)/26) | 沿用 joint attention mask；有效可见平均剪枝率为目标值。hidden/KV 物理长度不缩短，物理平均剪枝率单独为0。保护最后的换行标记。 |
| VisionZip | LLM前；最终 packed 总数 floor(N0(1-r)) | 复用现有 SigLIP received-attention 选择、dominant/contextual 合并、稀疏排列和原始行换行。扣除换行后，各 encoder view 的名额最多差1；对齐 native packed 分母，不能直接把原始每view剪枝率当层平均率。 |
| PyramidDrop | 第7/14/21层后；累计保留 a,a²,a³ | 求解 (1+a+a²+a³)/4=1-r；整数化实际值另记。前7层不剪，90%不可达。 |
| AnchorPrune | LLM前 | 沿用独立对照实现，目标为 native packed 层平均率，保护结构换行。 |
| CLSE | 第1层后 | 纯CLSE、单次剪枝，按28/27换算边界比例，Base/AnyRes联合选择。 |
| PriorTR | 第2层后 | 按28/26换算边界比例，沿用 prior-corrected 评分和位置/cache 处理。 |
| 基准模式 | MyPool LLM前＋第2层后蒸馏 | λ=.7，base_full，base_text，fast mapping，lean，anyres_first，tail=9额外删除。 |

基准模式的50/70/90%仍是**原两阶段**预算；tail9为额外消融。
因此它的最终实际平均率会更高，不能称为与其它方法严格等计算预算；`budget_summary.csv` 同时列出两阶段值、tail贡献、总实际值。
VisionZip仍为当前 OneVision/SigLIP 适配版本；稀疏打包不恢复 unpad/max9 resize，本入口只调整总名额，不改变原选择/合并公式。
FastV与物理压缩方法的“剪枝率”含义也存在上述差异，比较速度时必须保留此说明。

## 输出与单位

- `summary.csv`：`config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct`。
  ACC/BACC/F1为百分数，Lat为平均首token延迟毫秒，mem为峰值allocated显存GiB。
  相对ACC的分母是**本次实测的VisA不剪枝ACC**，不用MVTec91.18%。未完成/失败/不可达指标为空。
- `budget_summary.csv`：配置、状态、不可达原因、目标率、有效/物理实际平均率、两阶段值、tail贡献、token计数。
  按样本平均各自的比例；整数凑整差异保持原样。
- 每组目录的 `metrics.json`：完整ACC/PRE/Recall/TNR、F1/BACC可从计数复算、未解析数、Lat分位数、显存allocated/reserved等。
- 每组目录的 `predictions.jsonl`：逐样本ID、GT、回答、时间/显存、必要预算记录；用于复核和恢复。
- 每组 `.log`：模型加载、预热、每25张进度、耗时/ETA；终端每30秒报告当前已保存数。
- `run.json`：冻结配置、数据/源码/学生权重指纹、指标口径。

Lat 从已驻留GPU的输入进入 `model.generate` 到首个选择的token追加完成并同步。
包含vision/projector、在线评分、合并、蒸馏旁路、Base映射、剪枝、prefill；
不扣除任何在线必需计算。CPU读图/预处理、输入H2D、预先准备的静态几何/提示信息、模型加载、预热、结果保存不计入。
Mem 是完整回答生成期间 PyTorch peak allocated（含模型及学生权重），另存 reserved 和占显卡总容量比例；不是GPU运算利用率。
无法解析的回答计入ACC错误；PRE/Recall/TNR/F1/BACC仅用可解析A/B，单独记录未解析数。
