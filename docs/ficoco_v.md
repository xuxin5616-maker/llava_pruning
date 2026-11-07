# FiCoCo-V：MVTec、MMAD、MPDD 一键评测

入口 `benchmark_ficoco_v.py`，核心实现 `llava_pruning/ficoco_v.py`。
这是当前 OneVision/SigLIP 的论文方案适配，不把官方 CLIP 实现直接当作已支持 SigLIP。
FiCoCo-V 是独立、免训练的对照方法，不加载蒸馏学生，不使用 MyPool、λ两阶段预算或tail删除。

## 运行

服务器项目目录先更新，然后检查全部数据。检查无需显卡或模型权重，不写输出文件：

```bash
git pull origin ex
python benchmark_ficoco_v.py --include-visa --check-data
```

使用已配置的三个数据路径，一键运行：

```bash
python benchmark_ficoco_v.py \
  --gpus 5 6 7 \
  --include-visa \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --mvtec-input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --mvtec-data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --mmad-root /home/yz/xxy/data/datasets/MMAD \
  --mmad-mvtec-root /home/yz/xxy/data/datasets/MVTec-AD \
  --mpdd-root /home/yz/xxy/data/datasets/MPDD \
  --prune-rates 50 70 90 \
  --output-dir output/ficoco_v_mvtec_mmad_mpdd_with_visa_01
```

默认共12个有效实验：每套数据各自运行不剪枝参考，以及FiCoCo-V的50/70/90%。
GPU5固定运行MVTec四组、GPU6固定运行MMAD四组、GPU7固定运行MPDD四组；卡间并行，每卡一次只运行一个新进程。
不指定GPU时也默认5/6/7。单卡可用 `--gpus 5`；`--datasets mvtec` 等可只选部分数据集。
没有额外缩小到100张，没有保存PNG或曲线，没有重新训练。
`--include-visa` 将MMAD中的VisA来源二分类题目纳入FiCoCo-V及不剪枝参考，
不是另外运行一套独立VisA数据集。省略该选项时仍沿用原来的排除VisA设置。
两种来源筛选使用不同的新输出目录；不可将旧的no-VisA目录直接当作包含VisA的实验恢复。

恢复相同实验：

```bash
python benchmark_ficoco_v.py --resume --output-dir output/ficoco_v_mvtec_mmad_mpdd_with_visa_01
```

恢复保存的配置、GPU分配、输入和逐题记录；跳过已完成任务。
是否包含VisA同样从保存的配置恢复，不因恢复命令省略 `--include-visa` 而删除VisA。
源码、输入、标注或模型配置变化时拒绝混入旧实验。每次工作进程另检查实际图片的大小与修改时间。

## 数据和指标

- MVTec沿用之前的 `question_musc.jsonl`、图片目录和类别v0提示词。mask不进入冻结清单或模型推理。
- MMAD只读Anomaly Detection；上述命令包含VisA，省略 `--include-visa` 时排除VisA。保留冻结的真实问题、选项和答案映射，兼容先前修复的原始test目录及独立MVTec根目录。
- MPDD复用本地已有的六类test读取器及通用v0 A/B提示词；不使用训练图或mask。
- 所有任务：0-shot，单图，batch1，greedy，原生AnyRes max9；沿用backend的FP16/FlashAttention2和TF32设置。

`summary.csv` 每配置一行，config含数据集、方法、目标率、GPU；FiCoCo行另含起始视觉层、balance和quantile。
指标为ACC、PRE、Recall、TNR、BACC、F1、Lat、Mem、相对ACC、未解析/非二分类数量，MMAD另有来源宏平均分。
百分指标单位为%，Lat为平均首token延迟毫秒，Mem为完整生成的PyTorch峰值allocated GiB。
分位数、reserved峰值和占设备容量比例保留在每组 `metrics.json`。
相对ACC分母分别使用本次MVTec/MMAD/MPDD各自的不剪枝实测值，不使用固定91.18%。

`summary_by_source.csv` 按来源汇总；MMAD分别列出MVTec-AD、MVTec-LOCO、GoodsAD，其它两个benchmark各列自己的来源。
启用 `--include-visa` 后另有VisA行；MMAD的 `config` 标为 `exclude=none`，整体及不剪枝参考都使用这四个来源。
用config中的dataset区分“完整MVTec评测”和“MMAD中的MVTec来源子集”。
`budget_summary.csv` 记录实际物理层平均率、原始/保留token数、视觉起始层及打包规则。
`inputs/*.jsonl` 是冻结清单；每组目录包含原始回答、canonical答案、逐题时间/显存/预算及inference环境。

ACC中无法解析或Maybe/Unknown算错；PRE/Recall/TNR/F1/BACC只用可解析二分类。
MVTec/MPDD沿用原有严格A/B解析；MMAD根据其实际选项转换，严格解析，不强制模糊匹配。
MMAD的mmad_score对剩余来源的正常/异常正确率先平衡，再在来源间宏平均。

Lat完整包含视觉编码、独立QK评分、筛选、合并、projector、packing和LLM prefill。
CPU文件读取/预处理、输入H2D、静态几何准备、模型加载、预热10题及结果保存不计入。
不减去在线剪枝计算，不因token减少就声称实际速度一定改善。
每25题日志打印耗时、ETA及Lat；父进程每30秒显示各GPU进度。中断会停止全部活动子进程。

## 方法与适配边界

依据 [论文v4 §3、§9、§10.2、§10.3、Algorithm1](https://arxiv.org/html/2411.17686v4)，
结合 [官方仓库](https://github.com/kawhiiiileo/FiCoCo) 的实现核对。默认从第12个视觉block开始，在attention残差之后、MLP之前逐层压缩。
当前SigLIP有26个实际执行的视觉block，因此在12～26层均匀分配总删除数量；整数余数在前面的层各多删除一个。
这沿用逐层近似恒定删除数量的形式，用精确的最终token预算取代作者CLIP示例中固定每层r=42。

评分采用各view独立的attention-IN（对query轴平均），与v4 §10.3解释一致。
无CLS全局参照为跨head平均Key后，再求token均值与各token的负余弦相似度；所有729个位置均视作patch，包含位置0。
两个评分项按官方Filter的做法分别L2归一化，再以论文的 `balance*VR - (1-balance)*equivalent` 符号/系数组合。
balance默认0.35，关联分位数默认0.998；这里的balance与蒸馏实验的λ=0.7不是同一参数。
2×2窗口在原始稀疏坐标上选择最大冗余点并放大其排名，包含27×27网格的奇数边缘；相同分数确定性打破平局。
丢弃位置到保留位置的attention相关性，经分位数阈值筛选及逐来源归一化，以论文Eq2进行多对多加权回收；保留token仍按原始空间顺序输出。

官方CLIP源码与论文存在实现差异：其Filter使用pre-softmax分数、不同系数排列及ascending排序，带CLS/576网格和全局状态假设；
示例还通过`idx>11`控制开始层。本适配采用上述明确声明的论文解释，单独实现无CLS、27×27、逐view状态及第12层边界，
不能将本版本称为官方CLIP程序的逐位复现。

评分只复用native projection输出的Q/K，自行计算FP32 QK/softmax；不读取LLM attention，不修改LLM计算精度和attention后端。
特征融合保留视觉特征原dtype，随后执行原始MLP。运行期修改仅作用当前模型实例，正常或异常退出都恢复forward和hooks。

50/70/90指相对原生未剪枝Base＋AnyRes＋newline的LLM层平均物理token剪枝率。
所有LLM层输入同一个已压缩长度，所以等于最终packed序列的删除比例，整数舍入后的实际率另存。
精确保留总数扣除原始AnyRes行newline后均匀分配各view名额，最多差1；Base与各AnyRes view都参与压缩。
非零率采用保留空间锚点＋原始行newline的稀疏packing，不对缺失网格执行dense unpad/max9插值。
因此即使总token预算对齐，这仍是OneVision稀疏输入适配；不能忽略packing差异解释精度变化。
0%自然端点使用完整grid的原生unpad/max9/newline操作；不会把“直接调用未剪枝路径”冒充方法端点验证。

可选 `--vision-start-layer`、`--balance`、`--quantile` 覆盖默认方法参数；默认值与当前模型/论文依据已记录在run.json。
本地测试覆盖真实小模型decode、0%逐位一致性、实际27×27/26层边界及整数分组；正式CUDA/FP16数值和速度仍需服务器验证。
