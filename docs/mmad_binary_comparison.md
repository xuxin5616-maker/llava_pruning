# MMAD 二分类：排除VisA，0-shot 七种方法＋联合自身评分对照

入口为 `benchmark_mmad_binary.py`，只读用户服务器上的 MMAD，不下载或修改数据。
沿用现有七种剪枝实现、FP16 / FlashAttention2、原生 AnyRes max9，单张待测图、greedy、batch1。
不输入正常参考图、mask、GT 或其它问题的回答，不保存 PNG。
当前学生 checkpoint 由 VisA 训练，因此该入口默认排除 MMAD 的全部 VisA 来源题目。
不剪枝参考及所有剪枝方法都使用相同的筛选结果，只评测 MVTec-AD、MVTec-LOCO、GoodsAD。
需要补入VisA并复用旧结果时，使用下一节的显式选项。

## 在旧结果上补入VisA，不重跑已完成题目

更新服务器的ex分支后，使用新输出目录运行：

```bash
git pull origin ex

python benchmark_mmad_binary.py \
  --gpus 5 6 7 \
  --include-visa \
  --reuse-dir /home/yz/xxy/code/pruning/llava_pruning/output/mmad_no_visa_two_hybrids_gpu567_01 \
  --output-dir output/mmad_with_visa_two_hybrids_gpu567_01
```

`--reuse-dir` 恢复旧 `run.json` 的模型、学生、方法列表、50/70/90%率、两组hybrid配置、
seed及warmup，不需要再次填写这些路径或参数。只需选择新输出目录和GPU。
旧目录只读，输出必须使用与旧目录分开的新目录，不能嵌套其中。
要先检查数据与可复用数量，可在同一命令上加 `--check-data`，无需显卡或学生权重，也不写文件。

复用要求旧目录保留 `run.json`、冻结的 `mmad_binary.jsonl`，以及每组的
`predictions.jsonl` 和 `inference.json`。只有summary.csv不足以恢复逐题分数、时间分位数和显存峰值。
支持此前的1.3、1.4、1.5以及当前1.6版no-VisA清单；方法配置或输入不匹配时明确报错。
已完成题目全部复制其真实记录；旧任务尚未完成则继续补齐其剩余非VisA题目，然后补测VisA。
PyramidDrop90%仍标为不可达，不伪造记录。

旧问题顺序、提示词、选项及对应的逐题seed保持不变，不用包含VisA的新全量shuffle覆盖旧题。
新VisA问题追加到清单末尾，选项按原seed单独冻结一次，所有方法共享同一份选项。
每组若旧数据已完整，预热也只使用待补测的VisA，不对旧题重新推理。
只补入MMAD标注中的VisA Anomaly Detection题，不混入独立VisA benchmark的额外图片。

脚本重新根据全部逐题记录计算 `summary.csv`、`summary_by_source.csv` 和 `budget_summary.csv`：

- ACC/混淆矩阵按题汇总；Lat对全部旧题＋新题的逐题测量取均值，分位数也重新计算。
- Mem取全部测量的最大值；不将两个CSV的均值直接再平均，也不将显存求和。
- 相对ACC使用补全后的完整MMAD不剪枝参考；各来源的相对ACC仍使用各自的不剪枝参考。
- 来源CSV增加VisA行，同时保留另外三个来源；旧来源的已有原始回答及测量值保持不变。
- `metrics.json` 的 `reused_from_previous` 记录从旧目录导入的数量；`run.json/reuse_history` 保存导入数量、
  文件指纹、原GPU以及历史/当前源码指纹，`reused_run.json` 保留旧清单快照。

这是明确的历史结果复用，并不声称旧测量来自当前源码指纹。新推理仍须匹配该组原来的
精度、attention、依赖库、GPU型号和总显存身份，才允许合并Lat/Mem；环境不一致时拒绝混入。
即使型号相同，不同运行时的GPU负载/时钟仍会影响延迟，脚本不会将旧时间改写为新时间。
模型配置、标注、旧题元数据、学生及训练来源清单会核对；没有为原版全部主模型权重重新计算哈希。

两组hybrid仍用VisA训练权重，它们的VisA结果存在学生训练来源重叠。
CSV的hybrid配置标记 `training_overlap=VisA`，`run.json` 同时说明整体分数包含该来源；
其它六个方法和不剪枝参考不使用这个学生权重。

补测中断后直接恢复新目录，不重新导入旧目录：

```bash
python benchmark_mmad_binary.py \
  --resume \
  --output-dir output/mmad_with_visa_two_hybrids_gpu567_01
```

恢复使用新清单中保存的完整配置及GPU，旧目录随后移走也不影响恢复。
MMAD推理依赖或输入发生变化时，仍遵守恢复校验；无关的根目录实验脚本不阻止续跑。
若要从头评测完整MMAD，可只加 `--include-visa`，不传 `--reuse-dir`；默认不加时继续排除VisA。

## 一键运行

在服务器项目目录运行，默认使用已确认的 **VisA 训练旧学生权重**：

```bash
python benchmark_mmad_binary.py \
  --gpus 5 6 7 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/MMAD \
  --mvtec-root /home/yz/xxy/data/datasets/MVTec-AD \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --prune-rates 50 70 90 \
  --output-dir output/mmad_no_visa_two_hybrids_gpu567_01
```

只跑 Base 引导评分、取消末尾 image token 删除（`tail=0`），其余沿用 λ=.7、
base_full＋base_text＋fast＋anyres_first：

```bash
python benchmark_mmad_binary.py \
  --gpus 5 6 7 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/MMAD \
  --mvtec-root /home/yz/xxy/data/datasets/MVTec-AD \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --prune-rates 50 70 90 \
  --experiment-group base_guided \
  --base-guided-tail 0 \
  --output-dir output/mmad_base_guided_tail0_50_70_90_01
```

这条命令启动4组：不剪枝基准＋Base引导50/70/90%，不运行其它六种方法及joint_self。
不保存图片，仍保存全部指标、按来源汇总及预算；相对ACC分母来自本次不剪枝基准。
`tail=0` 表示第2层后保留下来的视觉token继续进入后续所有层，不再在最后9层前额外删除。
两阶段剪枝及原生AnyRes处理不变，不强制3×3，不重新训练学生。

`--experiment-group` 可选 `all`（默认完整对照）、`base_guided`、`joint_self`；
每种选择均自动包含不剪枝基准。
`--base-guided-tail` 为0～26，默认9，只作用于Base引导组；joint_self默认tail=0，
可用 `--joint-self-tails` 单独选择一个或多个tail。
所选组、tail及每组GPU记录在 `run.json`，CSV的 `config` 使用实际tail。
不传这两个参数时，默认任务和方法设置沿用原完整对照。

## 联合自身评分：包含VisA，扫描tail0～10

这组使用各自蒸馏预测的attention：旁路输入Base＋AnyRes＋Text，Base与AnyRes合并统一Top-k，
λ=.7、LLM前MyPool不变；不保存图片。使用GPU3/4，完整评测MMAD四个来源：

```bash
python benchmark_mmad_binary.py \
  --gpus 3 4 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/MMAD \
  --mvtec-root /home/yz/xxy/data/datasets/MVTec-AD \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --include-visa \
  --experiment-group joint_self \
  --prune-rates 50 70 90 \
  --joint-self-tails 0 1 2 3 4 5 6 7 8 9 10 \
  --output-dir output/mmad_all_joint_self_50_70_90_lambda07_tail0_10_01
```

共33组hybrid＋1组共用的不剪枝参考，汇总34行。只跑指定joint_self组，
不重复运行其它方法，也不为每个tail重复测不剪枝。
`summary.csv` 汇总全部配置，`summary_by_source.csv` 分别报告GoodsAD、MVTec-AD、MVTec-LOCO、VisA。
各行config包含真实tail；tail>0标为 `tail=N(extra)`，实际总剪枝率及额外tail贡献保留在预算表。
tail只控制何时删除第二阶段保留的所有image token，不删除LLM层、不改变原两阶段预算或评分模式。

`--joint-self-tails` 默认仅0，支持0～26的不同整数，原有默认任务及tail0任务ID保持兼容。
不剪枝参考也使用包含VisA的完整题集，Ratio相对于该实测参考，不使用91.18%。
沿用VisA训练学生，因此VisA来源的训练重叠标注与其它完整MMAD实验相同。
此命令从头运行新扫描，不传 `--reuse-dir`。若导入旧结果，脚本会恢复旧实验的tail列表，
不把不同tail配置的历史回答互相复用。中断后使用原目录和 `--resume`，无需重新列出tails。

先检查下载的数据结构（无需显卡、模型或学生权重，不产生输出文件）：

```bash
python benchmark_mmad_binary.py \
  --data-root /home/yz/xxy/data/datasets/MMAD \
  --mvtec-root /home/yz/xxy/data/datasets/MVTec-AD \
  --check-data
```

自动查找根目录及 `dataset/MMAD/`、`MMAD/`、`annotations/` 中的 `mmad.json`。
该文件必须为官方问题标注：以图片相对路径为键，值包含 `conversation` 列表、Question、Options、Answer、type。
不是普通分类 JSON，也不是 MMAD-BBox 或只有图片路径的 metadata CSV。
图片从根目录、`images/`、`dataset/MMAD/`、`MMAD/` 下查找；允许来源目录软链接。
已兼容用户下载的原始 `test` 布局，按固定规则精确映射，例如：

```text
标注：MVTec-LOCO/breakfast_box/image/logical_anomalies/000.png
文件：MMAD/MVTec-LOCO/breakfast_box/test/logical_anomalies/000.png

标注：DS-MVTec/bottle/image/broken_large/000.png
文件：--mvtec-root/bottle/test/broken_large/000.png
```

`--mvtec-root` 应指向包含 `bottle/`、`cable/` 等类别目录的原始 MVTec-AD 根目录。
它对标注中的 `DS-MVTec` 和 `MVTec-AD` 两种前缀都生效；LOCO、GoodsAD仍从 MMAD 中读取。
该参数显式覆盖 MVTec 来源，即使 MMAD 中也有 MVTec 图片，也按用户指定的目录读取。
不指定时先使用标注同名来源的 `image` 布局，再尝试原生 `test` 布局，最后尝试两个MVTec来源前缀互换。
始终保留类别、缺陷类型、文件名和扩展名，不做递归同名搜索，不使用 `train/validation/ground_truth` 替代查询图。
原始图片尺寸从实际文件读取，冻结清单记录解析后的绝对路径；评测ID、问题、选项、GT仍来自原标注。
若安装位置不同或找到多个候选，明确添加：

```bash
  --annotations /实际位置/mmad.json \
  --image-root /包含DS-MVTec等来源文件夹的目录
```

脚本报错时会指出缺少的是问题标注还是具体图片；不会换用另一个数据集凑数。
实际问题数按本地标注计算，不固定为图片总数或100张。

中断后使用原目录恢复：

```bash
python benchmark_mmad_binary.py --resume --output-dir output/mmad_no_visa_two_hybrids_gpu567_01
```

恢复沿用保存的 GPU 分配/参数和问题顺序，已完成实验跳过，未完成实验读取逐题记录后继续。
MMAD推理依赖、模型配置、问题标注、冻结清单、学生权重或训练来源清单变化时拒绝混用。
每个工作进程还检查图片大小及修改时间；没有对图片内容和主模型全部权重逐个校验哈希。
当前版本为1.6；旧版本结果可通过上面的 `--reuse-dir` 显式导入新实验，不能直接改旧目录的source筛选再恢复。
准备数据阶段因缺图退出时还未创建实验目录，可以修正路径后重新执行原命令。

### 修复 `Inference source changed` 后继续原来的tail扫描

旧检查将根目录所有Python实验脚本一起算入源码指纹。运行中更新不参与MMAD推理的
`benchmark_ficoco_v.py` 等文件，也会使后续工作进程在模型加载之前失败。
新检查追踪实际MMAD生成、剪枝、调度、指标函数及其本地导入，并保守检查vendor/third-party源码。
无关的根目录脚本变化不再阻止续跑；实际推理代码变化仍然拒绝混入。

对于只存整项目哈希的旧 `run.json`，恢复时在本地Git最近100个提交中寻找**整项目哈希精确匹配**的版本，
再比较当时与现在的MMAD推理依赖。全部一致才自动迁移；找不到旧版本或依赖有变化时明确报错，不跳过核对。
首次迁移保存 `run.before_scoped_source.json`，原 `source_identity` 保留，另记旧Git版本及核对时间。
新实验在 `mmad_source_identity` 保存各依赖文件指纹；正常恢复无需再扫描Git历史。

用户已生成的完整MMAD、joint-self、50/70/90%、lambda=.7、tail0～10实验可这样继续：

```bash
git pull origin ex
python benchmark_mmad_binary.py \
  --resume \
  --output-dir /home/yz/xxy/code/pruning/llava_pruning/output/mmad_all_joint_self_50_70_90_lambda07_tail0_10_01
```

沿用原GPU3/4、完整问题清单及所有方法参数，不需要重填。成功的未剪枝与50%/tail0两组直接跳过；
失败或未完成组继续各自保存的逐题进度。原有CSV会随剩余32组完成重新汇总，不把空白改成伪造的0。
运行环境、图片元数据、模型配置和学生检查均保留；修改这些内容不能靠源码恢复绕过。

`--gpus 5 6 7` 将实验固定轮转分配到三张卡，三张卡可以同时工作；每张卡最多一个任务，
每个任务都独立加载模型。不是把单个模型拆到三张卡上。
每个任务的 `config` 包含 `gpu=...`，`metrics.json`/`inference.json` 保留GPU与运行环境信息。
CSV由父进程统一更新；中断时会停止所有活动子进程，逐题记录供恢复使用。
若继续单卡串行，使用 `--gpu 5`；这两个选项不能同时给出。默认不指定时仍为GPU5。

## 任务及评测设置

仅选 `type == "Anomaly Detection"`；每张图片采用第一个 `conversation*` 字段，与官方示例的字段选择一致。
默认VisA 来源在图片加载、预热及推理之前排除，因此不要求安装被排除的 VisA 图片。
终端及 `run.json` 记录被排除的二分类题数和剩余来源题数；冻结清单中不含 VisA。
工作进程再次检查保存的来源筛选；只有显式启用 `--include-visa` 的新清单允许VisA。
保留问题和选项文本，包括少量问题中的 Maybe / Unknown 选项，正确答案必须为 Yes / No。
选项顺序用 seed42 **只打乱一次并冻结**，所有方法使用同一份顺序，不在方法之间重新抽取。
输出提示为工业检查说明、原问题、选项及直接回答选项字母的要求。
没有将本项目之前的固定“A=异常、B=正常”提示覆盖到 MMAD 上。

类别分组按查询图所在的 `good` / `normal` 文件夹识别正常，其余为异常；
原始来源 `DS-MVTec` 和 `MVTec-AD` 合并为 `MVTec-AD`。
原始回答先解析实际选项，再依据该问题的正确选项及正常/异常分组转换为内部二分类标签。
转换只发生在模型生成后，不向模型或剪枝器暴露答案。

两组都沿用 MyPool LLM前＋第2层后蒸馏、λ=.7、lean，同一份旧 VisA checkpoint，不重新训练：

| 对照组 | 评分旁路输入 | 第2层后评分/选择 | tail |
|---|---|---|---|
| `variant=base_guided`，原基准 | Base＋Text | base_full，fast Base映射；anyres_first，先剪AnyRes，预算不足再剪Base | 9，原预算之外额外删除 |
| `variant=joint_self`，新增 | Base＋AnyRes＋Text | anyres_self；Base及AnyRes各用自身预测的attention，合在一起统一Top-k，不固定两组保留比例，不保护全部Base；不做Base映射 | 0 |

两组的前置 MyPool 相同，都会压缩 Base 和 AnyRes。新增组的联合选择发生在第2层后的蒸馏阶段，
结构 newline 仍保护，保留 token 按原顺序送入后续 LLM。
这里的“自身 attention”仍是现有学生旁路的预测评分，不切换为直接读取主干 attention。
VisA 旧权重的训练来源保留在 `run.json`，其中明确记录 VisA 已从评测排除。
这一步排除的是学生已知训练来源，不据此保证主模型预训练数据无重叠。

默认 `--experiment-group all` 共25行：不剪枝参考1行＋原七种方法各3行＋新增联合自身评分组3行；
其中 PyramidDrop90%不可达，故实际启动24个 GPU 任务。
每卡串行、卡间并行，每组新进程，预热10题，不计入统计。
各方法边界与预算换算沿用 [七方法对照说明](visa_method_comparison.md#平均剪枝率及方法差异)：

- FastV 为第2层后 attention mask，50/70/90%指层平均**有效可见**剪枝率，物理序列/KV长度不变。
- VisionZip、AnchorPrune 为 LLM 前，CLSE 为第1层后，PriorTR 为第2层后。
- PyramidDrop 固定7/14/21层，90%明确标不可达，指标留空。
- 原基准的目标仍为原两阶段平均率，tail9额外提高总实际平均率；新增组tail0不产生这部分额外删除。
  两组同时改变了旁路输入、评分/选择及tail，能比较最终配置，但不能只将差异归因于某一个改动。

## 输出及口径

- `summary.csv`：`config,tp,fp,tn,fn,acc,Lat,mem,bacc,f1,relative_accuracy_pct,pre,recall,tnr,unparsed,nonbinary,mmad_score`。
- `summary_by_source.csv`：相同指标按 MVTec-AD / MVTec-LOCO / GoodsAD 分开汇总，附正常/异常数量。
  默认配置含 `exclude=VisA`，没有 VisA 指标行；显式包含VisA时为 `exclude=none`，并增加VisA来源行。
- `budget_summary.csv`：目标率、实际有效/物理平均率、两阶段值、tail贡献及实际序列计数；
  另列 `pre_base_tokens/pre_anyres_tokens/stage2_base_tokens/stage2_anyres_tokens/stage2_base_policy/protected_newlines`。
  联合组的Base/AnyRes保留数量来自实际选择，不按固定比例反推。
- `mmad_binary.jsonl`：冻结的问题、选项、正确选项、图片路径/尺寸/元数据，仅用于加载与评测。
- 每组目录：`metrics.json`、`predictions.jsonl`（原始回答、解析选项、内部标签、逐题时间/显存/预算）、`inference.json`。
- 每组 `.log`：加载、预热、每25题进度和 ETA；终端每30秒显示已保存数。

ACC/PRE/Recall/TNR/BACC/F1/mmad_score/相对ACC列均为百分数；Lat为毫秒，mem为峰值 allocated GiB。
相对ACC分母是**本次排除VisA后不剪枝 MMAD 的实测 ACC**，不用MVTec91.18%。

ACC计入全部二分类题；无法解析及选 Maybe/Unknown 均算错。
PRE/Recall/TNR/F1/BACC只用可解析为 Yes/No 的回答，与之前的二分类统计口径一致；
`unparsed` 为无法识别选项的数量，`nonbinary` 为选择 Maybe/Unknown 的数量，二者分别报告。
若这两项非零，混淆矩阵总和会小于题数。

`mmad_score` 另采用每来源 `(正常答对率+异常答对率)/2`，总览为剩余来源间等权平均，不给被排除的VisA填0。
正常/异常答对率的分母包含未解析及非二分类选项，因此它可能与只用可解析回答的 BACC 不同。
本入口使用严格解析，不用官方部分示例的强制模糊匹配，也不删除无效回答；因此**不宣称精确复现官方评分脚本**。
这是用户指定的0-shot子任务对照，也不是论文默认1-shot数字的直接复现。

Lat：GPU驻留输入进入 `model.generate` 至首token选择并追加完成、同步；
包括视觉编码、projector、packing、在线评分/合并/映射/蒸馏/剪枝和prefill。
不包括模型加载、CPU读图/预处理、输入H2D、预热、结果保存及准备好的静态几何。
Mem：完整回答生成的 PyTorch peak allocated，含模型/学生权重和输入；不是整卡 nvidia-smi 显存或GPU利用率。
分位数、reserved和占设备容量百分比另存 `metrics.json`。
多卡测得的Lat/Mem属于各任务分配的GPU，不做跨型号或跨卡负载归一化；
如果要统一到一张卡比较延迟，可改用 `--gpu 5`。

## 依据

- [MMAD 作者仓库](https://github.com/jam-cc/MMAD) 与 [数据安装说明](https://github.com/jam-cc/MMAD/blob/main/dataset/MMAD/README.md)。
- [论文 §4.1 评测设置](https://arxiv.org/html/2410.09453v3#S4.SS1)：默认1-shot，也有0-shot设置。
- [官方问题与选项解析示例](https://github.com/jam-cc/MMAD/blob/main/evaluation/examples/GPT4/gpt4v.py)。
- [官方评分](https://github.com/jam-cc/MMAD/blob/main/evaluation/examples/helper/summary.py)：来源归并、正常/异常平衡及来源宏平均。

本地验证涵盖数据筛题、动态选项、来源统计、进程调度、断点恢复和CPU小模型生成；正式GPU数值、Lat和Mem需服务器运行。
