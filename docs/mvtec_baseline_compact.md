# 全量MVTec：基准模式50/70%，每样本一张三栏图

入口 `visualize_mvtec_baseline.py`。使用输入JSON的全部记录，不限制100张、不筛ID或正常/异常。
默认沿用已确认的基准：MyPool LLM前剪枝＋第2层后蒸馏，lambda=.7，base_full、base_text、
fast映射、lean、anyres_first、tail9、AnyRes max9、v0、greedy。
沿用当前LLaVA主干的FP16/FlashAttention2及已有剪枝实现，不重新训练。

## 同时对比Base映射与联合自身评分

```bash
python visualize_mvtec_baseline.py \
  --gpus 5 6 \
  --prune-rates 50 70 \
  --selection-modes baseline joint_self \
  --output-dir output/mvtec_base_vs_joint_compact_50_70_gpu56_01
```

共4组：两种模式各跑50/70%，都是完整MVTec。GPU5先baseline50再joint_self50，
GPU6先baseline70再joint_self70，每张卡最多一个活动任务。

| 模式 | 旁路输入 | 蒸馏阶段评分与选择 |
|---|---|---|
| baseline | Base＋Text | Base分数映射到AnyRes，anyres_first |
| joint_self | Base＋AnyRes＋Text | 每个Base/AnyRes空间token使用其自身预测attention，合在一起统一Top-k，无固定两组配额 |

joint_self同样保留MyPool前置、lambda=.7、tail9和原两阶段层平均预算；
图像仍在tail删除前捕获。这里用的是现有蒸馏旁路的预测attention，不读取原生第22层attention。
两种模式图片标题分别标Baseline或Joint self-attention Top-k，逐题记录及图像索引也标selection_mode。

基准图片仍在 `<output>/prune_50/visualizations/`、`prune_70/visualizations/`，
新增组放在 `<output>/joint_self/prune_50/visualizations/`、`joint_self/prune_70/visualizations/`。
只跑新增组时使用 `--selection-modes joint_self`，其它路径及配置默认不变。

如果上一条仅基准的命令已经运行，请在它结束后，直接追加新增组：

```bash
python visualize_mvtec_baseline.py \
  --resume --add-joint-self \
  --output-dir output/mvtec_baseline_compact_50_70_gpu56_01
```

恢复原目录保存的GPU5/6、输入、权重、50/70%和基准进度，再追加joint_self两组。
已完成的基准PNG不会重跑；有未完成的基准组时，先在相应卡补完它，再运行新增组。
`--add-joint-self`重复执行不会重复添加任务，兼容上一版1.0的基准清单。
每个目录只运行一个入口；让此前的入口退出后再恢复/追加，避免两套入口在同卡同时加载模型。

## 查询之前是否生成过

助手无法直接读取服务器的实验目录，不能以本地说明推断服务器是否已运行。
更新后可先只读查询实际保存的MVTec元数据及PNG例子：

```bash
git pull --ff-only origin ex
python visualize_mvtec_baseline.py --find-existing
```

只扫描工程的 `output/outputs`，列出候选目录、元数据中的样本数和一张现存PNG的路径。
不加载模型、Torch或GPU，不重新推理，不写入旧目录。
这用于查找已有记录，不自动判定其方法/权重/剪枝率一定与新任务相同。
目录被改名、移走、超出扫描层级或未保留元数据时，查询不到也不能据此断言没有运行过。

## 一键启动两张卡

```bash
python visualize_mvtec_baseline.py \
  --gpus 5 6 \
  --prune-rates 50 70 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --checkpoint output/attention_predictor_twig_train_01/best.pt \
  --input-json /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl \
  --data-root /home/yz/xxy/data/datasets/Traid_eval_data/mvtec/ \
  --prompt-version v0 \
  --output-dir output/mvtec_baseline_compact_50_70_gpu56_01
```

路径均有上述默认值，完整简写是：

```bash
python visualize_mvtec_baseline.py --gpus 5 6 --prune-rates 50 70 \
  --output-dir output/mvtec_baseline_compact_50_70_gpu56_01
```

GPU5独立加载一次模型跑50%的全部样本，GPU6独立加载一次模型跑70%的全部样本。
两卡并行，每卡一个进程；不要在命令前加仅暴露单卡的CUDA环境变量。
没有另外的0%实验、预热、干净性能测量或重复绘图前向。
每个样本只执行一次完整greedy生成，模型实际仍执行tail9；快照在tail之前捕获。
这不是Lat/Mem benchmark；保存回答和图像索引，日志打印进度、耗时及ETA。

运行前仅检查数据：`python visualize_mvtec_baseline.py --check-data`，不创建目录，不加载模型或权重。

## 图像含义和文件

每个样本、每个剪枝率恰好一张PNG，从左到右：

1. 磁盘上的原始RGB图像，保持比例。
2. 真正的Base编码器输入，叠加MyPool＋蒸馏后的累计白色剪枝遮罩。
3. 真正的AnyRes输入块按原生网格拼接，叠加相同阶段的累计白色遮罩。

绘图阶段已由用户确认是**第2层后蒸馏完成、tail9删除前**；若画tail之后，Base/AnyRes都会全白。
顶部简要标出ID、GT、模型判断、剪枝率和lambda；不额外保存输入、逐crop或中间阶段PNG。
AnyRes给几块就显示几块，不强制九块。三栏等高，默认768像素，可用 `--panel-height 384` 缩小图像。
缩放使用最近邻，避免白色格子边界被混色。

白色为RGB(255,255,255)：不再独立保留的原始token位置，包括已合并进contextual特征的来源位置。
它表示独立空间锚点的保留情况，不等同于完全丢失所有来源信息。
SigLIP的27x27个14像素patch只覆盖378x378，384输入余下的6像素边缘保持原像素；
newline计入实际预算，但不假装成图像中的一块像素。raw输入视图的白色面积不等同于层平均剪枝率。

```text
output/mvtec_baseline_compact_50_70_gpu56_01/
  run.json                     # 固定基准、路径、样本总数、GPU/率、阶段定义
  input_frozen.jsonl            # 实际完整问题顺序、提示词、图片尺寸/元数据
  prune_50.log
  prune_70.log
  prune_50/
    visualizations/sample_000000000.png
    visualizations/sample_000000001.png
    ...
    predictions.jsonl          # 原始question_id、GT、回答、图像路径、少量token计数
    image_index.csv
    inference.json
    status.json
  prune_70/                    # 相同布局
```

文件名使用原清单索引，`predictions.jsonl/image_index.csv` 映射回真实question_id。
若输入有1725张，两组共3450张PNG。没有原始attention、完整来源矩阵等大数组。
不会删改之前的实验目录。

## 中断恢复

```bash
python visualize_mvtec_baseline.py --resume \
  --output-dir output/mvtec_baseline_compact_50_70_gpu56_01
```

恢复保存的GPU和所有配置，已记录的完整PNG直接跳过；仅补后续样本。
PNG先写临时文件并完成重命名，再提交逐题记录，避免中断留下已标完成但未写完的PNG。
输入、图片元数据、模型配置、学生权重及GPU/依赖环境仍会检查。
源码指纹作溯源记录，不以整个工程新增无关实验脚本为由中断可视化。
若改变实际剪枝方法或绘图样式，应选择新目录；诊断图不能当作受控Lat/Mem测量。
