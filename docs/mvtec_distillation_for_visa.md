# MVTec 全图蒸馏，独立 VisA 测评

本版本使用 `/home/yz/xxy/data/datasets/MVTec-AD` 的所有类别 `train/`、`test/` 图片，
包括正常和缺陷图；不划验证集，不读取 GT 或 ground_truth mask，不使用任何 VisA 图片训练。
目录需要为 `<根目录>/<类别>/train/good/...` 和 `<根目录>/<类别>/test/<类型>/...`。
不需要 question JSON，也不需要 VisA CSV。`train.jsonl` 记录实际使用的全部图片名单。

## 服务器训练命令

```bash
CUDA_VISIBLE_DEVICES=5 python train_attention_predictor_mvtec.py \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --data-root /home/yz/xxy/data/datasets/MVTec-AD \
  --epochs 1 \
  --output-dir output/attention_predictor_mvtec_train_01
```

新入口复用 `train_attention_predictor.py` 的训练实现，只覆盖数据源、数据根目录、
validation_fraction=0 和输出目录默认值。原 VisA 入口默认仍为 VisA＋90/10 内部划分。
通用入口也支持 `--dataset mvtec --validation-fraction 0`，此时需显式设置 MVTec 的 data-root。

沿用原训练方式：冻结主模型，H2 输入三个独立复制的 Qwen2 blocks3/4/5，
用第三层的 attention 对齐教师 block22 的末尾 prompt query、mean-head 空间 attention；
损失仍是 attention KL，不训练答案。训练旁路输入仍是 Base＋AnyRes＋Text。
基准推理的 Base＋Text 模式沿用现有做法，此次不改变训练输入分布。

参数沿用上一版：1 epoch、AdamW、峰值LR5e-5、betas=(.9,.98)、weight_decay=0、
gradient_accumulation=128（microbatch1）、3%线性warmup＋cosine、clip_grad_norm=1。
主模型FP16/FlashAttention2不改；独立学生FP32参数＋BF16 autocast，默认梯度检查点。
只使用一个可见GPU。显存是否足够需按真实序列长度在服务器验证。

## 不验证时的保存行为

- 每轮保存 `last.pt`；不生成 `best.pt`，不按训练loss冒充验证loss挑选模型。
- `method.json` 的 checkpoint 为相对路径 `last.pt`，可随文件夹整体搬运。
- `history.csv` 只有 train 行，记录每轮 KL、耗时、更新步数和学习率。
- 终端打印逐样本进度、loss、LR、step、elapsed 和 ETA。
- `training_status.json` 的 `best_validation_kl` 为null，记录最终完成状态和 `last.pt`。
- `train_run.json` 明确记录 MVTec test 已用于训练，VisA test 未用于训练。

训练与验证不共用任何隐含回退；禁用验证时不执行 validation forward。
使用最后一轮权重，不根据后续 VisA 测试准确率挑选权重。
由于MVTec test已参与训练，本权重后续的独立评估数据是VisA，不能再把MVTec test当作独立测试。

## 后续 VisA 对照

训练完成后，指定生成的 `last.pt`，即可启用此前可选的基准模式：

```bash
python benchmark_visa_methods.py \
  --gpu 5 \
  --checkpoint output/attention_predictor_mvtec_train_01/last.pt \
  --prune-rates 50 70 90 \
  --output-dir output/visa_methods_mvtec_student_01
```

该命令运行原七种方法及不剪枝参考；PyramidDrop90%仍明确不可达。
基准模式仍是MyPool＋第2层蒸馏、λ=.7、Base映射、base_text、fast、anyres_first、额外tail9。
模型和VisA默认路径沿用此前设定；本地CPU测试不代表服务器训练或准确率已完成。

## 两张卡并行的一键入口

下面以训练卡4、评测卡5为例，按实际可用编号替换：

```bash
python run_mvtec_visa_pipeline.py \
  --train-gpu 4 --eval-gpu 5 \
  --model-path /home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/ \
  --mvtec-root /home/yz/xxy/data/datasets/MVTec-AD \
  --visa-root /home/yz/xxy/data/datasets/VisA \
  --visa-split-csv /home/yz/xxy/data/datasets/VisA/split_csv/1cls.csv \
  --epochs 1 --prune-rates 50 70 90 \
  --output-dir output/mvtec_train_visa_two_gpu_01
```

两个子进程先同时启动：训练卡只运行单卡蒸馏；评测卡串行运行不需要学生权重的六种方法及不剪枝参考。
两者结束后确认训练完整完成，读取 `student/last.pt`，只在同一评测卡补跑三个基准模式实验。
通过输入/模型/提示词/seed/GPU/源码等配置匹配检查后，把前18个实验的回答、时间/显存、预算记录和指标
复制到最终目录；不重新测量、不覆盖原始结果。PyramidDrop90%仍保留不可达记录。

最终表为 `<output-dir>/visa_all_methods/summary.csv`，预算表在同目录。
训练输出为 `student/last.pt`、`student/history.csv`，训练日志为 `training.log`；
其余方法的首轮日志为 `visa_independent.log`，原始结果保留在 `visa_independent/`。
终端每30秒显示训练loss/ETA的最新进度和评测状态，补跑基准时沿用评测入口的进度输出。
最终manifest记录并行训练期间完成的独立实验与训练结束后完成的基准实验这一运行上下文。
训练失败或独立实验失败时保留日志和已有结果，不使用尚未训练完成的中间checkpoint冒充最终权重。
