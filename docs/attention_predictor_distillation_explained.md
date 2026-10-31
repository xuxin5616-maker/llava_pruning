# Attention Predictor：蒸馏方法与旁路结构详解

后续推理消融（沿用训练checkpoint）：[末尾删图与Base＋Text旁路](pool_distill_ablations.md)。
这些实验不修改本文的教师、学生权重结构或蒸馏训练目标。

你说的是我们之前做的 **`attention_predictor`：在主模型第2层后，用三层旁路预测第22层 attention，再据此剪枝**。

我核对了当前本地代码。先强调两个容易混淆的地方：

- 当前已经不是最初的“512维轻量 Transformer”，而是**复制主模型第3、4、5层，保持原尺寸**。
- **蒸馏训练与剪枝策略是两件事**：训练让旁路模仿深层 attention；推理再利用预测分数选择保留哪些视觉 token。

## 1．整体结构

训练时，数据流可以概括成：

```text
图片 → SigLIP → Projector ─┐
                         ├→ 完整多模态序列 → 主模型第1、2层 → H₂
固定提示词 → 文本embedding ┘                              │
                                                       ├→ 冻结的主模型第3～21层
                                                       │       ↓
                                                       │   第22层的Q/K
                                                       │       ↓
                                                       │   教师attention分布 pT
                                                       │
                                                       └→ 可训练的旁路
                                                           复制第3层
                                                               ↓
                                                           复制第4层
                                                               ↓
                                                           复制第5层的Q/K
                                                               ↓
                                                           学生attention分布 pS

                                      损失：KL(pT || pS)
                                      只更新旁路，不更新主模型
```

这里的 `H₂` 是**主模型第2个 decoder block 刚结束的原始输出**，不是模型最后的输出，也不是最终 RMSNorm 后的特征。

它包含完整序列：

> 系统提示词、Base 图像 token、AnyRes 图像 token、newline，以及问题和回答前缀等文本 token。

所以，旁路不是只看图片，也不是只看 Base；它仍然能利用完整图文上下文。[教师提取代码](E:/GPTspace/llava_pruning/llava_pruning/predictor_training.py:45)

## 2．这三层旁路具体是什么结构？

它们从当前加载的主模型中，复制第3、4、5层的预训练权重。

**复制后是独立参数**：初始数值相同，但训练旁路不会修改主模型，也不会与主模型同步更新。

按你的 Qwen2-7B 配置，旁路保持：

| 项目 | 当前结构 |
|---|---|
| 隐藏特征维度 | 3584 |
| Query heads | 28 |
| Key/Value heads | 4，采用 GQA |
| 每个 attention head 维度 | 128 |
| FFN 中间维度 | 18944 |
| 位置编码与归一化 | 原生 RoPE、RMSNorm |

没有先降维到512，也没有增加一个小模型的输入投影。

但“三层”在执行上有一个区别：

| 旁路层 | 实际执行内容 |
|---|---|
| 第一层：复制主模型第3层 | 完整 Transformer block，包含 attention、残差、FFN |
| 第二层：复制主模型第4层 | 完整 Transformer block |
| 第三层：复制主模型第5层 | 只执行输入 RMSNorm、Q/K、RoPE和 attention 评分 |

**为什么第三层不需要完整执行？**

因为我们要的是第三层的 attention 权重。它由 Q、K 决定：

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)
\]

后面的 `A×V`、输出投影、FFN 都不会反过来改变这一层已经算出的 attention。

因此，第三层这些用不到的部分保留在层副本中，但**冻结且不执行**；前两层完整训练，第三层只训练 RMSNorm、Q/K 这条有效路径。[旁路结构与初始化](E:/GPTspace/llava_pruning/vendor/llava/llava/model/language_model/attention_predictor.py:167)

它也**没有复制 LM head，不负责直接输出 A/B 答案**。

## 3．教师 attention 究竟是什么？

不是完整的“所有 token 两两之间的 attention 矩阵”，也不是第22层输出的特征。

我们提取的是：

> **第22层中，最后一个输入提示位置，对各视觉 token 的 attention。**

注意，这个位置仍然属于输入 prompt，**不是已经生成的第一个答案 token**。

设：

- 完整输入长度为 \(L\)；
- 空间视觉 token 的索引集合为 \(V\)，包含 Base 和 AnyRes，但不包含 newline；
- attention head 数为 \(H=28\)。

教师的计算步骤如下。

### ① 主模型正常运行第1～21层

得到第22层的输入特征。这个过程不剪枝，主模型全部冻结。

### ② 使用第22层自身的 RMSNorm、Q/K 和 RoPE

只提取最后一个提示位置的 Query，但计算所有输入位置的 Key。

形状大致是：

```text
最后位置的 Q： [1, 28, 1, 128]

所有位置的 K：[1, 4, L, 128]
        GQA展开后：[1, 28, L, 128]

QK分数：       [1, 28, 1, L]
```

所以我们只额外计算一行评分，而不是构造 `L×L` 的完整 attention。

### ③ 每个 head 先对全部输入位置做 Softmax，再平均 heads

\[
a_j=\frac{1}{H}\sum_{h=1}^{H}
\operatorname{softmax}_{j=1,\ldots,L}
\left(\frac{q_hk_{h,j}^{\top}}{\sqrt{d_h}}\right)
\]

### ④ 取出空间视觉位置，重新归一化

\[
p_T(i)=\frac{a_i}{\sum_{j\in V}a_j},\qquad i\in V
\]

最终 \(p_T\) 是一个长度等于空间视觉 token 数量的分布，总和为1。

**Base 和 AnyRes 在训练目标中联合归一化，不是各自归一化。**

这里的 attention 是使用原层权重**独立重算 QK 得到的评分**，不是要求 FlashAttention 返回内部矩阵；主模型原生计算模式保持不变。[独立 attention 计算](E:/GPTspace/llava_pruning/vendor/llava/llava/model/language_model/fastv_attention.py:19)

## 4．学生怎样学？损失是什么？

学生直接接收 `H₂`：

```text
H₂ → 旁路第一层 → 旁路第二层 → 第三层最后一个Query的attention
```

学生按照与教师相同的顺序，得到空间视觉分布 \(p_S\)：

> 全部 keys 上 Softmax → 平均 heads → 取视觉位置 → 视觉位置重新归一化。

但两者来源不同：

- 教师：原模型第22层的 attention。
- 学生：经过训练的旁路第三层 attention。

损失只有一个：

\[
\mathcal L_{\mathrm{KD}}
=D_{\mathrm{KL}}(p_T\Vert p_S)
=\sum_{i\in V}p_T(i)\log\frac{p_T(i)}{p_S(i)}
\]

直观上，如果教师认为某个视觉 token 很重要，而学生给它很低的权重，就会受到惩罚。

**我们是在蒸馏“视觉注意力的分配”，不是蒸馏第22层的全部能力。**

当前没有加入：

- 正常／缺陷标签损失；
- A/B 答案预测损失；
- 中间特征对齐损失；
- 单独的排序损失。

教师目标与 `H₂` 都会 `detach`，主模型也被冻结，所以梯度只进入旁路。[KL损失与训练循环](E:/GPTspace/llava_pruning/llava_pruning/predictor_training.py:170)

另外，训练时不需要真的执行 Top-k 剪枝再反传。**训练先学习连续的 attention 分布，Top-k 是推理阶段的选择操作。** 同一个训练好的旁路可以用于不同剪枝率。

## 5．训练数据与当前默认参数

训练源是你的 **VisA 完整 CSV 中的图片**，不读取 GT、mask 或正常／异常标签。

固定使用不含类别名称的提示词：

```text
Are there any defects in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly.
```

内部按类别划分约90%训练、10%验证，默认 seed42。由于这里合并使用了原 VisA 的 train/test 图片，**这些图片不能再视为独立测试集**；最终评估用独立的 MVTec。

当前训练默认值为：

| 参数 | 默认值 |
|---|---|
| Epoch | 1 |
| 优化器 | AdamW，betas=(0.9, 0.98) |
| 峰值学习率 | `5e-5` |
| Weight decay | 0 |
| Microbatch | 1张图片 |
| 梯度累计 | 128，有效 batch 为128 |
| 学习率调度 | 前3%更新步数线性 warmup，之后 cosine 衰减 |
| 梯度裁剪 | 1.0 |

主模型保持 **FP16 + FlashAttention2**；学生使用独立 FP32 参数，默认 BF16 autocast，最后评分的 QK、归一化和 KL 使用 FP32。学生前两层默认启用梯度检查点，以额外重算换取更低激活显存。[当前训练参数](E:/GPTspace/llava_pruning/train_attention_predictor.py:19)

保存的 `best.pt` 根据**验证集 KL 最低**选择，不根据 MVTec 准确率选择。

## 6．训练好后，实际怎样剪枝？

推理时不再运行完整的第22层教师来获取评分，而是在第2层结束后运行旁路。

**当前默认已更新为 `base_full`：Base全保留，Base预测概率映射到整个AnyRes后统一Top-k。**
原始整体层平均预算不变，所有删除额度由AnyRes承担，不可行的图片/目标会在运行前报错。
不使用AnyRes自身预测分数或逐crop分配；训练仍用原来的联合视觉KL。
Lat/Mem及最新命令见[Base-full说明](attention_predictor_base_full.md)。
下面保留的是显式 `selection_mode=base_crop` 的对照流程，不是当前默认：

1. 完整 `H₂` 输入旁路。
2. 取旁路对 **Base 和 AnyRes 空间 token** 的预测 attention。
3. Base与整个AnyRes仍按同一连续空间比例确定两组保留总额，Base内部按自己的分数选Top-k。
4. 用Base对应区域的平均attention，经正号Softmax、容量封顶重分配和整数凑整，把AnyRes总额
   分给各crop。**crop内部用AnyRes自身的预测attention选Top-k**，不加入空间先验。
5. 用这些索引，**从原始主模型 `H₂` 中选择特征**，继续运行主模型第3～28层。

关键区别是：

> **旁路只给“保留名单”，不会把旁路输出的特征替换进主干。**

Base决定各crop“留多少”，AnyRes自身分数决定“留哪些”。计算crop均值用的是剪Base之前的完整
预测分数；不因为某个Base token最终被剪掉就把它的评分先删掉。

不同AnyRes crop允许不同剪枝率，但总额精确。温度 `crop_temperature` 可配置，默认0.05，
小尺度attention在此温度下可能接近均匀分配，不做隐藏的min-max重标度。
前置剪枝本次未接通，只预留原始crop/坐标/来源接口。
上一版Base插值排名仍可显式使用 `selection_mode=base_interpolated`。
[新版crop分配实现](E:/GPTspace/llava_pruning/vendor/llava/llava/model/language_model/attention_predictor_crops.py)

剪枝后还会：

- 保留文本和结构 newline；
- 保留剩余 token 的原始位置编号；
- 物理压缩已有 KV cache；
- 不做 token 合并。

这里训练仍然是“Base＋AnyRes 联合 attention 蒸馏”，**Base-guided 只是后来修改的推理选取策略，没有改变训练损失**。

## 7．它最终想实现什么？

一句话总结：

> **用第2层特征，通过一个独立旁路，提前估计第22层的视觉关注分布，让主模型从第3层开始就只处理选中的视觉 token。**

它不是让三层旁路完整替代主模型第3～22层，也不是保证预测 attention 完全相同。

还有两点需要分开看：

- **拟合得好不一定等于答案准确率完全不掉**：KL衡量分布相似，最终 ACC 还取决于剪掉的 token 是否影响任务。
- **平均剪枝率不等于实际加速比**：旁路本身也有计算和显存开销。当前这个方法在第2层后剪枝，28层模型的平均70%对应单次目标约75.38%；旁路成本不计入28层 token 平均数，但会计入实际 Lat 和 Mem。
