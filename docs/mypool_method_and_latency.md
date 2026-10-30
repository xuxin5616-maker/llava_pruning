# MyPool 前置压缩：方法、执行位置与精简 Lat 路径

本文对应当前 `run_pool_distill.py` 混合入口。它在 LLM 之前使用 MyPool，在完成
LLM 第2层后使用蒸馏评分再剪一次。前置部分不训练，也不使用问题答案、GT 或 mask。
独立的 `runMyPruningPoolSpatial.py` 默认仍保留原诊断行为，作为参考实现。

## 1．执行位置

```text
原图 → Base + M个AnyRes视图 → 完整SigLIP
                                  │
                         池化余弦评分 / 原生视觉attention / Key
                                  │
                         分配crop名额 → 选取与合并
                                  │
                         Projector → 稀疏空间打包
                                  │
                         LLM第1、2层 → 蒸馏后置剪枝 → 后续LLM
```

MyPool 实际压缩位置是 **完整 SigLIP 编码之后、Projector 之前**。
它不减少 SigLIP 内部各层处理的 token 数；Projector 和 LLM 接收到更短的序列。
冗余特征/attention 可以从指定 SigLIP 层捕获，最终合并特征和合并用 Key 固定来自
当前第26个活动 block。默认评分层也是26；这与 LLM 第2/22层不是同一套层号。

当前384输入、patch14的每个视图有27×27=729个空间 token。输入/输出形状：

| 环节 | 形状（省略单图batch） |
|---|---|
| Base + AnyRes像素 | `[1+M,3,384,384]` |
| SigLIP特征 | `[1+M,729,d_v]` |
| 每视图压缩后 | 第i个视图 `[K_i,d_v]` |
| 合并所有视图后 | `[sum(K_i),d_v]` |
| Projector后 | `[sum(K_i),d]` |
| 加入行newline后 | `[sum(K_i)+P,d]` |

M由原图尺寸及checkpoint候选分辨率决定，不保证每张图都是3×3。

## 2．crop预算评分：两个自身池化参考的余弦一致性

记Base token为x_B,j，第i个AnyRes视图的token为x_i,j。
仅在FP32评分副本上做L2归一化：`z=x/||x||₂`。真实待合并/投影的特征不被替换。

把Base的27×27网格按原始AnyRes网格分成M个区域B_i。每个方向用floor划分，最后一行/列
接收余数；这是原脚本的预算分区规则，不是后置Base映射时的精确像素插值。

对Base区域与AnyRes视图各求一个自身平均方向：

\[
u_{B,i}=\operatorname{norm}\left(\frac1{|B_i|}\sum_{j\in B_i}z_{B,j}\right),\qquad
u_{A,i}=\operatorname{norm}\left(\frac1{729}\sum_{j=1}^{729}z_{i,j}\right).
\]

默认`mean`模式的冗余评分：

\[
R_i=\frac1{|B_i|}\sum_{j\in B_i}\cos(z_{B,j},u_{B,i})
   +\frac1{729}\sum_{j=1}^{729}\cos(z_{i,j},u_{A,i}).
\]

两个分量分别表示Base对应区域与AnyRes视图内部的平均方向一致性。
R越高，当前规则把这个crop视为越冗余，分配更少名额；它不是缺陷概率。
这里没有把每个AnyRes token直接与Base区域的池化向量做交叉余弦。
选择多个冗余层时，每层独立评分，再平均标量R_i；不先平均不同层的特征。

实现：`runMyPruningPoolSpatial.py::pooled_crop_redundancy`。

## 3．从冗余到整数crop名额

默认温度T=.05，冗余分配权重为：

\[
p_i^{R}=\operatorname{Softmax}_i(-R_i/T).
\]

可选空间先验使用另一参数η=`--spatial-mix-weight`：

\[
p_i=(1-\eta)p_i^{R}+\eta P_i/\sum_jP_j.
\]

η默认0。混合后不再做第二次Softmax。η与划分前后两阶段剪枝贡献的λ完全不同。
空间先验全零时沿用原冗余分配路径。

设AnyRes总名额为K_A，每个crop至少4个、最多729个。先保留每crop下限，再按p_i分配额外
名额；达到上限的crop封顶，把多余名额重分配给其他crop，最后用最大余数法取整。
整数名额满足`sum(K_i)=K_A`，不强制每个crop使用相同剪枝率。

等价的连续形式是`K_i=4+min(c*p_i,729-4)`，由c满足总预算，再做精确整数分配。
Base只有一个独立总名额，不参与这些crop之间的竞争。冗余R_i改变的是crop“留多少”，
不是crop内部的token“留哪些”。

实现：`allocate_spatial_probability_budgets`、`allocate_crop_budgets`及其容量重分配函数。
精简路径保留原来的分数传输、Softmax、封顶和取整运算，只不组装无用诊断记录。

## 4．crop内部选取：SigLIP收到的attention

每个视图使用自身SigLIP attention。对头和query位置依次平均：

\[
a_{i,j}=\frac1{HN}\sum_{h=1}^{H}\sum_{q=1}^{N}A_{i,h,q,j},\quad N=729.
\]

它表示token j收到的平均视觉注意力，不是最后一个文本token的LLM attention。
选择多个attention层时，再平均各层评分。评分直接来自当前SigLIP原生前向，不再次运行编码器。

视图名额K_i分为dominant与contextual：

\[
D_i=\operatorname{round}(K_i\cdot54/64),\qquad C_i=K_i-D_i.
\]

满保留K_i=729时特殊设置D_i=729、C_i=0，不引入合并近似。
按a_i选Top-D_i作为dominant，随后恢复其原空间顺序。

## 5．contextual不是丢弃：其余特征被合并

在非dominant的原空间序列中，使用均匀步长选C_i个锚点。
合并相似度使用第26层原始K投影先平均heads、再L2归一化得到的Key方向。
每个非锚点分配给余弦相似度最大的contextual锚点，合并仅在同一视图内部发生。

设S_c是分到锚点c的OTHER token集合，则输出：

\[
y_c=x_c+\frac{\sum_{j\in S_c}x_j}{\max(|S_c|,1)}.
\]

这是“锚点 + 其他成员均值”，不是所有成员连同锚点一起求平均。没有成员时输出锚点本身。
dominant保留原特征，contextual携带聚合信息。因此视觉token减少不等同于相同比例的信息被直接删除。
归一化Key只用于分组，特征合并沿用原模型精度。

实现：`runMyPruningPoolSpatial.py::compress_tokens`；精简路径复用它的数值运算。

## 6．投影与空间打包

各视图的dominant/contextual特征先拼接，再经过原Projector。
Base按原锚点顺序排列；AnyRes按原始全局行列坐标排列，每一条原始空间行后插入checkpoint的
`image_newline`，空行也保留；`spatial_unpad_add_newl`额外保留最后一个标记。
非零稀疏路径不对已经打出空洞的特征再做unpad/max9插值。

进入LLM时，位置编号按MyPool压缩后的序列连续生成；后续第2层蒸馏剪枝继续保留这些编号。
模型仍处理完整图文上下文，前置评分本身不依赖问题文本。

实现：原路径`pack_anchor_rows`；精简路径`pool_lean.pack_device_rows`。
两者使用相同锚点排序、原始newline向量和dtype。精简路径将打包索引保持在GPU。

## 7．混合入口的预算与独立MyPool CLI不同

独立MyPool的rate作用于原始encoder内容数量。混合入口按原版native packed长度N0作为
统一参考，其中包含原版unpad/max9后的Base/AnyRes及newline。

目标平均剪枝比例r、蒸馏贡献占比λ、深度D=28、边界k=2时：

\[
N_1=\lfloor N_0[1-(1-\lambda)r]\rfloor,\qquad
N_2=\left\lfloor\frac{DN_0(1-r)-kN_1}{D-k}\right\rfloor.
\]

前置阶段需要输出N1个图像序列位置。先预留原始稀疏行newline数P，再把N1-P个空间名额按
原始容量Base729和整个AnyRes M×729成比例分成两组，并做精确Hamilton整数取整。
其中AnyRes总名额才进入上述逐crop分配。后置阶段保护前置剩余Base和newline，继续压缩AnyRes。
不可行的组合报错，不改变分母或偷偷降低目标。

实际层平均剪枝比例为`1-[2*N1+26*N2]/(28*N0)`。

## 8．哪些辅助工作移出了Lat

默认新增`--pruning-audit lean`，Lat仍从已准备好的驻留输入进入`model.generate`到首token结束。
没有从测量值里减去某一段时间。

| 精简项 | 实施方式 |
|---|---|
| 尺寸、网格、Base分区、固定空间索引、行newline位置、整数总预算 | 通过`method.prepare_inputs`在generate前准备并缓存 |
| 空间先验权重为0 | 不生成/归一化未使用的先验模板 |
| 热图、完整importance列表、仅供CSV/绘图的辅助数组与字典 | 不生成，不在关键路径搬到CPU |
| 单个冗余层 | 直接用该层评分，省去单元素stack/mean |
| 重复的输入全量检查 | 压缩输入按全batch检查一次；L2合法性检查合并；保留关键非有限/预算检查 |
| 打包索引CPU列表→GPU tensor往返 | 使用原GPU索引直接打包 |
| Base映射必需的来源 | 只传必要整数索引/分组，一次建立分组表，保持原来源顺序和系数 |
| AnyRes自身评分模式 | 不建立未使用的完整来源列表和权重 |

**仍计入Lat**：SigLIP正常前向、图像内容相关冗余/attention/Key评分、实际CPU预算分配及必要
传输、Top-k、Key聚类与特征合并、Projector、空间打包、Base映射所需来源处理、LLM和蒸馏旁路。
缓存常驻GPU张量仍计入实际显存。不能据此把所有CPU操作/数据搬运视为计时外工作。

## 9．入口、对照与旧结果

```bash
# 在原来的完整命令上加此选项，并使用新目录：
python run_pool_distill.py --gpu 5 --pruning-audit lean \
  --output-dir output/pool_distill_lean_01

# 保留旧诊断准备路径作对照：
python run_pool_distill.py --gpu 5 --pruning-audit full \
  --output-dir output/pool_distill_full_01
```

模型/数据/checkpoint、rate、λ及GPU池仍使用调用者传入的参数或当前工程默认值，本次不修改它们。
新鲜输出的CSV和JSON包含`pruning_auxiliary_profile=lean_aux_v1/full_audit_v1`。
Lat/Mem及ACC/PRE/Recall/TNR/BACC/F1/相对ACC的公式不变，不保存可视化。

旧输出缺少此配置时，短`--resume`沿用full路径；显式把旧full结果改成lean续跑会被拒绝，避免
同一组Lat混合两个实现路径。要测精简版性能，应使用新输出目录。旧的prompt-format校验、
多GPU调度、断点、预算可行性和checkpoint校验保持有效。

CPU对照覆盖相同预算、压缩特征逐元素相同、来源记录相同、选择索引和微型原生模型生成一致。
未运行服务器7B模型，不能把这些测试写成实际加速比或准确率不变的实测结论。
