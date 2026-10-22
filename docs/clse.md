# CLSE: OneVision / AnyRes 适配

## 来源与采用的设置

- [论文 v1](https://arxiv.org/abs/2606.24165v1)，§3.2 式 (7)–(13)、§4.1。
- [作者仓库](https://github.com/zjubinchen/CLSE)，固定提交
  `cfc9710abe4a1bd5480e521f3acd86ded4855ff3`。
- 原始参考及 Apache-2.0 许可保存在 `third_party/clse/`，读取方法配置时验证 LF SHA-256。
- 公共方法：`run.py --method clse --method-config configs/clse.json`。

采用论文主设置：比较经过 projector 的输入视觉 embedding（LLM block 之前）和
第一个 LLM block 输出，**第一个 block 后只剪一次**，纯频谱演化评分，不做合并或回收。
这不是 SigLIP 层间的比较，也不是第二个 block 后剪枝。

作者此提交的 LLaVA-1.5 默认代码与论文主设置不同：`K_list=[1,11,21]`、
`score_type="clse_attn"`，`RETAIN_TOKEN=192` 映射到 `[330,210,62]`。
此适配不采用该多阶段调度或 attention 门控，也不使用固定图像起点35、576 token、24×24网格。

## 评分与实际空间几何

1. 不改变原始模型的预处理、projector 或 AnyRes packing，只在真实打包过程中记录几何。
2. Base 按自身完整网格；AnyRes 按**拼接、去 padding、max9 缩放后**的实际完整网格。
   不是逐个 AnyRes tile 独立计算，也不把 Base、AnyRes、newline 硬拼成矩形。
3. 对两个网格分别做二维 FFT、Gaussian 高通、逆变换取绝对值并对通道取均值，得到每个空间
   token 的高频残差强度。高通 cutoff_ratio 默认0.1。特征**不做 L2 归一化**。
4. 把两个网格的强度向量拼在一起，使用联合空间 token 均值计算跨层相对变化，最大截断为1。
5. 沿用作者 `get_evolution_factor`：对联合变化量做均值/标准差标准化，再做
   `sigmoid(z / temperature)`，默认 temperature=0.1、epsilon=1e-6。
6. 所有 Base + AnyRes 空间 token 联合 top-k，**不分别分配保留数**；保留结果恢复原始顺序。

第5步是作者代码在论文式(11)后额外执行的单调校准（有限精度下饱和可产生并列分数）；没有另加
min-max 排名。共享统计量和预算是 OneVision 多视图适配选择，不能称作作者原生 OneVision 复现。
频谱操作用 FP32，按通道分块限制临时复数张量的显存；与作者整通道求均值可有微小累加舍入差。
主模型仍为原有 FP16 + FlashAttention2，不请求任何 native attention，不改为 eager，也不重算 Q/K。

当前非零率只支持单图片、batch=1、无序列 padding、`anyres_max_9`、
`spatial_unpad` / `spatial_unpad_add_newl`、完整未截断网格和 greedy DynamicCache 生成。
支持 `base_first` 和已有 `anyres_first`，网格每条边至少2个 token；不支持随机 ROI、视频、beam
或输入的自定义已有 cache，不会静默回退。

## 平均剪枝率口径

设原始 packed 视觉 token 数为 N（包含 newline），LLM 深度为 D，第一个 block 后保留 K。

```text
实际平均剪枝率 = 100 * (1 - (N + (D-1)*K) / (D*N))
单次目标剪枝率 = 平均目标剪枝率 * D / (D-1)
K = floor(N * (1 - 单次目标剪枝率/100))
```

对28层模型，平均50/70/90对应单次51.85185185/72.59259259/93.33333333%。
浮点乘法有1e-10的近整数保护。先算 packed 总保留预算，再减去必须保留的所有 newline，
剩余名额才分给联合空间 top-k。每个样本至少保留一个空间 token；如果预算不足则明确报错。
因此 newline 不参加 FFT 或排名，但**计入每层真实序列成本**。这不同于只在空间 token 上定义
50/70/90；不要混用。该平均数是 token 层数口径，不等于 FLOPs、Lat 或 Mem 剪掉同样比例。

`prune_rate` 和目录 `prune_50/70/90` 表示平均目标。输出另外记录：

- `target_average_prune_rate`：平均目标。
- `requested_layer_prune_rate`：按实际模型深度换算的单次目标。
- `actual_layer_prune_rate`：整数预算下的实际单次移除比例。
- `actual_average_prune_rate`：从实际每层视觉输入数计算；汇总用逐样本等权平均。
- `layer_tokens.csv`：每个样本每层实际视觉/全序列输入输出数，便于独立核查。

## 缓存、基线和计时

剪后维持原始 position IDs；所有已写入的浅层 KV 同步物理删除，后续层直接接收缩短序列。
decode 的新 token 位置从原始 prompt 长度继续，和压缩后的物理 cache 长度分开维护。
配置0%时，直接执行原生 Qwen2 forward，不计算 FFT、评分、top-k 或 cache 重排。

沿用公共 `--benchmark-metrics --warmup 10 --no-plots --no-sample` 协议：
Lat 从已驻留输入进入 generate 到首个选定 token；包含视觉编码、projector、packing、频谱评分、
剪枝、prefill。不含模型加载、CPU数据读取/预处理、H2D和保存。Mem 是完整答案生成期间 PyTorch
峰值 allocated/reserved，包含权重和输入，不是整卡计算利用率。每率预热后评估输入中的全部样本。
频谱和选择有额外开销，不能仅由平均剪枝率推断一定加速。

`--exclude-pruning-time` 仍为公共诊断选项（不能与 benchmark 同用），会减去插桩的频谱、评分、
选择和 cache 重排耗时；真实模型 block 耗时不减。同步会扰动执行，不能作为真实端到端加速。

## 可视化与验证

不测 benchmark 时加 `--save-prune-vis --save-attention-vis`，使用新输出目录。
配置默认三个非零率均可导出；可用 `--visualize-samples` 只导出指定 ID，不减少评估样本。
每样本只有 `comparison.png` 和 `attention_overlay.png` 两类 PNG，加精简的 `decisions.json`。
图上带 GT / Pred，分数图是 pure CLSE 的 JET 叠加，**不是 attention / 缺陷概率**。

CPU 测试对照固定作者 FFT 和演化函数，覆盖通道分块、矩形网格、FP16/BF16/FP32评分、实际
去 padding/max9/两种顺序的 packing、0%原生路径、非零缓存及原始位置、独立全前缀 decode
oracle、方法切换、公共 CLI 的全部样本/预热排除/Lat/Mem/预算统计和两类 PNG。
CPU 环境不支持 FP16 linear，解码采用 FP32/BF16；可选的 native FP16 + FlashAttention2 测试
需在服务器运行。本地测试不代表已验证真实权重准确率或 A100 加速：

```bash
python -m unittest discover -s tests -p 'test_clse*.py' -q
```
