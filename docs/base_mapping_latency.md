# Base映射的额外延迟与可选快路径

用户提供的27组相同剪枝率、λ、旁路输入及末尾删除层数的配对结果中，Base映射比AnyRes自身评分
平均多42.88ms，范围28.71～59.23ms，相对差距4.77%～12.36%。CSV没有GPU型号/负载，单凭这些
配对值不能确定每部分的耗时或排除不同卡、同时运行任务带来的影响。

## 额外工作在哪里

`input=full`时，两种方式都执行完整H2上的同一旁路：前两层Transformer和第三层最后Query的
Q/K评分。`base_full`仅提取Base视觉位置的预测分数，并没有省掉旁路对全部上下文的处理。

MyPool先合并了token，所以Base映射还必须处理：

1. 读取每个合并token的原始来源及实际系数；
2. 将Base概率按归一化合并系数回投到27×27原始网格；
3. 根据真实resize/padding几何，双线性采样每个AnyRes crop；
4. 对每个剩余AnyRes token的全部来源求加权均值，然后统一排序。

原路径每张图构造大量Python字典、来源列表和权重，再转成GPU张量；插值按crop逐个调用。
本地合成9-crop输入的CPU剖析中，来源记录及权重组装占据这段路径的大部分时间。这支持优先
优化这些循环，但不等于服务器真实Lat的逐段测量。

`input=base_text;tail=0`与`input=full;tail=2/3/4`改了两个条件，不能直接推断缩短旁路是否更快。
比较旁路输入需保持tail相同；比较评分方式也需保持GPU型号、负载及其它配置相同。

## 可选实现

新增`--base-mapping fast`，仅在`--pruning-audit lean`的MyPool＋Base映射路径启用：

- 保留动态来源的必要GPU→CPU传输，用NumPy数组替代逐token字典和Python循环；
- 按原顺序生成完全相同的owner/source/FP32归一化系数，压紧后统一传回GPU；
- 按原公式预备、缓存固定几何坐标，在一次`grid_sample`中映射全部crop；
- 保留全部来源、FP32评分、Base回投、真实padding、合并权重及稳定Top-k，不改模型精度。

动态来源处理、传输和实际映射仍计入Lat，没有扣除耗时。固定几何准备与现有lean输入准备一致。
快路径不是只使用anchor的近似，也没有更改训练checkpoint或剪枝预算。

默认仍为`reference`，因为本机没有可运行该7B实验的CUDA环境，不能宣称GPU已提速或真实数据
ACC一定不变。CPU测试覆盖来源索引/权重和映射分数逐元素一致，以及微型原生生成/选中token一致。
CUDA浮点归约可能受操作顺序影响，因此先做下面的设备核对，再比较真实数据预测和完整Lat。

## 先测映射热点，不加载7B模型

```bash
python benchmark_base_mapping.py \
  --gpu 5 \
  --output-dir output/base_mapping_benchmark_01
```

默认测试30/50/70、λ0.7/0.8/0.9，9个case，3×3crop与27×27token网格。
使用真实MyPool压缩函数生成合成来源关系；不读取真实图片、不加载checkpoint。
逐次同步CUDA，交替测原路径与快路径，包含动态来源组装和传输。固定几何在计时前准备。
输出`benchmark.csv`、`benchmark.json`；分数检查或最终保留token检查失败会报错。

这里的`mapping_speedup`只表示映射热点加速，不能当作完整模型Lat加速。合成验证也不能替代
真实数据的逐条预测对齐。如果设备上的加速不明显，继续用默认reference路径。

## 真实模型复测

设备核对通过后，在原实验命令中加入：

```text
--base-mapping fast
```

并使用新的输出目录。`run_pool_distill.py`和`run_pool_distill_ablations.py`均支持此参数。
后者仍执行原63组实验，只改变Base映射实现；AnyRes自身评分路径不变。
精简CSV输出11列，新增百分数BACC、F1及相对原版91.18% ACC的百分比；
Base快路径配置末尾增加`mapping=fast`。

配置、metrics和断点记录包含实现选项；旧输出缺少此字段时视为reference，禁止将旧计时结果
混入fast续跑。原参考结果和统计口径不作修改。
