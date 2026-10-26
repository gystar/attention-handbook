# 6.5 数值误差与性能评估

FlashAttention 的分块公式在实数运算下等价于指定 Mask 的 Softmax Attention。实际 kernel 使用有限精度数值，运算顺序、累加精度和指数函数实现都会影响结果。评估需要分别回答：输出及梯度误差是否符合要求，以及给定负载的执行时间降低了多少。

## 数值检查覆盖输出与梯度

可将相同输入、Mask 与缩放系数下的高精度实现作为参照，分别比较 $$O$$、$$\bar Q$$、$$\bar K$$ 与 $$\bar V$$。FlashAttention-3 的误差实验采用 FP64 参照，并构造带 Outlier 的输入检查低精度行为。[论文 §4.3](https://arxiv.org/html/2407.08608v2)

对待测张量 $$X$$ 与参照 $$X_{\mathrm{ref}}$$，以下指标分别描述最坏偏差和整体偏差：

$$
E_{\max}=\max_i|X_i-X_{\mathrm{ref},i}|,\qquad
E_{\mathrm{RMS}}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(X_i-X_{\mathrm{ref},i})^2}.
$$

其中 $$n$$ 为元素数量。逐元素容差也可以采用 $$|X_i-X_{\mathrm{ref},i}|\le\mathrm{atol}+\mathrm{rtol}|X_{\mathrm{ref},i}|$$；接近零的参照值主要由绝对容差约束。容差应随数据类型和使用场景记录，不能从一次随机输入的结果直接推广到全部输入。

| 检查对象 | 需要覆盖的情况 |
| --- | --- |
| 分块边界 | 长度小于块大小、不能整除块大小、不同遍历顺序 |
| 分数范围 | 相近分数、极大差值、Outlier、整体平移 |
| Mask | Causal、非等长 Query/Key、首尾有效位置、整块或整行被屏蔽 |
| 梯度 | 分别核对 Q/K/V；小规模输入增加有限差分检查 |
| 随机性 | Dropout 的前后向 Mask 一致；重复执行检查确定性 |

<p align="center">表 6-6　Attention 数值检查范围</p>

非等长 Causal Attention 还需明确 Query 的实际位置：按左上角对齐的三角 Mask，与将 Query 视为序列末尾位置的 Mask，允许访问的 Key 不同。参考实现与待测实现必须采用相同位置定义。

本章 CPU 检查覆盖分块前向、树形合并及小规模反向有限差分；输入使用 Python float64。它验证正文代数和教学实现，尚未覆盖 GPU kernel、FP16/BF16/FP8 误差或随机数重放。

## 性能比较固定工作负载

性能记录首先应给出实际延迟，再说明吞吐量的计算口径。以下条件共同决定测试对象：

| 类别 | 应记录的条件 |
| --- | --- |
| 输入 | Batch Size、Query/KV Head 数、$$N_q,N_k,d_h,d_v$$、数据布局 |
| 语义 | Mask 及位置对齐、Dropout、是否返回权重矩阵 |
| 数值 | 输入、输出与累加精度，量化方式 |
| 执行 | GPU 型号、软件及 kernel 版本、前向或前向加反向 |
| 测量 | 预热、编译是否计时、GPU 完成同步、重复次数及延迟分布 |

<p align="center">表 6-7　Attention 性能比较条件</p>

若用 TFLOPs/s 表示速度，应注明分子中的 FLOPs 是按数学任务计数，还是包含反向重算等实际执行操作。还需说明 Causal Mask 是否扣除了被跳过的配对，否则不同报告可能采用不同的 FLOPs 计数口径。

## Kernel 收益与端到端收益

设 Attention 原本占一次完整执行时间的比例为 $$f$$，Attention kernel 加速 $$s$$ 倍，其余部分保持不变。将原总时间归一化为 1，优化后的时间为：

$$
T_{\mathrm{new}}=(1-f)+\frac{f}{s},\qquad
\mathrm{Speedup}=\frac{1}{(1-f)+f/s}.
$$

这是按各部分耗时相加得到的估算。例如 $$f=0.4$$、$$s=2$$ 时，整体加速为 $$1.25$$ 倍。若优化同时改变 Batch Size、通信重叠或其他算子的执行情况，需要重新测量完整流程，而不是继续沿用这个固定占比模型。

Training、Prefill 与 Decode 也应分开测试：前两者通常同时处理多个 Query，逐 Token Decode 的 Query 长度常为 1，对历史 KV 的访问及服务调度会产生不同影响。相关计算差异见[2.5 Training、Prefill 与 Decode 的计算差异](../transformer-attention/training-prefill-decode.md)，缓存组织在第八章展开。
