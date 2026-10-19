# 第 2 章 Transformer Attention 的基本原理与计算过程

> 状态：已迁入部分基础内容，完整章节待写。

2.1 Transformer 中的 Attention 结构与计算流程

2.2 Query、Key、Value 与 Scaled Dot-Product Attention

2.3 Self-Attention、Cross-Attention 与 Causal Attention

2.4 Multi-Head Attention 的结构与 Output Projection

2.5 Training、Prefill 与 Decode 的计算差异


## 从向量表示到 QKV

以下按每个 Token 的表示占据矩阵一行来书写。

标准 Transformer 的 Attention 层将输入表示分别映射为 Query、Key 和 Value 三类向量。Query 与 Key 用于计算匹配分数，Value 提供需要汇总的内容。在 Self-Attention 中，三者来自同一序列的表示，但通常使用不同的投影参数。在 Encoder–Decoder Cross-Attention 中，Query 来自 Decoder，Key 和 Value 来自 Encoder 的最终输出。

一个 Attention Head 是一组这样的匹配与汇总计算；Multi-Head Attention 并行执行多组计算，再组合输出。以下只考虑一个 Head，省略一次同时处理多条序列的 Batch 维度和网络的 Layer 维度。

| 对象 | 形状 | 在计算中的作用 |
| --- | --- | --- |
| Query，$$Q$$ | $$n_q \times d_k$$ | 参与匹配的当前位置表示 |
| Key，$$K$$ | $$n_k \times d_k$$ | 与 Query 匹配，决定读取权重 |
| Value，$$V$$ | $$n_k \times d_v$$ | 被加权汇总的内容 |
| Attention 输出，$$O$$ | $$n_q \times d_v$$ | 每个 Query 对应的汇总结果 |

<p align="center">表 2-1　QKV 与输出的形状</p>

这里，$$n_q$$ 和 $$n_k$$ 分别是 Query 与 Key 的数量，$$d_k$$ 是匹配向量维度，$$d_v$$ 是 Value 维度。Key 和 Value（合称 KV）按位置配对，所以行数相同；Query 与 Key 的数量可以不同。

Query 与 Key 的点积经缩放得到匹配分数；Softmax 将每行分数转换为和为 1 的非负权重，再对 Value 加权求和。相应的 Scaled Dot-Product Attention 为：

$$
O=\operatorname{softmax}_{\mathrm{row}}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

$$M\in\mathbb{R}^{n_q\times n_k}$$ 是加性 Mask：允许的位置取 $$0$$，禁止的位置在数学表达中取 $$-\infty$$。Softmax 沿每一行的 Key 维度计算；每个 Query 至少需要一个允许访问的 Key。[Vaswani et al., 2017，§3.2](https://arxiv.org/html/1706.03762v7)；[Shazeer, 2019，§2.3](https://arxiv.org/html/1911.02150v1)

<figure align="center">
<img src="https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png" alt="原论文图 2 左图：Q 与 K 经 MatMul、Scale、可选 Mask 和 Softmax 后，再与 V 相乘">
<figcaption><p align="center">图 2-1　Scaled Dot-Product Attention</p></figcaption>
</figure>

计算沿箭头依次完成分数计算、缩放、可选 Mask、Softmax 和 Value 汇总。

Q、K、V 是当前输入经过投影得到的激活值。训练学到的是投影等参数；推理时参数通常保持不变，Attention 权重仍会随输入改变。[Shazeer, 2019，§2.1—2.3](https://arxiv.org/html/1911.02150v1)

在自回归 Self-Attention 中，Causal Mask 只允许位置访问自身及此前位置；当前位置的表示用于 Next-token Prediction，待预测位置被 Mask 屏蔽。

## 一个最小算例

假设某个 Query 的归一化权重为 $$(0.1,0.2,0.7)$$，三个 Value 分别为 $$(1,0)$$、$$(0,1)$$、$$(1,1)$$，则输出为：

$$
o=0.1(1,0)+0.2(0,1)+0.7(1,1)=(0.8,0.9).
$$

这是用于说明向量加权组合的教学算例，权重与 Value 均为人为设定。

## 配合视频理解 QKV

3Blue1Brown 的 [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=392s) 可配合本节观看：

| 时间段 | 观察重点 |
| --- | --- |
| 06:32—10:40 | Query、Key、点积与归一化权重 |
| 11:30—12:45 | Causal Mask 对可访问位置的限制 |
| 13:40—15:10 | Value 的加权组合 |
| 20:30—22:30 | 多个 Head 如何并行工作 |

<p align="center">表 2-2　QKV 视频阅读指引</p>

视频的示意图使用了自己的向量排列方式；本书统一按 Token 占据矩阵行来书写，Softmax 沿 Key 维度归一化。视频及[作者配套图文](https://www.3blue1brown.com/lessons/attention/)用于理解计算过程，精确定义以上述公式为准。时间段依据[作者字幕](https://github.com/3b1b/captions/blob/main/2024/attention/english/captions.srt)核对。

图示来源与许可见[图表来源](../../appendices/references.md#图表来源)。

## Dense Attention 的计算与存储成本

令 $$B$$ 为 Batch Size，$$N$$ 为序列长度，$$h_q$$ 为 Query Head 数，$$d_h$$ 为每个 Head 的维度。这里假设 Key 与 Value 的维度均为 $$d_h$$，仅统计 Attention 核心运算；线性投影、FFN、反向传播和系统调度另行计算。

| 计算对象 | 主要算术复杂度 | 原因 |
| --- | --- | --- |
| 长度 $$N$$ 的 Dense Self-Attention 前向计算 | $$\Theta(Bh_qN^2d_h)$$ | Query–Key 匹配和 Value 汇总都涉及位置对 |
| 当前可访问 KV 共 $$N$$ 个位置时的一步 Decode | $$\Theta(Bh_qNd_h)$$ | 每个 Head 只有一个当前 Query，访问范围包含当前 Token 的 KV |
| 显式保存的 Attention 分数或权重矩阵 | $$\Theta(Bh_qN^2)$$ 个元素 | 每个 Head 有一个位置对矩阵 |

<p align="center">表 2-3　Dense Attention 的计算成本</p>

Causal Mask 将有效位置对限制为三角区域，但仍有 $$N(N+1)/2$$ 个位置对，渐近阶仍为二次。上表最后一行描述显式保存矩阵的实现；FlashAttention 可避免将完整矩阵写入显存，因此二次运算量不意味着必须占用二次规模的中间显存。[Vaswani et al., 2017，Table 1](https://arxiv.org/html/1706.03762v7)；[Dao et al., 2022，§3](https://arxiv.org/html/2205.14135v2)

单步 Decode 的成本随历史长度增长。生成整个回答的成本需要累加各步骤的计算量，并考虑历史长度随生成过程增长。

小 Batch 的 Decode 可能受到读取模型参数与 KV 的带宽限制；瓶颈随 Batch、上下文长度、硬件与 Kernel 实现变化。[Shazeer, 2019，§2.4.1](https://arxiv.org/html/1911.02150v1)
