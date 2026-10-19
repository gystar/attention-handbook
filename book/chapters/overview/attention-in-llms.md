# 1.2 Attention 机制的基本计算

上一节说明了 Attention 机制用于结合上下文更新表示。本节给出标准 Transformer 中 QKV、Mask 与加权求和的计算定义；每个 Token 的输入表示是一行向量。

## 从向量表示到 QKV

标准 Transformer 的 Attention 层将输入表示分别映射为 Query、Key 和 Value 三类向量。Query 与 Key 用于计算匹配分数，Value 提供需要汇总的内容。在 Self-Attention 中，三者来自同一序列的表示，但通常使用不同的投影参数。

一个 Attention Head 是一组这样的匹配与汇总计算；Multi-Head Attention 并行执行多组计算，再组合输出。以下只考虑一个 Head，省略一次同时处理多条序列的 Batch 维度和网络的 Layer 维度。

| 对象 | 形状 | 在计算中的作用 |
| --- | --- | --- |
| Query，$$Q$$ | $$n_q \times d_k$$ | 参与匹配的当前位置表示 |
| Key，$$K$$ | $$n_k \times d_k$$ | 与 Query 匹配，决定读取权重 |
| Value，$$V$$ | $$n_k \times d_v$$ | 被加权汇总的内容 |
| Attention 输出，$$O$$ | $$n_q \times d_v$$ | 每个 Query 对应的汇总结果 |

这里，$$n_q$$ 和 $$n_k$$ 分别是 Query 与 Key 的数量，$$d_k$$ 是匹配向量维度，$$d_v$$ 是 Value 维度。Key 和 Value（合称 KV）按位置配对，所以行数相同；Query 与 Key 的数量可以不同。

Query 与 Key 的点积经缩放得到匹配分数；Softmax 将每行分数转换为和为 1 的非负权重，再对 Value 加权求和。相应的 Scaled Dot-Product Attention 为：

$$
O=\operatorname{softmax}_{\mathrm{row}}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

$$M\in\mathbb{R}^{n_q\times n_k}$$ 是加性 Mask：允许的位置取 $$0$$，禁止的位置在数学表达中取 $$-\infty$$。Softmax 沿每一行的 Key 维度计算；每个 Query 至少需要一个允许访问的 Key。[Vaswani et al., 2017，§3.2](https://arxiv.org/html/1706.03762v7)；[Shazeer, 2019，§2.3](https://arxiv.org/html/1911.02150v1)

![原论文图 2 左图：Q 与 K 经 MatMul、Scale、可选 Mask 和 Softmax 后，再与 V 相乘](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png)

图 1-2　Scaled Dot-Product Attention，引用自 Vaswani et al.，Figure 2 左图。计算沿箭头依次完成权重计算与 Value 汇总。原论文首页给出了署名后用于学术作品的图表转载许可。[原图及出处](https://arxiv.org/html/1706.03762v7)

Q、K、V 是当前输入经过投影得到的激活值。训练学到的是投影等参数；推理时参数通常保持不变，Attention 权重仍会随输入改变。不能把 Attention 权重与模型参数混为一谈。[Shazeer, 2019，§2.1—2.3](https://arxiv.org/html/1911.02150v1)

在自回归 Self-Attention 中，Causal Mask 只允许位置访问自身及此前位置；当前位置的表示用于预测下一 Token，因此不会读取待预测的答案。

## 一个最小算例

假设某个 Query 的归一化权重为 $$(0.1,0.2,0.7)$$，三个 Value 分别为 $$(1,0)$$、$$(0,1)$$、$$(1,1)$$，则输出为：

$$
o=0.1(1,0)+0.2(0,1)+0.7(1,1)=(0.8,0.9).
$$

这是人为构造的计算示例。它说明 Attention 输出是向量的加权组合；这些数值不代表某个模型在真实文本上的 Attention Pattern。

## 本书的范围

本书以用于语言建模的 Transformer Attention 为基础，后续成本分析主要采用 Decoder-only 模型的常规自回归生成场景。

| 纳入范围 | 讨论重点 |
| --- | --- |
| Transformer Attention | QKV、Multi-Head、Mask，以及 Self-Attention 与 Cross-Attention |
| Position Encoding 与结构改进 | 位置如何参与计算，Head 与 KV 表示如何组织，Gate 的作用位置 |
| 计算与推理优化 | 运算量、显存访问、KV Cache 和服务系统中的资源管理 |
| Linear Attention 与 Hybrid Architectures | 历史信息如何进入状态，以及不同序列模块如何组合 |

视觉、语音和多模态中的 Attention 机制在需要比较机制时引入。State Space Models（SSMs）作为相关序列建模路线和混合模块讨论；具体联系需要说明模型形式与假设。[Dao & Gu, 2024，§4.1、§5.1—5.2](https://arxiv.org/html/2405.21060v1)

## 配合视频理解 QKV

3Blue1Brown 的 [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=392s) 可配合本节观看：

| 时间段 | 观察重点 |
| --- | --- |
| 06:32—10:40 | Query、Key、点积与归一化权重 |
| 11:30—12:45 | Causal Mask 对可访问位置的限制 |
| 13:40—15:10 | Value 的加权组合 |
| 20:30—22:30 | 多个 Head 如何并行工作 |

视频的示意图使用了自己的向量排列方式；本书统一按 Token 占据矩阵行来书写，Softmax 沿 Key 维度归一化。视频及[作者配套图文](https://www.3blue1brown.com/lessons/attention/)用于理解计算过程，精确定义以上述公式为准。时间段依据[作者字幕](https://github.com/3b1b/captions/blob/main/2024/attention/english/captions.srt)核对。

下一节：[1.3 从 Seq2Seq 中的 Attention 到 Transformer](from-seq2seq-to-transformer.md)。
