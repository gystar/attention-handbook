# 1.2 从 Seq2Seq 中的 Attention 到 Transformer

本节从神经机器翻译中的 Seq2Seq 模型切入。需要区分两个问题：Decoder 如何读取输入序列，以及序列内部的表示如何计算。早期 Attention 首先改善前者；Transformer 将 Attention 用于输入与输出序列内部的表示计算。

## Seq2Seq 中的动态读取

Sequence-to-Sequence（Seq2Seq）描述从一个序列生成另一个序列的建模框架，输入和输出长度可以不同。Sutskever et al. 的 2014 年模型用一个 Long Short-Term Memory（LSTM）网络编码输入，再用另一个 LSTM 根据固定维度的表示生成输出。[原论文 §2](https://arxiv.org/html/1409.3215v3)

Bahdanau et al. 同年公开的工作让 Decoder 在每个生成步骤读取不同的 Encoder 状态组合。下面对比的是两类设计，并不表示两篇论文按表格顺序先后发表。

| 设计 | Decoder 可读取的输入信息 | Context 的组织 |
| --- | --- | --- |
| 固定表示的 Encoder–Decoder | 输入序列的固定维度表示 | 输入信息经固定表示传给 Decoder |
| 带 Attention 的 Encoder–Decoder | Encoder 各位置的状态 $$h_j$$ | 每个生成步骤计算自己的 $$c_i$$ |

在 Bahdanau Attention 中，第 $$i$$ 个输出步骤的 Context 为：

$$
\begin{aligned}
c_i&=\sum_{j=1}^{n_x}\alpha_{ij}h_j,\\[4pt]
\alpha_{ij}&=
\frac{\exp(e_{ij})}{\sum_{r=1}^{n_x}\exp(e_{ir})},
\\[4pt]
e_{ij}&=a(s_{i-1},h_j).
\end{aligned}
$$

$$n_x$$ 是输入长度，$$h_j$$ 是 Encoder 状态，$$s_{i-1}$$ 是前一步 Decoder 状态，$$a$$ 是学习得到的打分函数。$$\alpha_{ij}$$ 随生成步骤变化，使 Decoder 能够直接组合不同输入位置的信息。[Bahdanau et al., 2014，§3.1，式 (4)—(6)](https://arxiv.org/html/1409.0473v7)

论文中的 Alignment 指输出步骤与输入位置之间学到的软对应关系。它是 Attention 的一种解释，并非 Seq2Seq 的同义词。该模型的 Encoder 与 Decoder 仍使用 Recurrent Neural Network（RNN）；动态读取并没有去掉 Recurrent 计算本身。

## Transformer 改变了序列表示的计算方式

2017 年的 Transformer 以 Self-Attention 处理序列内部关系，并保留 Encoder–Decoder Attention 连接输入与输出。其序列表示计算不再依赖沿 Token 位置逐步推进的 RNN。

![Transformer 原论文架构图：左侧 Encoder，右侧含 Masked Self-Attention 和 Encoder–Decoder Attention 的 Decoder](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png)

图 1-3　Transformer 架构，引用自 Vaswani et al.，Figure 1。图中三处 Attention 分别承担输入序列建模、输出序列建模和跨序列信息读取；FFN、残差连接与 Layer Normalization 共同构成层内计算。[原图、§3.1 与 §3.2.3](https://arxiv.org/html/1706.03762v7)

| 原始 Transformer 中的位置 | Query 来自哪里 | Key、Value 来自哪里 | 可访问范围 |
| --- | --- | --- | --- |
| Encoder Self-Attention | Encoder 当前层输入 | 同一序列的当前层输入 | 全部有效输入位置 |
| Decoder Masked Self-Attention | Decoder 当前层输入 | 同一输出序列的当前层输入 | 当前位置及此前位置 |
| Encoder–Decoder Attention | Decoder 表示 | Encoder 最终输出 | 全部有效输入位置 |

“Self”与“Cross”描述 Query 和 KV 的来源关系；“Causal”描述访问限制。这两个分类维度不同，不能把 Self-Attention 一律理解为双向读取。

原论文图中的 Decoder 输入经过右移，所以当前位置的表示用于预测下一个目标 Token。允许访问当前位置，不等于访问尚未给定的目标答案。

## 从原始架构到语言模型分析

本书后续以 Decoder-only 自回归模型为主要场景：通过 Causal Self-Attention 读取前缀，再预测下一 Token。原始 Transformer 的完整 Encoder–Decoder 图有助于理解组件关系，但不能原样当作所有 LLM 的架构图。

这种场景下，训练和生成的执行方式不同。训练时目标序列已知，多个位置可在 Causal Mask 下并行计算；常规生成时，新 Token 要由前一步输出确定，因此需要逐步推进。[Shazeer, 2019，§2.4](https://arxiv.org/html/1911.02150v1)

这种数据依赖差异决定了下一节中 Prefill 与 Decode 的计算形态。

延伸图解：[Jay Alammar，The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)。重点对应本节图 1-3 中的 Encoder、Decoder 和 Multi-Head 结构。

下一节：[1.3 LLM 中的计算瓶颈、KV Cache 与 Long Context](bottlenecks-and-context.md)。
