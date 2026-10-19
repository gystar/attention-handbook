# 1.2 从 Seq2Seq 中的 Attention 到 Transformer

本节从神经机器翻译中的 Seq2Seq 模型切入。编码器（Encoder）计算输入序列的表示，解码器（Decoder）利用这些表示生成输出序列；需要区分的问题是 Decoder 如何读取输入，以及序列内部的表示如何计算。早期 Attention 首先改善前者；Transformer 将 Attention 用于输入与输出序列内部的表示计算。

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

在 Transformer 的每一层中，Attention 在位置之间汇总信息，前馈网络（Feed-Forward Network，FFN）对各位置的表示分别进行变换；残差连接与归一化也参与层内计算。下图展示 Encoder 与 Decoder 的基本组成。[Vaswani et al., 2017，§3.1—3.3](https://arxiv.org/html/1706.03762v7)

![左侧 Encoder 包含 Self-Attention 和 Feed Forward；右侧 Decoder 在 Self-Attention 与 Feed Forward 之间加入读取 Encoder 输出的 Encoder-Decoder Attention](../../assets/overview/alammar-encoder-decoder.png)

图 1-4　Encoder 与 Decoder 中的 Attention。箭头由下向上表示层内计算，横向箭头表示 Decoder 读取 Encoder 的输出。图为单层结构的简化示意，省略了残差连接、归一化等组件；右侧 Self-Attention 的因果限制未在原图中标出。引自 Jay Alammar，*The Illustrated Transformer*，A High-Level Look，原图未修改，CC BY-NC-SA 4.0。[原文](https://jalammar.github.io/illustrated-transformer/#a-high-level-look)；[许可](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

| Attention 所在位置 | 读取的信息 | 作用 |
| --- | --- | --- |
| Encoder Self-Attention | 输入序列的全部有效位置 | 结合前后文计算输入表示 |
| Decoder Self-Attention | 输出序列中当前位置及此前位置 | 根据已有输出计算当前位置的表示 |
| Encoder–Decoder Attention（Cross-Attention） | Encoder 最终输出 | 将输入信息用于生成 |

以翻译为例，Decoder 的 Self-Attention 处理已经写出的译文，Cross-Attention 则读取原文的表示。“Self”与“Cross”区分同一序列内部读取与跨序列读取；“Causal”表示只能访问当前位置及此前位置的因果限制。因此，Self-Attention 不一定是双向的。[Vaswani et al., 2017，§3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS3)

Decoder 当前位置的表示用于预测下一个 Token，所以允许访问当前位置并不意味着读取待预测的答案。原论文的完整结构及右移后的输入见 [Figure 1](https://arxiv.org/html/1706.03762v7#S3.F1)。

## 从 Encoder–Decoder 到 Decoder-only 语言模型

Transformer 的常见结构包括同时使用 Encoder 与 Decoder 的 Encoder–Decoder、侧重输入表示的 Encoder-only，以及无需独立 Encoder、根据已有文本继续生成的 Decoder-only。这些结构采用不同的信息读取方式，以适应输入分析、条件生成等任务。[Hugging Face，Transformer Architectures](https://huggingface.co/learn/llm-course/chapter1/6)

本书后续重点讨论采用 Decoder-only 架构的生成式大语言模型（Large Language Model，LLM）。它将提示文本和已生成内容放在同一序列中，通过 Causal Self-Attention 汇总上下文，无需图中读取独立 Encoder 的 Cross-Attention；输出层再根据最终表示计算下一 Token 的概率。每次生成一个 Token 后，将其接入序列并继续预测，这就是自回归生成。[Jay Alammar，The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/#the-decoder-only-block)

训练时，目标序列已知，多个位置可以在 Causal Mask 下并行计算；常规生成时，新 Token 依赖前一步输出，因此需要逐步推进。这一区别将影响 1.4 节讨论的计算与缓存成本。[Shazeer, 2019，§2.4](https://arxiv.org/html/1911.02150v1)

下一节：[1.3 Transformer 中的 Attention 计算](attention-in-llms.md)。
