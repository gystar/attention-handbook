# 1.1 Attention 机制的基本思想与模型中的作用

注意力机制（Attention Mechanism）用于根据当前输入动态汇总信息。理解它的作用，需要先区分两个问题：模型如何表示文本，以及一个位置的表示如何结合其他位置的信息。本节从上下文依赖出发，再说明这一机制在 Transformer 和语言生成中的具体计算。

## 从上下文依赖到信息汇总

以“这杯茶太烫了，我想等它凉一些再喝”为例，“它”的含义需要结合前面的“茶”。如果每个位置只处理自身输入，而不接收其他位置的信息，就无法表达这种上下文关系。这个例子用于说明建模需求，不规定实际的文本切分，也不代表某个模型的实测结果。

序列模型需要在位置之间传递信息。Recurrent Neural Network（RNN）通过沿序列更新的状态传递信息；Attention 机制则计算当前位置与一组可访问位置的匹配关系，按权重组合对应的向量。它是一种信息交互方式，可以与其他组件共同使用；具体历史演进见下一节。[Vaswani et al., 2017，§1—3.2](https://arxiv.org/html/1706.03762v7)

| 基本步骤 | 计算目的 | 在文本例子中的含义 |
| --- | --- | --- |
| 确定可访问范围 | 指定能够参与计算的位置 | 逐步预测后续文本时，“它”所在位置可访问自身及前文 |
| 计算匹配权重 | 根据输入区分各位置的贡献 | 不同前文位置可以获得不同权重 |
| 汇总向量信息 | 用加权结果更新当前位置的表示 | 为结合“茶”等前文信息提供计算途径 |

权重由模型的向量计算得到，不是人工指定的语法规则，也不能直接视为词语重要性或模型推理过程的完整解释。模型可能通过多个层和多组 Attention 计算共同处理一种语言关系。

## 文本表示与 Transformer 中的 Attention 机制

模型首先通过 Tokenizer 将文本转换为 Token 序列及其整数编号。Token 是模型处理文本的基本单元，可能对应一个词、词的一部分或标点等，不必对应一个完整汉字或单词；具体划分取决于 Tokenizer。[Hugging Face，Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4?fw=pt)

生成时，当前已知的输入与已生成内容共同构成上下文。模型计算词表（可输出的 Token 集合）中各个候选 Token 的概率，生成策略据此选出下一个 Token，再将其加入上下文，继续预测。这个逐步依赖已有结果的过程称为自回归（Autoregressive）生成。[Shazeer, 2019，§2.4](https://arxiv.org/html/1911.02150v1)

| 环节 | 输入与处理结果 |
| --- | --- |
| Tokenization | 将文本转换为 Token 编号序列 |
| Embedding | 将每个编号映射为数值向量，作为网络的初始表示 |
| 多层 Transformer 计算 | 结合上下文与位置信息，逐层更新各位置的向量表示 |
| 输出层与生成策略 | 由最后一个位置的最终表示计算下一 Token 的概率，并选择一个 Token |
| 继续生成 | 将新 Token 加入上下文，重复预测，直至满足停止条件 |

本书主要讨论 Decoder-only 模型，即使用一组自回归 Transformer 层直接处理已有序列、无需另设输入编码网络（Encoder）的架构。表中概括的是其生成流程，省略了缓存等实现细节。

Embedding 是网络处理 Token 的入口；后续各层产生的中间向量称为隐藏表示（Hidden Representation）。输出概率由最终表示计算得到，Attention 机制则参与中间表示的更新。[Vaswani et al., 2017，§3.1—3.4](https://arxiv.org/html/1706.03762v7)

## Self-Attention 与自回归生成

在 Transformer 中，Self-Attention 的匹配与汇总对象来自同一序列，因此能够将不同位置的信息组合进各位置的表示。它的输出仍是向量表示，而不是直接生成的文字；下一 Token 的概率要由后续网络与输出层计算。

自回归模型使用 Causal Mask 限制信息访问：每个位置只能访问自身及此前位置。这样，训练时即使整段文本已经给定，预测下一 Token 也不能利用后面的答案。标准 Transformer 中的 Feed-Forward Network（FFN）则对各位置的表示分别进行变换；层内还包含残差连接和归一化等组件。Attention 层是其中负责跨位置汇总信息的模块。[Vaswani et al., 2017，§3.1—3.3](https://arxiv.org/html/1706.03762v7)

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

下一节：[从 Seq2Seq 中的 Attention 到 Transformer](from-seq2seq-to-transformer.md)。
