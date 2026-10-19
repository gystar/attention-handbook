# 第 1 章 大模型 Attention 的总体框架与技术演进

理解大模型中的 Attention，需要同时回答三个问题：模型怎样从上下文中读取信息，这个过程消耗哪些计算与存储资源，以及不同改进改变了其中的哪一部分。

本章以 Transformer Attention 为参照，先介绍语言模型的生成过程与 Attention 的基本计算，再解释后续方法的分工。详细推导和实现分别在后续章节展开。

![Attention 技术路线图：围绕 Transformer Attention，区分表示与位置、连接模式、计算实现、缓存管理、状态更新和模块组合](../../assets/overview/attention-landscape.svg)

图 1-1　本书绘制的技术路线图。分支表示设计问题与章节归属；连线不表示方法之间存在时间上的继承关系。

| 本节 | 要回答的问题 |
| --- | --- |
| [1.1 Attention 在 LLM 中的作用与研究范围](attention-in-llms.md) | Attention 如何参与上下文计算，本书讨论到什么范围？ |
| [1.2 从 Seq2Seq 中的 Attention 到 Transformer](from-seq2seq-to-transformer.md) | 从动态读取 Encoder 状态到 Self-Attention，架构发生了什么改变？ |
| [1.3 计算瓶颈、KV Cache 与 Long Context](bottlenecks-and-context.md) | 为什么长输入、逐 Token 生成和长上下文能力需要分别分析？ |
| [1.4 结构改进、计算优化与架构扩展](optimization-landscape.md) | MQA、FlashAttention、PagedAttention、Linear Attention 分别改变什么？ |
| [1.5 演进关系与比较维度](evolution-and-comparison.md) | 哪些路线能够组合，比较方法时需要控制哪些条件？ |

本章从 Token 与文本生成入手，不要求预先熟悉 LLM 架构。公式部分需要向量、矩阵乘法和基本概率知识；Softmax 的作用在 1.1 中说明。符号见[附录 A](../../appendices/foundations.md)，原始论文与教学材料见[附录 C](../../appendices/references.md)。
