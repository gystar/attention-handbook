# 第 1 章 大模型 Attention 的总体框架与技术演进

本章先从表示学习与上下文建模的需求引出注意力机制（Attention Mechanism），再建立基本计算模型，介绍历史演进、资源成本与优化路线。初次接触这一主题，可按小节顺序阅读；已有基础的读者可直接进入成本分析和方法比较。

| 本节 | 要回答的问题 |
| --- | --- |
| [1.1 从 Representation Learning 到 Attention 机制](introduction.md) | 模型为何需要上下文，Attention 机制处于什么位置？ |
| [1.2 Attention 机制的基本计算](attention-in-llms.md) | QKV、Mask 与加权求和如何形成 Attention 输出？ |
| [1.3 从 Seq2Seq 中的 Attention 到 Transformer](from-seq2seq-to-transformer.md) | 从动态读取 Encoder 状态到 Self-Attention，架构发生了什么改变？ |
| [1.4 LLM 中的计算瓶颈、KV Cache 与 Long Context](bottlenecks-and-context.md) | 为什么长输入、逐 Token 生成和长上下文能力需要分别分析？ |
| [1.5 Attention 的结构改进、计算优化与架构扩展](optimization-landscape.md) | MQA、FlashAttention、PagedAttention、Linear Attention 分别改变什么？ |
| [1.6 主要技术路线的演进关系与比较维度](evolution-and-comparison.md) | 哪些路线能够组合，比较方法时需要控制哪些条件？ |

本章从学习方法与序列建模问题入手，不要求预先熟悉 LLM 架构。公式部分需要向量、矩阵乘法和基本概率知识；Softmax 的作用在 1.2 中说明。符号见[附录 A 英文术语、数学符号与补充推导](../../appendices/foundations.md)，原始论文与教学材料见[附录 C 参考文献、教学资源与本地源文件索引](../../appendices/references.md)。
