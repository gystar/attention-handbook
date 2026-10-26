# 第 1 章 Attention 机制的演进与主要发展方向

语言模型需要结合上下文理解和生成文本。本章从这一需求引出注意力机制（Attention Mechanism），解释它如何从 RNN Encoder–Decoder 中的信息读取模块，发展为 Transformer 的核心机制，再介绍大模型中的计算成本与主要改进方向。

| 本节                                                              | 主要内容                                 |
| --------------------------------------------------------------- | ------------------------------------ |
| [1.1 从 Representation Learning 到 Attention 机制](introduction.md) | 向量表示、上下文与信息汇总                        |
| [1.2 从 RNN 到 Transformer](from-seq2seq-to-transformer.md)       | 从读取 Encoder 状态到以 Self-Attention 建模序列 |
| [1.3 LLM 中的 Attention 成本与长上下文需求](bottlenecks-and-context.md)    | 自回归生成、计算与缓存成本、长上下文能力                 |
| [1.4 Attention 机制的主要发展方向与技术路线](optimization-landscape.md)       | 各类方法改变的对象、组合关系与比较条件                  |

<p align="center">表 1-1　第 1 章阅读导航</p>

本书以用于语言建模的 Transformer Attention 为基础，成本讨论主要采用 Decoder-only 模型的常规自回归生成场景。后续章节分别展开基本计算、位置处理与结构改进、计算与存储优化，以及 Linear Attention 和 Hybrid Architectures。视觉、语音和多模态中的应用按机制比较的需要引入；State Space Models（SSMs）在讨论相关序列模块与混合架构时介绍。

本章先建立概念与方法之间的联系。QKV、Mask 和 Multi-Head 的完整计算见[第 2 章 Transformer Attention 的基本原理与计算过程](../../yuan-li-yu-jie-gou-gai-jin/transformer-attention/)，术语与符号见[附录 A 英文术语、数学符号与补充推导](../../fu-lu/foundations.md)。
