# 第 3 章 Position Encoding 的演进与 Context Extension

第二章介绍了 Attention 如何按内容匹配与汇总信息。本章进一步解释位置如何参与计算，以及模型在处理更长序列时，需要怎样调整位置表示并检验效果。

叙述分为三个层次：**表示位置与距离、改变 Attention 的匹配方式、扩展并评估上下文长度**。Absolute Position Encoding、Relative Position Encoding、RoPE 和 ALiBi 对位置信息的处理不同；Position Interpolation 与后续 RoPE Scaling 方法则主要用于调整已有 RoPE 模型。

| 小节 | 主要问题 |
| --- | --- |
| [3.1 Attention 中的顺序与位置信息](position-and-order.md) | 内容匹配为什么需要位置，Causal Mask 提供了什么约束 |
| [3.2 Absolute 与 Relative Position Encoding](absolute-and-relative.md) | 从逐位置向量到位置对之间的关系 |
| [3.3 RoPE 与 ALiBi 的位置建模](rope-and-alibi.md) | 旋转 Q/K 与添加距离偏置如何影响匹配 |
| [3.4 RoPE Scaling 与上下文扩展](context-extension.md) | 从统一插值到按频率调整，以及缓存的一致性 |
| [3.5 长度泛化与长上下文评估](evaluation.md) | 区分可接收长度、语言建模表现与有效信息利用 |

<p align="center">表 3-1　第 3 章阅读导航</p>

本章依据 Transformer、Shaw Relative Position Representations、T5、RoFormer、ALiBi、Position Interpolation、YaRN 和 LongRoPE 的原始资料展开；评估部分结合 Lost in the Middle、RULER 与 LongBench。各节引用给出核查位置，版本及图表来源见[附录 C 参考文献、教学资源与本地源文件索引](../../appendices/references.md)。

沿用第二章的矩阵约定：Token 占据行，单独写出的向量为列向量。本章位置索引从 0 开始，$$i$$ 为 Query 位置、$$j$$ 为 Key 位置；$$d$$ 表示当前 Head 的 Q/K 维度，$$d_{\mathrm{model}}$$ 表示模型维度。

置换等变、相对旋转、插值与缓存一致性的数学算例由仓库 `scripts/verify_position_encoding.py` 核算，结果记录在 `reviews/position-encoding-checks.json`；这些检查针对公式与教学构造。
