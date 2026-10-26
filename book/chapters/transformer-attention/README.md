# 第 2 章 Transformer Attention 的基本原理与计算过程

Attention 层把每个位置的输入表示转换为 Query、Key 和 Value，计算位置之间的权重，再组合信息。本章沿这一计算过程展开，建立后续结构改进、kernel 优化和 KV cache 管理共同使用的定义与符号。

| 小节 | 主要内容 |
| --- | --- |
| [2.1 Transformer 中的 Attention 结构与计算流程](structure-and-flow.md) | Attention 在网络中的位置，以及输入到输出的计算步骤 |
| [2.2 Query、Key、Value 与 Scaled Dot-Product Attention](scaled-dot-product.md) | 投影、缩放、Softmax 和加权汇总 |
| [2.3 Self-Attention、Cross-Attention 与 Causal Attention](attention-and-masks.md) | 信息来源、访问范围和 Mask |
| [2.4 Multi-Head Attention 的结构与 Output Projection](multi-head.md) | 多组独立计算、拼接和输出投影 |
| [2.5 Training、Prefill 与 Decode 的计算差异](training-prefill-decode.md) | 完整序列计算、增量计算和资源成本 |

<p align="center">表 2-1　第 2 章阅读导航</p>

计算定义以 [Attention Is All You Need，§3](https://arxiv.org/html/1706.03762v7) 为基础；[Dive into Deep Learning，§11.3、§11.5、§11.7](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)辅助解释张量和结构；[Shazeer，§2](https://arxiv.org/html/1911.02150v1)与 [Kwon et al.，§2.2](https://arxiv.org/html/2309.06180v1)用于核对增量推理。本章的数值示例为教学构造，核查记录与来源版本保存在仓库中。

矩阵按 Token 占据行排列，单独写出的向量按列向量处理。正文先省略 batch 和 Head 维度，随后逐步补齐；矩阵乘法与 Softmax 的补充说明见[附录 A 英文术语、数学符号与补充推导](../../appendices/foundations.md)。
