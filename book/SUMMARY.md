# Summary

* [大模型 Attention：原理、演进与优化](README.md)

## 总览

* [第 1 章 大模型 Attention 的总体框架与技术演进](chapters/overview/README.md)
  * [1.1 从 Representation Learning 到 Attention 机制](chapters/overview/introduction.md)
  * [1.2 Attention 机制的基本计算](chapters/overview/attention-in-llms.md)
  * [1.3 从 Seq2Seq 中的 Attention 到 Transformer](chapters/overview/from-seq2seq-to-transformer.md)
  * [1.4 LLM 中的计算瓶颈、KV Cache 与 Long Context](chapters/overview/bottlenecks-and-context.md)
  * [1.5 Attention 的结构改进、计算优化与架构扩展](chapters/overview/optimization-landscape.md)
  * [1.6 主要技术路线的演进关系与比较维度](chapters/overview/evolution-and-comparison.md)

## 原理与结构改进

* [第 2 章 Transformer Attention 的基本原理与计算过程](chapters/transformer-attention/README.md)
* [第 3 章 Position Encoding 的演进与 Context Extension](chapters/position-encoding/README.md)
* [第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA](chapters/kv-representations/README.md)
* [第 5 章 Softmax Attention 的结构改进：Gated Attention](chapters/gated-attention/README.md)

## 计算与存储优化

* [第 6 章 FlashAttention：Exact Attention 的 IO 优化](chapters/flash-attention/README.md)
* [第 7 章 Sparse Attention 的连接模式与动态选择](chapters/sparse-attention/README.md)
* [第 8 章 KV Cache 的存储管理与推理优化](chapters/kv-cache/README.md)

## Linear Attention 与混合架构

* [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](chapters/linear-attention/README.md)
* [第 10 章 Hybrid Architectures 的结构设计与方法比较](chapters/hybrid-architectures/README.md)

## 附录

* [附录 A 英文术语、数学符号与补充推导](appendices/foundations.md)
* [附录 B 技术演进时间线与方法对照表](appendices/timeline.md)
* [附录 C 参考文献、教学资源与本地源文件索引](appendices/references.md)

