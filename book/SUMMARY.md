# Table of contents

* [大模型 Attention：原理、演进与优化](README.md)

## 总览

* [第 1 章 Attention 机制的演进与主要发展方向](zong-lan/overview/README.md)
  * [1.1 从 Representation Learning 到 Attention 机制](zong-lan/overview/introduction.md)
  * [1.2 从 RNN 到 Transformer](zong-lan/overview/from-seq2seq-to-transformer.md)
  * [1.3 LLM 中的 Attention 成本与长上下文需求](zong-lan/overview/bottlenecks-and-context.md)
  * [1.4 Attention 机制的主要发展方向与技术路线](zong-lan/overview/optimization-landscape.md)

## 原理与结构改进

* [第 2 章 Transformer Attention 的基本原理与计算过程](yuan-li-yu-jie-gou-gai-jin/transformer-attention/README.md)
  * [2.1 Transformer 中的 Attention 结构与计算流程](yuan-li-yu-jie-gou-gai-jin/transformer-attention/structure-and-flow.md)
  * [2.2 Query、Key、Value 与 Scaled Dot-Product Attention](yuan-li-yu-jie-gou-gai-jin/transformer-attention/scaled-dot-product.md)
  * [2.3 Self-Attention、Cross-Attention 与 Causal Attention](yuan-li-yu-jie-gou-gai-jin/transformer-attention/attention-and-masks.md)
  * [2.4 Multi-Head Attention 的结构与 Output Projection](yuan-li-yu-jie-gou-gai-jin/transformer-attention/multi-head.md)
  * [2.5 Training、Prefill 与 Decode 的计算差异](yuan-li-yu-jie-gou-gai-jin/transformer-attention/training-prefill-decode.md)
* [第 3 章 Position Encoding 的演进与 Context Extension](yuan-li-yu-jie-gou-gai-jin/position-encoding.md)
* [第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA](yuan-li-yu-jie-gou-gai-jin/kv-representations.md)
* [第 5 章 Softmax Attention 的结构改进：Gated Attention](yuan-li-yu-jie-gou-gai-jin/gated-attention.md)

## 计算与存储优化

* [第 6 章 FlashAttention：Exact Attention 的 IO 优化](ji-suan-yu-cun-chu-you-hua/flash-attention.md)
* [第 7 章 Sparse Attention 的连接模式与动态选择](ji-suan-yu-cun-chu-you-hua/sparse-attention.md)
* [第 8 章 KV Cache 的存储管理与推理优化](ji-suan-yu-cun-chu-you-hua/kv-cache.md)

## Linear Attention 与混合架构

* [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](linear-attention-yu-hun-he-jia-gou/linear-attention.md)
* [第 10 章 Hybrid Architectures 的结构设计与方法比较](linear-attention-yu-hun-he-jia-gou/hybrid-architectures.md)

## 附录

* [附录 A 英文术语、数学符号与补充推导](fu-lu/foundations.md)
* [附录 B 技术演进时间线与方法对照表](fu-lu/timeline.md)
* [附录 C 参考文献、教学资源与本地源文件索引](fu-lu/references.md)
