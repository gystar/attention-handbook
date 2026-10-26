# 大模型 Attention：原理、演进与优化

本书讨论注意力机制（Attention Mechanism）在 Transformer 与大语言模型中的作用、演进和优化。Attention 机制根据输入计算权重并汇总信息，是模型中的计算组件；本书从其建模需求出发，解释不同方法的改变、能力与计算成本。

第 1 章介绍必要基础与总体框架；后续章节按原理与结构改进、计算与存储优化、Linear Attention 与混合架构三个主题展开。阅读需要线性代数、概率和神经网络的基本知识。

## 阅读入口

* [第 1 章 Attention 机制的演进与主要发展方向](zong-lan/overview/)
* [附录 A 英文术语、数学符号与补充推导](fu-lu/foundations.md)
* [附录 B 技术演进时间线与方法对照表](fu-lu/timeline.md)
* [附录 C 参考文献、教学资源与本地源文件索引](fu-lu/references.md)

## 章节导航

| 主题                     | 章节                                                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 总览                     | [第 1 章 Attention 机制的演进与主要发展方向](zong-lan/overview/)                                                                      |
| 原理与结构改进                | [第 2 章 Transformer Attention 的基本原理与计算过程](yuan-li-yu-jie-gou-gai-jin/transformer-attention/)                             |
| 原理与结构改进                | [第 3 章 Position Encoding 的演进与 Context Extension](yuan-li-yu-jie-gou-gai-jin/position-encoding.md)                       |
| 原理与结构改进                | [第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA](yuan-li-yu-jie-gou-gai-jin/kv-representations.md)                              |
| 原理与结构改进                | [第 5 章 Softmax Attention 的结构改进：Gated Attention](yuan-li-yu-jie-gou-gai-jin/gated-attention.md)                          |
| 计算与存储优化                | [第 6 章 FlashAttention：Exact Attention 的 IO 优化](ji-suan-yu-cun-chu-you-hua/flash-attention.md)                           |
| 计算与存储优化                | [第 7 章 Sparse Attention 的连接模式与动态选择](ji-suan-yu-cun-chu-you-hua/sparse-attention.md)                                     |
| 计算与存储优化                | [第 8 章 KV Cache 的存储管理与推理优化](ji-suan-yu-cun-chu-you-hua/kv-cache.md)                                                     |
| Linear Attention 与混合架构 | [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](linear-attention-yu-hun-he-jia-gou/linear-attention.md) |
| Linear Attention 与混合架构 | [第 10 章 Hybrid Architectures 的结构设计与方法比较](linear-attention-yu-hun-he-jia-gou/hybrid-architectures.md)                    |

<p align="center">章节导航</p>

## 内容进度

第 1 章已有概览和四节正文；第 2、3 章各有概览及五节正文，分别解释 Attention 计算，以及位置编码、上下文扩展与评估。第 4、9 章已接收 KV 表示与容量、Linear Attention 递推等内容，完整章节待写；其余专题章保留节级目录。附录 B 收录时间线与方法比较条件。本地原文归档和本轮 GitBook 发布检查尚待完成。
