# 大模型 Attention：原理、演进与优化

本书讨论注意力机制（Attention Mechanism）在 Transformer 与大语言模型中的作用、演进和优化。Attention 机制根据输入计算权重并汇总信息，是模型中的计算组件；本书从其建模需求出发，解释不同方法的改变、能力与计算成本。

第 1 章介绍必要基础与总体框架；后续章节按原理与结构改进、计算与存储优化、Linear Attention 与混合架构三个主题展开。阅读需要线性代数、概率和神经网络的基本知识。

## 阅读入口

- [第 1 章 大模型 Attention 的总体框架与技术演进](chapters/overview/README.md)
- [附录 A 英文术语、数学符号与补充推导](appendices/foundations.md)
- [附录 B 技术演进时间线与方法对照表](appendices/timeline.md)
- [附录 C 参考文献、教学资源与本地源文件索引](appendices/references.md)

## 章节导航

| 主题 | 章节 |
| --- | --- |
| 总览 | [第 1 章 大模型 Attention 的总体框架与技术演进](chapters/overview/README.md) |
| 原理与结构改进 | [第 2 章 Transformer Attention 的基本原理与计算过程](chapters/transformer-attention/README.md) |
| 原理与结构改进 | [第 3 章 Position Encoding 的演进与 Context Extension](chapters/position-encoding/README.md) |
| 原理与结构改进 | [第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA](chapters/kv-representations/README.md) |
| 原理与结构改进 | [第 5 章 Softmax Attention 的结构改进：Gated Attention](chapters/gated-attention/README.md) |
| 计算与存储优化 | [第 6 章 FlashAttention：Exact Attention 的 IO 优化](chapters/flash-attention/README.md) |
| 计算与存储优化 | [第 7 章 Sparse Attention 的连接模式与动态选择](chapters/sparse-attention/README.md) |
| 计算与存储优化 | [第 8 章 KV Cache 的存储管理与推理优化](chapters/kv-cache/README.md) |
| Linear Attention 与混合架构 | [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](chapters/linear-attention/README.md) |
| Linear Attention 与混合架构 | [第 10 章 Hybrid Architectures 的结构设计与方法比较](chapters/hybrid-architectures/README.md) |

## 内容进度

第 1 章已有概览和五节正文，已进行技术核对与作者自查修正；本地原文归档和 GitBook 发布检查尚待完成。第 2—10 章提供章节定位和节级目录，尚未展开正文。附录当前覆盖首章内容。

