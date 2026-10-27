# 大模型 Attention：原理、演进与优化

本书讨论注意力机制（Attention Mechanism）在 Transformer 与大语言模型中的作用、演进和优化。Attention 机制根据输入计算权重并汇总信息，是模型中的计算组件；本书从其建模需求出发，解释不同方法的改变、能力与计算成本。

全书围绕一个问题展开：语言模型如何利用不断增长的上下文，同时控制计算与存储成本？先解释 Attention 为什么出现、怎样计算，再分别考察位置关系、KV 表示与输出调节；随后进入硬件执行、稀疏读取和缓存管理，最后讨论固定状态与混合架构。各条路线按改变的计算对象组织，具体模型用于说明机制和取舍。阅读需要线性代数、概率和神经网络的基本知识。

## 阅读入口

- [第 1 章 Attention 机制的演进与主要发展方向](chapters/overview/README.md)
- [附录 A 英文术语、数学符号与补充推导](appendices/foundations.md)
- [附录 B 技术演进时间线与方法对照表](appendices/timeline.md)
- [附录 C 参考文献、教学资源与本地源文件索引](appendices/references.md)

## 章节导航

| 主题 | 章节 |
| --- | --- |
| 总览 | [第 1 章 Attention 机制的演进与主要发展方向](chapters/overview/README.md) |
| 原理与结构改进 | [第 2 章 Transformer Attention 的基本原理与计算过程](chapters/transformer-attention/README.md) |
| 原理与结构改进 | [第 3 章 Position Encoding 的演进与 Context Extension](chapters/position-encoding/README.md) |
| 原理与结构改进 | [第 4 章 MQA、GQA 与 MLA 的 KV 表示优化](chapters/kv-representations/README.md) |
| 原理与结构改进 | [第 5 章 Gated Attention 的结构与作用](chapters/gated-attention/README.md) |
| 计算与存储优化 | [第 6 章 FlashAttention 的计算与 IO 优化](chapters/flash-attention/README.md) |
| 计算与存储优化 | [第 7 章 Sparse Attention 的连接模式与动态选择](chapters/sparse-attention/README.md) |
| 计算与存储优化 | [第 8 章 KV Cache 的存储管理与推理优化](chapters/kv-cache/README.md) |
| Linear Attention 与混合架构 | [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](chapters/linear-attention/README.md) |
| Linear Attention 与混合架构 | [第 10 章 Hybrid Architectures 的结构设计与方法比较](chapters/hybrid-architectures/README.md) |

<p align="center">章节导航</p>

## 资料与查阅

各节将原论文、作者图文和必要的计算示例放在相应机制附近。附录 A 用于查术语和符号，附录 B 用于对照方法及时间线，附录 C 关联公开来源与本地原文。章节中的教学算例与论文实验分别标明。
