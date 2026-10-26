# 大模型 Attention：原理、演进与优化

本书讨论注意力机制（Attention Mechanism）在 Transformer 与大语言模型中的作用、演进和优化。Attention 机制根据输入计算权重并汇总信息，是模型中的计算组件；本书从其建模需求出发，解释不同方法的改变、能力与计算成本。

第 1 章介绍必要基础与总体框架；后续章节按原理与结构改进、计算与存储优化、Linear Attention 与混合架构三个主题展开。阅读需要线性代数、概率和神经网络的基本知识。

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

## 内容进度

十章正文和三个附录已写齐，并完成归档后的逐章来源与表达审校。当前工作树已保存 60 篇必备论文和 76 个图表文件。第 3—10 章已补充原论文图及读图说明；GitBook 页面呈现与视频播放按用户决定留待同步后检查。
