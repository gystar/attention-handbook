# 第 7 章 Sparse Attention 的连接模式与动态选择

FlashAttention 减少中间结果的存储与读写，但 Dense 路径仍处理给定 Mask 下的全部位置配对。要进一步减少这些配对，可以限制每个 Query 的读取集合，这就是本章讨论的 Sparse Attention。减少读取也可能遗漏信息，因此连接设计必须与模型训练、信息覆盖和实际执行成本一起分析。

本章先讨论按位置规定连接的通用方式，再说明如何按块执行、如何根据内容动态选择。随后以 DeepSeek-V4 的 CSA/HCA 为模型案例，分析序列压缩与历史读取的组合；其中 HCA 对压缩后的历史执行 Dense Attention，在模型中与 CSA 层交替配置。[DeepSeek-V4，§2.3](https://arxiv.org/html/2606.19348v1#S2.SS3)

- [7.1 稀疏连接与计算成本](connectivity-cost.md)
- [7.2 Sliding Window 与 Local–Global Attention](local-global.md)
- [7.3 Block-Sparse Attention 的组织与执行](block-sparse.md)
- [7.4 基于内容的动态选择](dynamic-selection.md)
- [7.5 DeepSeek-V4 的 CSA 与 HCA](compressed-attention.md)
- [7.6 信息覆盖与长上下文评估](evaluation.md)

固定连接的讨论以 [Sparse Transformer](https://arxiv.org/html/1904.10509v1)、[Longformer](https://arxiv.org/html/2004.05150v2) 和 [BigBird](https://arxiv.org/html/2007.14062v2) 为依据，并结合 [Mistral 7B](https://arxiv.org/html/2310.06825v1) 说明自回归模型中的局部窗口。

动态选择部分先介绍 Hash/Clustering 的候选分组，再比较 Quest 的页面估计、MInference 的 Prefill 模式选择、NSA 的压缩与块选择，以及 DSA 的可训练 Indexer；各方法的运算和训练依据见 7.4。
