# 第 7 章 Sparse Attention 的连接模式与动态选择

Sparse Attention 通过限制每个 Query 的读取范围减少位置配对。连接模式决定信息传播路径，选择器确定输入相关的读取集合，块级实现则决定能够跳过哪些计算。本章先建立连接与成本的关系，再讨论局部和全局模式、Block-Sparse 执行与动态选择，最后评估信息覆盖和模型质量。

- [7.1 稀疏连接与计算成本](connectivity-cost.md)
- [7.2 Sliding Window 与 Local–Global Attention](local-global.md)
- [7.3 Block-Sparse Attention 的组织与执行](block-sparse.md)
- [7.4 基于内容的动态选择](dynamic-selection.md)
- [7.5 信息覆盖与长上下文评估](evaluation.md)

固定连接的讨论以 [Sparse Transformer](https://arxiv.org/html/1904.10509v1)、[Longformer](https://arxiv.org/html/2004.05150v2) 和 [BigBird](https://arxiv.org/html/2007.14062v2) 为依据，并结合 [Mistral 7B](https://arxiv.org/html/2310.06825v1) 说明自回归模型中的局部窗口。

动态选择部分先介绍 Hash/Clustering 的候选分组，再比较 Quest 的页面估计、MInference 的 Prefill 模式选择、NSA 的压缩与块选择，以及 DSA 的可训练 Indexer；各方法的运算和训练依据见 7.4。
