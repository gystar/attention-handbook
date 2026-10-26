# 第 6 章 FlashAttention 的计算与 IO 优化

Attention 的执行效率同时受到运算量、数据访问和 GPU 并行方式的影响。本章先分析完整分数矩阵的成本，再推导 Online Softmax 如何支持分块汇总，随后展开前后向计算、硬件调度、低精度矩阵乘法及跨设备划分，最后讨论数值误差和性能比较条件。

- [6.1 Attention 的计算量与数据访问](io-cost.md)
- [6.2 Online Softmax 与分块计算](online-softmax.md)
- [6.3 FlashAttention 的前向与反向计算](forward-backward.md)
- [6.4 并行调度与硬件适配](parallelism.md)
- [6.5 数值误差与性能评估](evaluation.md)

主要方法依据为 [FlashAttention 原始论文](https://arxiv.org/html/2205.14135v2)与 [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf)。硬件小节结合 Hopper 与 Blackwell 的实现，解释优化重点如何随计算、带宽和执行单元的比例变化。
