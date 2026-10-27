# 第 6 章 FlashAttention 的计算与 IO 优化

第三至第五章讨论模型计算本身的设计。选定 Attention 公式后，硬件执行还需处理数据搬运问题：同样的矩阵乘法，如果把庞大的中间结果反复写入显存，就会增加数据搬运。本章以 FlashAttention 为主线，解释如何通过分块与 Online Softmax 保持目标运算，同时减少中间存储，再分析并行调度、数值实现与硬件条件。

- [6.1 Attention 的计算量与数据访问](io-cost.md)
- [6.2 Online Softmax 与分块计算](online-softmax.md)
- [6.3 FlashAttention 的前向与反向计算](forward-backward.md)
- [6.4 并行调度与硬件适配](parallelism.md)
- [6.5 数值误差与性能评估](evaluation.md)

主要方法依据为 [FlashAttention 原始论文](https://arxiv.org/html/2205.14135v2)与 [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf)。硬件小节结合 Hopper 与 Blackwell 的实现，解释优化重点如何随计算、带宽和执行单元的比例变化。
