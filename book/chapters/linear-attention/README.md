# 第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update

前面的路线仍让 Query 读取一组显式保存的历史条目：条目可以共享 Head、降低精度、减少数量或压缩多个位置。本章考察另一种历史组织方式：将写入累积到固定维度的状态，Query 直接读取状态。由此，长序列问题从“保存和访问哪些条目”转为“有限状态怎样保留后续需要的信息”。

本章先从 Kernel Attention 的特征分解建立加法状态，再分析衰减与误差修正如何改变记忆更新，最后讨论并行训练和检索能力。RetNet、GLA、DeltaNet 等论文提供不同的具体设计；各自的状态更新、归一化和 Gate 会在引入时说明。状态 Gate 调节历史保留，第五章的输出 Gate 则调节本步向后续网络传递的结果。

- [9.1 从 Softmax Attention 到 Kernel Attention](kernel-attention.md)
- [9.2 Causal Linear Attention 的递推计算](recurrence.md)
- [9.3 状态衰减与 Gated Linear Attention](state-decay.md)
- [9.4 DeltaNet 与 Gated DeltaNet 的状态更新](delta-rule.md)
- [9.5 并行训练与分块计算](chunkwise-computation.md)
- [9.6 状态容量与检索能力](capacity-retrieval.md)
