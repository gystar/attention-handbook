# 第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update

Linear Attention 将历史信息累积到状态中，以状态读取替代逐个访问历史 KV。核心问题从“保留哪些历史位置”转为“如何写入、保留和读取状态中的信息”。固定状态维度降低了长序列推理的存储需求，也带来信息叠加与检索能力的问题。

本章先从特征核推导计算重排与因果递推，再讨论状态衰减、Delta Rule、并行计算及状态容量。状态 Gate 调节历史的保留，与第五章 Softmax Attention 的输出 Gating 作用位置不同。

- [9.1 从 Softmax Attention 到 Kernel Attention](kernel-attention.md)
- [9.2 Causal Linear Attention 的递推计算](recurrence.md)
- [9.3 状态衰减与 Gated Linear Attention](state-decay.md)
- [9.4 DeltaNet 与 Gated DeltaNet 的状态更新](delta-rule.md)
- [9.5 并行训练与分块计算](chunkwise-computation.md)
- [9.6 状态容量与检索能力](capacity-retrieval.md)
