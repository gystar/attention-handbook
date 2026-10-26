# 第 5 章 Gated Attention 的结构与作用

Attention 权重决定从哪些位置汇总信息；Gate 进一步调节汇总结果传递多少、哪些维度被保留。本章以 Softmax Attention 的输出 Gating 为中心，先说明计算，再比较插入位置与参数化，最后检查 Attention Sink、训练稳定性和模型质量的证据。

- [5.1 Attention 输出的 Gating](output-gating.md)
- [5.2 Gate 的位置与参数化](placement.md)
- [5.3 输出调节与 Attention Sink](attention-sink.md)
- [5.4 训练证据与比较条件](evaluation.md)

机制讨论以 [Qiu et al.，§2.2](https://arxiv.org/html/2505.06708v1) 的实验设计为主要参照；Attention Sink 的背景结合 [Xiao et al.，§3.1](https://arxiv.org/html/2309.17453v4)。本章使用 Softmax 后的输出 Gate；控制递推状态保留的 Gate 在第九章展开。
