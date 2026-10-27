# 第 5 章 Gated Attention 的结构与作用

前两章分别改变位置匹配和历史 KV 的表示。即使选定了这些结构，还可以进一步调节 Attention 汇总结果向后续网络传递的幅度。Gating 根据输入生成调节系数，作用于相应张量。本章以 Qiu 等人研究的 Softmax Attention 输出 Gate 为例，解释汇总之后的调节，并比较其他插入位置及其训练证据。

- [5.1 Attention 输出的 Gating](output-gating.md)
- [5.2 Gate 的位置与参数化](placement.md)
- [5.3 输出调节与 Attention Sink](attention-sink.md)
- [5.4 训练证据与比较条件](evaluation.md)

机制讨论以 [Qiu et al.，§2.2](https://arxiv.org/html/2505.06708v1) 的实验设计为主要参照；Attention Sink 的背景结合 [Xiao et al.，§3.1](https://arxiv.org/html/2309.17453v4)。Gated Attention 包含多种结构，使用时需明确 Gate 的作用对象；本章重点是 SDPA 输出，控制递推状态保留的 Gate 在第九章展开。
