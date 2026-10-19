# 第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update

> 状态：已迁入部分基础内容，完整章节待写。

9.1 Linear Attention 的计算框架与主要路线

9.2 从 Kernel Feature Maps 到 Recurrent Formulation

9.3 Gated Linear Attention 的 State Update

9.4 DeltaNet 与 Gated DeltaNet 的更新机制

9.5 State Capacity、Retrieval 与 Length Generalization


## 从显式历史到状态更新

一个基本的 Causal Linear Attention 构造使用非负特征映射 $$\phi:\mathbb{R}^{d_k}\to\mathbb{R}^{d_\phi}$$，维护：

$$
\begin{aligned}
S_t&=S_{t-1}+\phi(k_t)v_t^\top,\\
z_t&=z_{t-1}+\phi(k_t),\\[4pt]
o_t^\top&=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}.
\end{aligned}
$$

其中 $$q_t,k_t\in\mathbb{R}^{d_k}$$、$$v_t\in\mathbb{R}^{d_v}$$ 均按列向量书写；$$S_t\in\mathbb{R}^{d_\phi\times d_v}$$，$$z_t\in\mathbb{R}^{d_\phi}$$，初始状态为零，分母需非零。状态累积历史信息，读取时无需再次逐个访问全部历史 KV。固定特征与状态维度后，该构造的状态大小不随序列长度增加。[Katharopoulos et al., 2020，§3.2—3.4](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)

从显式历史求和到上述递推式的步骤见[附录 A 英文术语、数学符号与补充推导](../../appendices/foundations.md)。固定大小描述的是逐步推理时的状态；训练所需的激活存储还取决于序列长度与实现。

上述递推依赖可分解的核形式。标准 Softmax Attention 不具备同样的分解，不能通过交换矩阵乘法顺序直接得到这一固定大小状态的实现。后续方法还会改变状态的衰减、写入与更新方式。

Gated DeltaNet 将控制状态衰减的 Gate 与 Delta Rule 结合；Softmax Attention 中的输出 Gating 则调节某个位置的中间结果。两者的 Gate 作用对象不同。[Yang et al., 2024，§3.1，式 (8)](https://arxiv.org/html/2412.06464v1)；[Qiu et al., 2025，§2.2](https://arxiv.org/html/2505.06708v1)
