# 6.3 FlashAttention 的前向与反向计算

上一节说明了如何合并分块结果。本节将这些统计量放入完整的计算流程：前向输出 Attention 结果并保存每行的归一化信息；反向按需重算权重，累加 Query、Key 和 Value 的梯度。流程采用 FlashAttention-2 的 Query 分块前向顺序。[论文 §3.1、Algorithms 1—2](https://tridao.me/publications/flash2/flash2.pdf)

## 前向保留输出与行统计量

考虑单个 Head，允许 Query 与 Key 的长度不同：

$$
Q\in\mathbb R^{N_q\times d_h},\quad
K\in\mathbb R^{N_k\times d_h},\quad
V\in\mathbb R^{N_k\times d_v},\quad
O\in\mathbb R^{N_q\times d_v}.
$$

令 $$c=1/\sqrt{d_h}$$，有效位置的分数为 $$S_{ij}=c\,q_i^\top k_j$$。本节先讨论固定 Mask、无 Dropout 的情形，并要求每行至少有一个有效位置。

| 步骤 | 片上计算 | 保留内容 |
| --- | --- | --- |
| 读取一块 Query | 初始化各行的最大值、分母和未归一化输出 | 当前 Query 块及其统计量 |
| 遍历 Key、Value 块 | 计算分数，应用 Mask，按 6.2 的公式重缩放并累加 | 更新后的统计量与部分输出 |
| 结束该 Query 块 | 将加权和除以分母，计算 LogSumExp | 向 HBM 写出该块的 $$O$$ 与 $$L$$ |

<p align="center">表 6-3　分块前向计算的数据流</p>

对第 $$i$$ 行，完成所有有效块后保存的 LogSumExp 为：

$$
L_i=m_i+\log\ell_i
=\log\sum_{j\text{ 有效}}e^{S_{ij}}.
$$

这里的等式用于说明含义；实现使用左侧的稳定形式计算。反向所需数据包括 $$Q,K,V,O$$ 以及长度为 $$N_q$$ 的 $$L$$。输入张量由训练系统保留或通过其他 Checkpoint 策略重建，完整 $$N_q\times N_k$$ 权重矩阵则可按块重新计算。

## 反向按需恢复权重

设标量损失为 $$\mathcal J$$，从后续网络传回的梯度为 $$G=\partial\mathcal J/\partial O\in\mathbb R^{N_q\times d_v}$$。给定某个 Query–Key 块，通过重新计算分数恢复有效位置的权重：

$$
P_{ij}=\exp(S_{ij}-L_i).
$$

被 Mask 屏蔽的位置取 $$P_{ij}=0$$。由于 $$L_i$$ 来自完整有效行，上式恢复的是整行归一化后的权重，即使当前只计算其中一块。

下面从 $$O=PV$$ 重新推导梯度。记 $$\bar V=\partial\mathcal J/\partial V$$，其余上横线符号含义相同。先对 Value 加权和求导：

$$
\bar V=P^\top G,\qquad \bar P=GV^\top.
$$

对行 Softmax 求导，需要扣除该行的加权平均：

$$
\bar S_{ij}=P_{ij}\left(\bar P_{ij}-D_i\right),\qquad
D_i=\sum_jP_{ij}\bar P_{ij}.
$$

直接按右式计算 $$D_i$$ 需要遍历整行权重，但它可以从已经保存的输出获得：

$$
\begin{aligned}
D_i
&=\sum_j P_{ij}\sum_{a=1}^{d_v}G_{ia}V_{ja}\\
&=\sum_{a=1}^{d_v}G_{ia}\left(\sum_jP_{ij}V_{ja}\right)\\
&=\sum_{a=1}^{d_v}G_{ia}O_{ia}.
\end{aligned}
$$

因此，反向开始时只需对 $$G\odot O$$ 按行求和，得到 $$D\in\mathbb R^{N_q}$$；每个分数块随后都能独立使用对应的 $$D_i$$。最后，对缩放点积求导：

$$
\bar Q=c\,\bar S K,\qquad
\bar K=c\,\bar S^\top Q.
$$

缩放因子 $$c$$ 在这两个梯度中都必须保留。上述化简与原始 FlashAttention 的反向算法一致。[原论文 Appendix B.2、B.4](https://arxiv.org/html/2205.14135v2#A2.SS4)

## 分块梯度需要归约

前面的矩阵公式描述完整梯度，实现时只生成当前块的 $$P$$、$$\bar P$$ 和 $$\bar S$$，使用后即可释放。一个 Query 块会接收多个 Key 块的梯度贡献；一个 Key 或 Value 块也会接收多个 Query 块的贡献。

| 梯度 | 单个块对的贡献 | 需要累加的方向 |
| --- | --- | --- |
| $$\bar Q_A$$ | $$c\,\bar S_{AB}K_B$$ | 全部有效 Key 块 $$B$$ |
| $$\bar K_B$$ | $$c\,\bar S_{AB}^{\top}Q_A$$ | 全部有效 Query 块 $$A$$ |
| $$\bar V_B$$ | $$P_{AB}^{\top}G_A$$ | 全部有效 Query 块 $$A$$ |

<p align="center">表 6-4　分块反向计算的梯度累加</p>

表中 $$A,B$$ 分别表示 Query 与 Key 的位置集合。并行 kernel 必须组织这些累加操作，避免多个执行单元直接覆盖同一梯度。选择由哪个块持有累加器、何时进行跨块归约，是后续并行调度需要解决的问题。

## Dropout 需要重现同一 Mask

若对 Attention 权重使用 Dropout，令 $$Z_{ij}$$ 为前向采样后固定的缩放 Mask：保留位置取 $$1/(1-p)$$，丢弃位置取零，其中 $$p$$ 是丢弃概率。输出变为 $$O=(P\odot Z)V$$，相应梯度为：

$$
\bar V=(P\odot Z)^\top G,\qquad
\bar P=(GV^\top)\odot Z.
$$

将这两个关系代入前面的求和，仍可得到 $$D_i=\sum_aG_{ia}O_{ia}$$，其中 $$O$$ 必须是实际使用 Dropout 后的前向输出。反向需要重现同一个 $$Z$$；只保存随机种子是否足够，取决于随机数计数器和元素映射是否也保持一致。原始算法保存随机数状态，并在反向重新生成 Dropout Mask。[原论文 Appendix B.3—B.4](https://arxiv.org/html/2205.14135v2#A2.SS3)

本节推导可用一个小规模检查验证：固定 $$G$$，定义 $$\mathcal J=\sum_{i,a}O_{ia}G_{ia}$$，分别微扰 $$Q,K,V$$ 的每个元素，用中心有限差分与解析梯度比较。对于 Dropout 检查，两次微扰使用同一张固定 Mask，避免把采样变化混入梯度。
