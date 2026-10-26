# 2.2 Query、Key、Value 与 Scaled Dot-Product Attention

单个 Attention Head 通过 Query–Key 匹配决定读取权重，再汇总对应的 Value。下面先定义一个 Head 的计算，再用可复算的小例子连接分数、权重和输出。

## QKV 的来源与形状

设提供 Query 的输入为 $$X\in\mathbb{R}^{n_q\times d_x}$$，提供 Key 和 Value 的输入为 $$Z\in\mathbb{R}^{n_k\times d_z}$$。省略投影偏置，有：

$$
Q=XW_Q,\qquad K=ZW_K,\qquad V=ZW_V.
$$

投影参数的形状分别为 $$W_Q\in\mathbb{R}^{d_x\times d_k}$$、$$W_K\in\mathbb{R}^{d_z\times d_k}$$、$$W_V\in\mathbb{R}^{d_z\times d_v}$$。Self-Attention 中 $$Z=X$$；Cross-Attention 可以从另一序列取得 $$Z$$。Query 和 Key 的末维相同，便于做点积；Value 的维度可以不同。[Vaswani et al.，§3.2.1—3.2.3](https://arxiv.org/html/1706.03762v7)

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| $$Q$$ | $$n_q\times d_k$$ | 每行对应一个查询位置 |
| $$K$$ | $$n_k\times d_k$$ | 每行对应一个可匹配的位置 |
| $$V$$ | $$n_k\times d_v$$ | 与 Key 按位置配对的内容 |
| $$A$$ | $$n_q\times n_k$$ | Query 对各 Key 的读取权重 |
| $$O$$ | $$n_q\times d_v$$ | 每个 Query 对应的汇总结果 |

<p align="center">表 2-3　单个 Head 的张量形状</p>

## 从匹配分数到加权汇总

Scaled Dot-Product Attention 的矩阵形式为：

$$
S=\frac{QK^\top}{\sqrt{d_k}}+M,\qquad
A=\operatorname{softmax}_{\mathrm{row}}(S),\qquad O=AV.
$$

$$M$$ 是形状为 $$n_q\times n_k$$ 的加性 Mask：允许的位置取 $$0$$，禁止的位置取 $$-\infty$$。每行至少需要一个允许访问的 Key。Softmax 沿 Key 维度计算：

$$
A_{ij}=\frac{\exp(S_{ij})}{\sum_{r=1}^{n_k}\exp(S_{ir})},\qquad
O_{i,:}=\sum_{j=1}^{n_k}A_{ij}V_{j,:}.
$$

因此，每个 Query 分别得到一组权重；同一个 Key 可以被多个 Query 读取。图 2-2 中，从 Q、K 开始的路径产生权重，V 路径提供被加权的内容。

<figure align="center">
<img src="https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png" alt="原论文 Figure 2 左图：MatMul、Scale、Mask、Softmax 后与 V 相乘">
<figcaption><p align="center">图 2-2　Scaled Dot-Product Attention</p></figcaption>
</figure>

这里的权重是对输入位置的分布，而非词表概率。上述归一化性质针对 attention dropout 之前的 $$A$$；训练中若对权重施加 dropout，单次采样后的行和通常不再恰为 1。[D2L，§11.3.2—11.3.3](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)

## 缩放因子控制分数的尺度

原论文用一个简化假设解释 $$1/\sqrt{d_k}$$：若 $$q$$、$$k$$ 的各分量相互独立，均值为 0、方差为 1，则各乘积的方差为 1，因而

$$
\operatorname{Var}(q^\top k)
=\sum_{r=1}^{d_k}\operatorname{Var}(q_rk_r)=d_k,
\qquad
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)=1.
$$

缩放使这组假设下的分数方差不随维度增长。过大的分数差异会使 Softmax 接近集中在少数位置，部分梯度变得很小。实际训练后的 QKV 不必满足上述独立同分布条件，这一推导解释的是缩放的设计动机。[Vaswani et al.，§3.2.1 及脚注](https://arxiv.org/html/1706.03762v7)；[D2L，§11.3.1](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)

计算 Softmax 时，可先从每行有效分数中减去最大值，再取指数。给整行减去同一常数不改变归一化结果，且能避免大正数的指数溢出。[Dao et al.，§3.1](https://arxiv.org/html/2205.14135v2)

## 一个完整的单 Query 算例

取 $$d_k=1$$，并人为设定：

$$
Q=[1],\qquad
K=\begin{bmatrix}0\\ \ln 2\\ \ln 7\end{bmatrix},\qquad
V=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.
$$

不加 Mask 时，缩放后的分数为 $$(0,\ln2,\ln7)$$，指数为 $$(1,2,7)$$，所以：

$$
A=(0.1,0.2,0.7),\qquad O=(0.8,0.9).
$$

如果屏蔽第三个 Key，前两个位置重新归一化，权重成为 $$(1/3,2/3,0)$$，输出成为 $$(1/3,2/3)$$。Mask 会改变有效位置的归一化分母，而不只是把旧权重中的一项改成零。本例由仓库中的 `scripts/verify_transformer_attention.py` 实际核算。

## 配合视频观察计算

3Blue1Brown 的 [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=392s) 可配合本节观看：

| 时间段 | 观察重点 |
| --- | --- |
| 06:32—10:40 | Query、Key、点积与归一化权重 |
| 11:30—12:45 | Causal Mask 对可访问位置的限制 |
| 13:40—15:10 | Value 的加权组合 |
| 20:30—22:30 | 多个 Head 如何并行工作 |

<p align="center">表 2-4　QKV 视频阅读指引</p>

视频的示意图使用了自己的向量排列方式；本书统一按 Token 占据矩阵行来书写，Softmax 沿 Key 维度归一化。视频及[作者配套图文](https://www.3blue1brown.com/lessons/attention/)用于理解计算过程，精确定义以上述公式为准。时间段依据[作者字幕](https://github.com/3b1b/captions/blob/main/2024/attention/english/captions.srt)核对。

图示来源与许可见[图表来源](../../appendices/references.md#图表来源)。
下一节：[2.3 Self-Attention、Cross-Attention 与 Causal Attention](attention-and-masks.md)。
