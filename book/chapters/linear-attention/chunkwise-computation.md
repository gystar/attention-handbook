# 9.5 并行训练与分块计算

递推形式适合 Decode：每次处理一个新 Token，更新一次状态。训练和 Prefill 则已知整段输入，可以同时计算本层所有位置的 Query、Key、Value。此时，逐位置更新状态会限制序列维度上的并行度。Chunkwise Computation 将序列分块，让块内计算使用矩阵乘法，块间通过状态传递历史信息。

## 块内并行与块间状态

先考虑加法更新，并省略归一化分母。令一块包含 $$C$$ 个位置，按行堆叠得到 $$Q,K\in\mathbb R^{C\times d_k}$$、$$V\in\mathbb R^{C\times d_v}$$；$$S_{\mathrm{in}}\in\mathbb R^{d_k\times d_v}$$ 是进入该块前的状态。若使用特征映射，这里的 Q、K 指映射后的特征。

块内第 $$i$$ 个位置只能读取本块前 $$i$$ 次写入：

$$
o_i^\top=q_i^\top S_{\mathrm{in}}+\sum_{j=1}^{i}(q_i^\top k_j)v_j^\top.
$$

令 $$M_{ij}=1$$ 当且仅当 $$j\le i$$，则整块输出和离开该块的状态为：

$$
\begin{aligned}
O&=Q S_{\mathrm{in}}+\big((QK^\top)\odot M\big)V,\\
S_{\mathrm{out}}&=S_{\mathrm{in}}+K^\top V.
\end{aligned}
$$

| 计算项 | 包含的信息 | 张量形状 |
| --- | --- | --- |
| $$Q S_{\mathrm{in}}$$ | 当前块之前的历史 | $$C\times d_v$$ |
| $$((QK^\top)\odot M)V$$ | 本块中当前及更早的位置 | $$C\times d_v$$ |
| $$K^\top V$$ | 整块新增的状态 | $$d_k\times d_v$$ |

<p align="center">表 9-5　分块计算中的历史读取与局部读取</p>

块内无需为每个位置保存一份完整状态。得到各块的入口状态后，各块输出可独立计算；入口状态可以逐块递推，也可以按适合该更新规则的 Prefix Scan 计算。Prefix Scan 利用可结合的操作并行计算各个前缀的累计结果；加法状态对应前缀和，其他更新需要相应的状态变换组合。上述分解沿用 Linear Attention 的 Chunkwise Form，并统一为本章的状态形状。[Yang et al.，§2.1](https://arxiv.org/html/2406.06484v2#S2.SS1)

若保留 9.2 的核归一化，还要传递 $$z_{\mathrm{in}}$$。块内分母向量为：

$$
d=Qz_{\mathrm{in}}+\big((QK^\top)\odot M\big)\mathbf1,
\qquad z_{\mathrm{out}}=z_{\mathrm{in}}+K^\top\mathbf1.
$$

在分母为正的条件下，将 $$O$$ 的每一行除以对应的 $$d_i$$，才得到归一化输出。

## Chunk Size 决定计算与并行度的取舍

对长度 $$N$$ 的序列，基本加法分块公式的运算量为：

$$
O\!\left(Nd_kd_v+NC(d_k+d_v)\right).
$$

第一项来自状态读写，第二项来自块内配对和 Value 汇总；这里不含输入投影、特征映射和反向计算。固定维度与 $$C$$ 后，运算量随 $$N$$ 线性增长。$$C=1$$ 回到逐位置递推，$$C=N$$ 则形成全序列的配对计算。

增大 Chunk 可以扩大矩阵乘法规模、减少逐块递推次数，同时增加块内计算和工作空间。实际速度还取决于 Tensor Core 利用率、寄存器压力、HBM 读写和 Kernel 调度。训练也需要处理反向传播与中间量重计算；单步推理状态的大小不足以描述训练内存。

## GLA 与 DeltaNet 的分块更新

GLA 需要把每次写入到读取之间的累计 Gate 纳入计算。逐通道衰减的前缀乘积可能下溢，直接取前缀乘积之比还可能放大数值问题。GLA 论文采用分块及更细一级的块内划分，协调衰减计算的稳定性与标准矩阵乘法的使用。[§3.2—3.3](https://arxiv.org/html/2312.06635v3#S3.SS3)

DeltaNet 的写入依赖已有状态，不能直接把 Value 代入加法公式。为看清依赖关系，把一块中的修正量记为 $$r_i=\beta_i(v_i-S_{i-1}^\top k_i)$$，并按行堆叠为 $$R$$。由 $$S_{i-1}=S_{\mathrm{in}}+\sum_{j<i}k_jr_j^\top$$ 可得：

$$
r_i^\top=\beta_i\left(v_i^\top-k_i^\top S_{\mathrm{in}}-\sum_{j<i}(k_i^\top k_j)r_j^\top\right).
$$

令 $$B=\operatorname{diag}(\beta_1,\ldots,\beta_C)$$，$$L=\operatorname{tril}(KK^\top,-1)$$ 为严格下三角部分，便得到：

$$
(I+BL)R=B(V-KS_{\mathrm{in}}).
$$

这是单位下三角线性系统。求得修正量后，输出与状态更新为：

$$
O=QS_{\mathrm{in}}+\big((QK^\top)\odot M\big)R,
\qquad S_{\mathrm{out}}=S_{\mathrm{in}}+K^\top R.
$$

这里的展开用于说明块内依赖如何集中到修正量中，实际求解无需显式构造矩阵逆。DeltaNet 论文进一步利用 WY Representation 紧凑表达状态变换的乘积，实现分块计算，避免逐 Token 保存完整矩阵状态。[§3.1—3.2](https://arxiv.org/html/2406.06484v2#S3.SS2)

以上 DeltaNet 公式对应无额外 Forget Gate 的更新。Gated DeltaNet 还需把累计衰减纳入块内修正与块间状态传递；判断实现是否等价，应将其输出和最终状态与逐步递推分别对照。
