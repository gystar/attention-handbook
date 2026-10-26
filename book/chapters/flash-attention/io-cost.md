# 6.1 Attention 的计算量与数据访问

第二章按矩阵运算解释了 Attention：计算分数、按行归一化，再汇总 Value。将这三步直接实现为独立的 GPU kernel，会产生另一个成本：中间矩阵需要写入显存，再由后续 kernel 读回。FlashAttention 将这些运算融合到分块计算中，使中间分数和权重在片上使用后即可释放。[原始论文 §3.1](https://arxiv.org/html/2205.14135v2#S3.SS1)

## 中间矩阵的存储成本

考虑一个 Head，序列长度为 $$N$$，Query、Key 和 Value 的维度均为 $$d_h$$，暂不使用 Mask 与 Dropout：

$$
Q,K,V\in\mathbb R^{N\times d_h},\qquad
S=\frac{QK^\top}{\sqrt{d_h}},\qquad
P=\operatorname{softmax}_{\mathrm{row}}(S),\qquad O=PV.
$$

按一次乘法与一次加法计为两个 FLOPs，两个矩阵乘法各需要约 $$2N^2d_h$$ FLOPs。Softmax 另需逐元素运算和行归约。因此，完整计算的主要算术成本仍随 $$N^2d_h$$ 增长。

存储需求由张量形状直接得到。以下为本书的容量算例：$$N=8192$$、$$d_h=128$$，假设每个张量元素占 2 bytes，且仅统计一个样本的一个 Head。

| 对象 | 形状 | 单个张量容量 |
| --- | --- | --- |
| Query、Key、Value 或输出 | $$N\times d_h$$ | 2 MiB |
| 分数或权重矩阵 | $$N\times N$$ | 128 MiB |

<p align="center">表 6-1　Attention 张量的容量</p>

一个完整分数矩阵的容量已是一个输入矩阵的 64 倍。若分数与权重都被保留，两者合计 256 MiB；实际峰值取决于 buffer 复用、数据类型和反向传播需要保留的内容。这里的容量不等于一次执行的总读写量，同一张量可能被多次访问。

## 分块计算减少中间结果读写

GPU 的 HBM（High Bandwidth Memory）保存较大的输入、输出及模型状态。计算过程中，数据通过缓存、Shared Memory 和 Registers 等片上存储供执行单元使用。分析 FlashAttention 时，可以先将其简化为“容量较大的 HBM”与“容量受限的片上工作区”两层；具体 kernel 还需考虑各级存储的带宽和访问方式。

| 执行方式 | 分数与权重的处理 | 主要工作区 |
| --- | --- | --- |
| 显式矩阵实现 | 写出完整 $$S$$、$$P$$，由后续运算读取 | 与 $$N^2$$ 成正比的中间张量 |
| 分块融合实现 | 计算一块分数，立即归一化并累加 Value | 当前块、部分输出与行统计量 |

<p align="center">表 6-2　Attention 中间结果的处理方式</p>

分块的难点在于 Softmax 的分母涉及整行：读取某一块时，后面的 Key 可能尚未参与计算。下一节将推导如何随块更新归一化统计量，最终恢复整行输出。

## IO 复杂度的适用条件

FlashAttention 原论文采用两层存储模型，以 $$M$$ 表示片上工作区能够容纳的**元素数量**。在 $$d_h\le M\le Nd_h$$ 的假设下，论文给出以下 HBM 元素访问量：

$$
\begin{aligned}
\text{显式矩阵实现}&:\quad \Theta(Nd_h+N^2),\\
\text{FlashAttention 原始分块算法}&:\quad
\Theta\!\left(\frac{N^2d_h^2}{M}\right).
\end{aligned}
$$

这两个结果来自论文 §3.2 的 Theorem 2；其分析对象是给定的存储模型与算法。换算字节数还需乘以元素大小，实际延迟则取决于硬件、块大小及调度。[原始论文 §3.2](https://arxiv.org/html/2205.14135v2#S3.SS2)

从式子可以直接看到，片上空间允许块内复用更多数据，从而减少 HBM 往返。与此同时，两个主要矩阵乘法仍需处理全部有效 Query–Key 配对。Causal Mask 可跳过未来位置对应的块，但有效配对数为 $$N(N+1)/2$$，随长度仍呈二次增长。
