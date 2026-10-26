# 4.1 KV 表示与缓存成本

自回归生成每处理一个新位置，都会向各层缓存追加 KV；随后的 Query 再读取这些历史表示。第二章解释了复用的计算条件，本节考察缓存中保存哪些张量，以及它们如何决定容量与访问成本。

## 从 Head 数到缓存形状

令 $$B$$ 为 Batch Size，$$N$$ 为本步骤可访问的位置数（包含当前位置），$$h_q$$ 为 Query Head 数，$$h_{kv}$$ 为 KV Head 数，$$d_h$$ 为每个 Head 的维度，假设 Key 与 Value 的维度均为 $$d_h$$。

在确定性因果计算下，若模型参数、输入前缀、位置处理规则与计算配置保持一致，Causal Transformer 已经算出的历史 KV 可以在后续步骤复用。KV Cache 按层保存这些 Key 与 Value，减少重复计算。[Kwon et al., 2023，§2.2](https://arxiv.org/html/2309.06180v1)

若各请求均已缓存 $$N$$ 个位置，一层 KV 可按下面的逻辑形状表示；实际物理布局可以不同。

| 张量 | 逻辑形状 | Head 维度的作用 |
| --- | --- | --- |
| 本步骤 Query | $$[B,h_q,d_h]$$ | 每个 Query Head 独立匹配历史 |
| 历史 Key | $$[B,h_{kv},N,d_h]$$ | 每组 KV 提供一套匹配表示 |
| 历史 Value | $$[B,h_{kv},N,d_h]$$ | 每组 KV 提供一套待汇总表示 |

<p align="center">表 4-1　增量 Attention 的张量形状</p>

MHA 中 $$h_{kv}=h_q$$。MQA 令 $$h_{kv}=1$$，GQA 使用若干组 KV；Query Head 的数量可以保持不变。[Shazeer，§2.4、§3](https://arxiv.org/html/1911.02150v1)；[Ainslie et al.，§2.2](https://arxiv.org/html/2305.13245v3)

## 统一容量口径

假设模型共有 $$L$$ 层，每层保留全部 $$N$$ 个 Token，各请求长度相同，实际存储 $$h_{kv}$$ 组 KV，且每个元素占 $$s$$ 字节，则：

$$
\text{KV Cache bytes}=2LBNh_{kv}d_hs.
$$

系数 $$2$$ 对应 Key 和 Value。该式由张量元素数直接相乘得到，不包括分页预留、元数据、量化参数、跨设备复制或其他模型内存；不同层采用不同配置时应逐层求和。

以下为假设配置的容量算例：$$L=32$$、$$B=1$$、$$N=8192$$、$$h_q=32$$、$$d_h=128$$，KV 使用每元素 $$2$$ 字节。

| KV 组织 | $$h_{kv}$$ | KV 数据容量 |
| --- | ---: | ---: |
| Multi-Head Attention（MHA） | 32 | 4 GiB |
| Grouped-Query Attention（GQA） | 8 | 1 GiB |
| Multi-Query Attention（MQA） | 1 | 128 MiB |

<p align="center">表 4-2　不同 KV Head 数的理论容量</p>

其中 $$1$$ GiB 为 $$2^{30}$$ 字节，$$1$$ MiB 为 $$2^{20}$$ 字节。这是根据缓存形状与数据类型得到的理论容量。GQA/MQA 通过共享 KV Head 改变这一项成本；实际推理加速还取决于计算、访存与实现。[Ainslie et al., 2023，§2.2](https://arxiv.org/html/2305.13245v3)

Multi-Head Latent Attention（MLA）采用另一种缓存表示，其容量需根据 Latent 表示与位置相关的缓存项单独计算。[DeepSeek-AI, 2024，§2.1.2—2.1.4](https://arxiv.org/html/2405.04434v5)


## 缓存容量与计算量

令 $$A$$ 为 Softmax 后的 Attention 权重。对于一个新 Query 位置，忽略 Mask、Softmax 和投影开销，将一次乘加记为两个 FLOPs，逐 Head 的两个主要运算为：

$$
\begin{aligned}
QK^{\mathsf T}&:\quad 2Bh_qNd_h,\\
AV&:\quad 2Bh_qNd_h.
\end{aligned}
$$

上述计数直接来自张量维度。共享 KV 后，每个 Query Head 仍需计算自己的分数和加权和，因此两项的常规逐 Head 运算量仍由 $$h_q$$ 决定；缓存元素数则由 $$h_{kv}$$ 决定。若 kernel 能在同组 Query Head 间复用已读取的 KV，减少缓存数据也能减少相应的显存访问。

| 量 | 单层、单步的计数口径 | 主要影响因素 |
| --- | --- | --- |
| 历史 KV 数据量 | $$2BNh_{kv}d_hs$$ 字节 | KV Head 数、长度、精度 |
| 两次主要矩阵运算 | 约 $$4Bh_qNd_h$$ FLOPs | Query Head 数、长度、Head 维度 |
| 新增缓存数据 | $$2Bh_{kv}d_hs$$ 字节 | 每次各请求追加一个位置 |

<p align="center">表 4-3　缓存数据与主要运算量</p>

这里的历史 KV 数据量是假设每个元素读取一次时的数据计数。实际显存流量还受缓存命中、重复加载、分块方式和跨设备复制影响；整模型延迟还包含投影、FFN 等计算。应分别报告缓存容量、算子耗时与端到端性能。[Shazeer，§2.4.1、§3.1](https://arxiv.org/html/1911.02150v1)
