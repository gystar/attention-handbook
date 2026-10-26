# 第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA

> 状态：已迁入部分基础内容，完整章节待写。

4.1 Key–Value 表示的设计目标与优化路线

4.2 MHA 的 Key–Value Projections 与缓存成本

4.3 MQA 与 GQA：Key–Value Sharing

4.4 MLA：Latent Compression

4.5 模型质量、KV Cache 成本与实现复杂度的比较

## KV 表示与 Head 共享

MQA 让多个 Query Head 共用一组 Key 和 Value；GQA 为每组 Query Head 分配一组 KV。两者改变模型的 Head 组织。[Shazeer, 2019，§3](https://arxiv.org/html/1911.02150v1)；[Ainslie et al., 2023，§2.2](https://arxiv.org/html/2305.13245v3)

<figure><img src="https://arxiv.org/html/2305.13245v3/images/gmq_architecture.png" alt="GQA 原论文图 2：MHA 的 Query 与 KV 一一对应，GQA 分组共享 KV，MQA 全部共享一组 KV"><figcaption><p align="center">图 4-1　MHA、GQA 与 MQA</p></figcaption></figure>

图中 Query Head 数保持一致，KV Head 数随共享方式变化；各 Query Head 仍分别执行 Attention 计算。[Ainslie et al.，Figure 2](https://arxiv.org/html/2305.13245v3)

MLA 联合压缩 Key 与 Value 的表示，推理时缓存压缩后的 Latent 及必要的位置相关信息。它与 Head 共享的实现方式不同。[DeepSeek-AI, 2024，§2.1](https://arxiv.org/html/2405.04434v5)

## KV Cache 的内容与容量

令 $$B$$ 为 Batch Size，$$N$$ 为序列长度，$$h_q$$ 为 Query Head 数，$$d_h$$ 为每个 Head 的维度，假设 Key 与 Value 的维度均为 $$d_h$$。

对于固定参数、固定前缀的 Causal Transformer，已经算出的历史 KV 可以在后续步骤复用。KV Cache 按层保存这些 Key 与 Value，减少重复计算；它不同于模型参数，也不同于本步骤的 Attention 权重矩阵。[Kwon et al., 2023，§2.2](https://arxiv.org/html/2309.06180v1)

假设模型共有 $$L$$ 层，每层保留全部 $$N$$ 个 Token，各请求长度相同，实际存储 $$h_{kv}$$ 组 KV，且每个元素占 $$s$$ 字节，则：

$$
\text{KV Cache bytes}=2LBNh_{kv}d_hs.
$$

系数 $$2$$ 对应 Key 和 Value。该式由张量元素数直接相乘得到，不包括分页预留、元数据、量化参数、跨设备复制或其他模型内存；不同层采用不同配置时应逐层求和。

以下为假设配置的容量算例：$$L=32$$、$$B=1$$、$$N=8192$$、$$h_q=32$$、$$d_h=128$$，KV 使用每元素 $$2$$ 字节。

| KV 组织                        | $$h_{kv}$$ | KV 数据容量 |
| ---------------------------- | ---------: | ------: |
| Multi-Head Attention（MHA）    |         32 |   4 GiB |
| Grouped-Query Attention（GQA） |          8 |   1 GiB |
| Multi-Query Attention（MQA）   |          1 | 128 MiB |

<p align="center">表 4-1　KV Cache 的内容与容量</p>

其中 $$1$$ GiB 为 $$2^{30}$$ 字节，$$1$$ MiB 为 $$2^{20}$$ 字节。这是根据缓存形状与数据类型得到的理论容量。GQA/MQA 通过共享 KV Head 改变这一项成本；实际推理加速还取决于计算、访存与实现。[Ainslie et al., 2023，§2.2](https://arxiv.org/html/2305.13245v3)

Multi-Head Latent Attention（MLA）采用另一种缓存表示，其容量需根据 Latent 表示与位置相关的缓存项单独计算。[DeepSeek-AI, 2024，§2.1.2—2.1.4](https://arxiv.org/html/2405.04434v5)

图示来源与许可见[图表来源](../fu-lu/references.md#图表来源)。
