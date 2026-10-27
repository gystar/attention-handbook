# 4.5 KV 表示的质量与效率比较

比较 KV 表示，需要分别检查保存多少数据、执行哪些运算，以及模型在指定任务上的表现。缓存容量可由张量形状计算；质量与速度则需要训练和测量条件。

## 相同口径下的缓存容量

沿用本章符号，假设各层配置相同、每个请求保留全部 $$N$$ 个位置、缓存每元素占 $$s$$ 字节。下表只统计 KV 数据，不含分页预留、元数据、临时张量或跨设备副本。

| 结构 | 每层每个位置保存的元素数 | 全模型缓存字节数 |
| --- | --- | --- |
| MHA | $$2h_qd_h$$ | $$2LBNh_qd_hs$$ |
| GQA | $$2h_{kv}d_h$$ | $$2LBNh_{kv}d_hs$$ |
| MQA | $$2d_h$$ | $$2LBNd_hs$$ |
| MLA 的 Latent 缓存 | $$d_c+d_r$$ | $$LBN(d_c+d_r)s$$ |

<p align="center">表 4-9　KV 表示的统一容量口径</p>

在这些条件下，MLA 的缓存比 GQA 小，当且仅当 $$d_c+d_r<2h_{kv}d_h$$。这个不等式直接来自元素计数，说明方法名称本身并不能确定容量排序。

例如，取一组教学配置 $$d_h=128$$、$$d_c=512$$、$$d_r=64$$。MLA 每层每个位置保存 $$576$$ 个元素；GQA 使用两组 KV 时保存 $$512$$ 个，使用四组时保存 $$1024$$ 个。该算例只比较容量，没有指定具有相同质量的模型。

## 表示约束与训练适配

| 改变 | 对模型的约束 | 需要核查的质量问题 |
| --- | --- | --- |
| MHA → MQA/GQA | 同组 Head 使用相同的 K/V 投影 | 原 Checkpoint 转换后的质量、适配训练预算 |
| 引入 MLA | 多个 Head 的内容 K/V 通过共同 Latent 生成 | Latent 维度、投影和位置分支的联合训练效果 |
| 同一 MLA 模型重排计算 | 参数及数学运算保持一致 | 浮点误差、Mask、缩放与位置处理是否一致 |

<p align="center">表 4-10　模型变化与实现变化</p>

GQA 原论文将 T5.1.1-XXL 转换为 GQA-8，追加原预训练步数 5% 的适配训练，再进行下游任务微调。在这些实验中，GQA-8 的平均任务表现接近 MHA-XXL，推理速度接近 MQA-XXL。计时使用 8 个 TPUv4，分别优化各模型的并行配置，并选择每个 TPU 不超过 32 的最大可容纳 batch。[Ainslie et al.，§3.1—3.2、Table 1](https://arxiv.org/html/2305.13245v3)

比较另一组模型时，应固定或记录模型规模、训练 Token 数、数据、适配步骤和评价任务。改变 KV 结构时，投影参数量也可能变化；“相同模型名称”或“相同 Head 数”不足以确定公平的比较口径。

## 通道压缩与序列压缩

MLA 为每个历史位置保存较小的表示，位置数仍为 $$N$$。Linformer 则沿序列维度投影，将 $$N$$ 个 KV 向量混合成 $$r$$ 个条目。对单个 Head，令 $$Q,K\in\mathbb R^{N\times d_h}$$、$$V\in\mathbb R^{N\times d_v}$$，使用 $$E,F\in\mathbb R^{r\times N}$$：

$$
\widetilde K=EK,\qquad \widetilde V=FV,\qquad
O=\operatorname{softmax}\!\left(\frac{Q\widetilde K^\top}{\sqrt{d_h}}\right)\widetilde V.
$$

Attention 权重的形状由 $$N\times N$$ 变为 $$N\times r$$。投影与 Attention 的计算量合计为 $$O(Nr(d_h+d_v))$$；只有固定 $$r,d_h,d_v$$ 时，才对 $$N$$ 呈线性。该结构需要学习或指定序列投影，不是同一 Softmax 算子的等价重排。[Wang et al.，§4，式 (7)](https://arxiv.org/html/2006.04768v3#S4)

三类方法压缩的对象不同：

- **MLA**：压缩每个位置的 KV 通道，保留逐位置读取。
- **Linformer**：沿位置维度混合 KV，Query 读取压缩条目。
- **Kernel Attention**：通过特征分解汇总 KV 外积，Query 读取状态，见[9.1 从 Softmax Attention 到 Kernel Attention](../linear-attention/kernel-attention.md)。

Causal 使用还需检查投影的信息来源。如果某个压缩条目混入未来位置，只对压缩后的分数施加 Mask 无法分离其中的未来贡献。例如，令 $$N=2,r=1,F=(1/2,1/2)$$，唯一 Value 条目为 $$(v_1+v_2)/2$$；第一个 Query 只要读取它，输出就依赖 $$v_2$$。这是由上式构造的反例。按前缀构造压缩条目可以约束来源，但需要相应的状态更新与计算设计；原论文的预训练实验采用 Masked Language Modeling。[§5.1](https://arxiv.org/html/2006.04768v3#S5.SS1)

## 从容量收益到执行性能

对标准 GQA/MQA 的逐 Head 计算，缩小 KV Head 数主要减少缓存和 K/V 投影，Query 对历史位置的匹配仍保留。MLA 的 Latent 路径则改变了点积与汇总的内部维度，还包含当前 Query 的变换和输出映射。其成本应按实际执行路径计算。

| 测量对象 | 应同时记录的条件 |
| --- | --- |
| KV 数据容量 | 缓存形状、dtype、有效长度、复制份数 |
| 峰值显存 | 模型权重、临时展开张量、工作区、缓存分配 |
| Prefill 时间 | 输入长度、batch、是否展开 K/V、kernel |
| Decode 每步时间 | 历史长度、并发请求、Latent/展开路径、并行策略 |
| 模型质量 | Checkpoint、训练与适配数据、任务、长度及指标 |

<p align="center">表 4-11　KV 表示的评估条件</p>

Prefill 与 Decode 分开测量，有助于判断收益来自哪段计算。若比较最大吞吐，还要同时报告 batch、延迟和显存约束；更小的缓存可能让系统容纳更多请求，这与固定 batch 下单步更快是不同的测量结果。

本章改变了历史表示及其读取计算；得到 Head 输出后，还可以调节它向后续网络传递的信息，见[第 5 章 Gated Attention 的结构与作用](../gated-attention/README.md)。缓存的分配、量化、淘汰和复用见[第 8 章 KV Cache 的存储管理与推理优化](../kv-cache/README.md)。
