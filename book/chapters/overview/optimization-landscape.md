# 1.4 Attention 的主要改进方向与方法关系

长输入增加位置对计算，逐步生成需要读取历史 KV，而更长的上下文还要求模型有效处理位置与相关信息。围绕这些问题，Attention 的改进形成了多条路线：有的改变模型读取和保存信息的方式，有的优化同一计算在硬件上的执行。

<figure align="center">
<img src="../../assets/overview/attention-landscape.svg" alt="Attention 技术路线图，按表示、连接、实现、缓存与状态更新划分">
<figcaption><p align="center">图 1-12　Attention 技术路线</p></figcaption>
</figure>

## 按设计问题定位方法

| 设计问题 | 代表路线 | 主要改变 | 后续章节 |
| --- | --- | --- | --- |
| 位置如何影响计算 | RoPE、ALiBi、Context Extension | 位置相关表示、分数或使用范围 | [第 3 章 Position Encoding 的演进与 Context Extension](../position-encoding/README.md) |
| KV 如何表示 | MQA、GQA、MLA | Head 共享或 Latent 表示 | [第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA](../kv-representations/README.md) |
| Attention 输出如何调节 | Gated Attention | 在特定计算位置引入 Gate | [第 5 章 Softmax Attention 的结构改进：Gated Attention](../gated-attention/README.md) |
| 运算怎样在硬件上执行 | FlashAttention | 分块、在线归一化及中间数据管理 | [第 6 章 FlashAttention：Exact Attention 的 IO 优化](../flash-attention/README.md) |
| 哪些位置之间建立连接 | Sparse Attention | 允许或选择参与计算的位置对 | [第 7 章 Sparse Attention 的连接模式与动态选择](../sparse-attention/README.md) |
| 历史 KV 怎样存储和复用 | PagedAttention、量化、淘汰、共享 | 布局、精度、保留范围或复用方式 | [第 8 章 KV Cache 的存储管理与推理优化](../kv-cache/README.md) |
| 历史怎样进入状态 | Linear Attention、Gated DeltaNet | 用状态更新与读取组织计算 | [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](../linear-attention/README.md) |
| 不同序列模块怎样组合 | Hybrid Architectures | Layer 或模块的组合 | [第 10 章 Hybrid Architectures 的结构设计与方法比较](../hybrid-architectures/README.md) |

<p align="center">表 1-5　Attention 优化路线</p>

## 模型结构、计算实现与缓存管理

一个 Attention Head 独立执行一组匹配与汇总，Multi-Head Attention 并行执行多组这样的计算。以 KV 成本为例，Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA）让多个 Query Head 共享 KV；Multi-Head Latent Attention（MLA）使用压缩后的 Latent 表示。它们改变模型中的 KV 组织方式。PagedAttention 则通过分页管理已有 KV 的物理存储，减少内存碎片并支持共享。[Ainslie et al.，§2.2](https://arxiv.org/html/2305.13245v3)；[DeepSeek-AI，§2.1](https://arxiv.org/html/2405.04434v5)；[Kwon et al.，§4.1—4.2](https://arxiv.org/html/2309.06180v1)

计算优化也有不同层次。FlashAttention 保持目标 Attention 运算，通过分块与中间数据管理减少显存读写；Sparse Attention 选择部分位置参与计算，从连接数量上降低成本。前者需要考察硬件执行效率，后者还需要评估被省略的信息对模型能力的影响。[Dao et al.，§3](https://arxiv.org/html/2205.14135v2)；[Child et al.，§4](https://arxiv.org/html/1904.10509v1)

Linear Attention 进一步改变历史信息的组织方式：在适用的核形式下，将历史累积到状态中，逐步更新并读取。固定状态维度时，推理状态大小可以保持不随历史长度增长；状态如何保存和读取所需信息，成为这一方向的重要设计问题。Hybrid Architectures 则在不同层或模块组合 Attention 与其他序列运算。[Katharopoulos et al.，§3.2—3.4](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)；[Dao & Gu，§9.2.3](https://arxiv.org/html/2405.21060v1)

## 不同改进方向的组合

不同方法可以作用于同一模型的不同部分。例如，Mistral 7B 同时采用 GQA 与 Sliding Window Attention：GQA 减少 KV Head 数，滑动窗口限制每层直接读取的位置范围。[Jiang et al.，§2](https://arxiv.org/html/2310.06825v1)

| 组合方向 | 各自改变的对象 | 配合使用的条件 |
| --- | --- | --- |
| KV 共享与 Sparse Attention | KV Head 组织、可访问的位置 | 计算实现支持相应 Head 分组和 Mask |
| FlashAttention 与 KV 分页管理 | 运算调度、缓存布局 | Kernel 支持对应缓存布局及推理阶段 |
| Attention 与状态更新模块 | 不同层或模块的信息读取方式 | 协调层间表示、训练与推理状态 |

<p align="center">表 1-6　不同改进方向的组合</p>

技术时间线及原始论文入口见[附录 B 技术演进时间线与方法对照表](../../appendices/timeline.md)。结合上面的设计对象，可以辨认同期工作解决的不同问题，以及后续模型采用的组合。

## 方法比较的对象与条件

比较方法时，先明确关注的是模型能力、Attention 运算、单次生成还是完整服务系统，再选择相应指标。

| 比较对象 | 主要指标 | 需要保持清楚的条件 |
| --- | --- | --- |
| 模型能力 | 任务质量、检索与长文本表现 | 数据、训练预算、输入长度与相关信息位置 |
| Prefill 与 Decode | 耗时、每步延迟、峰值显存 | 输入与输出长度、Batch Size、硬件与精度 |
| 服务系统 | 吞吐、并发与延迟 | 请求分布、缓存策略和延迟约束 |

<p align="center">表 1-7　方法比较的基本条件</p>

FlashAttention 论文同时评估 Attention 实现与训练性能，PagedAttention 论文评估服务系统中的吞吐等指标。阅读这些结果时，应将速度或内存收益与其测量范围对应起来。[Dao et al.，§4](https://arxiv.org/html/2205.14135v2)；[Kwon et al.，§6](https://arxiv.org/html/2309.06180v1) 详细比较检查表见[方法比较的条件与指标](../../appendices/timeline.md#方法比较的条件与指标)。

图示来源与许可见[图表来源](../../appendices/references.md#图表来源)。

下一章：[第 2 章 Transformer Attention 的基本原理与计算过程](../transformer-attention/README.md)。
