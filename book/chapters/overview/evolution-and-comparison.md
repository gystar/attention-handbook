# 1.6 主要技术路线的演进关系与比较维度

Attention 的发展包含多条并行路线。时间线说明工作出现的先后；比较方法时还需确定它们修改的对象，以及是否在相同前提下评估。

## 代表性时间节点

下表按代表性工作的首次公开年份排列，覆盖本章使用的主要例子。预印本年份与正式发表、后续修订年份可能不同；具体使用版本见[附录 C 参考文献、教学资源与本地源文件索引](../../appendices/references.md)。

| 年份 | 代表工作 | 在本书主线中的位置 |
| --- | --- | --- |
| 2014 | [Seq2Seq](https://arxiv.org/abs/1409.3215)、[Bahdanau Attention](https://arxiv.org/abs/1409.0473) | 固定表示与按生成步骤动态读取的 Encoder–Decoder 设计 |
| 2017 | [Transformer](https://arxiv.org/abs/1706.03762) | Self-Attention 成为序列表示计算的核心模块 |
| 2019 | [Sparse Transformer](https://arxiv.org/abs/1904.10509)、[MQA](https://arxiv.org/abs/1911.02150) | 分别调整连接模式与 KV Head 组织 |
| 2020 | [Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html) | 特征映射与 Recurrent Formulation |
| 2021 | [RoPE](https://arxiv.org/abs/2104.09864)、[ALiBi](https://arxiv.org/abs/2108.12409) | 位置信息进入 Attention 的不同方式 |
| 2022 | [FlashAttention](https://arxiv.org/abs/2205.14135) | Exact Attention 的 IO 优化 |
| 2023 | [GQA](https://arxiv.org/abs/2305.13245)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[Position Interpolation](https://arxiv.org/abs/2306.15595) | KV 共享、缓存管理和上下文扩展分别推进 |
| 2024 | [MLA](https://arxiv.org/abs/2405.04434)、[Mamba-2/SSD](https://arxiv.org/abs/2405.21060)、[Gated DeltaNet](https://arxiv.org/abs/2412.06464) | Latent 压缩、结构化状态计算与状态更新 |
| 2025 | [Native Sparse Attention](https://arxiv.org/abs/2502.11089)、[Gated Attention](https://arxiv.org/abs/2505.06708) | 硬件适配的稀疏选择，以及对 Softmax Attention 中 Gating 位置的系统研究 |

<p align="center">表 1-13　代表性时间节点</p>

表中 2025 年的 Gated Attention 条目对应该论文研究的具体门控设计，Gating 概念的使用早于这一工作。

## 可以组合的方法与需要验证的接口

Mistral 7B 的原始技术报告同时采用 GQA 与 Sliding Window Attention：前者改变 KV Head 的组织，后者限制每层直接访问的位置。这是不同优化维度组合的例子。[Jiang et al., 2023，§2](https://arxiv.org/html/2310.06825v1)

| 组合 | 分别处理的问题 | 需要检查的接口 |
| --- | --- | --- |
| GQA + Sparse Attention | KV 共享与位置选择 | Head 分组、Mask 和 Kernel 支持 |
| FlashAttention + KV 分页管理 | Attention 计算与 KV 布局 | Kernel 是否支持对应的缓存布局与运行阶段 |
| Position Encoding + Context Extension | 位置表达与更长范围的适配 | 位置索引、训练长度、精度及长文本评估 |
| Attention + Recurrent/SSM 模块 | 不同方式的历史信息读取 | 层间表示、状态维护、训练与推理实现 |

<p align="center">表 1-14　方法组合与接口</p>

组合上述方法时，需要核查表中列出的接口。将既有模型检查点（Checkpoint）转换为 MQA/GQA/MLA 等结构时，还需评估结构转换、训练适应与实现支持。[Ainslie et al., 2023，§2.1](https://arxiv.org/html/2305.13245v3)

## 方法比较的条件与指标

| 比较维度 | 必要报告项 | 常见误判 |
| --- | --- | --- |
| 数学目标 | 运算定义、Mask、精度与是否改变模型结构 | 将实现优化与算子改变混为一谈 |
| 模型能力 | 任务、数据、指标、基线与训练预算 | 用单个任务概括所有能力 |
| Prefill | Prompt 长度、Batch、耗时和峰值显存 | 用短输入结果推断长输入表现 |
| Decode | 历史长度、输出长度、Batch 与每步延迟 | 将 KV 容量缩小倍数等同于速度提升 |
| 服务性能 | 并发、请求分布、吞吐与延迟约束 | 将单请求延迟与服务吞吐直接比较 |
| 长上下文 | 长度范围、相关信息位置及任务类型 | 将可接受的窗口长度等同于有效利用能力 |
| 实现条件 | 硬件、软件版本、Kernel、数据类型 | 忽略理论复杂度与实际执行之间的差别 |

<p align="center">表 1-15　方法比较的条件与指标</p>

例如，报告“更快”之前，需要说明是 Attention Kernel、整层、单次请求还是完整服务系统更快。FlashAttention 与 PagedAttention 论文分别考察了运算实现和服务系统层面的成本，评估对象不同。[Dao et al., 2022，§4](https://arxiv.org/html/2205.14135v2)；[Kwon et al., 2023，§6](https://arxiv.org/html/2309.06180v1)

## 按问题进入后续章节

| 当前问题 | 阅读位置 |
| --- | --- |
| QKV、Mask 或 Multi-Head 的计算还不清楚 | [第 2 章 Transformer Attention 的基本原理与计算过程](../transformer-attention/README.md) |
| 位置处理与长度扩展如何影响模型 | [第 3 章 Position Encoding 的演进与 Context Extension](../position-encoding/README.md) |
| KV Head、Latent 表示与 Gate 分别怎样设计 | [第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA](../kv-representations/README.md)、[第 5 章 Softmax Attention 的结构改进：Gated Attention](../gated-attention/README.md) |
| 为什么显存访问、连接模式与缓存管理要分开分析 | [第 6 章 FlashAttention：Exact Attention 的 IO 优化](../flash-attention/README.md)—[第 8 章 KV Cache 的存储管理与推理优化](../kv-cache/README.md) |
| 用状态替代逐位置历史读取会改变什么 | [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](../linear-attention/README.md) |
| 怎样组合不同序列模块并比较整体收益 | [第 10 章 Hybrid Architectures 的结构设计与方法比较](../hybrid-architectures/README.md) |

<p align="center">表 1-16　后续章节阅读指引</p>
