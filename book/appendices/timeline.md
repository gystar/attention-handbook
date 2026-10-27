# 附录 B 技术演进时间线与方法对照表

正文按问题与计算对象组织，时间线则补充各项工作的公开顺序。下面先列代表性节点，再按改变的对象查找专题，最后列出比较条件；同年出现的方法也可能处理不同层次的问题。

## 代表性时间节点

下表按代表性工作的首次公开年份排列，对应各专题中的主要方法。预印本年份与正式发表、后续修订年份可能不同；具体使用版本见[附录 C 参考文献、教学资源与本地源文件索引](references.md)。

| 年份 | 代表工作 | 在本书主线中的位置 |
| --- | --- | --- |
| 2014 | [Seq2Seq](https://arxiv.org/abs/1409.3215)、[Bahdanau Attention](https://arxiv.org/abs/1409.0473) | 固定表示与按生成步骤动态读取的 Encoder–Decoder 设计 |
| 2017 | [Transformer](https://arxiv.org/abs/1706.03762) | Self-Attention 成为序列表示计算的核心模块 |
| 2019 | [Sparse Transformer](https://arxiv.org/abs/1904.10509)、[MQA](https://arxiv.org/abs/1911.02150) | 分别调整连接模式与 KV Head 组织 |
| 2020 | [Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html) | 特征映射与 Recurrent Formulation |
| 2021 | [RoPE](https://arxiv.org/abs/2104.09864)、[ALiBi](https://arxiv.org/abs/2108.12409) | 位置信息进入 Attention 的不同方式 |
| 2022 | [FlashAttention](https://arxiv.org/abs/2205.14135) | Exact Attention 的 IO 优化 |
| 2023 | [GQA](https://arxiv.org/abs/2305.13245)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[Position Interpolation](https://arxiv.org/abs/2306.15595) | KV 共享、缓存管理和上下文扩展分别推进 |
| 2024 | [Griffin](https://arxiv.org/abs/2402.19427)、[MLA](https://arxiv.org/abs/2405.04434)、[Mamba-2/SSD](https://arxiv.org/abs/2405.21060)、[Gated DeltaNet](https://arxiv.org/abs/2412.06464) | 递推与 Local Attention 的组合、Latent 压缩、结构化状态计算与状态更新 |
| 2025 | [Native Sparse Attention](https://arxiv.org/abs/2502.11089)、[Gated Attention](https://arxiv.org/abs/2505.06708) | 硬件适配的稀疏选择，以及对 Softmax Attention 中 Gating 位置的系统研究 |
| 2026 | [FlashAttention-4](https://arxiv.org/abs/2603.05451v1)、[DeepSeek-V4](https://arxiv.org/abs/2606.19348v1) | 分别研究 Blackwell 上的执行优化，以及 CSA/HCA 的序列压缩与历史读取 |

<p align="center">表 B-1　代表性时间节点</p>

表中 2025 年的 Gated Attention 条目对应该论文研究的具体门控设计，Gating 概念的使用早于这一工作。

## 按改变的对象查找

| 改变的对象 | 代表方法 | 主要比较问题 | 专题 |
| --- | --- | --- | --- |
| 位置表示与距离信息 | RoPE、ALiBi、Position Interpolation | 使用范围如何变化，长文本能力怎样评估 | [第 3 章 Position Encoding 的演进与 Context Extension](../chapters/position-encoding/README.md) |
| KV Head 或表示 | MQA、GQA、MLA | 缓存保存什么，能力与容量如何权衡 | [第 4 章 MQA、GQA 与 MLA 的 KV 表示优化](../chapters/kv-representations/README.md) |
| Attention 内部的 Gate | Gated Attention | Gate 的输入、作用位置与实验依据 | [第 5 章 Gated Attention 的结构与作用](../chapters/gated-attention/README.md) |
| 运算调度与中间存储 | FlashAttention | 目标算子是否相同，IO 减少多少 | [第 6 章 FlashAttention 的计算与 IO 优化](../chapters/flash-attention/README.md) |
| 可访问的位置对 | Sparse Attention | 选择开销、实际稀疏度与质量变化 | [第 7 章 Sparse Attention 的连接模式与动态选择](../chapters/sparse-attention/README.md) |
| 压缩历史条目与选择 | DeepSeek-V4 的 CSA/HCA | 压缩率、因果边界、选择与全量读取的成本 | [7.5 DeepSeek-V4 的 CSA 与 HCA](../chapters/sparse-attention/compressed-attention.md) |
| KV 的物理存储和保留 | PagedAttention、KV 量化与淘汰 | 分别改变布局、精度还是历史覆盖 | [第 8 章 KV Cache 的存储管理与推理优化](../chapters/kv-cache/README.md) |
| 状态更新与读取 | Linear Attention、Gated DeltaNet | 状态容量、计算方式与信息读取能力 | [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](../chapters/linear-attention/README.md) |
| 模块组合 | Full–Sparse Attention、Attention 与 Linear Attention 或 SSM | 配置、训练及整模型表现 | [第 10 章 Hybrid Architectures 的结构设计与方法比较](../chapters/hybrid-architectures/README.md) |
| FFN 参数的稀疏激活 | MoE 与 Expert Routing | 选择依据、输出加权、负载及通信；与序列模块分别统计 | [10.6 MoE 与 Expert Routing](../chapters/hybrid-architectures/moe-routing.md) |

<p align="center">表 B-2　按改变的对象查找</p>

各路线的机制区别与组合关系见 [1.4 Attention 机制的主要发展方向与技术路线](../chapters/overview/optimization-landscape.md)。性能比较需同时指定阶段、长度、Batch、硬件、精度及能力指标。

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

<p align="center">表 B-3　方法比较的条件与指标</p>

例如，报告“更快”之前，需要说明是 Attention Kernel、整层、单次请求还是完整服务系统更快。FlashAttention 论文同时报告 Attention 实现耗时与端到端训练时间；PagedAttention 论文则在不同请求到达率下比较服务的 normalized latency，即每个请求的端到端延迟除以输出长度后取均值。比较前应先对齐测量对象与指标定义。[Dao et al., 2022，§4](https://arxiv.org/html/2205.14135v2)；[Kwon et al., 2023，§6](https://arxiv.org/html/2309.06180v1)
