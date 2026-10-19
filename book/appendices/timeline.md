# 附录 B 技术演进时间线与方法对照表

本页提供跨章检索入口。当前时间线覆盖首章引用的代表工作，后续随专题章节补充，不作为完整文献年表。

## 时间线入口

[1.6 主要技术路线的演进关系与比较维度](../chapters/overview/evolution-and-comparison.md) 按首次公开年份列出 2014—2025 年的主要例子；[附录 C 参考文献、教学资源与本地源文件索引](references.md) 记录实际引用的论文版本。年份先后不能单独证明方法之间存在继承关系。

## 按改变的对象查找

| 改变的对象 | 代表方法 | 主要比较问题 | 专题 |
| --- | --- | --- | --- |
| 位置表示与距离信息 | RoPE、ALiBi、Position Interpolation | 使用范围如何变化，长文本能力怎样评估 | [第 3 章 Position Encoding 的演进与 Context Extension](../chapters/position-encoding/README.md) |
| KV Head 或表示 | MQA、GQA、MLA | 缓存保存什么，能力与容量如何权衡 | [第 4 章 Key–Value 表示优化：MHA、MQA、GQA 与 MLA](../chapters/kv-representations/README.md) |
| Attention 内部的 Gate | Gated Attention | Gate 的输入、作用位置与实验依据 | [第 5 章 Softmax Attention 的结构改进：Gated Attention](../chapters/gated-attention/README.md) |
| 运算调度与中间存储 | FlashAttention | 目标算子是否相同，IO 减少多少 | [第 6 章 FlashAttention：Exact Attention 的 IO 优化](../chapters/flash-attention/README.md) |
| 可访问的位置对 | Sparse Attention | 选择开销、实际稀疏度与质量变化 | [第 7 章 Sparse Attention 的连接模式与动态选择](../chapters/sparse-attention/README.md) |
| KV 的物理存储和保留 | PagedAttention、KV 量化与淘汰 | 分别改变布局、精度还是历史覆盖 | [第 8 章 KV Cache 的存储管理与推理优化](../chapters/kv-cache/README.md) |
| 状态更新与读取 | Linear Attention、Gated DeltaNet | 状态容量、计算方式与信息读取能力 | [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](../chapters/linear-attention/README.md) |
| 模块组合 | Attention、Recurrent、SSM 的组合 | 配置、训练及整模型表现 | [第 10 章 Hybrid Architectures 的结构设计与方法比较](../chapters/hybrid-architectures/README.md) |

方法区别的原始依据集中在 [1.5 Attention 的结构改进、计算优化与架构扩展](../chapters/overview/optimization-landscape.md)。上表用于查找讨论位置，不能用来直接给方法排名；性能比较需同时指定阶段、长度、Batch、硬件、精度及能力指标。
