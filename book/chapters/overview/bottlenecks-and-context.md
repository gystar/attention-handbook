# 1.3 LLM 中的 Attention 成本与长上下文需求

在 Transformer 语言模型中，Attention 机制负责从已有上下文中读取信息。随着输入文档变长、生成内容增多，模型需要处理更多位置，并保存更多可供后续读取的信息。本节以 Decoder-only 模型的常规自回归生成为例，说明这些计算与存储需求如何产生。

## Next-token Prediction 与自回归生成

Next-token Prediction 根据已有序列，估计紧接其后的 Token 在词表上的概率分布。Decoder-only 模型通过 Causal Self-Attention 汇总当前及此前位置的信息，再由输出层给出这一分布。生成时，系统按照解码策略选取一个 Token，将其追加到序列末尾，然后继续计算。这种逐步扩展序列的过程称为自回归生成（Autoregressive Generation）。[Kwon et al.，§2.1—2.2](https://arxiv.org/html/2309.06180v1)

例如，对于提示文本 `The cat sat`，一种可能的续写是 ` on the mat`。生成过程逐步扩展已有序列；每一步可以利用的上下文，都包含原始提示文本和此前已生成的内容。这是文本续写的教学示例，实际 Token 边界由 Tokenizer 决定。

## KV Cache 复用已有上下文

逐步生成意味着模型会反复读取相同的前缀。Attention 层将输入表示映射为 Query、Key 和 Value：Query 与 Key 用于计算匹配权重，Value 提供被汇总的内容。Key 和 Value 合称 KV，保存了后续位置读取该位置时所需的信息。

在参数和前缀固定的 Causal Transformer 中，已有位置的表示不受后来追加内容的影响。因此，各层已算出的历史 KV 可以保存下来，后续生成时直接复用。这样的缓存称为 KV Cache；每处理一个新 Token，模型计算其 Query 和 KV，读取历史 KV，并将新 KV 追加到缓存。[Shazeer，§2.3—2.4](https://arxiv.org/html/1911.02150v1)

KV Cache 减少了历史位置的重复计算，同时引入了持续增长的存储需求：保留的上下文越长，各层需要保存的 KV 就越多。

## Prefill 与 Decode 分别处理提示文本和生成内容

提示文本（Prompt）在生成开始前已经完整给定，可以一次处理多个位置；生成内容则依赖前一步选出的 Token，需要逐步推进。采用 KV Cache 时，这两种计算对应 Prefill 与 Decode。

| 阶段 | 处理的输入 | Attention 与缓存操作 |
| --- | --- | --- |
| Prefill | 已知 Prompt 的多个位置 | 在因果约束下计算各位置的表示，建立各层 KV Cache |
| Decode | 每个请求本轮新输入的一个 Token | 计算该位置的表示，读取历史 KV，并追加新 KV |

<p align="center">表 1-4　Prefill 与 Decode</p>

Prefill 完成后，模型利用 Prompt 最后一个位置的输出生成首个新 Token。随后，每轮 Decode 将刚生成的 Token 作为输入，继续生成后续内容。图 1-7 对照了这两个阶段在单层 Attention 中需要处理的位置。[Kwon et al.，§2.2](https://arxiv.org/html/2309.06180v1)

<figure align="center">
<img src="../../assets/overview/prefill-decode.svg" alt="Prefill 处理已知 Prompt 的多个位置；单步 Decode 处理一个新位置，并读取历史与当前 KV">
<figcaption><p align="center">图 1-7　Prefill 与 Decode</p></figcaption>
</figure>

图中省略了其他 Head、投影和前馈网络等组件；生成概率还需经过模型后续层及输出层。长 Prompt 也可以分块执行 Prefill，各块仍处理已知输入，并保留所需的历史 KV。[Jiang et al.，§2](https://arxiv.org/html/2310.06825v1)

## 更长的上下文增加计算、访存与缓存成本

采用 Dense Attention 时，Prefill 要为 Prompt 中的多个位置计算匹配与汇总；常规单步 Decode 只计算一个新位置，但仍需读取可访问的历史 KV。这使输入长度对两个阶段产生不同影响。

| 成本来源 | 随长度增长的变化 | 直接影响 |
| --- | --- | --- |
| 完整输入的位置对计算 | 序列长度加倍时，位置对数量约增至四倍 | 长输入的 Attention 运算量增加 |
| 单步 Decode 的历史读取 | 可访问的历史越长，需要读取的 KV 越多 | 每步 Attention 的计算和访存成本增加 |
| KV Cache | 保留的 Token 越多，缓存越大 | 单请求显存占用增加，可同时服务的请求数受限 |

<p align="center">表 1-5　序列长度与资源成本</p>

上述关系假设模型配置保持不变，采用 Dense Attention 并缓存全部历史 KV。Causal Mask 限制可访问的位置，但完整序列中的有效位置对仍随长度呈二次增长。FlashAttention 通过分块计算避免保存完整分数矩阵，因此 Attention 的运算量与中间显存占用需要分别分析。[Vaswani et al.，Table 1](https://arxiv.org/html/1706.03762v7)；[Dao et al.，§3](https://arxiv.org/html/2205.14135v2)

实际延迟还取决于硬件、一次同时处理的序列数（Batch Size）和计算实现。小 Batch 的 Decode 可能主要受模型参数与 KV 的读取带宽限制；减少缓存容量能节省显存，速度收益则需要在具体配置下测量。[Shazeer，§2.4.1](https://arxiv.org/html/1911.02150v1)

## Long Context 还需要位置适配与有效的信息使用

长上下文（Long Context）的目标，是让模型利用更长的文档、对话或代码完成任务。计算和缓存优化解决资源成本；随着输入变长，模型还会遇到超出训练范围的位置与距离，以及分散在长文本中的相关信息。

| 条件 | 要解决的问题 | 对应检查 |
| --- | --- | --- |
| 资源可承受 | 运算量、显存及缓存访问能否支撑目标长度 | Prefill 时间、Decode 时间、峰值显存 |
| 位置处理适配 | 更长的位置与距离是否超出训练时的使用范围 | Position Encoding、扩展方法及训练设置 |
| 信息能够有效使用 | 模型能否读取并利用长上下文中的相关信息 | 不同位置的检索、问答、多段信息整合等任务 |

<p align="center">表 1-6　Long Context 的三个条件</p>

例如，使用 Rotary Position Embedding（RoPE）的模型可以通过 Position Interpolation 重映射位置索引，并配合适应训练来扩展上下文窗口。[Chen et al., 2023，§2.3](https://arxiv.org/html/2306.15595v2)

Lost in the Middle 在其研究的模型和任务中发现，相关信息的位置会影响回答表现。这说明接受更长输入与有效利用其中信息需要分别评估；其他模型需要在相应任务上单独评估。[Liu et al., 2023，§3.1—3.3](https://arxiv.org/html/2307.03172v1)

图示来源与许可见[图表来源](../../appendices/references.md#图表来源)。

下一节：[1.4 Attention 的主要改进方向与方法关系](optimization-landscape.md)。
