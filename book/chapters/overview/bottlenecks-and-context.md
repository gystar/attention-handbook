# 1.3 LLM 中的 Attention 成本与长上下文需求

Transformer 通过 Attention 让不同位置交换信息。用于大语言模型（Large Language Model，LLM）时，这种计算需要反复处理提示文本与生成内容：输入越长，可供读取的信息越多，相应的计算和存储成本也会增加。

## 从下一 Token 预测到自回归生成

采用 Decoder-only 架构的语言模型将提示文本与已生成内容作为同一序列。每个位置通过 Causal Self-Attention 读取自身及此前的信息，输出层根据最终表示计算下一 Token 的概率。模型选出一个 Token 并加入序列，再继续预测，这就是自回归生成。[Jay Alammar，The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

例如，模型收到 `The cat sat` 后，可能先生成 ` on`，再以扩展后的序列继续预测。这是生成顺序的教学示例；实际文本切分取决于 Tokenizer。

生成过程中，已有前缀不断增长。理解 Attention 的成本，需要区分一次处理已知文本，以及逐步处理新生成的 Token。

## Prefill 与 Decode 的工作方式

Attention 层将输入表示映射为 Query、Key 和 Value：Query 与 Key 用于计算匹配权重，Value 提供被汇总的内容，Key 和 Value 合称 KV。自回归模型中，已有位置无法读取后来出现的 Token；对于固定参数和前缀，各层已算出的历史 KV 可以保留，供后续步骤复用。这部分缓存称为 KV Cache。[Shazeer，§2.3—2.4](https://arxiv.org/html/1911.02150v1)

Prefill 处理已知的提示文本（Prompt），建立 KV Cache 并预测首个新 Token；Decode 随后逐步处理新 Token，读取并更新缓存。

<figure align="center">
<img src="../../assets/overview/prefill-decode.svg" alt="Prefill 与 Decode 示意图：Prefill 同时处理已知前缀的多个 Query；Decode 处理一个新 Query，并读取缓存和当前 Token 的 KV">
<figcaption><p align="center">图 1-7　Prefill 与 Decode</p></figcaption>
</figure>

Prefill 的最后一个位置用于预测首个新 Token；随后每个 Decode 步骤处理新输入 Token，并预测再下一个 Token。图中仅展示单层 Attention 的位置关系，省略 Batch、其他 Head、投影与 FFN；预测还需要模型后续计算及输出层。[Kwon et al.，§2.2](https://arxiv.org/html/2309.06180v1)

| 阶段 | 已知输入与计算方式 | 与缓存相关的特点 |
| --- | --- | --- |
| Training | 已知训练序列，在 Mask 下并行计算并反向传播 | 需要处理梯度及激活保存或重算 |
| Prefill | 处理已知 Prompt 的多个位置 | 建立各层 KV Cache，得到首个新 Token 的预测 |
| Decode | 常规自回归场景中每个请求处理一个新 Token | 读取此前 KV，追加当前 KV，得到下一步预测 |

<p align="center">表 1-4　Training、Prefill 与 Decode</p>

Prefill 也可以按块执行；上述区别关注的是已知 Prompt 与新生成 Token 的数据依赖。[Kwon et al., 2023，§2.2](https://arxiv.org/html/2309.06180v1)；[Jiang et al., 2023，§2](https://arxiv.org/html/2310.06825v1)

## 序列长度如何影响资源成本

在 Dense Attention 中，一个位置会读取访问范围内的所有位置。处理完整输入时，每个位置都需要这样的计算；常规单步 Decode 则只有一个新位置需要读取历史。

| 成本来源 | 随长度增长的变化 | 直接影响 |
| --- | --- | --- |
| 完整输入的位置对计算 | 序列长度加倍时，位置对数量约增至四倍 | 长输入的 Attention 运算量增加 |
| 单步 Decode 的历史读取 | 可访问的历史越长，需要读取的 KV 越多 | 每步 Attention 的计算和访存成本增加 |
| KV Cache | 保留的 Token 越多，缓存越大 | 单请求显存占用增加，可同时服务的请求数受限 |

<p align="center">表 1-5　序列长度与资源成本</p>

上述关系假设模型配置保持不变，采用 Dense Attention 并缓存全部历史 KV。Causal Mask 限制可访问的位置，但完整序列中的有效位置对仍随长度呈二次增长。FlashAttention 通过分块计算避免保存完整分数矩阵，因此 Attention 的运算量与中间显存占用需要分别分析。[Vaswani et al.，Table 1](https://arxiv.org/html/1706.03762v7)；[Dao et al.，§3](https://arxiv.org/html/2205.14135v2)

实际延迟还取决于硬件、一次同时处理的序列数（Batch Size）和计算实现。小 Batch 的 Decode 可能主要受模型参数与 KV 的读取带宽限制；减少缓存容量能节省显存，速度收益则需要在具体配置下测量。[Shazeer，§2.4.1](https://arxiv.org/html/1911.02150v1)

## 长上下文的资源、位置与能力要求

长上下文（Long Context）让模型能够处理更长的文档、对话或代码。支持目标输入长度，需要同时考虑资源、位置处理和信息使用能力。

| 条件 | 要解决的问题 | 对应检查 |
| --- | --- | --- |
| 资源可承受 | 运算量、显存及缓存访问能否支撑目标长度 | Prefill 时间、Decode 时间、峰值显存 |
| 位置处理适配 | 更长的位置与距离是否超出训练时的使用范围 | Position Encoding、扩展方法及训练设置 |
| 信息能够有效使用 | 模型能否读取并利用长上下文中的相关信息 | 不同位置的检索、问答、多段信息整合等任务 |

<p align="center">表 1-6　Long Context 的三个条件</p>

Position Interpolation 针对 Rotary Position Embedding（RoPE）的位置范围提出重映射并配合适应训练，其作用应与计算、缓存优化区分。[Chen et al., 2023，§2.3](https://arxiv.org/html/2306.15595v2)

Lost in the Middle 在其研究的模型和任务中发现，相关信息的位置会影响回答表现。这说明接受更长输入与有效利用其中信息需要分别评估；其他模型需要在相应任务上单独评估。[Liu et al., 2023，§3.1—3.3](https://arxiv.org/html/2307.03172v1)

图示来源与许可见[图表来源](../../appendices/references.md#图表来源)。

下一节：[1.4 Attention 的主要改进方向与方法关系](optimization-landscape.md)。
