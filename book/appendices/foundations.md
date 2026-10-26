# 附录 A 英文术语、数学符号与补充推导

本页按主题整理正文使用的术语、数学约定和补充推导。原始论文的符号可能不同，引用时说明对应关系。

## 常用术语

### 序列建模与 Attention 基础

| 术语 | 本书中的含义 | 对应正文 |
| --- | --- | --- |
| Attention Mechanism | 根据当前输入计算匹配权重并汇总信息的机制；本书首先介绍 Softmax Attention | [1.1 从 Representation Learning 到 Attention 机制](../chapters/overview/introduction.md) |
| Large Language Model（LLM） | 大语言模型；本书主要分析 Decoder-only 自回归场景 | [1.3 LLM 中的 Attention 成本与长上下文需求](../chapters/overview/bottlenecks-and-context.md) |
| Token | 模型处理的离散序列单元，不必对应完整单词或汉字 | [1.1 从 Representation Learning 到 Attention 机制](../chapters/overview/introduction.md) |
| Sequence-to-Sequence（Seq2Seq） | 从输入序列生成输出序列的建模框架 | [1.2 从 RNN 到 Transformer](../chapters/overview/from-seq2seq-to-transformer.md) |
| Query / Key / Value（QKV） | 发起匹配、参与匹配和被汇总的向量表示 | [第 2 章 Transformer Attention 的基本原理与计算过程](../chapters/transformer-attention/README.md) |
| Self-Attention / Cross-Attention | 按 Query 与 KV 来自同一序列还是不同序列区分 | [1.2 从 RNN 到 Transformer](../chapters/overview/from-seq2seq-to-transformer.md) |
| Causal Mask | 只允许访问当前位置及此前位置的访问约束 | [1.2 从 RNN 到 Transformer](../chapters/overview/from-seq2seq-to-transformer.md) |
| Prefill / Decode | 处理已知 Prompt / 逐步处理生成 Token 的两个阶段 | [1.3 LLM 中的 Attention 成本与长上下文需求](../chapters/overview/bottlenecks-and-context.md) |
| Multi-Head Attention（MHA） | 多个 Head 分别计算 Attention 后组合输出 | [第 2 章 Transformer Attention 的基本原理与计算过程](../chapters/transformer-attention/README.md) |

<p align="center">表 A-1　序列建模与 Attention 基础术语</p>

### 位置编码与表示调整

| 术语 | 本书中的含义 | 对应正文 |
| --- | --- | --- |
| Position Encoding | 把位置或距离信息引入输入表示、Q/K 或 Attention 分数 | [3.1 Attention 中的顺序与位置信息](../chapters/position-encoding/position-and-order.md) |
| RoPE / ALiBi | 分别通过 Q/K 旋转与线性距离偏置建模位置 | [3.3 RoPE 与 ALiBi 的位置建模](../chapters/position-encoding/rope-and-alibi.md) |
| Position Interpolation（PI） | 将较长序列的位置坐标线性缩放到原训练长度范围 | [3.4 RoPE Scaling 与上下文扩展](../chapters/position-encoding/context-extension.md) |
| Length Generalization | 超出相关训练长度时保持任务能力 | [3.5 长度泛化与长上下文评估](../chapters/position-encoding/evaluation.md) |
| Multi-Query Attention（MQA） | 多个 Query Head 共享一组 KV | [4.2 MQA 与 GQA 的 Head 共享](../chapters/kv-representations/head-sharing.md) |
| Grouped-Query Attention（GQA） | 每组 Query Head 共享一组 KV | [4.2 MQA 与 GQA 的 Head 共享](../chapters/kv-representations/head-sharing.md) |
| Multi-Head Latent Attention（MLA） | 采用联合低秩 KV 表示，并处理位置相关分量的 Attention 设计 | [4.3 MLA 的 Latent Compression](../chapters/kv-representations/latent-compression.md) |
| Output Gating | 按输入生成系数，逐元素或逐 Head 调节 Attention 汇总结果 | [5.1 Attention 输出的 Gating](../chapters/gated-attention/output-gating.md) |

<p align="center">表 A-2　位置编码与表示调整术语</p>

### 执行与缓存管理

| 术语 | 本书中的含义 | 对应正文 |
| --- | --- | --- |
| KV Cache | 为后续推理复用而保存的逐层历史 KV 或相应压缩表示 | [8.1 KV Cache 的生命周期与容量](../chapters/kv-cache/lifecycle.md) |
| IO / HBM | 数据传输 / GPU 的 High Bandwidth Memory；区别于算术运算量 | [6.1 Attention 的计算量与数据访问](../chapters/flash-attention/io-cost.md) |
| Online Softmax | 逐块更新最大值与归一化分母；Attention 同时累积 Value 加权和 | [6.2 Online Softmax 与分块计算](../chapters/flash-attention/online-softmax.md) |
| Block-Sparse Attention | 按块组织可访问的位置对，使实现能够跳过未选块 | [7.3 Block-Sparse Attention 的组织与执行](../chapters/sparse-attention/block-sparse.md) |
| PagedAttention | 通过块表将逻辑 KV 位置映射到物理存储块 | [8.2 PagedAttention 与缓存分配](../chapters/kv-cache/paged-attention.md) |
| Cache Eviction | 本书 8.4 指移除活跃请求的历史 KV，改变后续可访问集合；闲置前缀缓存的回收见 8.5 | [8.4 Token Selection 与 Cache Eviction](../chapters/kv-cache/eviction.md) |
| Prefix Caching / Offloading | 复用已计算的前缀状态 / 在不同存储层之间迁移数据 | [8.5 Prefix Caching 与 Offloading](../chapters/kv-cache/reuse-offloading.md) |
| Continuous Batching / Chunked Prefill | 在迭代边界调整批次 / 将 Prompt 处理拆为多个 Chunk | [8.6 缓存策略与服务性能](../chapters/kv-cache/evaluation.md) |
| TTFT / ITL | 从请求到达到首次输出的时间 / 相邻输出 Token 的时间间隔；需固定观测端 | [8.6 缓存策略与服务性能](../chapters/kv-cache/evaluation.md) |

<p align="center">表 A-3　执行与缓存管理术语</p>

### 递推状态与混合架构

| 术语 | 本书中的含义 | 对应正文 |
| --- | --- | --- |
| Recurrent Formulation | 用上一步状态与当前输入表达下一步状态和输出 | [9.2 Causal Linear Attention 的递推计算](../chapters/linear-attention/recurrence.md) |
| Kernel Attention | 以特征核定义 Query–Key 权重；有限维特征映射允许重排计算 | [9.1 从 Softmax Attention 到 Kernel Attention](../chapters/linear-attention/kernel-attention.md) |
| Gated Linear Attention（GLA） | 通过输入相关 Gate 调节递推状态的历史保留 | [9.3 状态衰减与 Gated Linear Attention](../chapters/linear-attention/state-decay.md) |
| Delta Rule | 根据当前 Key 读取状态的误差修正状态写入 | [9.4 DeltaNet 与 Gated DeltaNet 的状态更新](../chapters/linear-attention/delta-rule.md) |
| State Space Model（SSM） | 通过状态转移、输入写入和状态读取描述序列计算 | [10.4 Attention 与 State Space Models 的组合](../chapters/hybrid-architectures/attention-ssm.md) |
| Hybrid Architecture | 在同一模型的不同层、Head 或分支中组合序列模块 | [10.1 序列模块的组合方式](../chapters/hybrid-architectures/composition.md) |

<p align="center">表 A-4　递推状态与混合架构术语</p>

## 符号与维度

矩阵按 Token 占据行来排列。单独写出的 $$q_t,k_t,v_t$$ 是列向量；因此它们作为矩阵行时写成转置。下标 $$t$$ 表示位置，暂时省略 Layer、Batch 和 Head 索引。

| 符号 | 含义 |
| --- | --- |
| $$B,L,N$$ | KV 容量语境中分别为 Batch Size、层数、序列长度 |
| $$n_q,n_k$$ | Query 数量与 Key 数量；Cross-Attention 中可以不同 |
| $$d_k,d_v$$ | Query/Key 维度与 Value 维度 |
| $$d_h$$ | 在明确假设 $$d_k=d_v$$ 时使用的统一 Head 维度 |
| $$h_q,h_{kv}$$ | Query Head 数与实际存储的 KV Head 数 |
| $$s$$ | 每个缓存元素占用的字节数 |
| $$M$$ | 加性 Mask，形状 $$n_q\times n_k$$；允许项为 $$0$$，禁止项为 $$-\infty$$ |
| $$d_\phi$$ | Linear Attention 特征映射的输出维度 |
| $$S_t,z_t$$（9.1—9.2） | 核特征历史状态与归一化状态，形状分别为 $$d_\phi\times d_v$$、$$d_\phi$$ |
| $$L,L',\rho$$（第 3 章） | 原训练长度、目标长度与扩展因子 $$\rho=L'/L$$ |
| $$d,b,\theta_r$$（第 3 章） | Head 的 Q/K 维度、RoPE 基底、第 $$r$$ 个维度对的角频率 |
| $$\lambda_r,\tau$$（第 3 章） | 波长 $$2\pi/\theta_r$$ 与 Attention Scaling 温度 |

<p align="center">表 A-5　符号与维度</p>

后续专题按局部公式使用以下符号。同一字母可能在不同章节表示不同对象，阅读跨章公式时应同时检查定义与形状。

| 位置 | 符号 | 含义与形状 |
| --- | --- | --- |
| 6.1 | $$S,P$$ | 分数与 Attention 权重矩阵，均为 $$N\times N$$ |
| 6.1 | $$M$$ | IO 分析中的片上容量，以元素数计；此处不表示加性 Mask |
| 6.2 | $$m_A,\ell_A,u_A$$ | 集合 $$A$$ 的最大分数、重缩放指数和、Value 加权和；前两者为标量，后者为 $$d_v$$ 维向量 |
| 7.1 | $$\mathcal A_i,E$$ | Query $$i$$ 的可访问位置集合，以及总配对数 $$E=\sum_i|\mathcal A_i|$$ |
| 9.3 | $$S_t,\alpha_t,\beta_t$$ | GLA 状态为 $$d_k\times d_v$$；一般 Gate 的行、列向量分别为 $$d_k$$、$$d_v$$ 维 |
| 9.4 | $$S_t,\alpha_t,\beta_t$$ | DeltaNet 状态为 $$d_k\times d_v$$；$$\beta_t$$ 是标量修正强度，Gated DeltaNet 的 $$\alpha_t$$ 是标量衰减 |
| 10.4 | $$h_t,\bar A_t,\bar B_t,C_t$$ | 标量输入输出 SSM 的状态为 $$d_s$$ 维，转移矩阵为 $$d_s\times d_s$$，写入与读取向量为 $$d_s$$ 维 |
| 10.5 | $$s_\ell,c_\ell,\mathcal C_P$$ | 第 $$\ell$$ 层核心与辅助状态的字节数，以及处理完位置 $$P$$ 后的全层必要状态快照 |

<p align="center">表 A-6　专题中的局部符号约定</p>

9.3 的 GLA 模型配置将列 Gate 固定为全 1 向量；9.4 的同名字母表示标量，不能直接代入 GLA 的逐通道公式。具体计算见[9.3 状态衰减与 Gated Linear Attention](../chapters/linear-attention/state-decay.md)和[9.4 DeltaNet 与 Gated DeltaNet 的状态更新](../chapters/linear-attention/delta-rule.md)。

$$\operatorname{softmax}_{\mathrm{row}}$$ 表示对每个 Query 沿 Key 维度归一化。对有限有效分数 $$x_j$$，可用减去最大值的形式计算：

$$
p_j=\frac{\exp(x_j-m)}{\sum_r\exp(x_r-m)},\qquad m=\max_r x_r.
$$

这与原始 Softmax 数学等价，并避免直接计算过大的指数；全部位置均被 Mask 时该式没有有效的归一化分母，需由具体实现单独处理。[Dao et al., 2022，§3.1](https://arxiv.org/html/2205.14135v2)

## 从加权历史到 Recurrent State

对非负相似度 $$\kappa(q,k)=\phi(q)^\top\phi(k)\geq0$$，且归一化分母大于零时，Causal Attention 的输出可写为：

$$
\begin{aligned}
o_t^\top
&=\frac{\sum_{j=1}^{t}\bigl(\phi(q_t)^\top\phi(k_j)\bigr)v_j^\top}
{\sum_{j=1}^{t}\phi(q_t)^\top\phi(k_j)}
\\[4pt]
&=\frac{\phi(q_t)^\top\left(\sum_{j=1}^{t}\phi(k_j)v_j^\top\right)}
{\phi(q_t)^\top\left(\sum_{j=1}^{t}\phi(k_j)\right)}.
\end{aligned}
$$

右侧两项历史求和分别定义为 $$S_t$$、$$z_t$$。从 $$t-1$$ 到 $$t$$，各追加当前项，就得到 [9.2 Causal Linear Attention 的递推计算](../chapters/linear-attention/recurrence.md) 的递推公式。这一步只使用分配律与有限求和；得到固定大小状态还要求 $$d_\phi,d_v$$ 不随序列长度增加。[Katharopoulos et al., 2020，§3.2—3.4，式 (9)—(20)](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)

## 容量计算的适用条件

[第 4 章 MQA、GQA 与 MLA 的 KV 表示优化](../chapters/kv-representations/README.md) 的 KV 容量是元素数乘以每元素字节数。单层的 Key 张量与 Value 张量各含 $$BNh_{kv}d_h$$ 个元素，乘以两份缓存、$$L$$ 层和 $$s$$ 字节，即得 $$2LBNh_{kv}d_hs$$。

这要求各层配置一致、请求长度一致且保留全部位置。按层、按请求长度不同的情况需要分别求和；MLA、窗口缓存及量化元数据也需按实际表示单独计算。

## Bahdanau Attention 的计算

在 Bahdanau Attention 中，第 $$i$$ 个输出步骤的 Context 为：

$$
\begin{aligned}
c_i&=\sum_{j=1}^{n_x}\alpha_{ij}h_j,\\[4pt]
\alpha_{ij}&=
\frac{\exp(e_{ij})}{\sum_{r=1}^{n_x}\exp(e_{ir})},
\\[4pt]
e_{ij}&=a(s_{i-1},h_j).
\end{aligned}
$$

$$n_x$$ 是输入长度，$$h_j$$ 是 Encoder 状态，$$s_{i-1}$$ 是前一步 Decoder 状态，$$a$$ 是学习得到的打分函数。$$\alpha_{ij}$$ 随生成步骤变化，使 Decoder 能够直接组合不同输入位置的信息。[Bahdanau et al., 2014，§3.1，式 (4)—(6)](https://arxiv.org/html/1409.0473v7)


这里的 Alignment 表示输出步骤与输入位置之间学到的软对应关系。

## MLA 的符号约定

第四章中，$$d_c$$ 为共同 KV Latent 的维度，$$d_r$$ 为单独 RoPE 分支的维度，$$d_h$$ 为每个 Head 的内容 Key 与 Value 维度。$$c_j$$ 是位置 $$j$$ 的 KV Latent，$$U_a^K,U_a^V$$ 为 Head $$a$$ 的 Up-projection。单独向量按列书写；图中按行排成矩阵时使用相应转置。

MLA 缓存的元素数按实际保存的 Latent 和位置 Key 相加。对于本书讨论的配置，每层每个位置为 $$d_c+d_r$$；Query 的压缩维度不属于历史 KV 缓存。具体推导见[4.3 MLA 的 Latent Compression](../chapters/kv-representations/latent-compression.md)和[4.4 MLA 中的 RoPE 与投影合并](../chapters/kv-representations/decoupled-rope.md)。
