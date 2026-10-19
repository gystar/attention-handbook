# 附录 A 英文术语、数学符号与补充推导

本页维护已写章节使用的术语和数学约定，当前覆盖第 1 章。原始论文的符号可能不同，引用时说明对应关系。

## 常用术语

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
| Multi-Query Attention（MQA） | 多个 Query Head 共享一组 KV | [1.4 Attention 的主要改进方向与方法关系](../chapters/overview/optimization-landscape.md) |
| Grouped-Query Attention（GQA） | 每组 Query Head 共享一组 KV | [1.4 Attention 的主要改进方向与方法关系](../chapters/overview/optimization-landscape.md) |
| Multi-Head Latent Attention（MLA） | 采用联合低秩 KV 表示，并处理位置相关分量的 Attention 设计 | [1.4 Attention 的主要改进方向与方法关系](../chapters/overview/optimization-landscape.md) |
| KV Cache | 为后续推理复用而保存的逐层历史 KV 或相应压缩表示 | [1.3 LLM 中的 Attention 成本与长上下文需求](../chapters/overview/bottlenecks-and-context.md) |
| IO / HBM | 数据传输 / GPU 的 High Bandwidth Memory；区别于算术运算量 | [1.4 Attention 的主要改进方向与方法关系](../chapters/overview/optimization-landscape.md) |
| Recurrent Formulation | 用上一步状态与当前输入表达下一步状态和输出 | [1.4 Attention 的主要改进方向与方法关系](../chapters/overview/optimization-landscape.md) |

<p align="center">表 A-1　常用术语</p>

## 符号与维度

矩阵按 Token 占据行来排列。单独写出的 $$q_t,k_t,v_t$$ 是列向量；因此它们作为矩阵行时写成转置。下标 $$t$$ 表示位置，暂时省略 Layer、Batch 和 Head 索引。

| 符号 | 含义 |
| --- | --- |
| $$B,L,N$$ | Batch Size、层数、序列长度 |
| $$n_q,n_k$$ | Query 数量与 Key 数量；Cross-Attention 中可以不同 |
| $$d_k,d_v$$ | Query/Key 维度与 Value 维度 |
| $$d_h$$ | 在明确假设 $$d_k=d_v$$ 时使用的统一 Head 维度 |
| $$h_q,h_{kv}$$ | Query Head 数与实际存储的 KV Head 数 |
| $$s$$ | 每个缓存元素占用的字节数 |
| $$M$$ | 加性 Mask，形状 $$n_q\times n_k$$；允许项为 $$0$$，禁止项为 $$-\infty$$ |
| $$d_\phi$$ | Linear Attention 特征映射的输出维度 |
| $$S_t,z_t$$ | 历史状态与归一化状态，形状分别为 $$d_\phi\times d_v$$、$$d_\phi$$ |

<p align="center">表 A-2　符号与维度</p>

$$\operatorname{softmax}_{\mathrm{row}}$$ 表示对每个 Query 沿 Key 维度归一化。对有限有效分数 $$x_j$$，可用减去最大值的形式计算：

$$
p_j=\frac{\exp(x_j-m)}{\sum_r\exp(x_r-m)},\qquad m=\max_r x_r.
$$

这与原始 Softmax 数学等价，并避免直接计算过大的指数；全部位置均被 Mask 时该式没有有效的归一化分母，需由具体实现单独处理。[Dao et al., 2022，§3.1](https://arxiv.org/html/2205.14135v2)

## 从加权历史到 Recurrent State

对特征核 $$\kappa(q,k)=\phi(q)^\top\phi(k)$$，且归一化分母非零时，Causal Attention 的输出可写为：

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

右侧两项历史求和分别定义为 $$S_t$$、$$z_t$$。从 $$t-1$$ 到 $$t$$，各追加当前项，就得到 [1.4 Attention 的主要改进方向与方法关系](../chapters/overview/optimization-landscape.md) 的递推公式。这一步只使用分配律与有限求和；得到固定大小状态还要求 $$d_\phi,d_v$$ 不随序列长度增加。[Katharopoulos et al., 2020，§3.2—3.4，式 (9)—(20)](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)

## 容量计算的适用条件

[1.3 LLM 中的 Attention 成本与长上下文需求](../chapters/overview/bottlenecks-and-context.md) 的 KV 容量是元素数乘以每元素字节数。单层的 Key 张量与 Value 张量各含 $$BNh_{kv}d_h$$ 个元素，乘以两份缓存、$$L$$ 层和 $$s$$ 字节，即得 $$2LBNh_{kv}d_hs$$。

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
