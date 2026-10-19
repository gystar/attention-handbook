# 1.1 Attention 在 LLM 中的作用与研究范围

在自回归大语言模型（Large Language Model，LLM）中，下一 Token 的预测依赖已知前缀。Attention 让当前位置的表示能够读取允许访问的其他位置，并根据当前输入计算读取权重。标准 Transformer 中的 Feed-Forward Network（FFN）则分别作用于各个位置；二者共同更新序列表示。[Vaswani et al., 2017，§3](https://arxiv.org/html/1706.03762v7)

## 从上下文到加权读取

先考虑一个 Attention Head，暂时省略 Batch 和 Layer 维度。Query 与每个 Key 计算匹配分数，Softmax 将分数归一化，再用这些权重对 Value 加权求和。

| 对象 | 形状 | 在计算中的作用 |
| --- | --- | --- |
| Query，$$Q$$ | $$n_q \times d_k$$ | 表示哪些位置正在发起读取 |
| Key，$$K$$ | $$n_k \times d_k$$ | 与 Query 匹配，决定读取权重 |
| Value，$$V$$ | $$n_k \times d_v$$ | 被加权汇总的内容 |
| Attention 输出，$$O$$ | $$n_q \times d_v$$ | 每个 Query 读取后的结果 |

这里，$$n_q$$ 和 $$n_k$$ 分别是 Query 与 Key 的数量，$$d_k$$ 是匹配向量维度，$$d_v$$ 是 Value 维度。Key 和 Value（合称 KV）按位置配对，所以行数相同；Query 与 Key 的数量可以不同。

Scaled Dot-Product Attention 写为：

$$
O=\operatorname{softmax}_{\mathrm{row}}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

$$M\in\mathbb{R}^{n_q\times n_k}$$ 是加性 Mask：允许的位置取 $$0$$，禁止的位置在数学表达中取 $$-\infty$$。Softmax 沿每一行的 Key 维度计算；每个 Query 至少需要一个允许访问的 Key。[Vaswani et al., 2017，§3.2](https://arxiv.org/html/1706.03762v7)；[Shazeer, 2019，§2.3](https://arxiv.org/html/1911.02150v1)

![原论文图 2 左图：Q 与 K 经 MatMul、Scale、可选 Mask 和 Softmax 后，再与 V 相乘](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png)

图 1-2　Scaled Dot-Product Attention，引用自 Vaswani et al.，Figure 2 左图。顺着箭头阅读：先产生权重，再汇总 Value。原论文首页给出了署名后用于学术作品的图表转载许可。[原图及出处](https://arxiv.org/html/1706.03762v7)

Q、K、V 是当前输入经过投影得到的激活值。训练学到的是投影等参数；推理时参数通常保持不变，Attention 权重仍会随输入改变。不能把 Attention 权重与模型参数混为一谈。[Shazeer, 2019，§2.1—2.3](https://arxiv.org/html/1911.02150v1)

## 一个最小算例

假设某个 Query 的归一化权重为 $$(0.1,0.2,0.7)$$，三个 Value 分别为 $$(1,0)$$、$$(0,1)$$、$$(1,1)$$，则输出为：

$$
o=0.1(1,0)+0.2(0,1)+0.7(1,1)=(0.8,0.9).
$$

这是人为构造的计算示例。它说明 Attention 输出是向量的加权组合；这些数值不代表某个模型在真实文本上的 Attention Pattern。

## 本书的范围

本书以用于语言建模的 Transformer Attention 为基础，后续成本分析主要采用 Decoder-only 模型的常规自回归生成场景。

| 纳入范围 | 讨论重点 |
| --- | --- |
| Transformer Attention | QKV、Multi-Head、Mask，以及 Self-Attention 与 Cross-Attention |
| Position Encoding 与结构改进 | 位置如何参与计算，Head 与 KV 表示如何组织，Gate 放在哪里 |
| 计算与推理优化 | 运算量、显存访问、KV Cache 和服务系统中的资源管理 |
| Linear Attention 与 Hybrid Architectures | 历史信息如何进入状态，以及不同序列模块如何组合 |

视觉、语音和多模态中的 Attention 在需要比较机制时引入。State Space Models（SSMs）作为相关序列建模路线和混合模块讨论；具体联系需要说明模型形式与假设。[Dao & Gu, 2024，§4.1、§5.1—5.2](https://arxiv.org/html/2405.21060v1)

## 配合视频理解 QKV

3Blue1Brown 的 [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=392s) 可配合本节观看：

| 时间段 | 观察重点 |
| --- | --- |
| 06:32—10:40 | Query、Key、点积与归一化权重 |
| 11:30—12:45 | Causal Mask 怎样限制可访问位置 |
| 13:40—15:10 | Value 怎样被权重组合 |
| 20:30—22:30 | 多个 Head 如何并行工作 |

视频的示意图使用了自己的向量排列方式；本书统一按 Token 占据矩阵行来书写，Softmax 沿 Key 维度归一化。视频及[作者配套图文](https://www.3blue1brown.com/lessons/attention/)用于理解计算过程，精确定义以上述公式为准。时间段依据[作者字幕](https://github.com/3b1b/captions/blob/main/2024/attention/english/captions.srt)核对。

下一节：[从 Seq2Seq 中的 Attention 到 Transformer](from-seq2seq-to-transformer.md)。
