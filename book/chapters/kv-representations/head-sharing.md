# 4.2 MQA 与 GQA 的 Head 共享

Multi-Query Attention（MQA）让所有 Query Head 共用一组 Key 和 Value。Grouped-Query Attention（GQA）保留多组 KV，每组服务一部分 Query Head。两者都保留多个 Query 的匹配过程，改变的是这些 Query 读取的表示。[Shazeer，§3](https://arxiv.org/html/1911.02150v1)；[Ainslie et al.，§2.2](https://arxiv.org/html/2305.13245v3)

## 从独立 Head 到分组共享

用 $$g(a)$$ 表示 Query Head $$a$$ 所属的 KV 组。对长度为 $$N$$、隐藏维度为 $$d$$ 的输入 $$X\in\mathbb{R}^{N\times d}$$，省略 batch 维度，设 Key 和 Value 的 Head 维度均为 $$d_h$$：

$$
\begin{aligned}
Q_a&=XW_a^Q, &W_a^Q&\in\mathbb{R}^{d\times d_h},\\
K_g&=XW_g^K, &W_g^K&\in\mathbb{R}^{d\times d_h},\\
V_g&=XW_g^V, &W_g^V&\in\mathbb{R}^{d\times d_h},\\
O_a&=\operatorname{Softmax}\!\left(\frac{Q_aK_{g(a)}^{\mathsf T}}{\sqrt{d_h}}+M\right)V_{g(a)}.
\end{aligned}
$$

这里 $$M\in\mathbb{R}^{N\times N}$$ 为 Mask，Softmax 沿 Key 位置归一化。各 Head 的 $$O_a\in\mathbb{R}^{N\times d_h}$$ 仍按第二章的方式拼接，再经 Output Projection 输出。式中展示内容投影；使用 RoPE 时，按模型约定在 Q/K 上施加位置变换。

假设 Query Head 均匀分组，且 $$h_q$$ 可被 $$h_{kv}$$ 整除。下表用八个 Query Head 说明映射，编号从零开始。

| 结构 | $$h_{kv}$$ | Query Head 0—7 对应的 KV 组 |
| --- | ---: | --- |
| MHA | 8 | 0、1、2、3、4、5、6、7 |
| GQA | 2 | 0、0、0、0、1、1、1、1 |
| MQA | 1 | 0、0、0、0、0、0、0、0 |

<p align="center">表 4-4　Query Head 与 KV 组的对应关系</p>

这个例子把 GQA 写成统一形式：令 $$h_{kv}=h_q$$ 得到 MHA，令 $$h_{kv}=1$$ 得到 MQA。组内的 Query 投影各自保留，因此即使读取同一套 KV，不同 Query Head 仍可产生不同的 Attention 权重。

## 共享 KV 与缓存复用

在上述固定维度的比较中，K/V 投影参数数为 $$2dh_{kv}d_h$$，每层每个历史位置的缓存元素数为 $$2h_{kv}d_h$$；两项都随 KV Head 数变化。参数量计数省略 bias，只计算 K/V 投影，不包含 Q、Output Projection 或 FFN。

实现公式时，可以在逻辑上将每组 KV 广播到其 Query Head。若先物化成 $$h_q$$ 份并长期保存，缓存占用会重新扩大。保留 $$h_{kv}$$ 份缓存、由 kernel 在组内复用，才能实现该表示对应的容量收益。广播的数学等价与物理存储是否复制是两个需要分别检查的实现问题。

## 从 MHA Checkpoint 转换

已有 MHA 权重可以用组内 K/V 投影的均值初始化共享投影，再进行额外预训练，使模型适应新结构。GQA 原论文将这一适配过程称为 Uptraining。[Ainslie et al.，§2.1—2.2](https://arxiv.org/html/2305.13245v3)

设组 $$g$$ 包含的原 Head 集合为 $$\mathcal H_g$$，则：

$$
\overline W_g^K=\frac{1}{|\mathcal H_g|}\sum_{a\in\mathcal H_g}W_a^K,
\qquad
\overline W_g^V=\frac{1}{|\mathcal H_g|}\sum_{a\in\mathcal H_g}W_a^V.
$$

线性投影满足 $$X\overline W_g^K=|\mathcal H_g|^{-1}\sum_a XW_a^K$$。但 Attention 包含 Softmax，平均投影一般不能保持原模型输出。下面是直接由公式构造的反例：单个 Query 为 $$1$$，两个 Key 位置，每个 Head 的 Value 都为 $$[0,1]^{\mathsf T}$$，Head 维度为 $$1$$。

| 计算对象 | 两个 Key 值 | 第二个位置的权重与 Head 输出 |
| --- | --- | ---: |
| 原 Head 0 | $$[0,0]$$ | $$0.5$$ |
| 原 Head 1 | $$[0,2]$$ | 约 $$0.8808$$ |
| 平均 Key 后 | $$[0,1]$$ | 约 $$0.7311$$ |

<p align="center">表 4-5　平均投影改变 Attention 输出的算例</p>

原来两个 Head 输出的平均值约为 $$0.6904$$，与平均 Key 后的结果不同。该例说明转换后的模型需要重新评估；实际质量还取决于适配训练及任务，不能从线性投影的均值关系推得。

原论文的适配实验使用 T5.1.1，在 Decoder Self-Attention 和 Cross-Attention 上应用共享；这给出了特定训练配置下的证据。对其他架构的转换，应重新检查质量、稳定性和部署条件。[Ainslie et al.，§3.1](https://arxiv.org/html/2305.13245v3)
