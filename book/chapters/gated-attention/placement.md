# 5.2 Gate 的位置与参数化

Gate 的效果取决于它乘在哪个张量上、由哪个位置的输入生成，以及是否在 Head 或特征之间共享。下面沿同一 Attention 计算路径区分这些选择。[Qiu et al.，§2.2](https://arxiv.org/html/2505.06708v1)

## 插入位置决定作用对象

| Gate 位置 | 直接改变的对象 | 位置依赖 |
| --- | --- | --- |
| Query 投影后 | 当前 Query 与各 Key 的匹配分数 | Query 位置 |
| Key 投影后 | 被访问位置参与匹配的表示 | Key 位置 |
| Value 投影后 | 各位置待汇总的信息 | Value 所在位置 |
| SDPA 输出后 | 每个 Query 汇总后的 Head 输出 | Query 位置 |
| Output Projection 后 | 已混合多个 Head 的输出 | Query 位置 |

<p align="center">表 5-2　Gate 的位置与作用对象</p>

以一个特征维度为例，Value Gating 计算 $$\sum_j a_{ij}g_jv_j$$；输出 Gating 计算 $$g_i\sum_j a_{ij}v_j$$。前者的 $$g_j$$ 随源位置变化，后者的 $$g_i$$ 对当前 Query 汇总的所有位置共同作用。只有满足额外条件时，两种形式才相等。

## 粒度与 Head 共享

令 Query Head 数为 $$h_q$$，每个 Head 的 Value 维度为 $$d_v$$，输入宽度为 $$d$$。下表使用直接线性投影生成 Sigmoid Gate，参数数目不含 bias。

| 参数化 | Gate 形状 | 投影参数数 |
| --- | --- | ---: |
| Head-specific、Elementwise | $$[N,h_q,d_v]$$ | $$dh_qd_v$$ |
| Head-specific、Headwise | $$[N,h_q,1]$$ | $$dh_q$$ |
| Head-shared、Elementwise | $$[N,1,d_v]$$ | $$dd_v$$ |
| Head-shared、Headwise | $$[N,1,1]$$ | $$d$$ |

<p align="center">表 5-3　直接投影生成 Gate 的参数化</p>

Headwise 的系数沿特征广播，Head-shared 的系数沿 Head 广播。上表是按输出维度直接构造投影的计数；如果先生成各 Head 的系数再求平均，实现的参数量不会按共享后的形状缩小。比较论文配置时，应查实际生成路径，而不能只看最终 Gate 张量。

## 动态系数与固定线性映射

将某位置的全部 Head 输出写成行向量 $$h_i$$。若 Gate $$g$$ 与输入无关且在推理时固定，则：

$$
(h_i\odot g)W^O=h_i\operatorname{diag}(g)W^O.
$$

此时可以将 $$\operatorname{diag}(g)$$ 合入输出权重。对输入自适应的 $$g_i=\operatorname{sigmoid}(x_iW^G)$$，该矩阵随输入变化，一般不能合并为一个固定权重。

类似地，固定 Attention 权重时，单个 Head 的 Value 与输出投影可写成 $$W_a^VW_a^O$$，秩不超过 Head 宽度。输入相关 Gate 改变了这条线性映射的组合方式。这个分析只针对固定匹配权重下的 Value 路径；完整 Attention 的权重本来就依赖输入。

Sigmoid、SiLU、加法与乘法还会改变系数范围和运算本身。讨论“Gated Attention”时，应给出具体公式；例如 $$H\odot\operatorname{sigmoid}(XW^G)$$ 与 $$H+\operatorname{SiLU}(XW^G)$$ 表达不同的调节方式。
