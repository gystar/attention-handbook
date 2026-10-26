# 7.1 稀疏连接与计算成本

FlashAttention 改变同一 Attention 运算的执行方式。Sparse Attention 进一步限制每个 Query 可以读取的位置，从而减少参与计算的 Query–Key 配对。Sparse Transformer 将这种连接关系写成每个位置的可访问集合，并研究跨层传播如何补充单层的稀疏连接。[Child et al.，§4.2—4.3](https://arxiv.org/html/1904.10509v1#S4.SS2)

## 在选定位置上重新归一化

令 $$\mathcal A_i$$ 为 Query $$q_i$$ 可以访问的 Key 位置集合，且 $$\mathcal A_i\ne\varnothing$$。单个 Head 的输出为：

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}},\qquad
p^{\mathcal A}_{ij}=\frac{e^{s_{ij}}}{\sum_{t\in\mathcal A_i}e^{s_{it}}},\qquad
o_i^{\mathcal A}=\sum_{j\in\mathcal A_i}p^{\mathcal A}_{ij}v_j.
$$

这里 $$d_h$$ 是 Query/Key 维度，$$v_j\in\mathbb R^{d_v}$$。Causal Attention 还要求 $$\mathcal A_i\subseteq\{1,\ldots,i\}$$。集合之外的权重为零，集合之内重新归一化；直接将原权重置零而不调整分母，会得到不同的输出。

以本书构造的单 Query 算例说明：原权重为 $$[0.2,0.3,0.5]$$，标量 Value 为 $$[1,2,10]$$，完整输出为 $$5.8$$。只保留前两个位置后，权重变为 $$[0.4,0.6]$$，输出为 $$1.6$$；若只删掉第三项而不重新归一化，结果则是 $$0.8$$。

## 按有效连接计数

令 $$E=\sum_i|\mathcal A_i|$$ 为一个 Head 的有效配对数。只计算这些配对时，分数与 Value 汇总的算术成本为 $$O(E(d_h+d_v))$$，Softmax 还需 $$O(E)$$ 的指数与归约运算。以下按长度为 $$N$$ 的 Causal Self-Attention 计数：

| 连接范围 | 每行最多读取 | 总配对数 |
| --- | --- | --- |
| 全部历史及自身 | $$N$$ | $$N(N+1)/2$$ |
| 含自身的窗口，宽度 $$w\le N$$ | $$w$$ | $$Nw-w(w-1)/2$$ |
| 每行预算最多 $$k$$ 个位置 | $$k$$ | 至多 $$Nk$$ |

<p align="center">表 7-1　Attention 连接数量</p>

窗口的精确计数来自 $$1+2+\cdots+w+(N-w)w$$。只有在 $$w$$ 或 $$k$$ 不随 $$N$$ 增长时，相应配对数才随长度线性增长。动态方法还要计算选择集合的开销，不能只统计选中位置的 Attention 运算。

有效配对数描述数学工作量。若实现仍先执行完整 $$QK^\top$$ 再施加稀疏 Mask，就已经支付了完整分数计算的成本；如何跳过未选中的块将在 7.3 展开。

## 删除权重的输出误差

对于同一组固定的分数与 Value，可以用被删除的权重质量解释输出变化。设完整分布为 $$p_j$$，删除集合为 $$R$$，保留集合为 $$A$$，删除质量为 $$\delta=\sum_{j\in R}p_j$$，且 $$0<\delta<1$$。定义两组条件均值：

$$
\mu_A=\frac{\sum_{j\in A}p_jv_j}{1-\delta},\qquad
\mu_R=\frac{\sum_{j\in R}p_jv_j}{\delta}.
$$

完整输出满足 $$o=(1-\delta)\mu_A+\delta\mu_R$$，稀疏输出为 $$o^A=\mu_A$$。因此：

$$
o-o^A=\delta(\mu_R-\mu_A).
$$

若所有 Value 满足 $$\|v_j\|\le B$$，由三角不等式可得 $$\|o-o^A\|\le2\delta B$$。这是本书针对固定输入的一步推导：误差同时取决于删除质量和两组 Value 的差异。它不直接给出整模型质量保证，也没有包含选择器、后续层或训练适应的影响。
