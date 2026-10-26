# 10.4 Attention 与 State Space Models 的组合

State Space Models（SSMs）用随输入更新的状态描述序列。用于语言模型的结构化 SSM 通过限制状态转移的形式，使长序列计算和增量推理更高效。与 Attention 组合时，需要同时考虑 SSM 如何更新历史，以及 Attention 保留多大范围的显式 KV。

## 从状态转移到输入相关的选择

先看一个输入和输出均为标量的离散 SSM：

$$
h_t=\bar A_t h_{t-1}+\bar B_t x_t,\qquad
 y_t=C_t^\top h_t.
$$

这里 $$h_t\in\mathbb R^{d_s}$$ 是状态，$$\bar A_t\in\mathbb R^{d_s\times d_s}$$ 描述状态转移，$$\bar B_t,C_t\in\mathbb R^{d_s}$$ 分别控制输入写入和状态读取。公式省略输入到输出的直接项及外围投影，用于解释 SSM 核心；完整模块通常包含多个通道、短卷积和 Gate。

若参数不随位置变化且初始状态为零，这一线性递推可用固定卷积核计算。初始状态非零时，还需加上它经状态转移传播到输出的响应。Mamba 的 Selective SSM 让 $$B_t,C_t$$ 和离散化步长 $$\Delta_t$$ 依赖输入；离散转移 $$\bar A_t$$ 因而也可随输入变化。其基础参数 $$A$$ 仍是训练得到的参数，而非每步从输入重新生成的任意矩阵。[Gu & Dao，§2、§3.2、Algorithm 2](https://arxiv.org/html/2312.00752v2#S3.SS2)

“Selective” 指输入相关的状态更新与读取。它不需要像动态 Sparse Attention 那样生成历史位置索引：过去输入通过状态影响后续计算，模型按当前输入调节这一过程。输入相关参数也使固定卷积核的计算形式不再普遍适用，Mamba 据此采用 Selective Scan。

## Mamba-2 与 Linear Attention 的关系

Mamba-2 引入 Structured State Space Duality（SSD）。其核心联系可以从上述递推的一个特例看出：令 $$\bar A_t=a_tI$$、$$h_0=0$$，则

$$
y_t=\sum_{j=1}^{t}\left(\prod_{r=j+1}^{t}a_r\right)(C_t^\top\bar B_j)x_j.
$$

每项由输入写入、期间累计衰减与当前读取共同决定。这个形式类似带衰减 Mask 的未归一化 Linear Attention：$$C_t$$ 对应读取特征，$$\bar B_j$$ 对应写入特征。SSD 将具有特定结构的 SSM 与 Structured Masked Attention 联系起来，并据此构造适合矩阵乘法的分块算法。[Dao & Gu，§5.1—5.3、§6](https://arxiv.org/html/2405.21060v1#S5)

这一对应要求状态转移为标量乘单位矩阵，且读取式不含 Softmax 分母。它说明特定 SSM 核心与 Linear Attention 的联系；完整模块中的投影、Gate 和短卷积仍需按各自定义计算。

## 全局 Attention 与窗口 Attention 的选择

Jamba 将 Mamba 与 Transformer Attention 层组合；原论文配置的 Attention:Mamba 层比例为 1:7，另有独立的 MoE 配置。[Lieber et al.，§2](https://arxiv.org/html/2403.19887v2#S2) Mamba 层保存递推状态，Attention 层保留其所需的历史 KV，因此总历史存储包含固定部分和随长度增长的部分。

Samba 则组合 Mamba、Sliding-window Attention 与 MLP，论文使用 2048 的局部窗口。[Ren et al.，§2.1](https://arxiv.org/html/2406.07522v1#S2.SS1) 如果局部层及时释放窗口外 KV，且各模块状态维度固定，其增量历史缓存可以保持有界；能够继续运行的序列长度与有效记忆长度仍需分别评价。

| 结构 | 历史访问方式 | 随长度增长的主要状态 |
| --- | --- | --- |
| SSM + Full Attention | SSM 状态与全前缀 KV | 全局 Attention 层的 KV |
| SSM + 固定窗口 Attention | SSM 状态与近期 KV | 窗口填满后无新增历史容量，前提是及时释放旧 KV |
| SSM + Linear Attention + 窗口 Attention | 多种递推状态与近期 KV | 同样取决于固定维度及窗口保留策略 |

<p align="center">表 10-6　SSM 混合模型的历史访问与状态</p>

最后一种配置的实例是上一节介绍的 Gated DeltaNet-H2。这里的分类按状态与访问路径组织，不把所有递推模块都归为同一种 SSM。Griffin 的 RG-LRU 也采用门控递推，但其论文明确说明该层并非由连续系统离散化定义。[De et al.，§2.4](https://arxiv.org/html/2402.19427v1#S2.SS4)

SSM 核心之外，短卷积也需要保存近期输入。分块 Prefill 和前缀复用必须恢复完整模块所需的状态，具体计数与恢复条件见[10.5 层间配置与状态管理](state-management.md)。
