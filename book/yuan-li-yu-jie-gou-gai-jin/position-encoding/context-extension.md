# 3.4 RoPE Scaling 与上下文扩展

对采用 RoPE 的预训练模型，扩展上下文需要处理两个变化：出现更大的位置与距离，单个 Query 也会面对更多 Key。只增加可接收的 Token 数，没有同时完成模型对这些变化的适配。

以下设原训练上下文长度为 $$L$$，目标长度为 $$L'>L$$，扩展因子记为 $$\rho=L'/L$$。本节的 $$L$$ 表示长度；KV 容量公式中的层数符号 $$L$$ 另按所在章节定义。

## Position Interpolation 压缩位置坐标

直接外推保留角度 $$i\theta_r$$，使模型接触训练范围之外的位置组合。Position Interpolation（PI）改用：

$$
i'=\frac{i}{\rho},\qquad
R(i\theta_r)\longrightarrow R\left(\frac{i\theta_r}{\rho}\right).
$$

索引范围 $$[0,L')$$ 被映射回 $$[0,L)$$。例如从 4 个位置扩展到 8 个位置，$$\rho=2$$：新索引 $$0,1,\ldots,7$$ 对应 $$0,0.5,\ldots,3.5$$。相邻 Token 的相位差也缩小为原来的一半。[Chen et al.，§2.3，式 (4)](https://arxiv.org/html/2306.15595v2)

<figure><img src="../../.gitbook/assets/position-interpolation (1).png" alt="原训练范围为零至四的半开区间；直接外推的八个位置跨出范围，二倍 PI 将它们映射到零至三点五"><figcaption><p align="center">图 3-5　位置外推与 Position Interpolation</p></figcaption></figure>

PI 在方法上缩放位置，等价于将所有频率除以 $$\rho$$。它保留了顺序，却改变了相邻位置的间隔；原论文使用继续微调来适配这种改变，并评估长文本与原长度范围内的任务。上述映射本身只说明相位如何变化，模型质量还取决于训练。[Chen et al.，§3.1—3.4](https://arxiv.org/html/2306.15595v2)

## 非均匀缩放保留不同频率的作用

RoPE 的各维度对具有不同波长 $$\lambda_r=2\pi/\theta_r$$。统一缩放会同时压缩高频与低频的相位间隔。NTK-aware scaling 采用改变频率基底的方式，使不同维度缩放不同。以 YaRN 论文整理的形式为例，对 $$d>2$$：

$$
b'=b\rho^{d/(d-2)},\qquad
\theta'_r=(b')^{-2r/d}
=\theta_r\rho^{-2r/(d-2)}.
$$

由该式可直接看出：最高频的 $$r=0$$ 保持不变，最低频的 $$r=d/2-1$$ 恰好除以 $$\rho$$，中间维度平滑过渡。这个名称来自相关方法，不表示对任意模型都具有长度泛化保证。[Peng et al.，2309.00071v2，§3.1](https://arxiv.org/html/2309.00071v2)

## YaRN 结合分频调整与 Attention Scaling

YaRN（Yet another RoPE extensioN）先按各维度在原长度内的旋转圈数 $$L/\lambda_r$$ 选择缩放程度：低频维度采用插值，高频维度保留原频率，中间维度过渡。用 $$\gamma_r\in[0,1]$$ 表示从低频到高频的过渡系数，其频率形式为：

$$
\theta'_r=(1-\gamma_r)\frac{\theta_r}{\rho}+\gamma_r\theta_r.
$$

这里 $$\gamma_r=0$$ 对应 PI，$$\gamma_r=1$$ 对应保留原频率；图中采用 $$u_r=L\theta_r/(2\pi)$$、$$\gamma_r=\operatorname{clip}((u_r-\alpha)/(\beta-\alpha),0,1)$$，其中 $$\alpha=1,\beta=32$$；这些参数用于该公式示例，实际模型应记录所用配置。[Peng et al.，v2，§3.2，式 (18)—(20)](https://arxiv.org/html/2309.00071v2)

<figure><img src="../../.gitbook/assets/frequency-scaling (1).png" alt="四倍扩展示例：PI 对所有维度使用四分之一频率，NTK-aware 从高频保持到低频四分之一，分频方法在两端保留对应策略"><figcaption><p align="center">图 3-6　RoPE 各维度的频率缩放</p></figcaption></figure>

图 3-6 依据公式计算，显示频率比值，不是模型得分。YaRN 还引入温度 $$\tau>0$$，对 Softmax 前的内容分数进行调整。令 $$\widetilde K$$ 的第 $$j$$ 行为 $$\widetilde k_j^\top$$，则：

$$
A_{i,:}=\operatorname{softmax}\left(
\frac{\widetilde q_i^\top\widetilde K^\top}{\tau\sqrt d}+M_{i,:}
\right).
$$

把旋转后的 Q、K 同时乘以 $$1/\sqrt\tau$$，即可得到相同的分数缩放。YaRN 是**分频插值与 Attention Scaling 的组合**；只改变 RoPE 基底不等于实现了完整 YaRN。[Peng et al.，v2，§3.4，式 (21)—(22)](https://arxiv.org/html/2309.00071v2)

## LongRoPE 将缩放配置纳入搜索

前述方法用规则指定各频率的缩放。LongRoPE 进一步搜索各维度的缩放因子，并搜索保留原位置处理的起始 Token 数量，以校准文本上的语言建模损失选择配置。它还采用分阶段扩展与短上下文配置调整。[Ding et al.，§3.1—3.3、§4.3](https://arxiv.org/html/2402.13753v1)

这类方法把扩展方案与具体模型、校准数据联系起来。评估时因此还要记录搜索数据和目标长度，并使用独立的评估集，才能判断配置能否泛化到其他文本。

| 方法        | 相位或频率调整         | 需要一并关注的变化         |
| --------- | --------------- | ----------------- |
| 直接外推      | 原频率、原位置索引       | 超出训练的位置与距离范围      |
| PI        | 所有频率除以 $$\rho$$ | 局部间隔也被压缩          |
| NTK-aware | 增大基底，逐维非均匀缩放    | 实际有效扩展长度需评估       |
| YaRN      | 按频率分段插值，并调整分数尺度 | 频率配置、温度与适配训练      |
| LongRoPE  | 搜索逐维缩放及起始位置处理   | 搜索数据、分阶段适配与短上下文保留 |

<p align="center">表 3-5　RoPE 扩展方法的主要变化</p>

## 位置规则与 KV cache 的一致性

固定频率时，第 $$j$$ 个位置缓存的 Key 为 $$R_jk_j$$。如果生成过程中改变频率，新 Query 使用新的旋转规则，而旧 Key 仍保留原规则，点积就不再等于同一配置下的完整计算。

YaRN 的 Dynamic Scaling 讨论指出这一问题，并提出保留旋转前的 Key 以便使用当前规则。实际系统还需明确缓存格式和更新策略：重新旋转某层 Key 可以统一该层的旋转，但多层模型的旧隐藏状态也可能已受原位置规则影响，完整前缀重算与增量复用的等价性需要进一步验证。[Peng et al.，v2，§3.3](https://arxiv.org/html/2309.00071v2)

这里后一项是由第二章缓存等价条件推得的实现限制。验证扩展配置时，应同时比较 prefill 与增量 decode，并覆盖跨越原长度阈值的情况。

图示来源与参数见[图表来源](../../appendices/references.md#%E5%9B%BE%E8%A1%A8%E6%9D%A5%E6%BA%90)。

下一节：[3.5 长度泛化与长上下文评估](evaluation.md)。
