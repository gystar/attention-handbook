# 3.2 Absolute 与 Relative Position Encoding

位置可以表示为序列中的索引，也可以表示为两个位置之间的偏移。前者为各位置提供向量，后者直接描述 Query 与 Key 的关系。

## Absolute Position Encoding 为位置赋予向量

设 Token embedding 为 $$e_i\in\mathbb{R}^{d_{\mathrm{model}}}$$，同维度的位置向量为 $$p_i$$，网络底层输入写成 $$x_i=e_i+p_i$$。这里的 $$e_i$$ 已包含模型规定的 embedding 缩放。位置向量可以通过查表学习，也可以由固定函数计算。

原始 Transformer 使用不同频率的正弦与余弦。对偶数维度 $$d_{\mathrm{model}}$$，令 $$r=0,\ldots,d_{\mathrm{model}}/2-1$$：

$$
p_{i,2r}=\sin(i\omega_r),\qquad
p_{i,2r+1}=\cos(i\omega_r),\qquad
\omega_r=10000^{-2r/d_{\mathrm{model}}}.
$$

相邻两维构成同一频率的 sin/cos 对。图 3-2 中，每行对应一个位置；左侧较高频的维度变化较快，右侧较低频的维度在这段位置范围内变化较慢。[Vaswani et al.，§3.5](https://arxiv.org/html/1706.03762v7)

<figure><img src="../../.gitbook/assets/alammar-sinusoidal (1).png" alt="Jay Alammar 的正弦位置编码热图：横轴为编码维度，纵轴为 Token 位置，颜色为编码值"><figcaption><p align="center">图 3-2　不同频率的 Sinusoidal Position Encoding</p></figcaption></figure>

固定偏移 $$\Delta$$ 对每对 sin/cos 的影响可用一个相同的线性变换表达：

$$
\begin{bmatrix}
\sin((i+\Delta)\omega_r)\\ \cos((i+\Delta)\omega_r)
\end{bmatrix}
=
\begin{bmatrix}
\cos(\Delta\omega_r)&\sin(\Delta\omega_r)\\
-\sin(\Delta\omega_r)&\cos(\Delta\omega_r)
\end{bmatrix}
\begin{bmatrix}\sin(i\omega_r)\\\cos(i\omega_r)\end{bmatrix}.
$$

这由三角函数加法公式得到，为学习相对关系提供了结构。但输入相加后再计算 Q/K，分数同时包含内容项、位置项及交叉项，因此它通常并非只依赖相对距离。[D2L，§11.6.3.2](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)

两类 Absolute 方法的长度限制不同：可学习位置表只为配置范围内的索引提供参数；Sinusoidal 公式可以为更大的索引计算向量，但模型是否能利用这些位置，仍需超出训练长度的评估。

## Relative Position Encoding 直接表示位置对

Shaw et al. 为偏移 $$j-i$$ 学习向量。令截断半径为 $$k$$，将偏移截断到 $$[-k,k]$$，分别查得 $$a^K_{ij}$$ 与 $$a^V_{ij}$$。为便于展示，这里取 $$d_k=d_v=d$$，各向量均为 $$d$$ 维：

$$
s_{ij}=\frac{q_i^\top(k_j+a^K_{ij})}{\sqrt d}+M_{ij},\qquad
A_{i,:}=\operatorname{softmax}(s_{i,:}),\qquad
o_i=\sum_j A_{ij}(v_j+a^V_{ij}).
$$

相对位置既参与匹配，也可以参与输出。截断后，距离超过阈值的位置共享同一方向的边界表示；这种表示保留了左右方向，但不再区分更远位置的精确距离。[Shaw et al.，§3.1—3.2，式 (3)—(4)](https://aclanthology.org/N18-2074.pdf)

## Relative Position Bias 简化为分数偏置

T5 将相对位置表示简化为每个 Head 的可学习标量，并直接加到 Attention logits。以 $$\operatorname{bucket}(j-i)$$ 表示偏移所属的区间：

$$
s^{(a)}_{ij}=c^{(a)}_{ij}+b^{(a)}_{\operatorname{bucket}(j-i)}+M_{ij}.
$$

这里 $$c^{(a)}_{ij}$$ 表示该模型第 $$a$$ 个 Head 的内容匹配分数；此式只描述位置项，不规定内容分数的缩放约定。较远距离被归入较宽的区间，位置参数在层间共享，各 Head 使用不同参数。[Raffel et al.，§2.1](https://arxiv.org/html/1910.10683v4)

| 方法                  | 位置参数或函数       | 主要作用             | 距离分辨方式       |
| ------------------- | ------------- | ---------------- | ------------ |
| Learned Absolute    | 每个索引一个可学习向量   | 输入表示             | 按绝对索引查表      |
| Sinusoidal Absolute | 固定 sin/cos 函数 | 输入表示             | 多频率组合        |
| Shaw Relative       | 按截断偏移学习向量     | 匹配及可选的 Value 汇总  | 阈值内逐偏移，阈值外共享 |
| T5 Relative Bias    | 按距离区间学习标量     | Attention logits | 多个偏移可落入同一区间  |

<p align="center">表 3-3　Absolute 与 Relative 方法的处理方式</p>

图示出处与许可见[图表来源](../../appendices/references.md#%E5%9B%BE%E8%A1%A8%E6%9D%A5%E6%BA%90)。

下一节：[3.3 RoPE 与 ALiBi 的位置建模](rope-and-alibi.md)。
