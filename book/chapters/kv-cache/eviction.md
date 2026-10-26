# 8.4 Token Selection 与 Cache Eviction

当缓存预算小于完整历史时，系统需要决定保留哪些位置。Token Selection 给出位置集合；Cache Eviction 则释放或覆盖被淘汰位置的 KV，改变后续计算可以直接访问的历史。这里的选择单位是 Token **位置**，同一个词在不同位置产生不同的 KV。

## 当前读取与历史保留

令第 $$t$$ 步之前可用的缓存位置集合为 $$C_{t-1}$$，本步追加位置 $$t$$，Attention 实际读取集合为 $$A_t$$。暂不考虑从外部存储恢复或重新计算，则：

$$
A_t\subseteq C_{t-1}\cup\{t\},\qquad
C_t\subseteq C_{t-1}\cup\{t\}.
$$

$$A_t$$ 决定当前计算，$$C_t$$ 决定下一步仍能访问什么。第七章的 Quest 保留候选页，只减少本次读取；淘汰策略则将部分位置从 $$C_t$$ 中移除。以后若需要这些位置，必须从其他存储恢复，或承担重新计算的成本。

例如，某步对位置 3 的 Attention 权重很小，下一步 Query 却可能转向位置 3。仅根据当前权重删除它，会将对当前 Query 的近似变成对未来 Query 的约束。因此，保留策略通常结合近期位置、历史评分或 Prompt 中可观察到的任务信息。

## StreamingLLM 的初始位置与近期窗口

StreamingLLM 保留少量初始位置的 KV，并维护近期位置的滚动窗口。初始位置承担 Attention Sink 的作用，近期窗口提供局部语言建模所需的上下文。[Xiao et al.，§3.1—§3.2](https://arxiv.org/html/2309.17453v4#S3)

若初始保留数为 $$s$$、近期保留数为 $$w$$，本书将其位置集合写为：

$$
C_t=\{0,\ldots,s-1\}\cup\{\max(s,t-w+1),\ldots,t\},\qquad t\ge s-1.
$$

空区间按空集处理，因此集合大小最多为 $$s+w$$。这是保留位置的表达；论文还按缓存内的连续位置处理相对位置编码。其 RoPE 方案缓存旋转前的 Key，读取时按缓存位置施加旋转；ALiBi 则按连续缓存距离计算偏置。[§3.2](https://arxiv.org/html/2309.17453v4#S3.SS2)

初始位置可以继续稳定归一化，但已淘汰的中间位置不再以原始 KV 形式供后续 Query 读取。评估长期运行时，应分别检查局部生成质量与远距离信息检索。Attention Sink 的形成与 Gated Attention 的关系见第五章。

## H₂O 的累计 Attention 评分

H₂O 同时保留近期位置和累计 Attention 权重较高的历史位置，后者称为 Heavy Hitters。生成过程中，系统更新评分并淘汰预算之外的 KV。[Zhang et al.，§3.2、§4.1、Appendix A](https://arxiv.org/html/2306.14048v3)

为说明累计统计，固定某层某个 Head，设本步可访问位置 $$j$$ 的归一化权重为 $$a_{t,j}$$，可以用以下记号表示其累计分数：

$$
H_{t,j}=H_{t-1,j}+a_{t,j}.
$$

该式说明统计量的含义；具体更新时机、近期区间和预算划分仍属于缓存策略。当前权重来自实际保留集合上的归一化，已经淘汰的位置也无法再参与正常评分。历史累计值因而是在线决策依据，不能视为未来需求的已知答案。

原论文的近似保证针对满足 Submodular 假设的目标函数；实际生成质量仍需通过模型与任务实验评估。[§4.2](https://arxiv.org/html/2306.14048v3)

## SnapKV 的 Prompt 选择

SnapKV 在 Prompt 尾部设置 Observation Window，汇总该窗口中的 Query 对前缀位置的 Attention 权重。随后沿位置方向做 Pooling，再按 Head 选择 Top-k 位置，与 Observation Window 的全部 KV 一起保留。[Li et al.，§4.1、§4.3、Listing 1](https://arxiv.org/html/2404.14469v2#S4)

Pooling 作用于位置评分，选中后收集的仍是相应位置的 KV。它使相邻位置的评分相互影响，帮助保留局部信息；不是将邻域 Value 平均成一个新 Value。

设 Prompt 缓存预算为 $$B$$、Observation Window 长度为 $$W<B$$。对需要压缩的 Prompt，伪代码从较早前缀选择 $$B-W$$ 个位置，再保留窗口中的 $$W$$ 个位置。这个预算约束的是 **Prompt KV**：若后续生成位置全部追加，总缓存仍随已处理的生成位置数增长。

## 保留策略的实现与评估

| 方法 | 主要决策依据 | 保留对象 | 需要检查的条件 |
| --- | --- | --- | --- |
| StreamingLLM | 位置与近期性 | 初始位置及滚动窗口 | 相对位置处理、远距离信息损失 |
| H₂O | 累计 Attention 权重与近期性 | 动态更新的历史集合 | 评分更新、统计偏差、淘汰成本 |
| SnapKV | Prompt 尾部的 Attention 观测 | 选中的 Prompt 位置及观察窗口 | 任务变化、生成阶段增长、Pooling 参数 |

<p align="center">表 8-7　KV 保留策略的决策依据</p>

位置选择落实到存储时，还需保留三个对应关系：Key 与 Value 成对收集；位置元数据与所采用的位置编码一致；各 Head 的选择集合与实际 KV Head 布局一致。尤其在 GQA 中，多个 Query Heads 共享同一 KV Head；独立选择后若复制 KV，会抵消部分共享收益。应明确实现采用共享集合、集合并集，还是允许额外副本。

评分、Top-k、索引收集和数据整理也占用时间。若沿用不显式保存完整 Attention 矩阵的 kernel，策略还需要获取评分所需的统计量。评估应计入这些额外操作，并分别报告 Prefill 后、Decode 期间的缓存峰值，以及长生成和任务切换后的质量变化。
