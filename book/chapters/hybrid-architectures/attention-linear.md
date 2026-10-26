# 10.3 Attention 与 Linear Attention 的组合

Linear Attention 层通过固定维度状态保存历史，Softmax Attention 层通过显式 KV 读取历史位置。把两者放在不同深度，可以同时保留状态压缩和逐位置读取的计算路径。本节中的 Attention 指 Softmax Attention，递推更新则沿用第九章的定义。

## 局部 KV 补充固定状态

考虑交替排列的 Linear Attention 与 Sliding-window Attention。递推层将每一步输入写入状态；局部 Attention 层保存窗口内各位置的 KV，在这些位置间进行 Query 相关的加权读取。窗口外信息仍可能经递推层传到后续表示中，但局部 Attention 不再直接访问这些位置原有的 KV。

这使近期逐位置比较与历史状态汇总有了不同的计算通道。例如，最近出现的多个相似标识符可以分别保存在窗口 KV 中，而较早输入通过递推状态影响当前表示。哪些信息实际由哪条通道承担，仍取决于训练，不能预先把每个事实分配给某类层。

DeltaNet 的一种混合配置交替使用 DeltaNet 层与 Sliding-window MQA 层。[Yang et al.，§3.4](https://arxiv.org/html/2406.06484v2#S3.SS4) Gated DeltaNet-H1 同样将 Gated DeltaNet 与 Sliding-window Attention（SWA）组合。[§3.3、Figure 1](https://arxiv.org/html/2412.06464v1#S3.SS3)

## 少量全局层保留前缀读取

另一种配置在以递推层为主的模型中保留少量 Full Attention 层。与局部窗口相比，这些层可以直接访问当前前缀内所有保留的 KV，但对应缓存仍随上下文增长。

DeltaNet 论文研究了仅在第 2 层和第 $$L/2+1$$ 层使用 Global Attention 的配置，其中 $$L$$ 是模型总层数，编号从 1 开始。[§3.4](https://arxiv.org/html/2406.06484v2#S3.SS4) 这是一项具体实验布局，层比例和位置应作为待比较的设计变量。

| 配置 | 读取路径 | 需要持续保存的历史 |
| --- | --- | --- |
| 纯 Linear Attention | 从递推状态读取 | 各递推层的状态及其他辅助缓存 |
| Linear Attention + Local Attention | 状态读取与窗口内 KV 读取 | 递推状态、局部 KV 和辅助缓存 |
| Linear Attention + Full Attention | 状态读取与全前缀 KV 读取 | 递推状态、全局层 KV 和辅助缓存 |

<p align="center">表 10-4　递推状态与显式 KV 的组合</p>

同一位置进入不同层时，表示已经改变，各层缓存不能仅因位置相同而直接共享。普通层间组合也不会自动把窗口外 KV 转存到另一层的递推状态；递推层按自己的输入和更新规则写入。若模型专门设计 KV 共享或跨模块状态传递，应按该接口另行分析。

## Gated DeltaNet 的混合配置实验

Gated DeltaNet 论文还研究了 H2：在层间组合 Mamba2、Gated DeltaNet 和 SWA。它同时改变递推模块的类型及组合布局，适合与 H1 一起观察不同指标的变化。[§3.3](https://arxiv.org/html/2412.06464v1#S3.SS3)

下表摘录该论文 Table 2。所列实验采用约 1.3B 参数、100B FineWeb-Edu 训练 Token、4K 训练长度；混合模型的 SWA 窗口为 2K。三行使用相同实验设置，数值为原论文报告，非本书复现。

| 模型 | WikiText Perplexity ↓ | 论文报告的 Avg. ↑ |
| --- | ---: | ---: |
| Gated DeltaNet | 16.42 | 55.32 |
| Gated DeltaNet-H1 | 16.07 | 56.40 |
| Gated DeltaNet-H2 | 15.91 | 56.18 |

<p align="center">表 10-5　Gated DeltaNet 混合配置的部分结果</p>

来源：[实验设置与 Table 2](https://arxiv.org/html/2412.06464v1#S4)。Avg. 是八项任务准确率的均值，其中 HellaSwag 和 ARC-Challenge 采用 normalized accuracy，其余任务采用 accuracy。H1 的八个展示值直接求均值约为 56.46，与表中 56.40 不一致；本表保留论文报告值。其余两行按展示值求均值与报告值一致到两位小数。

在这组实验中，H1、H2 的两项汇总结果均优于纯 Gated DeltaNet；H2 的 WikiText Perplexity 更低，H1 的准确率均值更高。因此，选择配置需要结合目标任务查看分项结果，并分别测量运行成本。仅凭这些质量指标，无法判断训练或 Decode 更快。

层比例、窗口和递推状态维度同时影响计算与存储，具体计数见[10.5 层间配置与状态管理](state-management.md)；训练预算控制与窗口内外检索的比较方法见[10.6 模型能力与资源成本的比较](evaluation.md)。
