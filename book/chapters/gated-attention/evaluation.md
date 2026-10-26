# 5.4 训练证据与比较条件

Gating 的数学形式说明它能调节哪些张量；训练稳定性、质量与长上下文表现需要实验支持。比较时应同时检查模型配置、训练预算和评估协议。

## 质量与稳定性分别评估

Qiu et al. 的一组 Dense 实验固定为 28 层、1.7B 参数、400B 训练 Token、batch size 1024，最大学习率为 $$4\times10^{-3}$$。为维持参数规模，加入 Gate 的模型缩小 FFN 宽度。下表摘录作者的评估结果。[§3.2.2、Table 2，行 (1)—(2)](https://arxiv.org/html/2505.06708v1)

| 配置 | Avg PPL ↓ | MMLU ↑ |
| --- | ---: | ---: |
| Baseline | 7.499 | 50.21 |
| SDPA Elementwise Gate | 7.404 | 51.15 |

<p align="center">表 5-5　指定训练设置下的质量比较</p>

这些数字来自原论文，未在本书中复现。参数量对齐伴随 FFN 宽度变化，因此比较对象是完整的两套配置。

稳定性的证据来自训练过程：作者在 3.5T Token 实验中观察到输出 Gate 减少 loss spikes；在提高最大学习率的设置中，部分 Baseline 出现收敛问题，而相应 Gated 配置能够训练并获得更好的评估结果。[§3.2.2、Figure 1、Table 2](https://arxiv.org/html/2505.06708v1) 这与上表的最终质量指标是不同证据；比较稳定性还需记录发散次数、梯度与激活统计。

## 长上下文的适配条件

同一论文从已训练 3.5T Token 的模型出发，将 RoPE base 从 10k 调整为 1M，在 32k 序列上继续训练 80B Token，再用 YaRN 扩展到 128k，并运行 RULER。经过 YaRN 扩展后，Baseline 与 Gated 模型在原有 32k 范围内的得分均下降，但 Gated 模型下降较少；在 64k 和 128k 上，Gated 模型得分也更高。[§4.4、Table 5](https://arxiv.org/html/2505.06708v1)

这些结果支持输出 Gate 在该适配流程中的作用。Attention Sink 减少如何影响长度扩展，作者仍以机制假设讨论，尚未给出严格的理论解释。[Limitations](https://arxiv.org/html/2505.06708v1)

| 要比较的因素 | 建议固定或分别报告的条件 |
| --- | --- |
| Gate 位置与粒度 | 参数量、FFN 宽度、归一化、初始化 |
| 训练稳定性 | 学习率、batch、精度、训练步数、随机种子 |
| 模型质量 | 数据、训练 Token 数、任务与打分协议 |
| 长上下文 | 原训练长度、继续训练长度、RoPE/YaRN 配置 |
| 执行开销 | Gate 投影、逐元素运算、融合方式、输入长度与并发 |

<p align="center">表 5-6　Gating 实验的比较条件</p>

## 与其他优化组合

输出 Gate 可以接在 GQA 的多个 Query Head 输出之后，仍按 Query Head 而非 KV Head 生成调节系数。它也可以接在数学输出相同的 SDPA 实现之后；具体融合方式影响额外内存和执行时间。

若原路径已利用第四章的投影合并，需要区分 Gate 的粒度。沿用 4.3 的列向量记法，逐元素 Gate 会得到 $$W_a^O\operatorname{diag}(g_{t,a})U_a^Vz_{t,a}$$；对角矩阵随输入变化，一般无法预先合并成一个固定投影。

如果每个 Head 只使用一个标量 Gate $$g_{t,a}$$，则可利用标量乘法与线性映射的交换关系：

$$
W_a^O\bigl(g_{t,a}U_a^Vz_{t,a}\bigr)
=(W_a^OU_a^V)\bigl(g_{t,a}z_{t,a}\bigr).
$$

此时仍可预先合并 $$W_a^OU_a^V$$，再对每个 Query 的 Latent 汇总结果施加标量 Gate。Gate 本身仍需随输入计算；保留固定投影合并与把 Gate 吸收到固定权重中是两件事。以上由 4.3 的线性式直接推得，实际性能还取决于执行方式。

对输出 Gating，额外投影与逐元素乘法是实际开销；Gate 接近零并不自动降低 Attention 的位置对计算量。第六章将讨论如何在保持给定 Attention 运算的条件下优化数据访问。
