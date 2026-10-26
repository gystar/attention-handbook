# 6.4 并行调度与硬件适配

分块算法确定了需要计算什么，GPU 调度则决定这些计算由谁执行、何时执行。HBM 读写减少之后，执行时间还可能受到并行任务不足、片上通信或 Softmax 运算的限制。FlashAttention 后续版本围绕这些具体成本调整工作分配。

## FlashAttention-2 的工作划分

前向中，每个 Query 块只写自己的输出行，可以分配给不同的 Thread Block。这样，即使 Batch Size 和 Head 数较小，较长序列仍能提供多个并行任务。Thread Block 内部再按 Query 行向不同 Warp 分配工作，使其持有各自的输出累加器，减少部分输出之间的归约通信。[FlashAttention-2 §3.2—3.3](https://tridao.me/publications/flash2/flash2.pdf)

这里，Warp 是 GPU 将一组线程共同调度的单位，Thread Block 则包含多个 Warp，可通过 Shared Memory 协作。分块有两个不同尺度：先将 Query 块分配给 Thread Block，再在块内分配线程的计算任务。

| 划分位置 | 分配对象 | 主要影响 |
| --- | --- | --- |
| Thread Block 之间 | 不同 Query 块 | 增加可并行的输出行任务 |
| Thread Block 内部 | 不同 Query 行的计算 | 减少输出部分和的跨 Warp 合并 |
| 反向梯度累加 | 同一输入收到的多个贡献 | 需要归约或原子累加 |

<p align="center">表 6-5　Attention 并行计算的任务划分</p>

例如，一个 Query 块中的所有行都需要读取同一批 Key 和 Value。按 Query 行划分时，每个工作单元最终得到完整的若干输出行；若按 Key 位置划分，同一个输出行会分散到多个工作单元，需要按 6.2 的合并关系汇总。选择划分方式时，应同时考虑可用并行度与合并成本。

## FlashAttention-3 的异步流水线

FlashAttention-3 面向 Hopper，利用 TMA（Tensor Memory Accelerator）异步搬运数据，利用 WGMMA（Warpgroup Matrix Multiply-Accumulate）异步执行矩阵乘法。Producer 负责准备下一块数据，Consumer 处理已经就绪的数据，并通过同步机制保护共享 buffer。[论文 §2.2、§3.1](https://arxiv.org/html/2407.08608v2)

异步还允许矩阵乘法与 Softmax 重叠。论文的 [Figure 1](https://arxiv.org/html/2407.08608v2/figs/pingpong_pipelining.png) 展示 Pingpong Scheduling：一个 Warpgroup 执行矩阵乘法时，另一个执行 Softmax；角色交替，利用不同执行单元。图中同色表示同一迭代，示意时间线用于解释调度关系，实际执行仍受同步和数据依赖约束。

## FlashAttention-4 的片上资源优化

FlashAttention-4 面向 Blackwell 的资源比例变化：Tensor Core 吞吐增长快于指数运算及 Shared Memory 带宽。其前向将部分指数计算交给普通 FMA 单元，以多项式近似分担专用单元的负载；还通过条件重缩放减少 Online Softmax 的额外操作。[论文 §3.1.1—3.1.4](https://arxiv.org/html/2603.05451v1)

条件重缩放保留一个仍可用的指数基准，直到需要更新时才同时调整分子和分母。它要求统计量始终处于一致尺度；直接省略缩放而继续套用新最大值，会破坏 6.2 的合并等式。指数近似则属于数值实现选择，需要单独评估误差。

片上 Tensor Memory 也允许重新安排反向矩阵乘法与逐元素运算的顺序，扩大可重叠的部分。[论文 §3.2.1—3.2.2](https://arxiv.org/html/2603.05451v1) 因而，分析 kernel 时除了 HBM，还需观察片上数据搬运、寄存器占用和执行单元是否存在长时间等待；同一分块公式可以对应不同的硬件瓶颈。

## 低精度矩阵乘法

除调度方式外，矩阵乘法的数据类型也影响硬件吞吐。SageAttention 的 SAGEAttn-B 采用如下混合精度路径：Q/K 按块量化为 INT8 → 计算点积并恢复尺度 → 高精度 Online Softmax → FP16 的加权求和。最后一步的矩阵乘法采用 FP16 累加器。[Zhang et al.，§4.1、§4.4，Table 6](https://arxiv.org/html/2410.02367v1#S4)

量化前，方法从所有 Key 中减去同一个均值向量。对固定 Query，这只给整行分数减去相同常量，因此在实数运算下保持 Softmax 权重；随后的量化仍引入近似误差。[§4.2](https://arxiv.org/html/2410.02367v1#S4.SS2)

这里改变的是计算格式；[8.3 KV Cache Quantization](../kv-cache/quantization.md)讨论历史数据的存储格式。两者可以组合，评估时分别记录缓存、乘法输入、累加器与输出的精度，并计入量化和尺度恢复的耗时。

## 跨设备的序列与 Head 划分

单卡分块减少中间矩阵的存储，跨设备划分进一步分担长序列的激活与计算。两种代表方式改变数据分布，同时保留指定 Mask 下的 Attention 运算：

- **Ring Attention**：各设备持有自己的 Query 块，KV 块沿环依次传递；本地每收到一个块，就更新 6.2 的行最大值、归一化分母与输出分子。不同 KV 块的结果按共同尺度合并，不能先各自归一化再直接平均。原论文 [§3、Figure 2](https://arxiv.org/html/2310.01889v4#S3) 展示 KV 轮转与计算通信重叠。
- **DeepSpeed-Ulysses**：先沿序列切分输入，用 All-to-All 将 QKV 改排为“完整序列、部分 Head”，各设备计算所分配 Head 的 Attention，再用反向布局的 All-to-All 恢复序列切分。原论文 [§3.1、Figure 2](https://arxiv.org/html/2309.14509v1#S3.SS1) 展示这两次布局转换；本地计算可使用 FlashAttention。

以标准 MHA 为例，若设备数 $$p$$ 整除序列长度 $$N$$ 和 Head 数 $$h$$，Ulysses 的本地 QKV 布局由 $$(N/p,h,d_h)$$ 变为 $$(N,h/p,d_h)$$，元素数不变。这个直接划分要求有足够的 Head；GQA/MQA 还需按实际 KV Head 数安排分片或复制，不能仅按 Query Head 数推算通信和容量。

Ring 的通信隐藏取决于每轮可重叠的计算。设当前 Query、KV 块分别含 $$b_q,b_k$$ 个位置，Key/Value 维度同为 $$d_h$$，每元素 $$s$$ 字节。两次矩阵乘法约需 $$4b_qb_kd_h$$ FLOPs，一份 KV 消息含 $$2b_kd_hs$$ 字节。若有效计算吞吐为 $$F$$、单向有效传输带宽为 $$B$$，并假设收发可并行，理想重叠至少需要：

$$
\frac{4b_qb_kd_h}{F}\gtrsim\frac{2b_kd_hs}{B}.
$$

这是块级估算，尚需计入启动延迟、同步、收发竞争及流水线首尾。Causal Mask 的有效配对与各设备工作量也可能不同。完整连接的总计算量仍随长度平方增长；多卡分担计算，并不自动改变该复杂度。性能报告应分别给出每卡峰值显存、通信量和端到端延迟。
