# Summary

* [大模型 Attention：原理、演进与优化](README.md)

## 总览

* [第 1 章 Attention 机制的演进与主要发展方向](chapters/overview/README.md)
  * [1.1 从 Representation Learning 到 Attention 机制](chapters/overview/introduction.md)
  * [1.2 从 RNN 到 Transformer](chapters/overview/from-seq2seq-to-transformer.md)
  * [1.3 LLM 中的 Attention 成本与长上下文需求](chapters/overview/bottlenecks-and-context.md)
  * [1.4 Attention 机制的主要发展方向与技术路线](chapters/overview/optimization-landscape.md)

## 原理与结构改进

* [第 2 章 Transformer Attention 的基本原理与计算过程](chapters/transformer-attention/README.md)
  * [2.1 Transformer 中的 Attention 结构与计算流程](chapters/transformer-attention/structure-and-flow.md)
  * [2.2 Query、Key、Value 与 Scaled Dot-Product Attention](chapters/transformer-attention/scaled-dot-product.md)
  * [2.3 Self-Attention、Cross-Attention 与 Causal Attention](chapters/transformer-attention/attention-and-masks.md)
  * [2.4 Multi-Head Attention 的结构与 Output Projection](chapters/transformer-attention/multi-head.md)
  * [2.5 Training、Prefill 与 Decode 的计算差异](chapters/transformer-attention/training-prefill-decode.md)
* [第 3 章 Position Encoding 的演进与 Context Extension](chapters/position-encoding/README.md)
  * [3.1 Attention 中的顺序与位置信息](chapters/position-encoding/position-and-order.md)
  * [3.2 Absolute 与 Relative Position Encoding](chapters/position-encoding/absolute-and-relative.md)
  * [3.3 RoPE 与 ALiBi 的位置建模](chapters/position-encoding/rope-and-alibi.md)
  * [3.4 RoPE Scaling 与上下文扩展](chapters/position-encoding/context-extension.md)
  * [3.5 长度泛化与长上下文评估](chapters/position-encoding/evaluation.md)
* [第 4 章 MQA、GQA 与 MLA 的 KV 表示优化](chapters/kv-representations/README.md)
  * [4.1 KV 表示与缓存成本](chapters/kv-representations/cache-cost.md)
  * [4.2 MQA 与 GQA 的 Head 共享](chapters/kv-representations/head-sharing.md)
  * [4.3 MLA 的 Latent Compression](chapters/kv-representations/latent-compression.md)
  * [4.4 MLA 中的 RoPE 与投影合并](chapters/kv-representations/decoupled-rope.md)
  * [4.5 KV 表示的质量与效率比较](chapters/kv-representations/comparison.md)
* [第 5 章 Gated Attention 的结构与作用](chapters/gated-attention/README.md)
  * [5.1 Attention 输出的 Gating](chapters/gated-attention/output-gating.md)
  * [5.2 Gate 的位置与参数化](chapters/gated-attention/placement.md)
  * [5.3 输出调节与 Attention Sink](chapters/gated-attention/attention-sink.md)
  * [5.4 训练证据与比较条件](chapters/gated-attention/evaluation.md)

## 计算与存储优化

* [第 6 章 FlashAttention 的计算与 IO 优化](chapters/flash-attention/README.md)
  * [6.1 Attention 的计算量与数据访问](chapters/flash-attention/io-cost.md)
  * [6.2 Online Softmax 与分块计算](chapters/flash-attention/online-softmax.md)
  * [6.3 FlashAttention 的前向与反向计算](chapters/flash-attention/forward-backward.md)
  * [6.4 并行调度与硬件适配](chapters/flash-attention/parallelism.md)
  * [6.5 数值误差与性能评估](chapters/flash-attention/evaluation.md)
* [第 7 章 Sparse Attention 的连接模式与动态选择](chapters/sparse-attention/README.md)
  * [7.1 稀疏连接与计算成本](chapters/sparse-attention/connectivity-cost.md)
  * [7.2 Sliding Window 与 Local–Global Attention](chapters/sparse-attention/local-global.md)
  * [7.3 Block-Sparse Attention 的组织与执行](chapters/sparse-attention/block-sparse.md)
  * [7.4 基于内容的动态选择](chapters/sparse-attention/dynamic-selection.md)
  * [7.5 信息覆盖与长上下文评估](chapters/sparse-attention/evaluation.md)
* [第 8 章 KV Cache 的存储管理与推理优化](chapters/kv-cache/README.md)
  * [8.1 KV Cache 的生命周期与容量](chapters/kv-cache/lifecycle.md)
  * [8.2 PagedAttention 与缓存分配](chapters/kv-cache/paged-attention.md)
  * [8.3 KV Cache Quantization](chapters/kv-cache/quantization.md)
  * [8.4 Token Selection 与 Cache Eviction](chapters/kv-cache/eviction.md)
  * [8.5 Prefix Caching 与 Offloading](chapters/kv-cache/reuse-offloading.md)
  * [8.6 缓存策略与服务性能](chapters/kv-cache/evaluation.md)

## Linear Attention 与混合架构

* [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](chapters/linear-attention/README.md)
  * [9.1 从 Softmax Attention 到 Kernel Attention](chapters/linear-attention/kernel-attention.md)
  * [9.2 Causal Linear Attention 的递推计算](chapters/linear-attention/recurrence.md)
  * [9.3 状态衰减与 Gated Linear Attention](chapters/linear-attention/state-decay.md)
  * [9.4 DeltaNet 与 Gated DeltaNet 的状态更新](chapters/linear-attention/delta-rule.md)
  * [9.5 并行训练与分块计算](chapters/linear-attention/chunkwise-computation.md)
  * [9.6 状态容量与检索能力](chapters/linear-attention/capacity-retrieval.md)
* [第 10 章 Hybrid Architectures 的结构设计与方法比较](chapters/hybrid-architectures/README.md)
  * [10.1 序列模块的组合方式](chapters/hybrid-architectures/composition.md)
  * [10.2 Full Attention 与 Sparse Attention 的组合](chapters/hybrid-architectures/full-sparse.md)
  * [10.3 Attention 与 Linear Attention 的组合](chapters/hybrid-architectures/attention-linear.md)
  * [10.4 Attention 与 State Space Models 的组合](chapters/hybrid-architectures/attention-ssm.md)
  * [10.5 层间配置与状态管理](chapters/hybrid-architectures/state-management.md)
  * [10.6 模型能力与资源成本的比较](chapters/hybrid-architectures/evaluation.md)

## 附录

* [附录 A 英文术语、数学符号与补充推导](appendices/foundations.md)
* [附录 B 技术演进时间线与方法对照表](appendices/timeline.md)
* [附录 C 参考文献、教学资源与本地源文件索引](appendices/references.md)
