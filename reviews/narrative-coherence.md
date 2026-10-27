# 全书背景与叙述衔接 Review

## 范围与版本

- 审阅日期：2026-10-27。
- 审阅方式：作者逐页自查，无独立审阅者。
- 范围：SUMMARY 指向的十章 63 个正文页面；同时核查首页、附录 A/B 的概念入口和附录 C 的来源组织。历史 GitBook 导出副本不作为当前书稿修订。
- 任务：核查必要背景、概念依赖、段落过渡、模型归属和结论范围。数学定义与实验数据沿用已有来源审校；本次不声称重新复现全部算法或实验。
- 前序未提交的 CSA/HCA/MoE 新内容保留，并纳入本轮叙述修订。旧 Review 中的 SHA-256 对应旧版本；本轮修改后的结论以 `narrative-coherence.sha256` 为准。
- 适用规则：CONTRIBUTING.md 的叙述结构、结构与标题、事实保真和表达专项核查。新增“背景、衔接与方法归属”规则，并在 Review 模板加入对应检查项。

## 目录与标题核查

全书保持“建模需求与计算基础—表示和结构—执行与存储—固定状态与混合架构”的组织方式。章节顺序用于建立概念依赖，各技术路线的并行、组合关系在正文具体说明。

7.5 改为“DeepSeek-V4 的 CSA 与 HCA”，导航、H1、整页链接及附录同步。MQA/GQA 与 MLA 分别作为共享和压缩路线，MLA 计算以 DeepSeek-V2 为依据；第五章明确以 Qiu 等人的输出 Gating 为例。第七章分别交代推理期选择、可训练结构和 DeepSeek 案例，第九章区分教学递推与具体论文模型，第十章区分序列模块与 FFN/MoE。

## 检查结果

| 维度 | 证据与处理 | 结果 |
| --- | --- | --- |
| 范围与结构 | 按阅读导航检查 63 页，36 页存在需改进的背景、归属或衔接；逐页结果见下表 | 本轮通过 |
| 事实与引用 | 机制比较沿用各节已引用的原论文；对新增或强化的模型归属、动机说明核对本地原文，定位见下表 | 本轮通过 |
| 数学与实现 | 与本轮开始前工作树对比，63 页的独立公式块、代码块、图片块逐项一致；未改算法和数值结果 | 未受影响部分保留；结构对比通过 |
| 实验与比较 | 保留窗口、模型、训练预算、原作者结果及未复现标记；并行路线未写成普遍替代关系 | 本轮通过 |
| 表达与教学素材 | 阅读各页正文、图表前后说明；添加衔接以具体问题和操作为依据，原有素材保留 | 本轮通过 |
| 资料与链接 | 严格归档检查报告 74 项来源、79 项素材，必备缺项为 0；导航 H1 一致。全库 81 项链接错误均位于旧导出副本 | 当前正文范围通过；全库未通过 |
| 跨章一致性 | 总览与附录加入 CSA/HCA 所处的序列压缩路线；MoE 单列 FFN 路由维度；共享、压缩、读取、状态分别说明 | 本轮通过 |
| 页面呈现 | 本轮未执行 GitBook 在线渲染检查，既有公式和图片块未变 | 待检查 |

### 关键归属与衔接的来源

| 修订对象 | 本地来源及定位 | 核查结论 |
| --- | --- | --- |
| MLA、Decoupled RoPE | `deepseek2024mla-v5.pdf` §2.1.2–2.1.3 | DeepSeek-V2 的具体设计；共同 Latent 与 Head 共享分开解释 |
| SDPA 输出 Gating | `qiu2025gatedattention-v1.pdf` §2.2，式 5 | 论文研究多个插入位置，本章以输出位置为主；保留粒度与参数化区别 |
| DSA | `deepseek2025v32.pdf` §2.1 | 报告采用独立 Indexer 与细粒度选择；不与 NSA 的压缩分支评分混写 |
| CSA/HCA | `deepseek2026v4-v1.pdf` §2.3、§3.5.1 | DeepSeek-V4 中的具体架构和异构缓存；HCA 对压缩条目使用 Dense Attention |
| PagedAttention | `kwon2023pagedattention-v1.pdf` §3–4 | 未知长度和连续预留带来分配浪费，分页改变物理布局与寻址 |
| GLA | `yang2023gla-v3.pdf` §3–3.1 | 明确为 Yang 等人的输入相关门控模型，保留与固定衰减的区别 |
| SageAttention | `zhang2024sageattention-v1.pdf` §4.1 | 量化乘法及恢复尺度属于额外数值选择，区别于单纯分块调度 |
| RG-LRU | `de2024griffin-v1.pdf` §2、§2.4 | 全称 Real-Gated Linear Recurrent Unit，属于 Griffin 使用的门控递推单元 |
| Expert Routing | `deepseek2026v4-v1.pdf` §2.1；归档 `deepseek-v4-model-b5968e9.py` Gate.forward | 通用 Token-choice 背景与 V4 特定评分、Hash 配置分别说明 |

## 表达专项核查

| 检查项 | 实际范围、发现与处理 | 结果 |
| --- | --- | --- |
| 信息增量 | 全部正文入口与节内过渡。删去首页写作进度式结尾，改为资料查阅方式；新增句子分别说明前提、方法动机、模型归属或后续问题 | 通过 |
| 否定与对比 | 保留因果边界、缓存一致性、固定维度和数学等价条件等实质限制。将“硬件仍有很大优化空间”改为具体数据搬运问题；未增加“不是完整历史”等范围辩解 | 通过 |
| 衔接与重复 | 核查通用机制到论文案例的转折。复查中发现 2.3 引言错误暗示前页未定义 Mask，已改为对应前页输入与 Mask 设置；NSA 对比改为准确的“压缩分支权重” | 通过 |
| 事实保真 | 独立公式、代码及图片块全部不变；保留模型版本、实验条件、教学示例和未验证状态。没有把原作者实验提升为通用结论 | 通过 |

## 逐页检查记录

“保留”表示该页已有足够背景或必要过渡，并非跳过审阅；“已修订并复查”包含引言、节内连接、术语归属或相关入口调整。

| 页面 | 处理 | 叙述检查与理由 |
| --- | --- | --- |
| [第 1 章 Attention 机制的演进与主要发展方向](../book/chapters/overview/README.md) | 保留 | 已有建模需求、演进与阅读范围，保留。 |
| [1.1 从 Representation Learning 到 Attention 机制](../book/chapters/overview/introduction.md) | 保留 | 表示、Token、上下文、动态汇总按依赖展开；DistilBERT 图例有具体语境，保留。 |
| [1.2 从 RNN 到 Transformer](../book/chapters/overview/from-seq2seq-to-transformer.md) | 保留 | RNN 固定表示—动态读取—Self-Attention 的职责变化清楚；Encoder/Decoder 结构已解释，保留。 |
| [1.3 LLM 中的 Attention 成本与长上下文需求](../book/chapters/overview/bottlenecks-and-context.md) | 保留 | 生成循环—缓存—成本—长上下文能力衔接完整，保留计算条件。 |
| [1.4 Attention 机制的主要发展方向与技术路线](../book/chapters/overview/optimization-landscape.md) | 已修订并复查 | 新增序列压缩在路线图中的位置，以 DeepSeek-V4 为明确案例；保留与 Head 共享、固定状态的区别。 |
| [第 2 章 Transformer Attention 的基本原理与计算过程](../book/chapters/transformer-attention/README.md) | 已修订并复查 | 章首从第一章的信息读取落实到可计算对象。 |
| [2.1 Transformer 中的 Attention 结构与计算流程](../book/chapters/transformer-attention/structure-and-flow.md) | 保留 | 原始 Encoder–Decoder 图与 Decoder-only 的使用范围已说明，保留。 |
| [2.2 Query、Key、Value 与 Scaled Dot-Product Attention](../book/chapters/transformer-attention/scaled-dot-product.md) | 已修订并复查 | 补出单个 Head 的含义与 QKV 职责，补 SDPA 全称缩写；公式及例子不变。 |
| [2.3 Self-Attention、Cross-Attention 与 Causal Attention](../book/chapters/transformer-attention/attention-and-masks.md) | 已修订并复查 | 从前页输入来源、Mask 设置转向三类 Attention 的语义；避免声称前页没有介绍 Mask。 |
| [2.4 Multi-Head Attention 的结构与 Output Projection](../book/chapters/transformer-attention/multi-head.md) | 保留 | 单 Head 到多 Head、拼接到输出投影已有完整过渡，保留。 |
| [2.5 Training、Prefill 与 Decode 的计算差异](../book/chapters/transformer-attention/training-prefill-decode.md) | 已修订并复查 | 从完整一层计算转入不同执行阶段，补出参数相同与输入形状变化的关系。 |
| [第 3 章 Position Encoding 的演进与 Context Extension](../book/chapters/position-encoding/README.md) | 保留 | 章首已承接内容匹配，并限定 RoPE 扩展对象，保留。 |
| [3.1 Attention 中的顺序与位置信息](../book/chapters/position-encoding/position-and-order.md) | 保留 | 语序例子先于置换推导；Mask 与位置表示的区别明确，保留。 |
| [3.2 Absolute 与 Relative Position Encoding](../book/chapters/position-encoding/absolute-and-relative.md) | 已修订并复查 | 补 Absolute 到相对偏移、Shaw 向量到 T5 标量偏置的动机；各方案归属明确。 |
| [3.3 RoPE 与 ALiBi 的位置建模](../book/chapters/position-encoding/rope-and-alibi.md) | 已修订并复查 | 补 RoFormer 的方法来源，并把 RoPE/ALiBi 写成不同设计选择。 |
| [3.4 RoPE Scaling 与上下文扩展](../book/chapters/position-encoding/context-extension.md) | 已修订并复查 | 先说明训练长度与相位使用范围，再进入插值、分频和搜索。 |
| [3.5 长度泛化与长上下文评估](../book/chapters/position-encoding/evaluation.md) | 保留 | 可执行长度、任务能力、评估设置已经区分，保留。 |
| [第 4 章 MQA、GQA 与 MLA 的 KV 表示优化](../book/chapters/kv-representations/README.md) | 已修订并复查 | 从位置处理转向每个位置的 KV 存储；明确两种结构选择及 DeepSeek-V2 案例。 |
| [4.1 KV 表示与缓存成本](../book/chapters/kv-representations/cache-cost.md) | 保留 | 已有缓存等价条件、形状与成本背景，保留。 |
| [4.2 MQA 与 GQA 的 Head 共享](../book/chapters/kv-representations/head-sharing.md) | 已修订并复查 | 先由容量式指出 KV Head 数的作用，再引入 MQA/GQA。 |
| [4.3 MLA 的 Latent Compression](../book/chapters/kv-representations/latent-compression.md) | 已修订并复查 | 对照 Head 共享引入 DeepSeek-V2 MLA；保留内容/位置分支的展开顺序。 |
| [4.4 MLA 中的 RoPE 与投影合并](../book/chapters/kv-representations/decoupled-rope.md) | 已修订并复查 | 首段已说明投影合并为何遇到 RoPE；明确 Decoupled RoPE 归属。 |
| [4.5 KV 表示的质量与效率比较](../book/chapters/kv-representations/comparison.md) | 已修订并复查 | 容量、结构与训练比较已齐备；补到输出 Gating 与缓存管理的不同后续问题。 |
| [第 5 章 Gated Attention 的结构与作用](../book/chapters/gated-attention/README.md) | 已修订并复查 | 从位置与 KV 设计转向汇总结果的幅度；明确以 Qiu 等人的输出 Gate 为例。 |
| [5.1 Attention 输出的 Gating](../book/chapters/gated-attention/output-gating.md) | 已修订并复查 | 补 Softmax 权重与输出幅度的关系，定义本节采用的 Gate；保留输入相关与粒度条件。 |
| [5.2 Gate 的位置与参数化](../book/chapters/gated-attention/placement.md) | 保留 | 已有位置、粒度、动态系数三层分析，原图实验标签有范围说明，保留。 |
| [5.3 输出调节与 Attention Sink](../book/chapters/gated-attention/attention-sink.md) | 已修订并复查 | 用读取位置与输出幅度的区别引出 Sink；限定 StreamingLLM 的观测对象。 |
| [5.4 训练证据与比较条件](../book/chapters/gated-attention/evaluation.md) | 保留 | 实验、长上下文适配与组合条件均明确，保留参数对齐和未复现标记。 |
| [第 6 章 FlashAttention 的计算与 IO 优化](../book/chapters/flash-attention/README.md) | 已修订并复查 | 从结构设计转向给定算子的执行问题，以中间数据搬运引出分块。 |
| [6.1 Attention 的计算量与数据访问](../book/chapters/flash-attention/io-cost.md) | 保留 | 矩阵成本、硬件层级、IO 模型的前提已说明，保留。 |
| [6.2 Online Softmax 与分块计算](../book/chapters/flash-attention/online-softmax.md) | 保留 | 整行分母问题先于在线统计量；数值例子支撑合并，保留。 |
| [6.3 FlashAttention 的前向与反向计算](../book/chapters/flash-attention/forward-backward.md) | 保留 | 已有前向统计量到反向重算的依赖解释，保留。 |
| [6.4 并行调度与硬件适配](../book/chapters/flash-attention/parallelism.md) | 已修订并复查 | 保留 FA2/3/4 硬件背景；补 SageAttention 的独立路线与量化误差定位。 |
| [6.5 数值误差与性能评估](../book/chapters/flash-attention/evaluation.md) | 保留 | 数值与性能评估分开；Kernel 到端到端的条件完整，保留。 |
| [第 7 章 Sparse Attention 的连接模式与动态选择](../book/chapters/sparse-attention/README.md) | 已修订并复查 | 由 FlashAttention 仍计算完整有效配对引出稀疏；明确 HCA 是案例中配合的 Dense 压缩层。 |
| [7.1 稀疏连接与计算成本](../book/chapters/sparse-attention/connectivity-cost.md) | 保留 | 集合定义先于计数与误差，已有条件说明，保留。 |
| [7.2 Sliding Window 与 Local–Global Attention](../book/chapters/sparse-attention/local-global.md) | 保留 | 局部、Global、长距离路径的职责与 Encoder/Causal 差异明确，保留。 |
| [7.3 Block-Sparse Attention 的组织与执行](../book/chapters/sparse-attention/block-sparse.md) | 保留 | 逻辑连接与物理块的矛盾先于实现分析，保留。 |
| [7.4 基于内容的动态选择](../book/chapters/sparse-attention/dynamic-selection.md) | 已修订并复查 | 补分组—页面评分—Prefill 模式—训练结构之间的比较关系；DSA 归属于 DeepSeek-V3.2。 |
| [7.5 DeepSeek-V4 的 CSA 与 HCA](../book/chapters/sparse-attention/compressed-attention.md) | 已修订并复查 | 标题改为 DeepSeek-V4；首段交代 DSA 背景与模型归属，再展开 CSA/HCA，保留原图和参数。 |
| [7.6 信息覆盖与长上下文评估](../book/chapters/sparse-attention/evaluation.md) | 已修订并复查 | 保留通用评价方法；补模型压缩表示与逐位置剪枝指标不直接对应的条件。 |
| [第 8 章 KV Cache 的存储管理与推理优化](../book/chapters/kv-cache/README.md) | 已修订并复查 | 承接读取集合与保留集合，再引入请求生命周期；区分第四章表示与服务系统管理。 |
| [8.1 KV Cache 的生命周期与容量](../book/chapters/kv-cache/lifecycle.md) | 保留 | Prefill、Decode、回收与共享的时间边界明确，保留。 |
| [8.2 PagedAttention 与缓存分配](../book/chapters/kv-cache/paged-attention.md) | 已修订并复查 | 先说明未知请求长度与预留浪费，再引入 vLLM 的分页设计。 |
| [8.3 KV Cache Quantization](../book/chapters/kv-cache/quantization.md) | 保留 | 分页到字节表示的衔接已有；KIVI 限于论文观察与实现，保留。 |
| [8.4 Token Selection 与 Cache Eviction](../book/chapters/kv-cache/eviction.md) | 已修订并复查 | 承接分页和量化的预算限制；补固定位置、历史评分、Prompt 观察三类决策依据的衔接。 |
| [8.5 Prefix Caching 与 Offloading](../book/chapters/kv-cache/reuse-offloading.md) | 已修订并复查 | 从请求内部管理转向跨请求重复计算与外部存储。 |
| [8.6 缓存策略与服务性能](../book/chapters/kv-cache/evaluation.md) | 保留 | 服务目标、调度、指标及质量条件的依赖关系明确，保留。 |
| [第 9 章 Linear Attention 的 Recurrent Formulation 与 State Update](../book/chapters/linear-attention/README.md) | 已修订并复查 | 从显式历史条目转向固定状态；指出具体论文的状态、归一化和 Gate 需要分别定义。 |
| [9.1 从 Softmax Attention 到 Kernel Attention](../book/chapters/linear-attention/kernel-attention.md) | 已修订并复查 | 用有限维特征分解引出状态读取；明确换核与近似 Softmax 两条路线。 |
| [9.2 Causal Linear Attention 的递推计算](../book/chapters/linear-attention/recurrence.md) | 保留 | 前缀和自然导出递推，归一化与边界有说明，保留。 |
| [9.3 状态衰减与 Gated Linear Attention](../book/chapters/linear-attention/state-decay.md) | 已修订并复查 | 先说明持续累积的限制，再比较固定/输入相关衰减；GLA 明确归属 Yang 等人的模型。 |
| [9.4 DeltaNet 与 Gated DeltaNet 的状态更新](../book/chapters/linear-attention/delta-rule.md) | 已修订并复查 | 用同一 Key 更新 Value 的问题引出 Delta Rule；保留对已有状态的误差修正与 TTT 条件。 |
| [9.5 并行训练与分块计算](../book/chapters/linear-attention/chunkwise-computation.md) | 保留 | Decode 与已知完整输入的并行需求先于公式，保留块内/块间解释。 |
| [9.6 状态容量与检索能力](../book/chapters/linear-attention/capacity-retrieval.md) | 保留 | 从干扰到秩、有限比特与任务评估逐层限定；自然引出混合架构，保留。 |
| [第 10 章 Hybrid Architectures 的结构设计与方法比较](../book/chapters/hybrid-architectures/README.md) | 已修订并复查 | 由显式条目与固定状态的取舍进入组合；案例模型与独立 FFN/MoE 维度明确。 |
| [10.1 序列模块的组合方式](../book/chapters/hybrid-architectures/composition.md) | 已修订并复查 | 已有层间、Head、分支的组织关系；补 RG-LRU 全称及 Griffin 归属。 |
| [10.2 Full Attention 与 Sparse Attention 的组合](../book/chapters/hybrid-architectures/full-sparse.md) | 已修订并复查 | 以 Gemma 2 明确说明 Full/Local 组合案例，再推导通用计数。 |
| [10.3 Attention 与 Linear Attention 的组合](../book/chapters/hybrid-architectures/attention-linear.md) | 保留 | 局部 KV 与全局 KV 分别补足固定状态，实验条件齐全，保留。 |
| [10.4 Attention 与 State Space Models 的组合](../book/chapters/hybrid-architectures/attention-ssm.md) | 已修订并复查 | 先交代与前节递推模块的关系，给出 SSM 必需基础，再进入具体混合模型。 |
| [10.5 层间配置与状态管理](../book/chapters/hybrid-architectures/state-management.md) | 已修订并复查 | 补 CSA/HCA 状态计数入口，保留快照、边界和共享条件。 |
| [10.6 MoE 与 Expert Routing](../book/chapters/hybrid-architectures/moe-routing.md) | 已修订并复查 | 先回顾 FFN 的位置，说明 MoE 与序列模块的独立配置；限定 Token-choice 评分例子与 DeepSeek 公式。 |
| [10.7 模型能力与资源成本的比较](../book/chapters/hybrid-architectures/evaluation.md) | 保留 | 评价依次区分约束、任务、阶段及消融，保留；不泛化模型实验结果。 |

首页改以统一问题串联阅读路径，附录 A 明确专属模型术语并补 CSA/HCA、MoE 与路由入口，附录 B 对齐序列压缩和 FFN 路由的分类。附录 C 为来源索引，保留其来源与本地文件的组织方式。

## 问题与修正

| 编号 | 级别 | 问题 | 修正与复查 | 状态 |
| --- | --- | --- | --- | --- |
| N1 | 需修正 | 通用类别与具体模型案例的边界不够显眼 | 7.5 标题与入口明确 V4；强化 MLA、DSA、GLA、Gating 的归属，核对论文 | 已关闭 |
| N2 | 需修正 | 章首有方法清单，但部分缺少前一问题到当前问题的连接 | 按表示、计算、存储、状态分别补背景，保留已清楚的页面 | 已关闭 |
| N3 | 需修正 | 7.4、8.4 内部切换论文时缺少比较依据 | 按候选组织、评分对象、处理阶段、训练条件解释切换 | 已关闭 |
| N4 | 需修正 | 少数缩写先出现后解释 | 在 2.2 补 SDPA，在 Griffin 案例补 RG-LRU 全称及职责 | 已关闭 |
| N5 | 需修正 | 新增模型案例与总览、附录检索不完全同步 | 同步总览分类、术语表、时间线和状态管理入口，检查标题与链接 | 已关闭 |
| N6 | 需修正 | 初稿衔接出现不准确回顾与宽泛措辞 | 改正 2.3 对前节 Mask 的描述、NSA 分支名称、GLA “数量”表述；复查最终差异 | 已关闭 |
| N7 | 后续改进 | 旧 GitBook 导出副本残留 81 项链接错误 | 与本次修改分开记录，不声称全库检查通过 | 后续改进 |

## 未完成项与结论

- 本轮叙述与表达 Review 通过，结论限于记录的文件版本与修订范围；未替代既有的完整事实、公式和实验审校。
- GitBook 页面呈现待发布后检查；旧导出副本仍需单独清理。
- 未执行 commit、push。具体检查结果见 `narrative-coherence-checks.json`，当前版本见 `narrative-coherence.sha256`。
