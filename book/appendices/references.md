# 附录 C 参考文献、教学资源与本地源文件索引

本页收录已写章节所使用的来源。论文年份按首次公开年份填写；引用依据为列出的具体版本。核查位置记录于各条目，范围限于相关正文、公式和图表。

## 必备原文

以下论文支撑正文的核心机制、推导与比较，按专题排列；跨章使用的来源只列一次。本地下载状态由仓库 `references/catalog.yaml` 记录。

### 序列建模与 Transformer

- **Sutskever, Vinyals & Le，2014**：[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215v3)。使用版本：1409.3215v3；核查位置：§2。[全文](https://arxiv.org/html/1409.3215v3) · [PDF](https://arxiv.org/pdf/1409.3215v3)。来源 ID：`sutskever2014seq2seq`。

- **Bahdanau, Cho & Bengio，2014**：[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7)。使用版本：1409.0473v7；核查位置：§3.1, equations (4)–(6)。[全文](https://arxiv.org/html/1409.0473v7) · [PDF](https://arxiv.org/pdf/1409.0473v7)。来源 ID：`bahdanau2014attention`。

- **Vaswani et al.，2017**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762v7)。使用版本：1706.03762v7；核查位置：§3.1–3.2.3、§3.5；Table 1；Figures 1–2；Figure reuse statement on page 1。[全文](https://arxiv.org/html/1706.03762v7) · [PDF](https://arxiv.org/pdf/1706.03762v7)。来源 ID：`vaswani2017attention`。

### 位置编码与长上下文

- **Shaw, Uszkoreit & Vaswani，2018**：[Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/)。使用版本：NAACL-HLT 2018, N18-2074；核查位置：§3.1–3.2, equations (3)–(4), clipping rule。[全文](https://aclanthology.org/N18-2074.pdf) · [PDF](https://aclanthology.org/N18-2074.pdf)。来源 ID：`shaw2018relative`。

- **Raffel et al.，2019**：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683v4)。使用版本：1910.10683v4；核查位置：§2.1 relative scalar bias and parameter sharing。[全文](https://arxiv.org/html/1910.10683v4) · [PDF](https://arxiv.org/pdf/1910.10683v4)。来源 ID：`raffel2019t5`。

- **Su et al.，2021**：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5)。使用版本：2104.09864v5；核查位置：§3.2。[全文](https://arxiv.org/html/2104.09864v5) · [PDF](https://arxiv.org/pdf/2104.09864v5)。来源 ID：`su2021rope`。

- **Press, Smith & Lewis，2021**：[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409v2)。使用版本：2108.12409v2；核查位置：§3。[全文](https://arxiv.org/html/2108.12409v2) · [PDF](https://arxiv.org/pdf/2108.12409v2)。来源 ID：`press2021alibi`。

- **Chen et al.，2023**：[Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595v2)。使用版本：2306.15595v2；核查位置：§2.3。[全文](https://arxiv.org/html/2306.15595v2) · [PDF](https://arxiv.org/pdf/2306.15595v2)。来源 ID：`chen2023pi`。

- **Peng, Quesnelle, Fan & Shippole，2023**：[YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071v2)。使用版本：2309.00071v2；核查位置：§2.2–2.3; §3.1–3.4; equations (10)–(22)。[全文](https://arxiv.org/html/2309.00071v2) · [PDF](https://arxiv.org/pdf/2309.00071v2)。来源 ID：`peng2023yarn`。

- **Ding et al.，2024**：[LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753v1)。使用版本：2402.13753v1；核查位置：§3.1–3.3 search and progressive extension; §4.3 short-context adjustment。[全文](https://arxiv.org/html/2402.13753v1) · [PDF](https://arxiv.org/pdf/2402.13753v1)。来源 ID：`ding2024longrope`。

- **Liu et al.，2023**：[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172v1)。使用版本：2307.03172v1；核查位置：§3.1–3.3。[全文](https://arxiv.org/html/2307.03172v1) · [PDF](https://arxiv.org/pdf/2307.03172v1)。来源 ID：`liu2023lostmiddle`。

- **Hsieh et al.，2024**：[RULER: What’s the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654v3)。使用版本：2404.06654v3；核查位置：§3 task definitions; §4 evaluation configuration and effective context size。[全文](https://arxiv.org/html/2404.06654v3) · [PDF](https://arxiv.org/pdf/2404.06654v3)。来源 ID：`hsieh2024ruler`。

- **Bai et al.，2023**：[LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508v2)。使用版本：2308.14508v2；核查位置：§3.1–3.2; Table 1 task categories and length units。[全文](https://arxiv.org/html/2308.14508v2) · [PDF](https://arxiv.org/pdf/2308.14508v2)。来源 ID：`bai2023longbench`。

### KV 表示与输出 Gating

- **Noam Shazeer，2019**：[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150v1)。使用版本：1911.02150v1；核查位置：§2.1–2.4.1；§3。[全文](https://arxiv.org/html/1911.02150v1) · [PDF](https://arxiv.org/pdf/1911.02150v1)。来源 ID：`shazeer2019mqa`。

- **Ainslie et al.，2023**：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v3)。使用版本：2305.13245v3；核查位置：§2.1—2.2、§3.1—3.2；Figure 2、Table 1。[全文](https://arxiv.org/html/2305.13245v3) · [PDF](https://arxiv.org/pdf/2305.13245v3)。来源 ID：`ainslie2023gqa`。

- **DeepSeek-AI，2024**：[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434v5)。使用版本：2405.04434v5；核查位置：§2.1.1—2.1.4、§3.1.2。[全文](https://arxiv.org/html/2405.04434v5) · [PDF](https://arxiv.org/pdf/2405.04434v5)。来源 ID：`deepseek2024mla`。

- **Qiu et al.，2025**：[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708v1)。使用版本：2505.06708v1；核查位置：§2.2、§3.1—3.2.2、§4.1—4.4；Table 2 行 (1)—(2)。[全文](https://arxiv.org/html/2505.06708v1) · [PDF](https://arxiv.org/pdf/2505.06708v1)。来源 ID：`qiu2025gatedattention`。

- **Wang et al.，2020**：[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/html/2006.04768v3)。使用 2006.04768v3；核查 §4 式 (7) 的序列投影及 §5.1 的 MLM 预训练设置。[PDF](https://arxiv.org/pdf/2006.04768v3)。来源 ID：`wang2020linformer`。

### Attention 计算与硬件优化

- **Milakov & Gimelshein，2018**：[Online normalizer calculation for softmax](https://arxiv.org/html/1805.02867v2)。使用版本：1805.02867v2；核查 §2—3.1、Algorithms 2—3 与式 (2)—(4)。[PDF](https://arxiv.org/pdf/1805.02867v2)。来源 ID：`milakov2018onlinesoftmax`。

- **Dao et al.，2022**：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135v2)。使用版本：2205.14135v2；核查位置：§3.1, Algorithm 1 与 Theorem 1；§3.2, Theorem 2 与 Proposition 3；§3.3；§4；Appendix B.2—B.4。[全文](https://arxiv.org/html/2205.14135v2) · [PDF](https://arxiv.org/pdf/2205.14135v2)。来源 ID：`dao2022flashattention`。

- **Dao，2023**：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf)。使用作者网站日期为 2023-07-18 的 PDF，核查 §3.1—3.3、Algorithms 1—2 与 §4.1 的比较条件。来源 ID：`dao2023flashattention2`。

- **Shah et al.，2024**：[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/html/2407.08608v2)。使用版本 2407.08608v2；核查 §2.2、§3.1—3.3、§4.1—4.3。Figure 1 用于解释 Pingpong Scheduling，正文提供原图链接。来源 ID：`shah2024flashattention3`。

- **Zadouri et al.，2026**：[FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/html/2603.05451v1)。使用版本 2603.05451v1；核查 §3.1.1—3.1.4、§3.2.1—3.2.2。来源 ID：`zadouri2026flashattention4`。

- **Zhang et al.，2024**：[SageAttention: Accurate 8-bit attention for Plug-and-Play Inference Acceleration](https://arxiv.org/html/2410.02367v1)。使用 2410.02367v1；核查 §4.1—4.4、Table 6 的 SAGEAttn-B 精度路径与 Key 均值平移。[PDF](https://arxiv.org/pdf/2410.02367v1)。来源 ID：`zhang2024sageattention`。

- **Hao Liu, Matei Zaharia & Pieter Abbeel，2023**：[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/html/2310.01889v4)。使用 2310.01889v4；核查 §3 block rotation, merging and compute/communication overlap; Figure 2。[PDF](https://arxiv.org/pdf/2310.01889v4)。来源 ID：`liu2023ringattention`。

- **Sam Ade Jacobs et al.，2023**：[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/html/2309.14509v1)。使用 2309.14509v1；核查 §3.1–3.4; Figure 2; sequence/head layout and all-to-all。[PDF](https://arxiv.org/pdf/2309.14509v1)。来源 ID：`jacobs2023ulysses`。

### 稀疏连接与动态选择

- **Nikita Kitaev, Łukasz Kaiser & Anselm Levskaya，2020**：[Reformer: The Efficient Transformer](https://arxiv.org/html/2001.04451v2)。使用 2001.04451v2；核查 §2 shared-QK, LSH sorting/chunking, multiple rounds and causal mask; Figure 2。[PDF](https://arxiv.org/pdf/2001.04451v2)。来源 ID：`kitaev2020reformer`。

- **Aurko Roy, Mohammad Saffar, Ashish Vaswani & David Grangier，2020**：[Efficient Content-Based Sparse Attention with Routing Transformers](https://arxiv.org/html/2003.05997v3)。使用 2003.05997v3；核查 §4.1 clustering, balanced top-k routing, EMA centroids; Algorithm 1 and Figure 1。[PDF](https://arxiv.org/pdf/2003.05997v3)。来源 ID：`roy2020routing`。

- **Child et al.，2019**：[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509v1)。使用版本：1904.10509v1；核查位置：§4。[全文](https://arxiv.org/html/1904.10509v1) · [PDF](https://arxiv.org/pdf/1904.10509v1)。来源 ID：`child2019sparse`。

- **Beltagy, Peters & Cohan，2020**：[Longformer: The Long-Document Transformer](https://arxiv.org/html/2004.05150v2)。使用版本 2004.05150v2；核查位置：§3.1 local, dilated and global patterns; Figure 2。来源 ID：`beltagy2020longformer`。

- **Zaheer et al.，2020**：[Big Bird: Transformers for Longer Sequences](https://arxiv.org/html/2007.14062v2)。使用版本 2007.14062v2；核查位置：§2 random, window and global connectivity; Figure 1。来源 ID：`zaheer2020bigbird`。

- **Jiang et al.，2023**：[Mistral 7B](https://arxiv.org/abs/2310.06825v1)。使用版本：2310.06825v1；核查位置：§2 的 Sliding Window、Rolling Buffer Cache 与 Table 1。[全文](https://arxiv.org/html/2310.06825v1) · [PDF](https://arxiv.org/pdf/2310.06825v1)。来源 ID：`jiang2023mistral`。

- **Tang et al.，2024**：[Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/html/2406.10774v1)。使用版本 2406.10774v1；核查 §3.3–3.5, Algorithm 1 and Figure 4; §4.1 evaluation setup。来源 ID：`tang2024quest`。

- **Jiang et al.，2024**：[MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](https://arxiv.org/html/2407.02490v1)。使用 2407.02490v1；核查 §2.2—3.2、Algorithms 1—3 与 §4 Baselines。[PDF](https://arxiv.org/pdf/2407.02490v1)。来源 ID：`jiang2024minference`。

- **Yuan et al.，2025**：[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089v1)。使用版本：2502.11089v1；核查位置：§3.2—3.4、§4.1—4.3 与 §5。[全文](https://arxiv.org/html/2502.11089v1) · [PDF](https://arxiv.org/pdf/2502.11089v1)。来源 ID：`yuan2025nsa`。

- **DeepSeek-AI，2025**：[DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/html/2512.02556v1)。使用版本 2512.02556v1；核查 §2.1 indexer and MLA; §2.1.1 training objectives; §2.3 quadratic indexer cost。来源 ID：`deepseek2025v32`。

### KV Cache 与推理服务

- **Kwon et al.，2023**：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180v1)。使用版本：2309.06180v1；核查位置：§2.1—2.2、§3.1、§4.1—4.5、§5.1—5.2、§6.1。[全文](https://arxiv.org/html/2309.06180v1) · [PDF](https://arxiv.org/pdf/2309.06180v1)。来源 ID：`kwon2023pagedattention`。

- **Liu et al.，2024**：[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/html/2402.02750v2)。使用版本 2402.02750v2；核查 §3.1—3.3、Figures 1—3、§4.1—4.2.1。[PDF](https://arxiv.org/pdf/2402.02750v2)。来源 ID：`liu2024kivi`。

- **Xiao et al.，2023**：[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453v4)。使用版本：2309.17453v4；核查位置：§3.1—3.3。[全文](https://arxiv.org/html/2309.17453v4) · [PDF](https://arxiv.org/pdf/2309.17453v4)。来源 ID：`xiao2023streamingllm`。

- **Zhang et al.，2023**：[H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/html/2306.14048v3)。使用版本 2306.14048v3；核查 §2.1、§3.2、§4.1—4.2、Appendix A 的实现说明及伪代码。[PDF](https://arxiv.org/pdf/2306.14048v3)。来源 ID：`zhang2023h2o`。

- **Li et al.，2024**：[SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/html/2404.14469v2)。使用版本 2404.14469v2；核查 §4.1、Listing 1、§4.2.1、§4.3。[PDF](https://arxiv.org/pdf/2404.14469v2)。来源 ID：`li2024snapkv`。

- **Zheng et al.，2023**：[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/html/2312.07104v2)。使用版本 2312.07104v2；核查 §3 radix tree, reference counts, LRU eviction and cache-aware scheduling; Appendix A.1。[PDF](https://arxiv.org/pdf/2312.07104v2)。来源 ID：`zheng2023sglang`。

- **Sheng et al.，2023**：[High-throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/html/2303.06865v1)。使用版本 2303.06865v1；核查 §4.1–4.3 tensor placement, computation delegation and overlap cost model。[PDF](https://arxiv.org/pdf/2303.06865v1)。来源 ID：`sheng2023flexgen`。

- **Agrawal et al.，2024**：[Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/html/2403.02310v3)。使用版本 2403.02310v3；核查 §2.4—2.5、§4.1—4.3、Algorithm 3。[PDF](https://arxiv.org/pdf/2403.02310v3)。来源 ID：`agrawal2024sarathiserve`。

### Linear Attention 与状态更新

- **Katharopoulos et al.，2020**：[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html)。使用版本：PMLR 119, 2020；核查位置：§3.2—3.4，式 (4)—(12)、(16)—(20)。[全文](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf) · [PDF](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)。来源 ID：`katharopoulos2020linear`。

- **Choromanski et al.，2020**：[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794v4)。使用 2009.14794v4；核查 §2、Lemma 1 的正随机特征及 Appendix B.2 的 Orthogonal Random Features。[PDF](https://arxiv.org/pdf/2009.14794v4)。来源 ID：`choromanski2020performer`。

- **Schlag et al.，2021**：[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)。使用 PMLR 139 出版版；核查 §4.1—4.2 的容量与 Delta 更新、§6.1 的检索任务。[PDF](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf)。来源 ID：`schlag2021fastweights`。

- **Yang et al.，2023**：[Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/html/2312.06635v3)。使用版本 2312.06635v3；核查 §2.2、§3.1—3.4、式 (8)—(13)、Head 内 LayerNorm 与 Gate 参数化。[PDF](https://arxiv.org/pdf/2312.06635v3)。来源 ID：`yang2023gla`。

- **Yang et al.，2024**：[Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/html/2406.06484v2)。使用版本 2406.06484v2；核查 §2.1—2.2 的加法分块与 DeltaNet 更新，以及 §3.1—3.2 的重参数化与 WY Representation。[PDF](https://arxiv.org/pdf/2406.06484v2)。来源 ID：`yang2024deltanet`。

- **Yang et al.，2024**：[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464v1)。使用版本：2412.06464v1；核查位置：§2.3；§3.1, equation (8)；§3.3 的 H1/H2；§4 Setup、Table 2 与 Appendix B.1 的评估定义。[全文](https://arxiv.org/html/2412.06464v1) · [PDF](https://arxiv.org/pdf/2412.06464v1)。来源 ID：`yang2024gateddelta`。

- **Arora et al.，2023**：[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)。使用 ICLR 2024 出版版；核查 §3 的关联检索分析与 Definition 3.1、§4 的模型比较范围。[PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/448fc91f669c15d10364ee01d512cc10-Paper-Conference.pdf)。来源 ID：`arora2023zoology`。

- **Yutao Sun et al.，2023**：[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/html/2307.08621v4)。使用 2307.08621v4；核查 §2.1–2.2; fixed per-head decay, positional rotation, GroupNorm and output gate; Figure 3。[PDF](https://arxiv.org/pdf/2307.08621v4)。来源 ID：`sun2023retnet`。

- **Bo Peng et al.，2023**：[RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/html/2305.13048v2)。使用 2305.13048v2；核查 §3.1–3.3; WKV equation (16), output gate; Appendix D state layout。[PDF](https://arxiv.org/pdf/2305.13048v2)。来源 ID：`peng2023rwkv`。

- **Yu Sun et al.，2024**：[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/html/2407.04620v1)。使用 2407.04620v1；核查 §2.1–2.3; Figure 6; §2.6 Theorem 1; §2.7 internal model and initialization。[PDF](https://arxiv.org/pdf/2407.04620v1)。来源 ID：`sun2024ttt`。

### 混合架构与 State Space Models

- **De et al.，2024**：[Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/html/2402.19427v1)。使用 2402.19427v1；核查 §2—3 的序列模块及 Griffin 层比例、默认窗口，以及 Appendix F.4 的 KV、递推与短卷积状态；§6.2、Figure 6 的两种检索实验。[PDF](https://arxiv.org/pdf/2402.19427v1)。来源 ID：`de2024griffin`。

- **Lieber et al.，2024**：[Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/html/2403.19887v2)。使用 2403.19887v2；核查 §2、Figure 1 的 Attention/Mamba 层比例和 MoE 配置；§6.1、Table 4 的无 MoE 消融。[PDF](https://arxiv.org/pdf/2403.19887v2)。来源 ID：`lieber2024jamba`。

- **Gemma Team，2024**：[Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/html/2408.00118v3)。使用 2408.00118v3；核查 §2、Table 1 的 Local–Global 交替、窗口及上下文范围。[PDF](https://arxiv.org/pdf/2408.00118v3)。来源 ID：`gemmateam2024gemma2`。

- **Gu & Dao，2023**：[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/html/2312.00752v2)。使用 2312.00752v2；核查 §2; §3.2 Algorithm 2; §3.3 Selective Scan。[PDF](https://arxiv.org/pdf/2312.00752v2)。来源 ID：`gu2023mamba`。

- **Dao & Gu，2024**：[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/html/2405.21060v1)。使用 2405.21060v1；核查 §5.1–5.3 scalar-identity SSM and structured masked attention; §6 algorithm。[PDF](https://arxiv.org/pdf/2405.21060v1)。来源 ID：`dao2024ssd`。

- **Ren et al.，2024**：[Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/html/2406.07522v1)。使用 2406.07522v1；核查 §2.1 Mamba/SWA/MLP architecture and window size。[PDF](https://arxiv.org/pdf/2406.07522v1)。来源 ID：`ren2024samba`。

## 教学资源

下表按概念背景、序列建模、Transformer 计算与自回归生成排列，列出本书使用的教学材料及阅读位置。

| 形式与来源 | 阅读或观看位置 | 对应内容 |
| --- | --- | --- |
| 书籍：[Goodfellow、Bengio、Courville，Deep Learning](https://www.deeplearningbook.org/contents/intro.html) | Chapter 1 的 Machine Learning、Representation Learning 与 Deep Learning 介绍；来源 ID：`goodfellow2016deep-learning` | 1.1 的概念背景 |
| 课程：[Hugging Face，Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4?fw=pt) | Word-based、Character-based、Subword；Encoding and decoding；来源 ID：`huggingface-tokenizers` | 1.1 的文本切分、Token 编号与文本还原 |
| 图文：[Jay Alammar，A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) | How a single prediction is calculated；Flowing Through DistilBERT；来源 ID：`alammar2019bertguide` | 1.1 的文本编码与用于分类的向量表示 |
| 书籍：[Zhang、Lipton、Li、Smola，Dive into Deep Learning](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)，在线版 1.0.3 | §10.7 的架构说明与 Figure 10.7.1；来源 ID：`d2l-seq2seq` | 1.2 的 RNN Encoder–Decoder 与固定表示传递 |
| 图文与动画：[Jay Alammar，Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) | Looking under the hood；Let’s Pay Attention Now；来源 ID：`alammar2018seq2seq` | 1.2 中按生成步骤读取 Encoder 状态的过程 |
| 图文：[Jay Alammar，The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，2018 | Encoder–Decoder、Multi-Head 与 Position Encoding 图解；来源 ID：`alammar2018illustrated` | 1.2、2.4、3.2 的结构与计算 |
| 课程：[Hugging Face，Transformer Architectures](https://huggingface.co/learn/llm-course/chapter1/6) | Encoder models；Decoder models；Sequence-to-sequence models；来源 ID：`huggingface-transformer-architectures` | 1.2、2.1 的三种架构及信息访问范围 |
| 书籍：[Zhang、Lipton、Li、Smola，Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)，在线版 1.0.3 | §11.1 QKV；§11.3 Scoring；§11.5 Multi-Head；§11.6.3 位置编码；§11.7 Transformer；来源 ID：`d2l-attention` | 第 2、3 章的公式、张量布局与代码示例 |
| 视频：[3Blue1Brown，Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)，2024 | 06:32—10:40 Q/K；11:30—12:45 Mask；13:40—15:10 Value；20:30—22:30 Multi-Head；来源 ID：`sanderson2024attention` | 第 2 章的逐步计算；时间段依据作者英文字幕核对 |
| 图文与动画：[Jay Alammar，The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)，2019 | One Difference From BERT；Evaluation Time: Processing One Token at a Time；来源 ID：`alammar2019gpt2` | 1.3、2.5 的自回归生成循环与历史 KV 复用 |

<p align="center">表 C-1　教学资源</p>

教学材料用于帮助理解，核心公式和方法归属仍以原始论文为据。D2L 已核查 §10.7 的架构说明、§11.1、§11.3 的缩放与 Mask、§11.5 的公式与张量布局，§11.6.3 的位置编码，以及 §11.7.1—11.7.3 的结构说明。书中数值检查使用本仓库脚本，D2L 的模型训练代码未运行。

## 图表来源

| 本书图号 | 来源与原图位置 | 解释对象 |
| --- | --- | --- |

| 图 1-1 | [Jay Alammar，A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)，CC BY-NC-SA 4.0；原图未修改 | 英文句子的 Token、编号及模型输出向量 |
| 图 1-2 | 本书使用 AI 生成的上下文词义示意；概念参照 [3Blue1Brown，Motivating Examples](https://www.3blue1brown.com/lessons/attention/) | 同一词形在水果与公司语境中的不同含义，非实验数据 |
| 图 1-3 | [Jay Alammar，The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，CC BY-NC-SA 4.0；原图未修改 | Self-Attention 可视化 |
| 图 1-4 | [Zhang, Lipton, Li & Smola，Dive into Deep Learning，§10.7，Figure 10.7.1](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)，[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)；SVG 白底栅格化，图中内容未改动 | RNN Encoder–Decoder 的固定表示传递 |
| 图 1-5 | 本书依据 [Bahdanau et al.，§3.1，式 (4)—(6)](https://arxiv.org/html/1409.0473v7#S3.SS1) 绘制；词语切分与权重为教学设定 | 英中翻译中逐步变化的读取权重 |
| 图 1-6 | [Jay Alammar，The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/#a-high-level-look)，CC BY-NC-SA 4.0；原图未修改 | Attention 承担原文、译文及跨序列的信息交互 |
| 图 1-7 | [Jay Alammar，The Illustrated GPT-2，One Difference From BERT](https://jalammar.github.io/illustrated-gpt2/#one-difference-from-bert)，CC BY-NC-SA 4.0；原 GIF 未修改 | 生成结果追加到输入，形成自回归反馈循环 |
| 图 1-8 | [Jay Alammar，The Illustrated GPT-2，Evaluation Time: Processing One Token at a Time](https://jalammar.github.io/illustrated-gpt2/#evaluation-time-processing-one-token-at-a-time)，CC BY-NC-SA 4.0；原图未修改 | 历史 KV 在各层的复用 |
| 图 1-9 | 本书依据 [Kwon et al.，§2.2](https://arxiv.org/html/2309.06180v1) 绘制 | Prefill 与 Decode 的位置关系 |
| 图 1-10 | 本书依据 [Vaswani et al.，Table 1](https://arxiv.org/html/1706.03762v7) 与 [Shazeer，§2.4](https://arxiv.org/html/1911.02150v1) 绘制 | 因果位置对与单步 KV 读取量，非实测性能 |
| 图 1-11 | 本书依据 [Liu et al.，§3.1](https://arxiv.org/html/2307.03172v1) 绘制 | 多文档问答的位置对照设置，七份文档为教学设定，未描绘得分 |
| 图 1-12 | 本书综合绘制；计算参照 [Vaswani §3.2](https://arxiv.org/html/1706.03762v7)，各路线依据见 1.4 的对应论文；SVG 与生成脚本随书保存 | 方法作用于表示、匹配、输出、计算存储及状态更新的主要位置 |
| 图 2-1 | [Zhang, Lipton, Li & Smola，D2L，Figure 11.7.1](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)，[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)；原 SVG 白底栅格化，图中内容未修改 | Attention 在 Encoder–Decoder 中的位置 |
| 图 2-2 | [Vaswani et al.，Figure 2 左图](https://arxiv.org/html/1706.03762v7)；原图未修改 | Scaled Dot-Product Attention 的计算步骤 |
| 图 2-3 | 本书依据 [Vaswani et al.，§3.2.3](https://arxiv.org/html/1706.03762v7) 绘制；网格表示允许读取的位置 | 三类 Attention 的信息来源与访问范围 |
| 图 2-4 | [Zhang, Lipton, Li & Smola，D2L，Figure 11.5.1](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html)，[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)；原 SVG 白底栅格化，图中内容未修改 | 独立投影、并行汇总与 Output Projection |
| 图 2-5 | 本书依据 [Kwon et al.，§2.2](https://arxiv.org/html/2309.06180v1) 绘制，复用图 1-9 | 完整 Prompt 与单个新 Query 的读取范围 |
| 图 3-1 | 本书依据 [Vaswani §3.5](https://arxiv.org/html/1706.03762v7)、[RoFormer §3.2](https://arxiv.org/html/2104.09864v5)、[T5 §2.1](https://arxiv.org/html/1910.10683v4) 与 [ALiBi §3](https://arxiv.org/html/2108.12409v2) 绘制 | 位置项进入 Attention 的主要位置 |
| 图 3-2 | [Jay Alammar，The Illustrated Transformer，Position Encoding 段落的 July 2020 Update](https://jalammar.github.io/illustrated-transformer/#representing-the-order-of-the-sequence-using-positional-encoding)，[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)；原图未修改 | sin/cos 交错排列的编码热图 |
| 图 3-3 | [Su et al.，RoFormer: Enhanced Transformer with Rotary Position Embedding，Figure 1](https://arxiv.org/pdf/2104.09864v5#page=5)；[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | 从归档 PDF 第 5 页截取图区并栅格化，保留原标注；RoPE 的成对维度旋转 |
| 图 3-4 | 本书依据 [Su et al.，§3.2](https://arxiv.org/html/2104.09864v5) 绘制；单位内容向量、角频率 π/6 为教学设定 | 共同平移与相对旋转 |
| 图 3-5 | 本书依据 [Press et al.，§3](https://arxiv.org/html/2108.12409v2) 绘制；Query 位置 4，斜率 0.25 与 0.5 | ALiBi 的分数偏置，非 Attention 权重 |
| 图 3-6 | 本书依据 [Chen et al.，§2.3，式 (4)](https://arxiv.org/html/2306.15595v2) 绘制；长度 4 扩展到 8 为教学设定 | PI 的坐标压缩 |
| 图 3-7 | 本书依据 [Peng et al.，v2，§3.1—3.2](https://arxiv.org/html/2309.00071v2) 的公式绘制；d=64、L=2048、ρ=4、b=10000、α=1、β=32 | 各维度频率比值；未包含 YaRN 温度，无模型得分 |
| 图 3-8 | 本书依据 [Liu et al.，§3.1](https://arxiv.org/html/2307.03172v1) 绘制，复用图 1-11；七份文档为教学设定 | 证据位置对照，非实验得分 |
| 图 4-1 | [Ainslie et al.，Figure 2](https://arxiv.org/html/2305.13245v3) | MHA、GQA、MQA 的 Head 组织 |
| 图 4-2 | [DeepSeek-AI，DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model，Figure 3](https://arxiv.org/pdf/2405.04434v5#page=7) | 从归档 PDF 第 7 页截取图区并栅格化，保留原标注；MHA、GQA、MQA 与 MLA 的缓存表示 |
| 图 4-3 | 本书依据 [DeepSeek-V2 §2.1.2](https://arxiv.org/html/2405.04434v5) 自行绘制；非原图改绘 | 内容分支两种计算顺序；图用行向量，正文列向量；SVG 与 PNG 保存于 assets/kv-representations/，脚本 scripts/render_kv_representations.py |
| 图 4-4 | 本书依据 [DeepSeek-V2 §2.1.3](https://arxiv.org/html/2405.04434v5) 自行绘制；非原图改绘 | 内容与位置分数先相加后统一归一化；颜色区分缓存用途，非测量数据；源文件及脚本同上 |
| 图 5-1 | 本书依据 [Qiu et al. §2.2](https://arxiv.org/html/2505.06708v1) 自行绘制，非原图改绘 | SDPA 与 Gate 分支；SVG、PNG 位于 assets/gated-attention/，脚本 scripts/render_gated_attention.py |
| 图 5-2 | 本书按逐元素乘法绘制教学算例 | 输入与 Gate 值为构造数据，非实验结果；源文件及脚本同上 |
| 图 5-3 | [Qiu et al.，Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free，Figure 1 left panel](https://arxiv.org/pdf/2505.06708v1#page=2) | 从归档 PDF 第 2 页截取图区并栅格化，保留原标注；Attention 中的 Gate 插入位置 |
| 图 6-1 | [Dao et al.，FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，Figure 1 left and middle panels](https://arxiv.org/pdf/2205.14135v2#page=2) | 从归档 PDF 第 2 页截取图区并栅格化，保留原标注；FlashAttention 的存储层级与分块数据流 |
| 图 6-2 | 本书依据 [FlashAttention-2 §3.1.1](https://tridao.me/publications/flash2/flash2.pdf) 自行绘制，非原图改绘 | 三位置的教学算例；SVG、PNG 位于 assets/flash-attention/，脚本 scripts/render_flash_attention.py；数值经 CPU 检查 |
| 图 7-1 | [Beltagy, Peters & Cohan，Longformer: The Long-Document Transformer，Figure 2](https://arxiv.org/pdf/2004.05150v2#page=3) | 从归档 PDF 第 3 页截取图区并栅格化，保留原标注；Longformer 的局部、扩张与全局连接 |
| 图 7-2 | 本书依据 [Longformer §3.1](https://arxiv.org/html/2004.05150v2) 与单向 Mask 定义自行绘制，非原图改绘 | N=8；Causal 窗口包含自身为 3，双向局部半径 1、Global 位置 1；SVG/PNG 位于 assets/sparse-attention/，脚本 scripts/render_sparse_attention.py |
| 图 7-3 | 本书依据 [Quest §3.4](https://arxiv.org/html/2406.10774v1) 自行构造反例，非原图改绘 | 页面上界与真实最大点积排序不同；SVG/PNG 位于 assets/sparse-attention/，生成脚本 scripts/render_sparse_attention.py，数值已经检查 |
| 图 8-1 | 本书依据 [PagedAttention §4.1—4.3](https://arxiv.org/html/2309.06180v1) 自行绘制 | 块表 `[7, 2, 5]` 与 9 个有效位置的教学例子；[SVG](../assets/kv-cache/paged-mapping.svg) · [PNG](../assets/kv-cache/paged-mapping.png) |
| 图 8-2 | 本书依据 [KIVI §3.1、Figure 1](https://arxiv.org/html/2402.02750v2) 自行绘制 | 4×4 矩阵、组长 4 的量化分组，不显示实验数据；[SVG](../assets/kv-cache/quantization-groups.svg) · [PNG](../assets/kv-cache/quantization-groups.png) |
| 图 8-3 | [Liu et al.，KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache，Figure 2](https://arxiv.org/pdf/2402.02750v2#page=4)；[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 从归档 PDF 第 4 页截取图区并栅格化，保留原标注；KIVI 观察到的 Key 与 Value 数值分布 |
| 图 9-1 | 本书依据 [Katharopoulos et al. §3.2](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf) 自行绘制 | 核特征计算的重排与中间张量；[SVG](../assets/linear-attention/kernel-reordering.svg) · [PNG](../assets/linear-attention/kernel-reordering.png) |
| 图 9-2 | [Yutao Sun et al.，Retentive Network: A Successor to Transformer for Large Language Models，Figure 3](https://arxiv.org/pdf/2307.08621v4#page=4) | 从归档 PDF 第 4 页截取图区并栅格化，保留原标注；RetNet 的并行计算与递推计算 |
| 图 9-3 | 本书依据 [Gated DeltaNet §3.1](https://arxiv.org/html/2412.06464v1#S3.SS1) 自行绘制 | 衰减、读取误差与状态修正的教学示意；[SVG](../assets/linear-attention/delta-update.svg) · [PNG](../assets/linear-attention/delta-update.png) |
| 图 10-1 | 本书参照 [Griffin §3](https://arxiv.org/html/2402.19427v1#S3) 的模块组合自行构造层排列 | 三种教学布局与各层历史状态，非完整发布模型；[SVG](../assets/hybrid-architectures/layer-state-layouts.svg) · [PNG](../assets/hybrid-architectures/layer-state-layouts.png) |
| 图 10-2 | [De et al.，Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models，Figure 2](https://arxiv.org/pdf/2402.19427v1#page=3)；[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 从归档 PDF 第 3 页截取图区并栅格化，保留原标注；Griffin 的残差块与递推模块 |
| 图 10-3 | 本书根据 10.5 的状态恢复条件自行绘制，递推定义参照 [Mamba §2—3](https://arxiv.org/html/2312.00752v2#S2) | 同一前缀边界的快照与分支更新，非特定服务实现；[SVG](../assets/hybrid-architectures/prefix-state-branches.svg) · [PNG](../assets/hybrid-architectures/prefix-state-branches.png) |

<p align="center">表 C-2　图表来源</p>

Vaswani et al. 原论文首页允许署名后的学术图表转载；GQA 原图随论文以 CC BY 4.0 提供。图表的文件路径、转载条件和存储状态见仓库的 `references/catalog.yaml` 与 `book/assets/README.md`。

## 本地资料的查阅方式

公开书页直接链接原论文和作者页面。本地原文放在仓库 `sources/` 中，由 `sources/INDEX.md` 提供真实文件链接；来源 ID 与上述条目一致。原文文件默认不随 Git 仓库发布。

当前工作树已归档 60 篇必备论文 PDF 和清单中的 76 个图表文件；正文中的自回归 GIF、Scaled Dot-Product Attention 与 GQA 原图均使用本地相对路径。另存有 3Blue1Brown 的英文字幕，视频仍通过原始链接观看。重新克隆后，本地资料是否齐备应以 `python3 scripts/check_handbook.py --require-archives` 的检查结果为准。

在允许联网的环境中，可在仓库根目录执行 `python3 scripts/sources.py --download` 获取必备原文及引用图片，再执行 `python3 scripts/sources.py` 更新索引。选读资料使用 `--download --all`。工具说明和后续归档要求见仓库 `references/README.md`。

## KV Cache 实现补充

- [vLLM Paged Attention 历史设计文档](https://docs.vllm.ai/en/latest/design/paged_attention/)，访问日期 2026-10-26。核查历史文档声明、Inputs、Key 与 Softmax 部分；仅用于解释原始寻址与归一化机制。来源 ID：`vllm-paged-attention-history`。
