# 图表来源与源文件

随书图表按章节放置。引用图的作者、定位与许可集中列于附录 C，正文提供入口，改绘注明来源及改绘，原创示意图保留可编辑 SVG 或生成脚本。

本书自制图表为结构、计算流程与方法关系的教学示意。无法下载的外部原图保留原始链接，来源索引如实记录归档状态。

## 正文素材清单

| 图号 | 文件或原始位置 | 来源与存储状态 |
| --- | --- | --- |
| 1-1 | [alammar-sentence-representation.png](overview/alammar-sentence-representation.png) | Jay Alammar，A Visual Guide to Using BERT for the First Time；英文文本到逐 Token 向量，CC BY-NC-SA 4.0，原图未修改 |
| 1-2 | [context-apple.png](overview/context-apple.png) | 本书使用内置 imagegen 生成；上下文词义教学示意，非真实事件或实测数据；提示词见 [context-apple.prompt.txt](overview/context-apple.prompt.txt) |
| 1-3 | [alammar-self-attention.png](overview/alammar-self-attention.png) | Jay Alammar 教程原图，CC BY-NC-SA 4.0；Encoder 场景的可视化 |
| 1-4 | [d2l-seq2seq.png](overview/d2l-seq2seq.png)／[SVG 原文件](overview/d2l-seq2seq.svg) | Zhang, Lipton, Li & Smola，Dive into Deep Learning，Figure 10.7.1；CC BY-SA 4.0，SVG 白底栅格化，图中内容未改动 |
| 1-5 | [PNG](overview/translation-attention-en-zh.png)／[SVG 源文件](overview/translation-attention-en-zh.svg) | 本书依据 Bahdanau §3.1 绘制；I like tea → 我喜欢茶，切分与权重为教学设定 |
| 1-6 | [alammar-encoder-decoder.png](overview/alammar-encoder-decoder.png) | Jay Alammar 教程原图，CC BY-NC-SA 4.0；Encoder 与 Decoder 中三处 Attention 的位置 |
| 1-7 | [自回归 GIF](https://jalammar.github.io/images/xlnet/gpt-2-autoregression-2.gif) | Jay Alammar，The Illustrated GPT-2；CC BY-NC-SA 4.0，原图远程引用，本地归档待完成 |
| 1-8 | [alammar-kv-cache.png](overview/alammar-kv-cache.png) | Jay Alammar，The Illustrated GPT-2；CC BY-NC-SA 4.0，原图未修改 |
| 1-9 | [prefill-decode.svg](overview/prefill-decode.svg) | 本书依据 [Kwon et al. §2.2](https://arxiv.org/html/2309.06180v1) 绘制；只展示位置关系，省略其余网络计算 |
| 1-10 | [PNG](overview/context-costs.png)／[SVG](overview/context-costs.svg) | 本书绘制；单 Head、固定配置、缓存全部历史的计数示意 |
| 1-11 | [PNG](overview/context-evidence-position.png)／[SVG](overview/context-evidence-position.svg) | 本书依据 Lost in the Middle §3.1 绘制实验设置；未描绘实验结果 |
| 1-12 | [PNG](overview/attention-design-locations.png)／[SVG](overview/attention-design-locations.svg) | 本书综合绘制；生成脚本 scripts/render_attention_routes.py；方法依据见 1.4 对应引用 |
| 2-1 | [PNG](transformer-attention/d2l-transformer.png)／[原始 SVG](transformer-attention/d2l-transformer.svg) | Zhang, Lipton, Li & Smola，D2L Figure 11.7.1；CC BY-SA 4.0，白底栅格化 |
| 2-2 | [Transformer Figure 2 左图](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png) | Vaswani et al.，原图远程引用，待本地归档 |
| 2-3 | [PNG](transformer-attention/attention-masks.png)／[SVG](transformer-attention/attention-masks.svg) | 本书依据 Vaswani §3.2.3 绘制；网格仅表示访问范围 |
| 2-4 | [PNG](transformer-attention/d2l-multi-head-attention.png)／[原始 SVG](transformer-attention/d2l-multi-head-attention.svg) | Zhang, Lipton, Li & Smola，D2L Figure 11.5.1；CC BY-SA 4.0，白底栅格化 |
| 2-5 | [prefill-decode.svg](overview/prefill-decode.svg) | 复用图 1-9，方法依据为 Kwon §2.2 |
| 3-1 | [PNG](position-encoding/position-paths.png)／[SVG](position-encoding/position-paths.svg) | 本书依据公式绘制；生成脚本 scripts/render_position_encoding.py；方法与参数见附录 C |
| 3-2 | [PNG](position-encoding/alammar-sinusoidal.png) | Jay Alammar 原图；CC BY-NC-SA 4.0；未修改 |
| 3-3 | [PNG](position-encoding/rope-relative-rotation.png)／[SVG](position-encoding/rope-relative-rotation.svg) | 本书依据公式绘制；生成脚本 scripts/render_position_encoding.py；方法与参数见附录 C |
| 3-4 | [PNG](position-encoding/alibi-bias.png)／[SVG](position-encoding/alibi-bias.svg) | 本书依据公式绘制；生成脚本 scripts/render_position_encoding.py；方法与参数见附录 C |
| 3-5 | [PNG](position-encoding/position-interpolation.png)／[SVG](position-encoding/position-interpolation.svg) | 本书依据公式绘制；生成脚本 scripts/render_position_encoding.py；方法与参数见附录 C |
| 3-6 | [PNG](position-encoding/frequency-scaling.png)／[SVG](position-encoding/frequency-scaling.svg) | 本书依据公式绘制；生成脚本 scripts/render_position_encoding.py；方法与参数见附录 C |
| 3-7 | [PNG](overview/context-evidence-position.png)／[SVG](overview/context-evidence-position.svg) | 复用图 1-11；Lost in the Middle 证据位置实验设置 |
| 4-1 | [GQA Figure 2](https://arxiv.org/html/2305.13245v3/images/gmq_architecture.png) | Ainslie et al.，CC BY 4.0；原图远程引用，待本地归档 |

Transformer 图表使用条件见原论文首页的转载声明。三张论文原图的归档目标路径和来源 ID 已记录在仓库 `references/catalog.yaml`，没有将尚不存在的本地文件作为链接。

原有两张原创 SVG 已通过 librsvg 2.58 本地栅格预览检查；GitBook 页面显示待检查。


新增概念关系图 `attention-context.svg` 已经 GdkPixbuf 加载、生成 PNG 并检查文字与连线；复查证据见仓库 `reviews/previews/attention-context.png`。

三张 Alammar 原图来自 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，以 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 原样收录；署名见正文图注，禁止据本书其他文件的许可将这些素材视为可商用。源仓库为 `jalammar/jalammar.github.io`，来源路径、Git blob 与 SHA-256 见 catalog.yaml。原 `attention-context.svg` 保留为历史素材，当前正文不再引用。

原 `alammar-embeddings.png`、`representation-learning.svg` 和 `alammar-token-embeddings.png` 保留为历史素材，当前正文不再引用。图 1-1 使用 [BERT 入门教程](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)的英文原图，下载路径与校验值见 catalog.yaml；展示分词、编号与模型输出向量。

原 `alammar-encoder.png` 保留为历史素材，图 1-6 改用同一教程 A High-Level Look 的 Encoder–Decoder 示意图；原图未修改，许可与校验值见 catalog.yaml。

原论文 Transformer Figure 1 保留归档计划。现行图号以本页表格和正文为准，旧编号见历史 Review。

新增 D2L SVG 来自作者仓库 `d2l-ai/d2l-en/img/seq2seq.svg`；许可见该仓库 LICENSE。新增翻译 Attention 图来自 `jalammar/jalammar.github.io/images/attention.png`；两者均保留原文件，Git blob 与 SHA-256 见 catalog.yaml。

图 1-5 的生成脚本为仓库 `scripts/render_translation_attention.py`。原法英翻译图 `alammar-translation-attention.png` 保留为历史素材。

图 1-10、1-11 的可编辑 SVG 与 PNG 由 `scripts/render_context_figures.py` 生成。图 1-8 通过作者 GitHub 仓库下载，源路径与校验值见 catalog.yaml。

原 `autoregressive-steps.svg` 与 PNG 保留为历史素材，当前正文已由 Alammar 自回归动画替换。

原章节导航图 `attention-landscape.svg` 保留为历史素材；当前图 1-12 按方法作用于计算的主要位置组织，不再列章号。

第二章新增素材由 `scripts/render_transformer_attention.py` 生成或栅格化。D2L SVG 原文件保持不变；[源仓库许可](https://github.com/d2l-ai/d2l-en/blob/master/LICENSE)为 CC BY-SA 4.0，版本与文件校验值见 catalog.yaml。
