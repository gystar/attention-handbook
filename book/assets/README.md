# 图表来源与源文件

随书图表按章节放置。引用图的作者、定位与许可集中列于附录 C，正文提供入口，改绘注明来源及改绘，原创示意图保留可编辑 SVG 或生成脚本。

本书自制图表为结构、计算流程与方法关系的教学示意。无法下载的外部原图保留原始链接，来源索引如实记录归档状态。

## 首章素材清单

| 图号 | 文件或原始位置 | 来源与存储状态 |
| --- | --- | --- |
| 1-1 | [alammar-sentence-representation.png](overview/alammar-sentence-representation.png) | Jay Alammar，A Visual Guide to Using BERT for the First Time；英文文本到逐 Token 向量，CC BY-NC-SA 4.0，原图未修改 |
| 1-2 | [context-apple.png](overview/context-apple.png) | 本书使用内置 imagegen 生成；上下文词义教学示意，非真实事件或实测数据；提示词见 [context-apple.prompt.txt](overview/context-apple.prompt.txt) |
| 1-3 | [alammar-self-attention.png](overview/alammar-self-attention.png) | Jay Alammar 教程原图，CC BY-NC-SA 4.0；Encoder 场景的可视化 |
| 1-4 | [d2l-seq2seq.png](overview/d2l-seq2seq.png)／[SVG 原文件](overview/d2l-seq2seq.svg) | Zhang, Lipton, Li & Smola，Dive into Deep Learning，Figure 10.7.1；CC BY-SA 4.0，SVG 白底栅格化，图中内容未改动 |
| 1-5 | [PNG](overview/translation-attention-en-zh.png)／[SVG 源文件](overview/translation-attention-en-zh.svg) | 本书依据 Bahdanau §3.1 绘制；I like tea → 我喜欢茶，切分与权重为教学设定 |
| 1-6 | [alammar-encoder-decoder.png](overview/alammar-encoder-decoder.png) | Jay Alammar 教程原图，CC BY-NC-SA 4.0；Encoder 与 Decoder 中三处 Attention 的位置 |
| 1-7 | [Transformer Figure 2 左图](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png) | Vaswani et al.，原图远程引用，待本地归档 |
| 1-8 | [prefill-decode.svg](overview/prefill-decode.svg) | 本书依据 [Kwon et al. §2.2](https://arxiv.org/html/2309.06180v1) 绘制；只展示位置关系，省略其余网络计算 |
| 1-9 | [attention-landscape.svg](overview/attention-landscape.svg) | 本书原创，移至方法分类小节 |
| 1-10 | [GQA Figure 2](https://arxiv.org/html/2305.13245v3/images/gmq_architecture.png) | Ainslie et al.，CC BY 4.0；原图远程引用，待本地归档 |

Transformer 图表使用条件见原论文首页的转载声明。三张论文原图的归档目标路径和来源 ID 已记录在仓库 `references/catalog.yaml`，没有将尚不存在的本地文件作为链接。

原有两张原创 SVG 已通过 librsvg 2.58 本地栅格预览检查；GitBook 页面显示待检查。


新增概念关系图 `attention-context.svg` 已经 GdkPixbuf 加载、生成 PNG 并检查文字与连线；复查证据见仓库 `reviews/previews/attention-context.png`。

三张 Alammar 原图来自 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，以 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 原样收录；署名见正文图注，禁止据本书其他文件的许可将这些素材视为可商用。源仓库为 `jalammar/jalammar.github.io`，来源路径、Git blob 与 SHA-256 见 catalog.yaml。原 `attention-context.svg` 保留为历史素材，当前正文不再引用。

原 `alammar-embeddings.png`、`representation-learning.svg` 和 `alammar-token-embeddings.png` 保留为历史素材，当前正文不再引用。图 1-1 使用 [BERT 入门教程](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)的英文原图，下载路径与校验值见 catalog.yaml；展示分词、编号与模型输出向量。

原 `alammar-encoder.png` 保留为历史素材，图 1-6 改用同一教程 A High-Level Look 的 Encoder–Decoder 示意图；原图未修改，许可与校验值见 catalog.yaml。

原论文 Transformer Figure 1 保留归档计划。现行图号以本页表格和正文为准，旧编号见历史 Review。

新增 D2L SVG 来自作者仓库 `d2l-ai/d2l-en/img/seq2seq.svg`；许可见该仓库 LICENSE。新增翻译 Attention 图来自 `jalammar/jalammar.github.io/images/attention.png`；两者均保留原文件，Git blob 与 SHA-256 见 catalog.yaml。

图 1-5 的生成脚本为仓库 `scripts/render_translation_attention.py`。原法英翻译图 `alammar-translation-attention.png` 保留为历史素材。
