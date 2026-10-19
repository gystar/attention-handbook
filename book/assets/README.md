# 图表来源与源文件

随书图表按章节放置。引用图在图注中标明原作者与定位信息，改绘注明来源及改绘，原创示意图保留可编辑 SVG 或生成脚本。

本书自己的图表用于解释结构、计算流程或方法关系，示意图不代表实测性能。无法下载的外部原图保留原始链接，来源索引如实记录归档状态。

## 首章素材清单

| 图号 | 文件或原始位置 | 来源与存储状态 |
| --- | --- | --- |
| 1-1a | [alammar-embeddings.png](overview/alammar-embeddings.png) | Jay Alammar 教程原图，本地保存，CC BY-NC-SA 4.0 |
| 1-1b | [context-apple.png](overview/context-apple.png) | 本书使用内置 imagegen 生成；上下文词义教学示意，非真实事件或实测数据；提示词见 [context-apple.prompt.txt](overview/context-apple.prompt.txt) |
| 1-1c | [alammar-self-attention.png](overview/alammar-self-attention.png) | 同上；Encoder 场景的可视化 |
| 1-1d | [alammar-encoder.png](overview/alammar-encoder.png) | 同上；简化 Encoder 层，非 Decoder-only 全图 |
| 1-1e | [attention-landscape.svg](overview/attention-landscape.svg) | 本书原创，移至方法分类小节 |
| 1-2 | [Transformer Figure 2 左图](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png) | Vaswani et al.，原图远程引用，待本地归档 |
| 1-3 | [Transformer Figure 1](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png) | Vaswani et al.，原图远程引用，待本地归档 |
| 1-4 | [prefill-decode.svg](overview/prefill-decode.svg) | 本书依据 [Kwon et al. §2.2](https://arxiv.org/html/2309.06180v1) 绘制；只展示位置关系，省略其余网络计算 |
| 1-5 | [GQA Figure 2](https://arxiv.org/html/2305.13245v3/images/gmq_architecture.png) | Ainslie et al.，CC BY 4.0；原图远程引用，待本地归档 |

Transformer 图表使用条件见原论文首页的转载声明。三张原图的目标路径和来源 ID 已记录在仓库 `references/catalog.yaml`，没有将尚不存在的本地文件作为链接。

原有两张原创 SVG 已通过 librsvg 2.58 本地栅格预览检查；此记录只覆盖图本身，不代表 GitBook 页面中的实际显示已检查。


新增概念关系图 `attention-context.svg` 已经 GdkPixbuf 加载、生成 PNG 并检查文字与连线；复查证据见仓库 `reviews/previews/attention-context.png`。

三张 Alammar 原图来自 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，以 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 原样收录；署名见正文图注，禁止据本书其他文件的许可将这些素材视为可商用。源仓库为 `jalammar/jalammar.github.io`，来源路径、Git blob 与 SHA-256 见 catalog.yaml。原 `attention-context.svg` 保留为历史素材，当前正文不再引用。
