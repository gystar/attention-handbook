# 图 1-1 来源与英文示例复查

2026-10-19，作者自查。范围为 1.1 的 Embedding 介绍、图 1-1 及素材索引；替代此前原创中文查表图，历史 Review 保留。

| 检查项 | 结果与依据 |
| --- | --- |
| 来源与使用条件 | 核对 Jay Alammar《The Illustrated GPT-2》的 Input Encoding 与页尾 CC BY-NC-SA 4.0 声明；原图未修改，正文署名并链接许可。 |
| 文件一致性 | 从作者 GitHub 仓库取得原图，Git blob SHA-1 与返回值一致；记录 SHA-256 和字节数。实际查看本地图片，英文标注与矩阵清晰。 |
| 概念边界 | 原图右侧为简化词条，不能作为实际 GPT-2 分词结果；图注明确限定。正文引入 Hugging Face 官方课程 Subword tokenization 中 tokenization → token / ization 的示例，说明切分取决于 Tokenizer 与词表，不声称是本地运行结果。 |
| 编号与引用 | 保留图 1-1；同步正文、素材清单、参考附录与 catalog。旧 SVG 保留历史归档。 |
| 本地结构 | check_handbook.py 通过；既有 12 项资料归档缺口未在本轮解决。 |
| 发布检查 | 未检查 GitBook 实际页面，待同步后检查尺寸、清晰度与链接。 |

本次修改内容已核查，发布检查待完成；未变更数学公式与计算脚本。版本见 [representation-source.sha256](representation-source.sha256)。

## 英文句子到向量图的替换复查

2026-10-19，作者自查；后续版本见 [representation-sentence.sha256](representation-sentence.sha256)。图 1-1 改用同一作者 BERT 入门教程的 Flowing Through DistilBERT 原图。已阅读该节、前置分词说明、分类向量选取及对应代码，核对页尾许可。原图从作者仓库取得，Git blob 校验一致，SHA-256 已登记；实际查看本地原图，英文句子、子词、编号、逐位置向量及 768 维标注均清晰。

正文改为从一句英文影评出发，并明确：rumination 被拆成两个 Token；编号不是语义大小；图中输出是上下文化表示，不是初始 Embedding 查表结果；用于分类的整句特征取自特殊 Token [CLS] 的输出。引用仅用于解释原教程流程，未宣称本地复现或证明分类性能。三种架构介绍及后续章节未改动，图号仍为 1-1，素材索引与参考附录同步。旧图保留为历史素材。

本次适用的内容、来源、图示和跨文件一致性核查已完成，本地结构及 diff 格式检查通过；无新增公式或实验需要验证。GitBook 页面呈现仍待同步后核查，既有资料归档缺口未改变。
