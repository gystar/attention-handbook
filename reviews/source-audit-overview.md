# 第一章归档后的来源对照审校

## 范围与版本

- 日期及方式：2026-10-26，作者自查。
- 范围：`book/chapters/overview/` 全部五页、图 1-1—1-12、表 1-1—1-7；共享图 `prefill-decode.svg` 同时用于 2.5。
- 依据：CONTRIBUTING 的结构、原文核查、数学、图文和表达专项要求。
- 本轮修订版本：[校验值清单](source-audit-overview.sha256)。既有下载身份和版本检查见 [归档记录](source-archive.md)。

## 目录与标题核查

概览定位范围，1.1 建立表示与上下文需求，1.2 解释机制职责的变化，1.3 引入生成成本，1.4 分类主要改进对象。同级小节分别围绕概念、演进、成本和技术路线组织；机制计算留在第二章。标题、导航及图号未变。

本章的完备性按概览职责检查：具备 Token 与表示、因果生成、QKV 初步含义、缓存与长度成本、位置适配、信息使用及后续各章入口。没有把并行发展的优化方法写成单一替代链。各专题的完整方法覆盖仍须在对应章节审校。

## 逐段来源对照

本地论文名以 `references/catalog.yaml` 中的来源 ID 标识；版本均采用当前归档版本。表中覆盖正文论述、相邻图文及比较表；章节导航、素材索引入口仅作文件与名称检查。

| 页面与段落范围 | 论断与核对依据 | 结果 |
| --- | --- | --- |
| 概览的范围、四节导航与末段 | 与四节内容及第二至十章导航核对；明确 Decoder-only 常规生成的成本场景 | 通过 |
| 1.1 引言、表示学习首段、训练与深度学习段 | [Deep Learning Chapter 1](https://www.deeplearningbook.org/contents/intro.html)，表示选择、学习特征与多层表示部分 | 将表示定义改为数值形式描述输入及特征，补像素与特征向量的区别；避免将表示学习简化为文本转编号 |
| 1.1 影评例子及图 1-1 前后四段 | [Alammar 教程](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)，How a single prediction is calculated、Flowing Through DistilBERT、Slicing the important part；实际查看本地图 | 768 维、rum/##ination、CLS/SEP、首位置分类特征均对应。正文没有采用教程中对 NSP 与 CLS 作用的推测 |
| 1.1 上下文三段及图 1-2 | [3Blue1Brown 教学页](https://www.3blue1brown.com/lessons/attention/)，Recap、Motivating Examples；实际查看词义图 | 同 Token ID 的初始查表向量与上下文表示区分准确；保留 AI 示意和 Tokenizer 限定 |
| 1.1 Attention 定义、图 1-3、Head 与动态权重三段 | vaswani2017attention v7 §3.2—3.2.2；shazeer2019mqa v1 §2.2—2.3；[Alammar 原图上下文](https://jalammar.github.io/illustrated-transformer/) Self-Attention at a High Level | 加权汇总和参数/权重区别成立；本地图选中 it，Input–Input 及层标记与 Encoder 解读一致。没有据单个 Head 断言完整解释模型，也没有采用博客“自身权重必然最高”的错误泛化 |
| 1.1 末段与 1.2 引言 | 后续实际从 RNN 输入读取进入 Self-Attention | 承担概念衔接，保留 |
| 1.2 RNN 三段及图 1-4 | bahdanau2014attention v7 §2.1、Eq. (1)—(3)；[D2L §10.7.1—10.7.3](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)；实际查看图 | 固定维度摘要及潜在瓶颈成立；D2L 图每步接收 Context，正文没有误写成仅初始化一次 |
| 1.2 动态读取四段及图 1-5 | bahdanau2014attention v7 §3.1—3.2、Eq. (4)—(6)；查看本地教学图 | Context 随 Decoder 步骤变化；原模型仍采用 RNN。两行示意权重均和为 1，明确是教学设定 |
| 1.2 Self-Attention 三段及表 1-2 | vaswani2017attention v7 §2、§3.2.3 | 同序列读取、已知层输入时的并行性成立；Transformer 之前的使用场景在原文 Background 列明 |
| 1.2 核心机制、职责表、并行与 FFN 段 | vaswani2017attention v7 §3.1—3.3、§4 Table 1 | 三种 Attention 职责对应，FFN 按位置变换，常规生成仍有顺序依赖；未把训练并行推及生成 |
| 1.2 图 1-6 后的因果解释与三种结构 | vaswani2017attention v7 §3.1 输入偏移说明；[Hugging Face 架构课](https://huggingface.co/learn/llm-course/chapter1/6) | 补充 Decoder 输入右移和目标词不可见的例子；三种结构作为常见类型，不作能力互斥分类 |
| 1.3 引言、自回归、QKV 与缓存段、图 1-7—1-8 | kwon2023pagedattention v1 §2.1—2.2；shazeer2019mqa v1 §2.4；本地 KV 图及已有 GIF 帧核对记录 | SC-01—SC-02 修订后通过。动画是追加循环的示意；缓存图为逐层缓存 |
| 1.3 Prefill/Decode 两段及图 1-9 | kwon2023pagedattention v1 §2.2；jiang2023mistral v1 §2 Pre-fill and Chunking；SVG 源码与实际渲染 | Prompt 末位生成首个新 Token，后续处理新位置；修正 SVG desc 中遗漏当前 K/V 的文字，画面原已写 QKV |
| 1.3 成本两段及图 1-10 | vaswani2017attention v7 Table 1；shazeer2019mqa v1 §2.4.1；dao2022flashattention v2 §3.1 | 独立计算 4×5/2=10、8×9/2=36；图注明包含当前位置。运算量、缓存、访存和延迟没有合并为同一指标 |
| 1.3 长上下文首段、表 1-4 与 PI 段 | chen2023pi v2 §2.3、Eq. (4)；上一成本段与后一能力实验 | 资源、位置和信息使用是本书的组织维度；PI 已明确缩放及微调，不声称任意长度能力保证 |
| 1.3 Lost in the Middle 段及图 1-11 | liu2023lostmiddle v1 §3.1—3.3、Figures 4—5 | 原实验改变相关文档位置；七文档图是本书设置示意，未伪装成原文得分或原实验数量 |
| 1.4 开头、位置与 Gating 两段及图 1-12 | su2021rope v5 §3.2；press2021alibi v2 §3；qiu2025gatedattention v1 §2.2、Eq. (5) | RoPE 作用于 Q/K、ALiBi 作用于分数，图中已区分。Gating 是原文所研究的一种 Head-specific Elementwise 设计，无无条件收益断言 |
| 1.4 历史信息三类、表 1-5、MQA/GQA/MLA 段 | shazeer2019mqa v1 §3；ainslie2023gqa v3 §2.2；deepseek2024mla v5 §2.1.2—2.1.3 | Head 共享与低秩 KV 表示区分准确；概览未声称 MLA 仅缓存 Latent、也未遗漏后文 Decoupled RoPE 入口 |
| 1.4 Sparse 与 Linear 段 | child2019sparse v1 §4.2—4.3 的访问集合；katharopoulos2020linear PMLR 版 §3.4、Eq. (16)—(20) | 稀疏连接与缓存淘汰分别统计；固定维度状态结论限于适用核形式，未套用于精确 Softmax 的无限维映射 |
| 1.4 FlashAttention、PagedAttention 与表 1-6 | dao2022flashattention v2 §3.1；kwon2023pagedattention v1 §4.1—4.2 | 保留数学等价与浮点差异；分页描述存储映射，不混同量化与淘汰 |
| 1.4 Prefix Caching 与 Offloading 段 | zheng2023sglang v2 §3；sheng2023flexgen v1 §1 的内存层次和张量迁移说明；8.5 复用条件 | 概览正确区分复用与迁移，细节入口有效 |
| 1.4 Mistral 与 Hybrid 段 | jiang2023mistral v1 §2；SC-06 的 Gemma 2、DeltaNet、SSD 对照 | 方法组合与各模块作用对应；没有从结构组合直接推出速度或质量优势 |
| 1.4 比较条件段、表 1-7 与末段链接 | 与 1.3、各专题评价小节交叉检查 | 分清模型能力、运行阶段与服务系统的测量对象；指标是比较要求，不是论文实测结果 |

## 表达专项核查

逐段重读五页正文与图表说明。保留各章入口和承担衔接的引言；没有新增态度性总结。以下否定或限定具有技术作用：Token 不固定对应词、单 Head 不能代表全部计算、因果目标不可见、固定配置下历史表示不受追加影响、示意图不展示实测得分、浮点顺序改变可能带来差异。它们分别限定输入单位、解释对象、依赖关系和证据范围。

本轮将表示学习改为“学习如何提取特征”，补充因果输入右移，修复 SVG 描述遗漏。改写保留原引用、图号、教学设定及参数条件；未扩写成与第二章重复的推导。

## 问题与修正

| 编号 | 级别 | 位置与问题 | 修正及复查 | 状态 |
| --- | --- | --- | --- | --- |
| OV-S01 | 需修正 | 1.1 表示定义偏向输入数值化，未突出特征表示 | 依据 Deep Learning Chapter 1 区分像素与特征，训练学习特征提取；与影评实例衔接复查 | 已关闭 |
| OV-S02 | 需修正 | 1.2 “当前位置译文”未解释与目标的错位 | 依据 Transformer §3.1 增补右移例子；与 2.5 的因果计算对照 | 已关闭 |
| OV-S03 | 需修正 | 图 1-9 / 2-5 SVG desc 遗漏当前 K/V 计算 | 改为计算 QKV；源码解析及 GdkPixbuf 渲染查看通过，画面与可访问性说明一致 | 已关闭 |

前轮 SC-01—SC-06 见 [总记录](source-content-audit.md)，本轮保留修订并核对其上下文。

## 检查结果与结论

第一章正文的本轮来源对照、结构、表达和本地图文内容复查完成。图 1-1—1-6、1-8、1-10—1-12 实际查看，图 1-9 实际渲染查看；图 1-7 的 53 帧解码及代表帧查看沿用未变文件的归档记录。内容判断对应校验值清单，不表示独立审阅或训练复现。

目标 GitBook 的公式、图表、侧栏和视频呈现仍待实际检查。本地图像查看不代替目标页面验收。全书其余章节、附录及最终完备性审计继续进行。
