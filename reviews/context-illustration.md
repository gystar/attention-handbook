# 1.1 上下文插图复查

2026-10-19，作者自查。将“语言表示为什么需要上下文”的四段文字改为两段短文及一张双场景插图，以“苹果”的水果与公司两种含义说明上下文作用。

使用 imagegen 内置工具生成，提示词保存在 book/assets/overview/context-apple.prompt.txt。生成图已实际查看：两句中文、关键词着色及“水果／公司”标签正确，场景与句子对应。按用户反馈重新生成学术插图版本，移除顶部大标题及摄影场景，改为低饱和插画、等色关键词标注与双栏编号；实际查看确认文字正确、布局清晰。图片已保存到 book/assets/overview/context-apple.png，生成原文件保留。

图注明确是 AI 生成的教学示意，不是真实发布会照片或模型实验；不假定“苹果”对应一个 Token。正文限定“相同词表与嵌入参数”，区分初始查表表示与结合上下文后的表示。该例只说明语义需求，不将颜色或画面解释为模型向量、权重或测量结果。概念来源沿用已核查的 3Blue1Brown Motivating Examples。

图号同步调整为 1-1a 至 1-1e，来源索引与素材清单已更新。实际执行本地结构检查、名称一致性检查及 git diff --check，结果通过；GitBook 同步后的图片缩放与页面呈现待检查。本轮文件快照见 [context-illustration.sha256](context-illustration.sha256)。此前未提交的 Representation Learning 修改保留，未回退。
