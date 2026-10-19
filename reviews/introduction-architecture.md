# 1.1 架构介绍复查

## 范围与版本

2026-10-19，作者自查。范围为 introduction.md 最后一小节及新增课程来源记录；版本见 [introduction-architecture.sha256](introduction-architecture.sha256)。此前 Review 的文件快照保留为历史版本。

## 检查结果

| 检查维度 | 对应位置与核查依据 | 结果 |
| --- | --- | --- |
| 范围与结构 | 标题改为“基于 Transformer 的语言模型”；先说明架构与模型类别的关系，再引出三种结构、层内计算与自回归生成。 | 通过 |
| 事实与引用 | 核对 Hugging Face 官方课程三种架构的对应段落；保留此前已核查的 Vaswani §3.1—3.4 与 Shazeer §2.4 引用。未新增性能、历史或实验结论。 | 通过 |
| 数学与实现 | 未修改公式、算例或实现。 | 不适用 |
| 实验与比较 | 无新增实验数据或性能比较。 | 不适用 |
| 表达与教学素材 | 用两句话介绍 Encoder、Decoder 及三种结构；首次解释自回归与 Causal Self-Attention。图 1-4 内容未变，图注仍说明省略组件。 | 通过 |
| 资料与链接 | 课程全文已在线核查并登记；本地结构检查通过。既有 12 项待归档资料未在本轮处理。 | 本次修改通过 |
| 跨章一致性 | 对照 1.3：均以 Decoder-only 自回归模型为主要场景，访问范围包含当前位置，下一 Token 由输出层预测；无旧小节标题的其他正文引用。 | 通过 |
| 页面呈现 | 未检查 GitBook 目标页面。 | 待核查 |

## 问题与修正

| 编号 | 级别 | 问题 | 修正及复查结果 | 状态 |
| --- | --- | --- | --- | --- |
| A1 | 需修正 | 标题并列 Transformer 与 LLM，层次关系不清 | 标题与首段明确架构用于构建模型的关系 | 已关闭 |
| A2 | 需修正 | 未介绍架构分类即使用 Decoder-only | 先说明 Encoder 与 Decoder，再用一句话列出三种结构及作用 | 已关闭 |
| A3 | 需修正 | 叙述后置概念区分、衔接生硬 | 删除末尾重复分类，按架构、组件、生成过程组织 | 已关闭 |

## Review 结论

本次修改内容已核查；发布检查待完成。需在同步后检查 GitBook 图文与链接显示；本记录不代表整章既有归档缺口已经解决。

## “Transformer 中的 Attention 机制”重写复查

2026-10-19，作者自查。后续版本见 [introduction-attention-role.sha256](introduction-attention-role.sha256)，先前快照保留为历史版本。

- 结构：以 Attention 的层内作用开篇，简述 Encoder/Decoder 与三种架构，再解释图中三处 Attention 的信息来源，最后连接 Decoder-only 自回归生成。小节标题与内容一致。
- 技术：依据 Vaswani §3.1—3.3 核对 FFN 的逐位置作用、Encoder 全输入访问、Decoder 因果访问与 Encoder–Decoder Attention；Decoder 访问范围包含当前位置。依据 Alammar GPT-2 教程核对 Decoder-only 不含读取独立 Encoder 的 Attention，以及逐步追加 Token 的生成流程。未将所有 LLM 限定为该架构。
- 图示：替换为作者教程 A High-Level Look 原图，已下载并实际查看。Git blob 与来源返回值一致，SHA-256 和字节数登记入 catalog；正文保留许可与署名。图注明确省略组件及未画出的 Decoder 因果限制，避免把示意图当完整结构。
- 一致性：图号保留 1-4，附录与素材索引同步，旧图留作历史资料。与 1.3 的 Attention 信息来源分类一致；本节不提前引入 QKV 公式。
- 验证：check_handbook.py 与 git diff --check 通过；没有新增公式、实现或实验数据。GitBook 页面尚未实际预览，发布检查待完成；既有 12 项资料归档缺口不在本次修订范围。

本次修改内容已核查，发布检查待完成。
