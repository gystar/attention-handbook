# 第 1 章 Review：总体框架与技术演进

后续维护：书籍首页、部署说明与仓库检查工具的适配见 [GitBook 接入适配复查](gitbook-adaptation.md)。本记录及原 SHA-256 清单保留首章当时的审阅版本，后续维护不覆盖历史快照。

## 范围与版本

- 审阅日期：2026-10-19。
- 审阅者与方式：Codex，作者自查；没有独立审阅者，不标称独立 Review。
- 正文：[章节概览及五节正文](../book/chapters/overview/README.md)。相关文件包括首章图表、三个附录、来源清单、维护脚本与全书导航。
- 初稿快照：[overview-before.sha256](overview-before.sha256)，覆盖本轮内容修正前的六个正文文件和两张 SVG。
- 复查版本：[overview-after.sha256](overview-after.sha256)，覆盖本轮完成后的正文、图表、来源记录、脚本及规范。当前目录尚未初始化 Git，使用文件 SHA-256 记录版本。Review 记录自身不纳入该清单，避免循环依赖。
- 原始资料版本及核查位置：[catalog.yaml](../references/catalog.yaml)。以下“核查”指核对支持本章论述的相关正文，不代表整篇资料的全面审计。

## 检查结果

| 检查维度 | 对应位置与核查依据 | 结果 |
| --- | --- | --- |
| 范围与结构 | 对照 GOAL 与已定目录：总览、历史基础、成本、分类、演进与比较五节齐全；第 2—10 章明确标为正文待写 | 通过 |
| 事实与引用 | 逐项依据见下表；历史按首次公开年份组织，未宣称覆盖所有论文或后续最新工作 | 通过，限表中范围 |
| 数学与实现 | QKV、Mask、Bahdanau 权重、KV 容量、Linear Attention 递推均检查条件与维度；附录保留推导，算例实际运行 | 通过 |
| 实验与比较 | 本章未复现论文训练；容量例子明确为假设配置，没有把论文实验或示意图表述为本书实测 | 通过 |
| 表达与教学素材 | 两张原创图、三张原论文图的远程引用、16 个对照表、视频片段；补齐缩写、图注与公式条件 | 本地文本与原创图通过；原图实际呈现待查 |
| 资料与链接 | 本地文件与导航检查通过；19 篇原文、3 项教学资源有来源记录；9 篇必备 PDF 与 3 张引用原图尚未归档 | 需修正：R06 |
| 跨章一致性 | 首章与附录的符号、分类一致；导航与各章标题一致；其他九章仅检查框架，没有评价尚不存在的正文 | 通过 |
| 页面呈现 | 已检查原创 SVG 的本地渲染；已对照 GitBook 官方配置与公式语法；未连接目标 Space | 待核查：R07 |

## 核心结论与公式的证据

来源 ID 对应 [catalog.yaml](../references/catalog.yaml) 中的公开全文与原文版本。

| 论述或公式 | 位置 | 核查依据与检查结论 |
| --- | --- | --- |
| QKV 的作用及 Scaled Dot-Product Attention | 1.1 | `vaswani2017attention` v7 §3.2、`shazeer2019mqa` v1 §2.1—2.3；核对缩放、Mask、Softmax 方向与输出形状 |
| Attention 权重与模型参数的区别 | 1.1 | Shazeer v1 §2.1—2.3 的投影与输入相关计算；表述没有将激活值写成固定参数 |
| 加权 Value 算例 | 1.1 | 人为设定三个二维 Value，手算与脚本均得约 (0.8, 0.9)；不解释为真实语言模型权重 |
| 固定表示与动态 Context | 1.2 | `sutskever2014seq2seq` v3 §2；`bahdanau2014attention` v7 §3.1 式 (4)—(6)；区分概念对照与发表先后 |
| Bahdanau 的 Context 与归一化 | 1.2 | 检查 Encoder 长度、前一步 Decoder 状态、权重求和与 Context 维度；不将 Alignment 等同于 Seq2Seq |
| Self/Cross 与 Causal 的分类 | 1.2 | Vaswani v7 §3.2.3；核对三处 Attention 的 QKV 来源及右移输入；两类标签代表不同维度 |
| Prefill 与 Decode | 1.3、图 1-4 | `kwon2023pagedattention` v1 §2.2；`jiang2023mistral` v1 §2；核对首个输出 Token、当前 KV 与缓存追加位置 |
| Dense 算术与显式矩阵存储 | 1.3 | Vaswani v7 Table 1、`dao2022flashattention` v2 §3.1 Theorem 1；重新计算位置对数与张量规模，区分运算量与 HBM 中间存储 |
| Decode 的带宽条件 | 1.3 | Shazeer v1 §2.4.1；保留小 Batch、长度、硬件与实现条件，不宣称所有 Decode 都由同一瓶颈限制 |
| KV Cache bytes | 1.3、附录 A | 由两份 L×B×N×hkv×dh 张量乘以每元素字节数推导；脚本验证三个容量结果，列出未计入项 |
| MQA、GQA 与 MLA 的区别 | 1.3—1.4 | Shazeer v1 §3；`ainslie2023gqa` v3 §2.1—2.2；`deepseek2024mla` v5 §2.1.2—2.1.4；没有用 MQA 公式代替 MLA 的 Latent 与位置缓存 |
| FlashAttention 与 Sparse Attention | 1.4 | Dao v2 §3.1、§3.3；`child2019sparse` v1 §4；`yuan2025nsa` v1 §3.2—3.3.2；区分数学目标、位置选择与具体执行 |
| PagedAttention | 1.4 | Kwon v1 §4.1—4.2；核对逻辑块、物理块与 KV Cache Manager，没有将分页误作历史删除或量化 |
| RoPE、ALiBi、Position Interpolation | 1.3—1.4 | `su2021rope` v5 §3.2；`press2021alibi` v2 §3；`chen2023pi` v2 §2.3；分别说明旋转、偏置、重映射和适应训练 |
| Long Context 的能力边界 | 1.3 | `liu2023lostmiddle` v1 §3.1—3.3；结论限定在论文所测模型和任务，未作为后续所有模型的定论 |
| Linear Attention 的 Recurrent Formulation | 1.4、附录 A | `katharopoulos2020linear` PMLR 119 §3.2—3.4，式 (9)—(20)；逐步重排求和，检查列向量约定、零初始状态、非零分母与固定状态维度 |
| 两类 Gate 的作用对象 | 1.4 | `yang2024gateddelta` v1 §3.1 式 (8)；`qiu2025gatedattention` v1 §2.2；状态衰减与 Softmax Attention 的输出调节分别描述 |
| SSM 关系与 Hybrid 实例 | 1.1、1.4 | `dao2024ssd` v1 §4.1、§5.1—5.2、§9.2.3；只说明特定形式的联系与模块组合，没有把任意 SSM 等同于标准 Softmax Attention |
| 方法组合与性能比较 | 1.5 | Mistral v1 §2 的 GQA + Sliding Window；GQA v3 §2.1 的转换与适应训练；FlashAttention §4 与 PagedAttention §6.1 的评估范围 |
| 教学视频、图文和书籍 | 1.1—1.2、附录 C | 3Blue1Brown 作者英文字幕与页面、Jay Alammar 原文、D2L 1.0.3 §11.1 及目录；视频时间段由字幕核对，没有声称观看或下载了完整视频 |

## 实际执行的检查

环境：Python 3.12.3、PyYAML 6.0.1；原创 SVG 使用 librsvg 2.58 生成本地预览并查看。完整命令输出保存在 [overview-checks.txt](overview-checks.txt)。

| 检查 | 实际结果及边界 |
| --- | --- |
| `python3 scripts/check_handbook.py` | 本地文件链接、导航、来源 ID 与归档哈希、SVG XML、GitBook 内容映射及公式定界符检查；不检测外链当前连通性、链接锚点或目标页面布局 |
| `python3 scripts/verify_examples.py` | 加权和约 (0.8, 0.9)；MHA/GQA/MQA 容量分别为 4294967296/1073741824/134217728 bytes；N=5 时有 15 个 Causal 位置对 |
| Linear Attention 数值对照 | 4 个位置，Query/Key 维度 2、Value 维度 3，φ=ELU+1；显式历史求和与递推状态的最大绝对误差 2.220446049250313e-16，小于 1e-12；数值对照不替代附录中的代数推导 |
| `python3 scripts/sources.py` | 无网络模式检查现有字幕并生成本地索引；1 项已归档，9 篇必备 PDF 未归档；没有执行下载成功声明 |
| `python3 -m py_compile ...` | 三个维护脚本通过语法编译；下载分支未在联网条件下完成端到端验证 |
| 两张 SVG 预览 | 查看[技术路线图](previews/attention-landscape.png)与[Prefill/Decode 图](previews/prefill-decode.png)，文字无截断、连接关系与图例清楚；不是 GitBook 页面截图 |
| `--require-archives` | 应返回失败并列出 9 篇必备 PDF 和 3 张引用原图；本轮实际输出见记录，不能将其算作通过 |

## 问题与修正

| 编号 | 级别 | 位置与问题 | 修正及复查结果 | 状态 |
| --- | --- | --- | --- | --- |
| R01 | 需修正 | 1.1 与 1.3 的合并引用标签列了两篇论文，链接只指向一篇 | 拆为各自原文链接，核对公式与计算阶段的出处 | 已关闭 |
| R02 | 需修正 | 1.3 的一步 Decode 使用“缓存长度 N”，没有明确包含当前 KV；图注也未说明预测需要其余网络计算 | 明确 N 是当前可访问 KV 的总位置数；图注补充省略的模型计算与方法来源 | 已关闭 |
| R03 | 需修正 | 1.4 的递推式未明确单个向量的方向，固定状态大小易被误读为训练全程常数内存 | 补列向量维度、训练/推理区别，附录给出历史求和到状态的推导，并运行数值对照 | 已关闭 |
| R04 | 需修正 | LLM、Seq2Seq、LSTM、RNN、MQA/GQA/MLA 等缩写的展开不完整 | 在相关正文首次解释处补全，并建立附录术语表；概览图与标题保留短称 | 已关闭 |
| R05 | 需修正 | 初稿采用单美元行内公式，与 GitBook 官方示例不一致；多项递推公式横向过长 | 按官方文档统一为双美元行内公式，长公式改为 aligned；检查分隔符配对。实际渲染仍归 R07 | 已关闭，限语法修正 |
| R06 | 需修正 | 必备本地原文和引用原图未归档；不满足写作规范的资料条件 | 当前终端在创建网络 Socket 时报 Operation not permitted；网页工具可核查原文。保留清单、公开链接和真实状态，未创建伪造 PDF 或缺失文件链接 | 待修正 |
| R07 | 需修正 | GitBook 公式、图片、表格宽度和视频入口尚无目标页面检查 | 尚未连接远程仓库与 GitBook Space；本地 SVG 检查和官方语法核对不替代实际发布预览 | 待修正 |
| R08 | 需修正 | 保存的 SRT 比作者 Git blob 多一个末尾换行 | 去掉额外换行后文件为 42646 bytes，Git blob 为 c0e2617102bdfac0227ad747a65d5c882de6bcc5，与作者文件一致；重新记录 SHA-256 | 已关闭 |

## 未完成项

1. 在允许终端联网的环境执行 `python3 scripts/sources.py --download`，核对 9 篇必备 PDF 的内容和版本，以及 3 张原图的图号、清晰度与署名；下载成功后将原图改为本地相对路径，再更新索引和版本记录。10 篇选读论文的本地归档不属于本阶段必备条件。
2. 连接 GitBook 后，逐页检查首章与三个附录的公式、SVG、原图、表格、视频链接和导航，保存实际预览结果。当前没有发布、推送或目标环境测试结果。
3. 资料归档后关闭 R06，完成目标预览后关闭 R07；任何正文或共享符号变动按影响范围重新核查。

## Review 结论

- 内容核查：**尚未全部通过**。范围、相关技术事实、数学、表达与比较条件已自查并修正；必备资料归档仍有 R06。
- 发布检查：**待检查**，R07 未关闭。
- 版本与日期：以 `overview-after.sha256` 的文件版本及 2026-10-19 的实际检查记录为准。初稿完成不等于具备发布通过结论。
