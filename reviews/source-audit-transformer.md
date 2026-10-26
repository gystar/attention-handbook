# 第二章归档后的来源对照审校

## 范围与版本

2026-10-26，作者自查。逐段检查 `book/chapters/transformer-attention/` 六页、五图、八表、相关本地论文与字幕，遵循 CONTRIBUTING 的逐章 Review 和表达专项要求。版本见 [校验值清单](source-audit-transformer.sha256)。

本轮补充既有 [第二章 Review](transformer-attention.md)，不改写其中当时的下载失败记录。论文和原图现已归档，身份检查见 [归档记录](source-archive.md)。

## 目录与标题核查

网络结构与流程 → 单 Head 计算 → 来源和 Mask → 多 Head 与输出组合 → 训练及推理阶段，概念依赖完整。2.1 定位原始架构，2.2 引入列向量/行矩阵和维度，2.3 分开来源与访问约束，2.4 补 Head 及 batch 维度，2.5 解释成本。没有将位置编码细节或 KV 表示变体重复展开；对应第三、四章入口保留。标题和导航未修改。

## 逐段核查依据

论文均为 `sources/papers/` 中的归档版本：Transformer v7、Shazeer MQA v1、PagedAttention v1、Mistral v1、FlashAttention v2。网页核对日期为本记录日期。

| 位置与段落范围 | 实际证据或推导 | 结果 |
| --- | --- | --- |
| 概览引言、导航、来源及符号段 | 与五节正文和附录 A 对照 | 符号约定一致；比较范围为基础计算，不泛指所有 Attention 变体 |
| 2.1 网络位置三段及图 2-1 | Transformer §3.1—3.4；[D2L §11.7](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) 的 Encoder Block、AddNorm；实际查看本地图 | Post-Norm、逐位置 FFN 和 FC 含义对应。图未绘输出 Softmax，正文已明确说明 |
| 2.1 向量来源、表 2-2 与末段 | Transformer §3.2 的投影及汇总定义 | 输入、分数、Mask、归一化、Value 汇总、Head 合并对应；权重和为 1 限于 dropout 前 |
| 2.2 QKV 来源及表 2-3 | Transformer §3.2.1—3.2.3；手工维度乘法 | X/Z 输入维度允许不同，Q/K 末维相同，V 可取不同维度；输出行数等于 Query 数 |
| 2.2 分数、Softmax、输出公式及图 2-2 | Transformer Eq. (1)、Figure 2；Mask 规则见 §3.2.3 | 逐 Key 归一化与矩阵形状正确；原图本地查看沿用未变文件的归档记录 |
| 2.2 位置权重与 dropout 段 | [D2L §11.3.3](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html) 的 `masked_softmax` 后 `dropout` 再乘 V | 单次 dropout 结果不必和为 1；没有混同词表分布 |
| 2.2 缩放三段 | Transformer §3.2.1、脚注 4；FlashAttention §3.1 | 独立且各分量零均值、单位方差时，E(qᵣkᵣ)=0，E(qᵣ²kᵣ²)=1；独立求和方差为 d_k。同分布不是必要条件，已修正表述。减行最大值保持比例并避免正指数溢出 |
| 2.2 教学算例三段 | 独立复算 exp(0,ln2,ln7)=(1,2,7)，并运行现有脚本 | 未 Mask 输出 (0.8,0.9)，屏蔽第三项后输出 (1/3,2/3)；分母重新计算正确 |
| 2.2 视频表与排列说明 | 本地 `sources/videos/3blue1brown-attention.en.srt` 的四个时间区间 | Q/K、Mask、Value 和 Multi-Head 主题与时间范围对应；视频采用按列归一化的画法，正文已说明本书按行。未宣称实际播放验证 |
| 2.3 引言、来源表与解释段 | Transformer §3.2.3 | Self/Cross 定义与 Causal 约束是两个维度；矩形分数矩阵成立 |
| 2.3 因果公式、图 2-3 与错位段 | Transformer §3.1；实际查看本地 Mask 图 | Query 行、Key 列及对角线允许访问正确；输入位置 i 预测 i+1 与第一章补充说明一致 |
| 2.3 Padding 两段 | D2L §11.3.2.1 实现沿末维 Mask；由 Softmax 公式独立分析 | 分开 Key 屏蔽和补齐 Query 的后续忽略；全 Mask 行的基础公式无定义，未将特定实现的有限大负数处理写成通用行为 |
| 2.3 缓存偏移公式及例子 | Shazeer §2.4；将绝对 Query 位置 p+i 代入因果关系 | 3 个缓存位置加 2 个新位置时范围分别至 4 和 5；漏偏移的脚本反例能检测错误 |
| 2.4 引言、逐 Head 公式、图 2-4 | Transformer §3.2.2；[D2L §11.5](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html) 的投影、reshape 与 transpose；实际查看原图 | 图中投影前 Q/K/V 与正文 X/Z/Z 已区分；各 Head 有不同参数，并非把原始输入直接切块 |
| 2.4 Concat 与分块输出三段 | 块矩阵乘法；Shazeer §2.2 的输出 einsum；数值对照 | W_O 按行拆分，各 Head 投影后求和等于 Concat 后投影；输出投影逐位置操作 |
| 2.4 batch 表、布局解释和参数量例子 | D2L transpose_qkv；四个 d_model×d_model 矩阵计数 | B/N/h/d_h 轴对应；512/8=64，权重总数 4d_model²，已排除偏置并限定标准 MHA |
| 2.5 引言、阶段表、并行和分块段 | Shazeer §2.3—2.4；PagedAttention §2.2；Mistral §2 Pre-fill and Chunking | 训练已知目标不等于允许读取未来；Prefill 末位预测首个新 Token；Decode 读取历史加当前 KV |
| 2.5 缓存追加、输出公式、等价条件及脚本段 | Shazeer §2.4；逐层因果依赖与向量形状检查 | q_tᵀKᵀ 为 1×t，乘 V 为 1×d_v；缓存条件包括相同前缀、参数、位置和关闭随机性。固定单 Head 检查没有写成完整模型验证 |
| 2.5 表 2-8 与成本解释 | Transformer Table 1；Shazeer §2.3.1/2.4.1；FlashAttention §3.1；矩阵乘法计数 | 完整 Attention 核心二次、单步线性、KV 元素数 2Bhtd_h，投影另计；没有将 Shazeer 的 n≤d 假设误扩展为任意长度的整层成本 |

## 数学与数值检查

已读取 `scripts/verify_transformer_attention.py` 后重新运行：它覆盖教学算例、平移不变性、全 Mask 拒绝、矩形计算、单步和分块等价、前缀不变、Concat 分块求和，并含错误偏移反例。固定 seed 20261026，Python 3.12.3，容差 1e-12；最大误差 1.11e-16，漏偏移反例误差 0.7243。

另用两行零 Query、两个零 Key、Value 为 2 和 999 的反例，屏蔽第二个 Key 后，两行输出都为 2。该检查证实只屏蔽 Key 不会把补齐 Query 的输出自动置零。结果见 [数值记录](source-audit-transformer-checks.json)。这些是数学算例，不是训练或 GPU 性能实验。

## 表达专项核查与问题修正

| 编号 | 级别 | 问题与处理 | 复查结果 |
| --- | --- | --- | --- |
| TA-S01 | 需修正 | 2.2 缩放解释写“QKV 独立同分布”，但推导只涉及 Q/K 的独立性和一、二阶矩 | 改为 Query/Key 的独立、零均值及单位方差假设；与公式及原论文一致，已关闭 |
| TA-S02 | 需修正 | 2.3 Padding 段未区分 Key 屏蔽、Query 行输出以及全 Mask 的数值处理 | 拆成两个短段，补充非零输出反例核验；基础公式与实现约定分开，已关闭 |

逐段检查概览、五节及图表说明：没有需要删除的写作自评或无信息总结。保留 dropout 后行和、实际 Q/K 不必满足统计假设、全 Mask 行无定义、核心成本不等于整层成本等限制，它们直接影响数学或比较结论。修订未删来源、偏置省略、固定配置、示意标记和浮点差异；段落拆分用于分开数值边界，不重复前文定义。

## 结论与未完成项

第二章本轮正文来源对照、数学、表达与本地图文核查完成。图 2-1、2-3、2-4 本轮实际查看；图 2-2 与共享图 2-5 沿用未变文件的归档及第一章查看记录。相关原文和字幕已在本地。

GitBook 当前工作树的公式、图注、表格与视频呈现，以及 YouTube 实际播放和跳转仍待验证。字幕时间核查不等于视频播放通过。全书其余章节和附录继续审校。
