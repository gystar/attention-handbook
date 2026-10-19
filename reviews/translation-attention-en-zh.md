# 图 1-5 英中翻译示例复查

2026-10-19，作者自查。范围：1.2 的 Attention 按需读取输入信息、图 1-5、素材索引与来源记录。版本见 [translation-attention-en-zh.sha256](translation-attention-en-zh.sha256)。

采用原创可编辑 SVG 与 PNG，保留生成脚本。例子为 I like tea → 我喜欢茶，以两个生成步骤比较对 like 和 tea 的读取权重。每行示意权重均非负且和为 1，脚本已执行断言；较高权重对应深色，同时保留数值。实际查看 PNG：中英文、数字、箭头与图例清晰，未见截断。

机制沿用已核查的 Bahdanau §3.1：Decoder 状态影响当前读取权重，权重用于汇总 Encoder 状态；预测同时使用 Context 与 Decoder 状态。正文明确图中省略状态更新与输出层，分词和权重均为教学设定，未声称实际模型推理结果。两行是不同生成步骤，不是两个并行 Decoder。图名与图号保持不变，源文件、脚本和 SHA-256 已登记；旧法英图留作历史素材。

内容复查完成；check_handbook.py 与 git diff --check 通过。GitBook 实际页面显示待同步确认，既有归档缺口保留。本轮未提交或推送。
