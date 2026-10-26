# Attention Handbook

《大模型 Attention：原理、演进与优化》的书稿与资料索引。

- [大模型 Attention：原理、演进与优化](book/README.md)
- [Summary](book/SUMMARY.md)
- [建设目标与本阶段进度](GOAL.md)
- [写作及逐章 Review 规范](CONTRIBUTING.md)
- [本地资料说明](sources/README.md)
- [GitBook 部署说明](DEPLOYMENT.md)

## 当前内容

第 1 章已有概览和四节正文；第 2、3 章各有概览及五节正文，分别解释 Attention 计算，以及位置编码、上下文扩展与评估。第 4、9 章已接收 KV 表示与容量、Linear Attention 递推等内容，完整章节待写；其余专题章保留节级目录。附录 B 收录时间线与方法比较条件。本地原文归档和本轮 GitBook 发布检查尚待完成。

结构与表达复查见[第一章结构重写 Review](reviews/overview-restructure.md)、[第二章 Review](reviews/transformer-attention.md)及[第三章 Review](reviews/position-encoding.md)。

## 目录职责

| 目录 | 内容 |
| --- | --- |
| `book/` | GitBook 发布正文、图表和公开参考资料 |
| `references/` | 来源元数据和下载清单 |
| `sources/` | 本地原始资料及索引；下载文件不提交到 Git |
| `scripts/` | 文档检查、资料下载和索引维护工具 |
| `reviews/` | 逐章问题、核查依据和修正记录 |

## 本地维护

在仓库根目录执行，需 Python 3.10+ 和 PyYAML：

```sh
python3 -m pip install -r requirements.txt
python3 scripts/check_handbook.py
python3 scripts/verify_examples.py
python3 scripts/verify_transformer_attention.py
python3 scripts/verify_position_encoding.py
python3 scripts/sources.py
```

安装依赖需要访问包源；上述检查与索引命令本身不访问网络。刚克隆的仓库不包含本地原文，普通结构检查会提示缺失的本地归档；严格归档检查仍要求实际文件存在。下载与严格归档检查见[来源记录说明](references/README.md)。

## GitBook 接入

`gitbook-docs.yaml` 将仓库根目录映射为一个 Space，`.gitbook.yaml` 的内容根目录设为 `book/`。两个配置文件都需推送到同步分支。在 GitBook 选择对应仓库、分支并核对映射，首次同步选择 Repository → GitBook。[GitBook 内容配置](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

远程同步及 GitBook 页面呈现仍待验证；发布检查在同步成功后执行。

首次连接的目录设置、预览项目和更新流程见 [DEPLOYMENT.md](DEPLOYMENT.md)。

