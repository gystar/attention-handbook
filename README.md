# Attention Handbook

《大模型 Attention：原理、演进与优化》的书稿与资料索引。

- [开始阅读](book/README.md)
- [全书目录](book/SUMMARY.md)
- [建设目标与本阶段进度](GOAL.md)
- [写作及逐章 Review 规范](CONTRIBUTING.md)
- [本地资料说明](sources/README.md)

## 当前内容

第 1 章概览和五节正文已写出，配有两张原创示意图、三张原论文图的引用、对照表、公式算例和教学视频入口。第 2—10 章已建立节级目录，正文待写；三个附录已覆盖首章所需内容。

首章已执行作者自查与修正。必备 PDF、引用原图的本地归档及 GitBook 发布检查尚未完成，不能标为全项通过。依据与遗留事项见[第 1 章 Review](reviews/overview.md)。

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
python3 scripts/check_handbook.py
python3 scripts/verify_examples.py
python3 scripts/sources.py
```

这些命令分别检查文件与导航、首章教学算例、已有资料索引，不访问网络。下载与严格归档检查见[来源记录说明](references/README.md)。

## GitBook 接入

初始配置将仓库根目录作为一个 Space 的映射目录，`.gitbook.yaml` 的内容根目录设为 `book/`。将仓库推送到 GitHub 或 GitLab 后，在 GitBook 选择对应仓库、分支并核对映射，首次同步选择 Repository → GitBook。站点映射文件按实际连接结果维护。[GitBook 内容配置](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

目前尚未接入远程仓库或 GitBook 站点；发布检查在接入后执行。

