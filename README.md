# Attention Handbook

《大模型 Attention：原理、演进与优化》的书稿与资料索引。

- [大模型 Attention：原理、演进与优化](book/README.md)
- [Summary](book/SUMMARY.md)
- [建设目标与本阶段进度](GOAL.md)
- [写作及逐章 Review 规范](CONTRIBUTING.md)
- [本地资料说明](sources/README.md)
- [GitBook 部署说明](DEPLOYMENT.md)

## 当前内容

十章正文和三个附录已写齐，并完成归档后的逐章来源与表达审校。当前工作树已保存 60 篇必备论文和 68 个图表文件。全书最终一致性检查正在进行，当前版本的 GitBook 页面呈现与视频播放仍待验收。

逐章来源核查、修订和未完成项见[归档后来源审校](reviews/source-content-audit.md)。

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
python3 scripts/verify_kv_representations.py
python3 scripts/verify_gated_attention.py
python3 scripts/sources.py
```

安装依赖需要访问包源；上述检查与索引命令本身不访问网络。刚克隆的仓库不包含本地原文，普通结构检查会提示缺失的本地归档；严格归档检查仍要求实际文件存在。下载与严格归档检查见[来源记录说明](references/README.md)。

## GitBook 接入

`gitbook-docs.yaml` 将仓库根目录映射为一个 Space，`.gitbook.yaml` 的内容根目录设为 `book/`。两个配置文件都需推送到同步分支。在 GitBook 选择对应仓库、分支并核对映射，首次同步选择 Repository → GitBook。[GitBook 内容配置](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

远程同步及 GitBook 页面呈现仍待验证；发布检查在同步成功后执行。

首次连接的目录设置、预览项目和更新流程见 [DEPLOYMENT.md](DEPLOYMENT.md)。

