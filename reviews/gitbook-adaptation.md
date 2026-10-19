# GitBook 接入适配复查

日期：2026-10-19。方式：Codex 作者自查。范围：仓库部署说明、书籍首页导航、依赖清单和检查工具；没有修改首章技术正文、公式或图表。

## 修改与依据

| 项目 | 处理结果 |
| --- | --- |
| 发布目录 | 核对 `.gitbook.yaml`，保留 `root: ./book/` 及现有首页、导航配置；维护规范继续保留在仓库根目录 |
| 首页阅读入口 | 移除指向导航配置文件 `SUMMARY.md` 的阅读链接，增加十章直接入口；检查工具要求正文 Markdown 链接指向导航中配置的页面 |
| 干净仓库检查 | 复现新克隆副本缺少被忽略的 SRT、普通检查错误失败的问题；现在只提示缺少本地归档，已有文件的哈希不符仍报错 |
| 严格归档检查 | 同时检查必备文件的实际存在性，避免仅依赖其他机器记录的下载状态 |
| 接入操作 | 新增根目录 [DEPLOYMENT.md](../DEPLOYMENT.md)，说明仓库映射、首次导入、预览、发布及更新步骤 |
| 工具依赖 | 新增 `requirements.txt`，记录 PyYAML 6.x 依赖；实际验证使用 Python 3.12.3、PyYAML 6.0.1 |

配置与公式语法核对自 [GitBook 内容配置](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)、[Quickstart](https://gitbook.com/docs/getting-started/quickstart)、[行内公式说明](https://gitbook.com/docs/create-content/formatting/inline#math-and-tex)。不将当前托管方案解释为 HonKit 或 GitHub Pages 构建。

## 验证结果

- 按 `.gitignore` 导出可提交文件到临时目录，模拟新克隆环境；修改前普通检查返回 1，错误指向缺失的本地字幕；修改后返回 0。
- 干净副本使用 `--require-archives` 返回 1，列出未归档的必备资料，符合当前实际状态。
- 在临时副本中加入损坏的字幕文件，普通检查返回 1，校验值检查仍生效。
- 当前目录的导航、文件链接、公式定界符及 SVG 检查通过；本轮最终输出保存在 [gitbook-adaptation-checks.txt](gitbook-adaptation-checks.txt)。

本轮文件版本见 [gitbook-adaptation.sha256](gitbook-adaptation.sha256)。原有 `overview-after.sha256` 保留为首章当时的审阅快照，不覆盖历史证据；其中首页、仓库说明、来源维护说明和检查脚本的后续变化由本记录补充。

## 未完成项

GitBook 尚未连接，无法给出实际导入、搜索、公式渲染或页面预览通过结论。9 篇必备 PDF 和 3 张引用原图仍待归档，首章 Review 的 R06、R07 保持未关闭。本轮通过的是本地接入适配检查，不是网站发布验收。
