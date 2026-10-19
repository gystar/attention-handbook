# GitBook 部署说明

本仓库面向 GitBook 官方托管：GitHub 或 GitLab 保存书稿，GitBook 同步并发布在线阅读站点。远程同步与页面呈现仍待验证；没有配置 HonKit 或 GitHub Pages 构建。

## 发布目录

直接推送当前项目目录，保留 `AGENTS.md`、`GOAL.md`、写作规范、来源清单和 Review 记录。仓库根目录的 `.gitbook.yaml` 已配置：

```yaml
root: ./book/
structure:
  readme: README.md
  summary: SUMMARY.md
```

这些设置对应 `book/README.md` 和 `book/SUMMARY.md`。前者是阅读首页，后者生成侧栏导航。维护规范位于正文根目录之外，不会作为本书内容导入。[GitBook 内容配置](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

本地原文、`build/`、Python 缓存和环境专用隐藏目录已由 `.gitignore` 排除；`sources/README.md` 保留在 Git 中作为说明。

## 首次接入

1. 将项目推送到 GitHub 或 GitLab，确认分支中的书稿与本地版本一致。
2. 在 GitBook 创建 Site，选择 **Sync with Git**，授权并选择仓库、分支。
3. 将 Git Sync 的 Project directory 和该 Space 的映射目录设为仓库根目录（默认或 `./`）；正文子目录已由 `.gitbook.yaml` 指定为 `book/`。不要将 Space 映射目录再设成 `book/`，否则根目录配置将不在该 Space 的映射范围内。
4. 首次同步选择 **Repository → GitBook**，从现有仓库导入。核对首页及三篇、十章、三个附录的侧栏结构。
5. 使用 **Preview** 检查页面，确认后再点击 **Publish**。发布入口与站点地址以 GitBook 界面为准。[官方 Quickstart](https://gitbook.com/docs/getting-started/quickstart)

仓库已提供站点配置 `gitbook-docs.yaml`：一个 Space，`key: attention-handbook`、`path: handbook`、`content.directory: ./`，语言为 `zh`。章节分组由 `book/SUMMARY.md` 管理，不另建站点级 Section。`key` 是首次接入选择的标识，成功接入后应保持稳定；若平台已有对应 Space 的映射，沿用其原有 key。[站点映射规则](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

### 提示 gitbook-docs.yaml does not exist on this branch

确认 GitHub 仓库 `gystar/attention-handbook` 的 `main` 分支根目录已包含 `gitbook-docs.yaml` 和 `.gitbook.yaml`。本地新增文件必须提交并推送到该分支后，GitBook 才能读取。GitBook 的 Project directory 保持仓库根目录，再重试同步。

配置关系为：`gitbook-docs.yaml` → Space 映射目录 `./` → `.gitbook.yaml` → 正文 `book/`。`AGENTS.md`、Review 和来源维护文件不另建 Space。

## 发布前检查

仓库检查工具需要 Python 3.10+。以下命令用于本地检查，不是 GitBook 的网站构建命令：

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/check_handbook.py
python3 scripts/verify_examples.py
```

刚克隆的仓库不包含本地原始资料。普通结构检查会提示清单中已记录、但本机未保存的资料；已存在文件的哈希不符仍然报错。要验收必备原文与引用图片的归档，使用 `python3 scripts/check_handbook.py --require-archives`；该模式要求实际文件存在，不能只凭清单中的历史状态通过。

| 目标页面检查 | 验收内容 |
| --- | --- |
| 导航与搜索 | 首页和章节链接可达；搜索正文术语能定位相关页面 |
| 公式 | 行内公式、独立公式和 aligned 多行公式正常显示 |
| 图片 | 两张 SVG 和三张远程引用图完整、清晰，图注与署名可见 |
| 表格 | 桌面及移动端没有不可读的截断，宽表可正常浏览 |
| 教学视频 | 原始视频链接及时间定位可用 |
| 发布边界 | `AGENTS.md`、`GOAL.md`、Review 等维护文件未出现在书籍导航中 |

公式采用 GitBook 文档中的双美元语法；图片和视频保留来源与原始链接。[行内公式语法](https://gitbook.com/docs/create-content/formatting/inline#math-and-tex)

当前必备 PDF 和三张引用原图的本地归档尚未完成，严格归档检查会失败。远程原图的可用性和 GitBook 显示仍需在 Preview 中确认；本地结构检查通过不表示发布验收通过。

## 后续更新

本地修改 → 执行相应检查 → 提交并推送到同步分支 → 查看 GitBook 同步及预览结果。已发布站点会随同步内容更新。[GitBook 更新流程](https://gitbook.com/docs/getting-started/quickstart)

章节 Review 的 SHA-256 清单记录特定审阅版本；后续改动按影响范围另记复查，不能把旧快照当作当前文件的通过证明。
