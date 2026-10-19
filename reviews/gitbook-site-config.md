# GitBook 站点配置修正

日期：2026-10-19。方式：作者自查。用户在首次同步时收到 `gitbook-docs.yaml does not exist on this branch`，此前仅提供 Space 配置，遗漏站点配置。

新增根目录 `gitbook-docs.yaml`：0 个 Section、1 个 Space，映射仓库根目录，由现有 `.gitbook.yaml` 指向 `book/`。首次接入标识选择为 `attention-handbook`，路径选择为 `handbook`；这些是配置选择，不代表已读取远程 Space 的标识。同步修正 README 和 DEPLOYMENT 中配置不完整及远程状态的表述；正文未改动。

依据：[官方内容配置说明](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)、[官方仓库配置](https://github.com/GitbookIO/public-docs/blob/main/gitbook-docs.yaml)。完整 schema 地址通过 Web 工具读取失败，未声称完成 schema 校验。

实际检查：PyYAML 解析通过；唯一映射目录存在，映射目录内 `.gitbook.yaml` 可读取，内容根目录正确解析至 `book/`，首页和导航文件存在。`python3 scripts/check_handbook.py` 返回 0，检查 32 个 Markdown 文件、126 个本地链接及 19 个导航入口（新增本记录前）。该脚本尚不包含站点配置检查，本次另行执行上述路径断言。

未执行 commit、push 或远程同步，GitBook 平台校验及页面预览待完成。之前的适配 Review 与哈希清单保留为历史版本，本次变更版本见 [gitbook-site-config.sha256](gitbook-site-config.sha256)。
