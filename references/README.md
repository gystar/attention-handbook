# 来源记录与归档

[catalog.yaml](catalog.yaml) 是正文引用、公开来源与本地原文的关联清单。`checked_sections` 记录本章引用时实际核对的位置，不表示整篇论文已逐句审核；`checked_on` 与下载状态分别维护。

| 字段 | 用途 |
| --- | --- |
| `id` | 稳定的来源标识 |
| `first_public_year` / `version` | 首次公开年份 / 实际使用版本，二者不混用 |
| `url` / `full_text_url` / `download_url` | 元数据页、核查正文与原文件下载地址 |
| `local_path` / `required_local` | 目标路径 / 是否为当前阶段必备原文 |
| `archive_status` / `sha256` / `bytes` | 实际归档状态、文件校验值和字节数 |
| `checked_sections` / `checked_on` | 证据核对范围与日期 |
| `assets` | 引用图片的来源 ID、图号、转载条件及文件状态 |

当前已保存 3Blue1Brown 作者提供的英文字幕；字幕不等于视频文件。论文 PDF 和三张引用原图仍待归档，原始网页已用于内容核查。终端创建网络 Socket 被当前运行环境禁止，不能由此判断用户代理本身异常。

## 维护命令

环境：Python 3.10+，并安装 PyYAML。命令在仓库根目录执行。

依赖列在根目录 `requirements.txt`。清单中的 `archive_status` 记录归档时的状态，不保证新克隆环境已有该文件。普通结构检查允许缺少未随 Git 提交的本地资料，但仍检查已有文件的哈希；严格归档检查同时要求必备文件实际存在。

```sh
# 只检查已有资料并更新本地索引，不访问网络
python3 scripts/sources.py

# 在允许联网的环境中下载必备原文与三张引用原图
python3 scripts/sources.py --download

# 同时下载清单中的选读原文
python3 scripts/sources.py --download --all

# 检查导航、文件链接、来源记录与 SVG
python3 scripts/check_handbook.py

# 将必备原文与引用图片的归档状态纳入检查
python3 scripts/check_handbook.py --require-archives
```

下载工具保留已有文件，检查 PDF/PNG 文件头和 SHA-256；文件头检查不能代替人工确认内容与版本。已记录的文件校验值不符时停止接受该文件，不自动覆盖旧记录。

下载后从 `sources/INDEX.md` 打开本地原文。三张引用原图取得后，将正文中的远程图片地址改为对应相对路径，并重新检查页面；下载工具不会自行修改正文。公开读者使用[附录 C](../book/appendices/references.md)中的原始来源链接。
