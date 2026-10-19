#!/usr/bin/env python3
"""检查本地导航、Markdown 文件链接、来源记录和 SVG；不代替内容 Review 或 GitBook 预览。"""
import argparse
import hashlib
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-archives", action="store_true", help="必备原文或引用图片未归档时返回失败")
    args = parser.parse_args()
    errors = []
    documents = sorted(BOOK.rglob("*.md")) + sorted(ROOT.glob("*.md"))
    for directory in ("references", "reviews"):
        documents += sorted((ROOT / directory).glob("*.md"))
    documents += sorted((ROOT / "sources").glob("*.md"))
    count = 0
    for doc in documents:
        body = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
        if doc.is_relative_to(BOOK) and re.search(r"(?<![\\$])\$(?!\$)", body):
            errors.append(f"正文存在单美元公式定界符：{doc.relative_to(ROOT)}")
        if doc.is_relative_to(BOOK) and body.count("$$") % 2:
            errors.append(f"公式定界符未配对：{doc.relative_to(ROOT)}")
        for target in LINK.findall(body):
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            path = (doc.parent / unquote(url.path)).resolve()
            count += 1
            if not path.is_file():
                errors.append(f"文件链接不存在：{doc.relative_to(ROOT)} → {target}")
            if doc.is_relative_to(BOOK) and not path.is_relative_to(BOOK):
                errors.append(f"发布正文链接超出 book/：{doc.relative_to(ROOT)} → {target}")
    summary_targets = [(BOOK / target).resolve() for target in LINK.findall((BOOK / "SUMMARY.md").read_text(encoding="utf-8"))]
    if len(summary_targets) != len(set(summary_targets)):
        errors.append("SUMMARY.md 存在重复入口")
    required_pages = list((BOOK / "chapters").rglob("*.md")) + list((BOOK / "appendices").glob("*.md")) + [BOOK / "README.md"]
    for page in required_pages:
        if page.resolve() not in summary_targets:
            errors.append(f"导航未收录：{page.relative_to(ROOT)}")
    config = yaml.safe_load((ROOT / ".gitbook.yaml").read_text(encoding="utf-8"))
    if (ROOT / config["root"]).resolve() != BOOK or config["structure"] != {"readme": "README.md", "summary": "SUMMARY.md"}:
        errors.append("GitBook 内容映射与本书目录不一致")
    catalog = yaml.safe_load((ROOT / "references/catalog.yaml").read_text(encoding="utf-8"))
    entries = catalog["sources"] + catalog.get("assets", [])
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        errors.append("来源 ID 不唯一")
    for entry in entries:
        if "source_id" in entry and entry["source_id"] not in ids:
            errors.append(f"图表来源不存在：{entry['id']}")
        if entry.get("archive_status") == "downloaded":
            path = ROOT / entry["local_path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
                errors.append(f"资料缺失或校验值不符：{entry['id']}")
    for path in (BOOK / "assets").rglob("*.svg"):
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            errors.append(f"SVG 解析失败：{path.name}: {exc}")
    missing = [e["id"] for e in entries if (e.get("required_local") or "source_id" in e) and e.get("archive_status") != "downloaded"]
    if args.require_archives and missing:
        errors.append(f"必备资料及引用图片未归档：{', '.join(missing)}")
    print(f"检查 {len(documents)} 个 Markdown 文件、{count} 个本地文件链接、{len(summary_targets)} 个导航入口。")
    print(f"来源记录 {len(catalog['sources'])} 项；引用图片记录 {len(catalog.get('assets', []))} 项；必备资料及引用图片待归档 {len(missing)} 项。")
    for error in errors:
        print(f"错误：{error}")
    print("本地结构检查：" + ("未通过" if errors else "通过") + "。外链可用性、链接锚点和 GitBook 页面呈现须另行核查。")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
