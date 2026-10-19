#!/usr/bin/env python3
"""维护实际存在的资料索引；仅 --download 访问网络。依赖 PyYAML。"""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "references/catalog.yaml"


def local_path(entry):
    path = (ROOT / entry["local_path"]).resolve()
    allowed = (ROOT / "book/assets" if "source_id" in entry else ROOT / "sources").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError(f"资料路径超出允许目录: {entry['id']}")
    return path


def validate(path, expected_hash=None):
    data = path.read_bytes()
    signatures = {".pdf": b"%PDF-", ".png": b"\x89PNG\r\n\x1a\n"}
    # 临时文件沿用最终扩展名，以便检查实际格式。
    if path.suffix in signatures and not data.startswith(signatures[path.suffix]):
        raise ValueError("文件头与扩展名不符")
    if path.suffix == ".srt" and " --> " not in data.decode("utf-8"):
        raise ValueError("未发现 SRT 时间标记")
    if not data:
        raise ValueError("文件为空")
    digest = hashlib.sha256(data).hexdigest()
    if expected_hash and expected_hash != digest:
        raise ValueError("SHA-256 与已记录版本不同；请人工核查，不覆盖原记录")
    return digest, len(data)


def download(entry, path):
    url = entry["download_url"]
    if not url.startswith("https://"):
        raise ValueError("下载地址必须使用 HTTPS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        request = Request(url, headers={"User-Agent": "AttentionHandbook/1.0"})
        with urlopen(request, timeout=30) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as temp:
                temp_path = Path(temp.name)
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 128 * 1024 * 1024:
                        raise ValueError("下载超过 128 MiB 上限")
                    temp.write(chunk)
        validate(temp_path, entry.get("sha256"))
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def write_index(entries):
    lines = [
        "# 本地资料索引", "",
        "由 `python3 scripts/sources.py` 根据实际文件生成。完整校验值见 `references/catalog.yaml`。",
        "只对存在且校验通过的文件提供本地链接；待下载条目保留公开来源。", "",
        "| 来源 ID | 资料或图表 | 本地文件 | 状态 |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        title = entry.get("title", entry.get("locator", entry["id"]))
        url = entry.get("url", entry.get("download_url", ""))
        public = f"[{title}]({url})" if url else title
        local = "—"
        if entry.get("archive_status") == "downloaded":
            path = local_path(entry)
            relative = os.path.relpath(path, ROOT / "sources")
            local = f"[{path.name}]({relative})"
        status = {"downloaded": "已归档", "pending": "待下载", "link_only": "仅公开链接", "invalid": "校验失败"}.get(entry.get("archive_status"), "待核查")
        lines.append(f"| `{entry['id']}` | {public} | {local} | {status} |")
    (ROOT / "sources/INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="下载必备原文及引用图片")
    parser.add_argument("--all", action="store_true", help="同时下载选读资料，须与 --download 一起使用")
    args = parser.parse_args()
    if args.all and not args.download:
        parser.error("--all 须与 --download 一起使用")
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    entries = catalog["sources"] + catalog.get("assets", [])
    errors = []
    for entry in entries:
        if not entry.get("local_path"):
            continue
        path = None
        try:
            path = local_path(entry)
            if not path.exists():
                entry["archive_status"] = "pending"
                selected = args.all or entry.get("required_local") or "source_id" in entry
                if args.download and selected:
                    download(entry, path)
                    entry["archive_note"] = "已下载；文件格式标记与 SHA-256 已检查。"
            if path.exists():
                digest, size = validate(path, entry.get("sha256"))
                entry.update(archive_status="downloaded", sha256=digest, bytes=size)
        except (OSError, ValueError) as exc:
            entry["archive_status"] = "invalid" if path and path.exists() else "pending"
            entry["archive_note"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{entry['id']}: {entry['archive_note']}")
    CATALOG.write_text(yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False, width=110), encoding="utf-8")
    write_index(entries)
    archived = sum(e.get("archive_status") == "downloaded" for e in entries)
    missing = sum(e.get("required_local", False) and e.get("archive_status") != "downloaded" for e in entries)
    print(f"已归档 {archived} 项；必备原文待归档 {missing} 项。索引：sources/INDEX.md")
    for error in errors:
        print(f"错误：{error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
