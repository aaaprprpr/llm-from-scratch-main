"""下载阶段的 JSONL 正文读取；bin 构建不再读取原始 JSONL。"""

from __future__ import annotations

import json
from pathlib import Path


def read_text(line: bytes, path: Path, line_number: int) -> str:
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON at {path}:{line_number}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        raise ValueError(f"Expected a string 'text' field at {path}:{line_number}")
    if record["text"] == "":
        raise ValueError(
            f"Empty 'text' at {path}:{line_number}; input was not modified"
        )
    return record["text"]
