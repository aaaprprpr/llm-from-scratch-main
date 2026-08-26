from __future__ import annotations

import hashlib
import html
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def tokenizer_file(path: Path) -> Path:
    target = path / "tokenizer.json" if path.is_dir() else path
    if not target.is_file():
        raise FileNotFoundError(f"Tokenizer file does not exist: {target}")
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_report_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    report_dir = output_root / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.55; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f5f5;
           border: 1px solid #ddd; border-radius: 6px; padding: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ccc; padding: .45rem; text-align: left; }}
    .ok {{ color: #18794e; }} .warn {{ color: #9a6700; }}
    .fail {{ color: #cf222e; }} code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{body}
</body>
</html>
"""


@dataclass
class AuditSection:
    name: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def add(
        self,
        name: str,
        status: str,
        detail: str,
        **metrics: Any,
    ) -> None:
        if status not in {"pass", "warn", "fail", "skip"}:
            raise ValueError(f"Unsupported audit status: {status}")
        item: dict[str, Any] = {
            "name": name,
            "status": status,
            "detail": detail,
        }
        if metrics:
            item["metrics"] = metrics
        self.checks.append(item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checks": self.checks,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
