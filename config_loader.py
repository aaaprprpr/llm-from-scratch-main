import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
_MISSING = object()


class Config:
    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

        self.config_path = config_path
        self.data = json.loads(config_path.read_text(encoding="utf-8"))

    def get(self, *keys: str, default: Any = None) -> Any:
        value = self.data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def require(self, *keys: str) -> Any:
        value = self.get(*keys, default=_MISSING)
        if value is _MISSING:
            name = ".".join(keys)
            raise KeyError(f"config 缺少字段：{name}")
        return value

    def resolve_path(self, *keys: str) -> Path:
        path = Path(self.require(*keys))
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def optional_path(self, *keys: str) -> Path | None:
        value = self.get(*keys, default=None)
        if value in (None, ""):
            return None

        path = Path(value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


def resolve_recorded_path(value: str | Path) -> Path:
    """Resolve paths saved by older training runs after directory moves."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data_pipeline":
        return PROJECT_ROOT / "dataset" / path

    parts = path.parts
    if parts[:2] == ("tokenize", "bpe"):
        parts = parts[2:]
    elif parts and parts[0] in {"bpe", "tokenize", "tokenizer"}:
        parts = parts[1:]

    old_names = {"tokenizer": "bpe_8192", "tokenizer_24576": "bpe_24576"}
    if parts and parts[0] in old_names:
        return PROJECT_ROOT / "tokenizer" / old_names[parts[0]] / Path(*parts[1:])
    return PROJECT_ROOT / path
