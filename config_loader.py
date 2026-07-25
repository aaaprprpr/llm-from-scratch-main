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
