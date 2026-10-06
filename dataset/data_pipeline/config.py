"""下载阶段的配置与来源选择；不参与 bin 输入解析。"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from configs.config_loader import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_pipeline.json"
SOURCE_KINDS = {"jsonl", "hf_dataset"}
MINIMIND_FILES = {"pretrain_t2t.jsonl", "pretrain_t2t_mini.jsonl"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _check_keys(value: dict, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{where}: missing={sorted(missing)}, unused={sorted(extra)}")


def validate_config(data: dict[str, Any]) -> None:
    _check_keys(data, {"sources", "download"}, set(), "download pipeline")
    if not isinstance(data["sources"], dict) or not data["sources"]:
        raise ValueError("sources must be a nonempty object")
    for name, source in data["sources"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Source names must be nonempty strings")
        if not isinstance(source, dict) or source.get("kind") not in SOURCE_KINDS:
            raise ValueError(f"sources.{name}.kind must be jsonl or hf_dataset")
        if not isinstance(source.get("path"), str) or not source["path"]:
            raise ValueError(f"sources.{name}.path must be a nonempty string")
        if source["kind"] == "jsonl":
            _check_keys(source, {"kind", "path"}, {"repo", "revision", "filename", "sha256"}, f"sources.{name}")
            if name == "minimind":
                if source.get("repo") != "jingyaogong/minimind_dataset":
                    raise ValueError("MiniMind must use jingyaogong/minimind_dataset")
                if source.get("filename") not in MINIMIND_FILES:
                    raise ValueError("MiniMind accepts only pretrain_t2t JSONL files, not SFT/RL files")
                if not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source["sha256"]):
                    raise ValueError("MiniMind requires the source SHA256")
            if source.get("filename") and Path(source["path"]).name != source["filename"]:
                raise ValueError(f"sources.{name}.path must end with the selected filename")
        else:
            _check_keys(source, {"kind", "repo", "split", "path", "adapter"},
                        {"config", "data_files", "revision"}, f"sources.{name}")
    _check_keys(data["download"], {"sources", "cleanup_cache"}, set(), "download")
    names = data["download"]["sources"]
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError("download.sources must be a list without duplicates")
    for name in names:
        if name not in data["sources"]:
            raise KeyError(f"Unknown source {name!r} in download.sources")


def load_config() -> Config:
    config = Config(CONFIG_PATH)
    validate_config(config.data)
    return config


def selected_sources(config: Config, stage: str) -> list[dict[str, Any]]:
    catalog = config.require("sources")
    return [{"source_id": name, **catalog[name]} for name in config.require(stage, "sources")]
