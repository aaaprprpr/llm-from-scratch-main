"""Persist model choices for interactive and batch cleaning."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

from .llm_cleaning import CleaningConfig

ModelSource = Literal["deepseek", "qwen_api", "local"]
SOURCES = ("deepseek", "qwen_api", "local")


@dataclass(frozen=True)
class ModelSelection:
    single: ModelSource = "deepseek"
    batch: ModelSource = "local"

    def __post_init__(self):
        if self.single not in SOURCES or self.batch not in SOURCES:
            raise ValueError("未知的 LLM 来源")

    def as_dict(self) -> dict[str, str]:
        return {"single": self.single, "batch": self.batch}


class ModelSettings:
    def __init__(self, root: Path):
        self.path = root / "model_settings.json"
        self.project_root = Path(__file__).resolve().parents[3]

    def read(self) -> ModelSelection:
        if not self.path.exists():
            return ModelSelection()
        return ModelSelection(**json.loads(self.path.read_text(encoding="utf-8")))

    def write(self, selection: ModelSelection) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(selection.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def config(self, source: ModelSource) -> CleaningConfig:
        if source not in SOURCES:
            raise ValueError("未知的 LLM 来源")
        base = CleaningConfig.from_file()
        if source == "deepseek":
            return base
        if source == "local":
            return replace(
                base, provider="llamacpp", base_url="http://127.0.0.1:8080", model="qwen-local",
                api_key="", risk_fallback=None, context_tokens=32768, max_output_tokens=4096,
                timeout_seconds=600, max_chunk_characters=2000, max_units_per_chunk=128,
                max_attempts_per_chunk=2, max_parallel_chunks=1,
            )
        env = {**dotenv_values(self.project_root / ".env"), **os.environ}
        return replace(
            base, provider="dashscope",
            base_url=(env.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(),
            model=(env.get("DEFAULT_MODEL") or "qwen-plus").strip(),
            api_key=(env.get("DASHSCOPE_API_KEY") or "").strip(), risk_fallback=None,
        )

    def available(self) -> dict[str, dict[str, str | bool]]:
        available = {}
        for source in SOURCES:
            config = self.config(source)
            available[source] = {"model": config.model,
                                 "configured": source == "local" or bool(config.api_key)}
        return available
