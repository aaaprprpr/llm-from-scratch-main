"""Persist model choices for interactive and batch cleaning."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

from .llm_cleaning import CleaningConfig
from .deepseek_web import WEB_AUTH_SOURCES

ModelSource = Literal["deepseek", "qwen_api", "local", "deepseek_web", "qwen_web", "kimi_web", "doubao_web", "chatglm_web", "spark_web", "wenxin_web", "yuanbao_web", "laya"]
SOURCES = ("deepseek", "qwen_api", "local", "laya", "deepseek_web", *WEB_AUTH_SOURCES)


@dataclass(frozen=True)
class ModelSelection:
    single: ModelSource = "deepseek"
    batch: tuple[ModelSource, ...] = ("local",)
    failure_fallback: bool = True

    def __post_init__(self):
        # Existing settings stored one source as a string.
        batch = (self.batch,) if isinstance(self.batch, str) else tuple(self.batch)
        if self.single not in SOURCES or not batch or any(source not in SOURCES for source in batch):
            raise ValueError("未知的 LLM 来源或未选择批量模型")
        if len(batch) != len(set(batch)):
            raise ValueError("批量模型不能重复")
        object.__setattr__(self, "batch", batch)

    def as_dict(self) -> dict[str, str | list[str] | bool]:
        return {"single": self.single, "batch": list(self.batch),
                "failure_fallback": self.failure_fallback}


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

    def _web_auth_file(self, env: dict, variable: str, default_name: str) -> str:
        requested = Path((env.get(variable) or default_name).strip()).expanduser()
        candidates = ([requested] if requested.is_absolute() else
                      [self.project_root / requested,
                       self.project_root / "deepseek-web-api" / requested.name])
        return str(next((path for path in candidates if path.is_file()), ""))

    def config(self, source: ModelSource) -> CleaningConfig:
        if source not in SOURCES:
            raise ValueError("未知的 LLM 来源")
        base = CleaningConfig.from_file()
        if source == "deepseek":
            return base
        if source == "laya":
            return replace(
                base, provider="laya", base_url="local://laya", model="laya-wiki-cleaning-v1",
                api_key="", risk_fallback=None, context_tokens=8192, max_output_tokens=128,
                max_chunk_characters=8000, max_units_per_chunk=64, max_parallel_chunks=1,
            )
        if source == "local":
            return replace(
                base, provider="llamacpp", base_url="http://127.0.0.1:8080", model="qwen-local",
                api_key="", risk_fallback=None, context_tokens=32768, max_output_tokens=4096,
                timeout_seconds=600, max_document_characters=100000,
                max_chunk_characters=2000, max_units_per_chunk=128,
                max_attempts_per_chunk=2, max_parallel_chunks=1,
            )
        env = {**dotenv_values(self.project_root / ".env"), **os.environ}
        if source == "deepseek_web":
            return replace(
                base, provider="deepseek_web", base_url="https://chat.deepseek.com",
                model="deepseek-web-default", api_key=(env.get("DEEPSEEK_WEB_TOKEN") or "").strip(),
                risk_fallback=None, context_tokens=131072, max_output_tokens=8192,
                timeout_seconds=300, max_document_characters=100000,
                max_chunk_characters=16000,
                max_units_per_chunk=512, max_attempts_per_chunk=2, max_parallel_chunks=1,
            )
        if source in WEB_AUTH_SOURCES:
            variable, filename, base_url, model = WEB_AUTH_SOURCES[source]
            return replace(
                base, provider=source, base_url=base_url, model=model,
                api_key=self._web_auth_file(env, variable, filename),
                risk_fallback=None, context_tokens=32768, max_output_tokens=4096,
                timeout_seconds=300, max_document_characters=100000,
                max_chunk_characters=8000,
                max_units_per_chunk=256, max_attempts_per_chunk=2, max_parallel_chunks=1,
            )
        return replace(
            base, provider="dashscope",
            base_url=(env.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(),
            model=(env.get("DEFAULT_MODEL") or "qwen-plus").strip(),
            api_key=(env.get("DASHSCOPE_API_KEY") or "").strip(), risk_fallback=None,
            max_document_characters=100000, max_chunk_characters=100000,
        )

    def available(self) -> dict[str, dict[str, str | bool]]:
        available = {}
        for source in SOURCES:
            config = self.config(source)
            available[source] = {"model": config.model,
                                 "configured": (source == "local" or (source == "laya" and
                                                (self.project_root / "dataset/label/models/laya_wiki_cleaning_v1/model.safetensors").is_file())
                                                or bool(config.api_key))}
        return available
