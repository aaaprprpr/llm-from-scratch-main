from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn

from fastapi import HTTPException

from .database import CurationDatabase, RevisionConflictError, UndoConflictError
from .documents import DatasetRepository
from .llm_cleaning import CleaningConfig, LlmCleaner, LlmCleaningError


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def automatic_source_id(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    name = resolved.stem if resolved.is_file() else resolved.name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "dataset"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{slug[:54]}-{digest}"


class ApiContext:
    def __init__(self, root: Path, tokenizer_path: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "curation.sqlite3"
        with CurationDatabase(self.database_path):
            pass
        self.repository = DatasetRepository()
        self.llm_cleaner = LlmCleaner(CleaningConfig.from_file(), self.root / "llm_suggestions")
        self.tokenizer_path = tokenizer_path.resolve()
        self._tokenizer: Any | None = None
        self._tokenizer_lock = threading.Lock()
        self._simplified_converter: Any | None = None
        self._simplified_converter_lock = threading.Lock()

    @contextmanager
    def open_database(self) -> Iterator[CurationDatabase]:
        with CurationDatabase(self.database_path) as database:
            yield database

    @staticmethod
    def raise_http(exc: Exception) -> NoReturn:
        if isinstance(exc, LlmCleaningError):
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(
            exc,
            (RevisionConflictError, UndoConflictError, sqlite3.IntegrityError),
        ):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, (ValueError, FileNotFoundError, IndexError)):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise exc

    def add_token_counts(self, value: dict[str, Any]) -> dict[str, Any]:
        if not self.tokenizer_path.exists():
            value["token_counts"] = None
            return value
        with self._tokenizer_lock:
            if self._tokenizer is None:
                from tokenizer import Tokenizer

                self._tokenizer = Tokenizer(str(self.tokenizer_path))
            tokenizer = self._tokenizer
        value["token_counts"] = {
            "raw": len(tokenizer.encode(value["raw_text"])),
            "review": len(tokenizer.encode(value["review_text"])),
            "materialized": len(tokenizer.encode(value["materialized_text"])),
        }
        value["tokenizer_path"] = str(self.tokenizer_path)
        return value

    def simplify_texts(self, texts: list[str]) -> dict[str, Any]:
        with self._simplified_converter_lock:
            if self._simplified_converter is None:
                from opencc import OpenCC

                self._simplified_converter = OpenCC("t2s")
            converted = [self._simplified_converter.convert(text) for text in texts]
        changed = sum(
            sum(left != right for left, right in zip(source, target))
            + abs(len(source) - len(target))
            for source, target in zip(texts, converted)
        )
        return {"texts": converted, "changed_characters": changed}

    def add_simplified_view(self, value: dict[str, Any]) -> dict[str, Any]:
        """Normalize the initial editor and diff baseline without changing source identities."""
        texts = list(dict.fromkeys([
            value["raw_text"], value["materialized_text"],
            *(block["text"] for block in value["blocks"]),
        ]))
        converted = dict(zip(texts, self.simplify_texts(texts)["texts"]))
        value["simplified"] = {
            "raw_text": converted[value["raw_text"]],
            "materialized_text": converted[value["materialized_text"]],
            "block_texts": [converted[block["text"]] for block in value["blocks"]],
        }
        return value
