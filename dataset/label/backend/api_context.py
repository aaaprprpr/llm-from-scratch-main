from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NoReturn

from fastapi import HTTPException

from .batch_jobs import BatchJobManager
from .database import CurationDatabase, RevisionConflictError, UndoConflictError
from .documents import DatasetRepository
from .llm_cleaning import LlmCleaner, LlmCleaningError
from .local_model import LocalModelService
from .model_settings import ModelSelection, ModelSettings


def automatic_source_id(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    name = resolved.stem if resolved.is_file() else resolved.name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "dataset"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{slug[:54]}-{digest}"


class ApiContext:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "curation.sqlite3"
        with CurationDatabase(self.database_path):
            pass
        self.repository = DatasetRepository()
        self.model_settings = ModelSettings(self.root)
        selection = self.model_settings.read()
        self.llm_cleaner = LlmCleaner(self.model_settings.config(selection.single), self.root / "llm_suggestions")
        self.local_model = LocalModelService(self.root)
        self.batch_jobs = BatchJobManager(self.root, self.database_path,
                                          self.model_settings.config(selection.batch), self.local_model)
        self._model_lock = threading.Lock()
        self._prepare_progress: dict[str, dict[str, Any]] = {}
        self._prepare_lock = threading.Lock()

    def set_model_selection(self, selection: ModelSelection) -> None:
        with self._model_lock:
            single = LlmCleaner(self.model_settings.config(selection.single), self.root / "llm_suggestions")
            batch = self.model_settings.config(selection.batch)
            self.batch_jobs.update_config(batch, lambda: self.model_settings.write(selection))
            self.llm_cleaner = single

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

    def set_prepare_progress(self, progress_id: str, processed: int, total: int, stage: str = "running") -> None:
        with self._prepare_lock:
            self._prepare_progress[progress_id] = {
                "processed": processed, "total": total, "stage": stage,
            }

    def get_prepare_progress(self, progress_id: str) -> dict[str, Any]:
        with self._prepare_lock:
            return self._prepare_progress.get(progress_id, {"processed": 0, "total": 0, "stage": "opening"})
