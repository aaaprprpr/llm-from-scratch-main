from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

from pydantic import BaseModel, Field

from .database import (
    BlockReviewInput,
    CurationDatabase,
    DocumentReviewInput,
    RevisionConflictError,
    UndoConflictError,
)
from .documents import DatasetRepository, DocumentService
from .import_service import ImportService
from .importers import ADAPTERS, SourceSpec, get_adapter
from .materialize import MaterializeService
from .prepare import PrepareConfig, PrepareService
from .queueing import create_source_quota_queue, create_uniform_random_queue
from .schema import FieldMapping


class ProjectCreateRequest(BaseModel):
    name: str
    guideline_version: str = "1"
    project_id: str | None = None


class SourceRequest(BaseModel):
    adapter: str
    path: str
    options: dict[str, Any] = Field(default_factory=dict)


class MappingRequest(BaseModel):
    text_fields: list[str]
    text_separator: str = "\n\n"
    title_field: str | None = None
    url_field: str | None = None
    local_id_field: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)

    def to_core(self) -> FieldMapping:
        return FieldMapping(
            text_fields=tuple(self.text_fields),
            text_separator=self.text_separator,
            title_field=self.title_field,
            url_field=self.url_field,
            local_id_field=self.local_id_field,
            metadata_fields=tuple(self.metadata_fields),
        )


class InspectRequest(SourceRequest):
    pass


class PreviewRequest(SourceRequest):
    mapping: MappingRequest
    limit: int = Field(default=20, ge=1, le=200)


class ImportRequest(SourceRequest):
    source_id: str
    license: str = "unknown"
    mapping: MappingRequest
    max_shard_size: str = "1GB"


class PrepareRequest(BaseModel):
    source_revision_directory: str
    config: dict[str, Any] = Field(default_factory=dict)
    max_shard_size: str = "1GB"
    read_batch_size: int = Field(default=1024, ge=1)


class AttachSourceRequest(BaseModel):
    prepare_revision_directory: str


class QueueCreateRequest(BaseModel):
    name: str
    policy: Literal["uniform_random", "source_quota"]
    seed: int
    sample_size: int | None = Field(default=None, ge=1)
    quotas: dict[str, int] | None = None
    queue_id: str | None = None


class DocumentReviewRequest(BaseModel):
    queue_id: str
    ordinal: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
    decision: Literal["keep", "drop", "unsure"]
    quality: int | None = Field(default=None, ge=0, le=3)
    primary_category: str | None = None
    flags: list[str] = Field(default_factory=list)
    notes: str = ""
    actor: str = "local-web"


class BlockReviewRequest(BaseModel):
    queue_id: str
    ordinal: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
    decision: Literal["keep", "drop"]
    reason: str | None = None
    actor: str = "local-web"


class UndoRequest(BaseModel):
    project_id: str
    actor: str = "local-web"


class MaterializeRequest(BaseModel):
    project_id: str
    policy: Literal["keep_only", "drop_rejected"]
    snapshot_event_seq: int | None = Field(default=None, ge=0)
    max_shard_size: str = "1GB"
    read_batch_size: int = Field(default=1024, ge=1)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def create_app(data_root: str | Path | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install project requirements first."
        ) from exc

    if data_root is None:
        data_root = os.environ.get(
            "LABEL_DATA_ROOT",
            str(Path(__file__).resolve().parents[1] / "data"),
        )
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    database_path = root / "curation.sqlite3"
    # Apply migrations before the server starts accepting concurrent requests.
    with CurationDatabase(database_path):
        pass
    repository = DatasetRepository()
    configured_tokenizer_path = Path(
        os.environ.get(
            "LABEL_TOKENIZER_PATH",
            str(Path(__file__).resolve().parents[2] / "bpe" / "tokenizer_24576"),
        )
    ).resolve()
    tokenizer_holder: dict[str, Any] = {}
    tokenizer_lock = threading.Lock()
    app = FastAPI(title="LLM Dataset Curation", version="0.1.0")
    app.state.dataset_repository = repository
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @contextmanager
    def open_database() -> Iterator[CurationDatabase]:
        with CurationDatabase(database_path) as database:
            yield database

    def raise_http(exc: Exception) -> None:
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

    def add_token_counts(value: dict[str, Any]) -> dict[str, Any]:
        if not configured_tokenizer_path.exists():
            value["token_counts"] = None
            return value
        with tokenizer_lock:
            tokenizer = tokenizer_holder.get("tokenizer")
            if tokenizer is None:
                from tokenizer import Tokenizer

                tokenizer = Tokenizer(str(configured_tokenizer_path))
                tokenizer_holder["tokenizer"] = tokenizer
        value["token_counts"] = {
            "raw": len(tokenizer.encode(value["raw_text"])),
            "review": len(tokenizer.encode(value["review_text"])),
            "materialized": len(tokenizer.encode(value["materialized_text"])),
        }
        value["tokenizer_path"] = str(configured_tokenizer_path)
        return value

    @app.get("/api/health")
    def health():
        return {"status": "ok", "data_root": str(root)}

    @app.get("/api/projects")
    def list_projects():
        with open_database() as database:
            return database.list_projects()

    @app.get("/api/catalog")
    def source_catalog():
        catalog = []
        sources_root = root / "sources"
        if not sources_root.is_dir():
            return catalog
        for source_manifest_path in sorted(sources_root.glob("*/*/manifest.json")):
            revision_directory = source_manifest_path.parent
            try:
                source_manifest = json.loads(
                    source_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            prepares = []
            review_root = revision_directory / "review"
            if review_root.is_dir():
                for prepare_manifest_path in sorted(
                    review_root.glob("*/manifest.json")
                ):
                    try:
                        prepare_manifest = json.loads(
                            prepare_manifest_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        continue
                    prepares.append(
                        {
                            "revision_directory": str(prepare_manifest_path.parent),
                            "manifest": prepare_manifest,
                        }
                    )
            catalog.append(
                {
                    "revision_directory": str(revision_directory),
                    "manifest": source_manifest,
                    "prepares": prepares,
                }
            )
        return catalog

    @app.post("/api/projects")
    def create_project(request: ProjectCreateRequest):
        try:
            with open_database() as database:
                project_id = database.create_project(
                    request.name,
                    guideline_version=request.guideline_version,
                    project_id=request.project_id,
                )
                return database.get_project(project_id)
        except Exception as exc:
            raise_http(exc)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        try:
            with open_database() as database:
                return {
                    "project": database.get_project(project_id),
                    "sources": database.list_project_sources(project_id),
                    "queues": database.list_queues(project_id),
                }
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/imports/inspect")
    def inspect_source(request: InspectRequest):
        try:
            adapter = get_adapter(request.adapter)
            return _jsonable(
                adapter.inspect(SourceSpec(Path(request.path), request.options))
            )
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/imports/preview")
    def preview_source(request: PreviewRequest):
        try:
            adapter = get_adapter(request.adapter)
            return _jsonable(
                adapter.preview(
                    SourceSpec(Path(request.path), request.options),
                    request.mapping.to_core(),
                    limit=request.limit,
                )
            )
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/imports")
    def import_source(request: ImportRequest):
        try:
            return _jsonable(
                ImportService(root).import_source(
                    source_id=request.source_id,
                    source_license=request.license,
                    adapter=get_adapter(request.adapter),
                    spec=SourceSpec(Path(request.path), request.options),
                    mapping=request.mapping.to_core(),
                    max_shard_size=request.max_shard_size,
                )
            )
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/prepares")
    def prepare_source(request: PrepareRequest):
        try:
            return _jsonable(
                PrepareService().prepare_source(
                    request.source_revision_directory,
                    config=PrepareConfig(**request.config),
                    max_shard_size=request.max_shard_size,
                    read_batch_size=request.read_batch_size,
                )
            )
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/projects/{project_id}/sources")
    def attach_source(project_id: str, request: AttachSourceRequest):
        try:
            revision_directory = Path(request.prepare_revision_directory).resolve()
            manifest_path = revision_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with open_database() as database:
                database.attach_source(
                    project_id=project_id,
                    source_id=manifest["source_id"],
                    source_revision=manifest["source_revision"],
                    dataset_path=revision_directory / manifest["dataset_path"],
                    manifest_path=manifest_path,
                    row_count=int(manifest["output_records"]),
                )
                return database.list_project_sources(project_id)
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/projects/{project_id}/queues")
    def create_queue(project_id: str, request: QueueCreateRequest):
        try:
            with open_database() as database:
                if request.policy == "uniform_random":
                    if request.sample_size is None:
                        raise ValueError("uniform_random requires sample_size")
                    queue_id = create_uniform_random_queue(
                        database,
                        project_id=project_id,
                        name=request.name,
                        sample_size=request.sample_size,
                        seed=request.seed,
                        queue_id=request.queue_id,
                    )
                else:
                    if not request.quotas:
                        raise ValueError("source_quota requires quotas")
                    queue_id = create_source_quota_queue(
                        database,
                        project_id=project_id,
                        name=request.name,
                        quotas=request.quotas,
                        seed=request.seed,
                        queue_id=request.queue_id,
                    )
                return database.get_queue(queue_id)
        except Exception as exc:
            raise_http(exc)

    @app.get("/api/queues/{queue_id}/items/{ordinal}")
    def get_queue_document(queue_id: str, ordinal: int):
        try:
            with open_database() as database:
                return add_token_counts(
                    DocumentService(database, repository).queue_document(
                        queue_id, ordinal
                    )
                )
        except Exception as exc:
            raise_http(exc)

    @app.put("/api/reviews/documents/{doc_id}")
    def review_document(doc_id: str, request: DocumentReviewRequest):
        try:
            with open_database() as database:
                value = DocumentService(database, repository).queue_document(
                    request.queue_id, request.ordinal
                )
                row = value["document"]
                if row["doc_id"] != doc_id:
                    raise ValueError("request doc_id does not match queue item")
                project_id = value["item"]["project_id"]
                state, event_seq = database.set_document_review(
                    project_id=project_id,
                    review=DocumentReviewInput(
                        doc_id=doc_id,
                        source_row=int(row["source_row"]),
                        content_sha256=row["content_sha256"],
                        decision=request.decision,
                        quality=request.quality,
                        primary_category=request.primary_category,
                        flags=tuple(request.flags),
                        notes=request.notes,
                        guideline_version=database.get_project(project_id)[
                            "guideline_version"
                        ],
                    ),
                    expected_revision=request.expected_revision,
                    actor=request.actor,
                )
                return {"review": state, "event_seq": event_seq}
        except Exception as exc:
            raise_http(exc)

    @app.put("/api/reviews/blocks/{block_id}")
    def review_block(block_id: str, request: BlockReviewRequest):
        try:
            with open_database() as database:
                value = DocumentService(database, repository).queue_document(
                    request.queue_id, request.ordinal
                )
                matches = [
                    block for block in value["blocks"] if block["block_id"] == block_id
                ]
                if len(matches) != 1:
                    raise ValueError("block_id does not belong to queue document")
                block = matches[0]
                state, event_seq = database.set_block_review(
                    project_id=value["item"]["project_id"],
                    review=BlockReviewInput(
                        doc_id=block["doc_id"],
                        block_id=block_id,
                        base_block_hash=block["content_sha256"],
                        decision=request.decision,
                        reason=request.reason,
                    ),
                    expected_revision=request.expected_revision,
                    actor=request.actor,
                )
                return {"review": state, "event_seq": event_seq}
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/events/{event_seq}/undo")
    def undo(event_seq: int, request: UndoRequest):
        try:
            with open_database() as database:
                undo_event_seq = database.undo_event(
                    project_id=request.project_id,
                    event_seq=event_seq,
                    actor=request.actor,
                )
                return {"event_seq": undo_event_seq}
        except Exception as exc:
            raise_http(exc)

    @app.post("/api/materializations")
    def materialize(request: MaterializeRequest):
        try:
            with open_database() as database:
                return _jsonable(
                    MaterializeService(root / "exports").materialize(
                        database,
                        project_id=request.project_id,
                        policy=request.policy,
                        snapshot_event_seq=request.snapshot_event_seq,
                        max_shard_size=request.max_shard_size,
                        read_batch_size=request.read_batch_size,
                    )
                )
        except Exception as exc:
            raise_http(exc)

    @app.get("/api/materializations/{materialization_id}")
    def get_materialization(materialization_id: str):
        try:
            with open_database() as database:
                return database.get_materialization(materialization_id)
        except Exception as exc:
            raise_http(exc)

    frontend_distribution = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if frontend_distribution.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=frontend_distribution, html=True))

    return app
