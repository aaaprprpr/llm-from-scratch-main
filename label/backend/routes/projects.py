from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from ..api_context import ApiContext
from ..api_models import AttachSourceRequest, ProjectCreateRequest, QueueCreateRequest
from ..queueing import create_full_dataset_queue


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/projects")
    def list_projects():
        with context.open_database() as database:
            return database.list_projects()

    @router.post("/api/projects")
    def create_project(request: ProjectCreateRequest):
        try:
            with context.open_database() as database:
                project_id = database.create_project(
                    request.name,
                    guideline_version=request.guideline_version,
                    project_id=request.project_id,
                )
                return database.get_project(project_id)
        except Exception as exc:
            context.raise_http(exc)

    @router.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        try:
            with context.open_database() as database:
                return {
                    "project": database.get_project(project_id),
                    "sources": database.list_project_sources(project_id),
                    "queues": database.list_queues(project_id),
                }
        except Exception as exc:
            context.raise_http(exc)

    @router.post("/api/projects/{project_id}/sources")
    def attach_source(project_id: str, request: AttachSourceRequest):
        try:
            revision_directory = Path(request.prepare_revision_directory).resolve()
            manifest_path = revision_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with context.open_database() as database:
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
            context.raise_http(exc)

    @router.post("/api/projects/{project_id}/queues")
    def create_queue(project_id: str, request: QueueCreateRequest):
        try:
            with context.open_database() as database:
                queue_id = create_full_dataset_queue(
                    database,
                    project_id=project_id,
                    name=request.name,
                    queue_id=request.queue_id,
                )
                return database.get_queue(queue_id)
        except Exception as exc:
            context.raise_http(exc)

    return router
