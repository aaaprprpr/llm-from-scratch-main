from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_context import ApiContext


class StartRequest(BaseModel):
    workers: int | None = Field(default=None, ge=1, le=16)
    max_requests: int | None = Field(default=None, ge=1, le=24)
    limit: int | None = Field(default=None, ge=1)


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/auto-clean")
    def overview():
        return context.batch_jobs.overview()

    @router.get("/api/auto-clean/{queue_id}")
    def status(queue_id: str):
        with context.open_database() as database:
            database.get_queue(queue_id)
        return context.batch_jobs.status(queue_id)

    @router.post("/api/auto-clean/{queue_id}/start")
    def start(queue_id: str, request: StartRequest):
        try:
            return context.batch_jobs.start(queue_id, workers=request.workers,
                                            max_requests=request.max_requests, limit=request.limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            context.raise_http(exc)

    @router.post("/api/auto-clean/{queue_id}/export")
    def export(queue_id: str):
        try:
            return context.batch_jobs.export(queue_id)
        except Exception as exc:
            context.raise_http(exc)

    @router.post("/api/auto-clean/{queue_id}/stop")
    def stop(queue_id: str):
        try:
            return context.batch_jobs.stop(queue_id)
        except Exception as exc:
            context.raise_http(exc)

    return router
