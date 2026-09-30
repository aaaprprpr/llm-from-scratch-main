"""Model choice and local Qwen service status for the settings page."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..api_context import ApiContext
from ..model_settings import ModelSelection, ModelSource


class ModelSettingsRequest(BaseModel):
    single: ModelSource
    batch: ModelSource | list[ModelSource]
    failure_fallback: bool | None = None


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    def current() -> dict:
        return {
            **context.model_settings.read().as_dict(),
            "available": context.model_settings.available(),
            "local_model": context.local_model.status(),
        }

    @router.get("/api/settings/models")
    def get_models():
        return current()

    @router.put("/api/settings/models")
    def update_models(request: ModelSettingsRequest):
        try:
            previous = context.model_settings.read()
            selection = ModelSelection(
                single=request.single, batch=request.batch,
                failure_fallback=(previous.failure_fallback if request.failure_fallback is None
                                  else request.failure_fallback),
            )
            context.set_model_selection(selection)
            return current()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            context.raise_http(exc)

    return router
