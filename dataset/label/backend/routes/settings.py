"""Model choice and local Qwen service status for the settings page."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..api_context import ApiContext
from ..model_settings import ModelSelection, ModelSource


class ModelSettingsRequest(BaseModel):
    single: ModelSource
    batch: ModelSource | list[ModelSource]


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
            selection = ModelSelection(single=request.single, batch=request.batch)
            previous = context.model_settings.read()
            context.set_model_selection(selection)
            if ((selection.single == "local" and previous.single != "local")
                    or (selection.batch != previous.batch and "local" in selection.batch)):
                context.local_model.ensure_running()
            return current()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            context.raise_http(exc)

    return router
