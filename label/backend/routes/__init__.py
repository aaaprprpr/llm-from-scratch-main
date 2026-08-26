from fastapi import FastAPI

from ..api_context import ApiContext
from .datasets import build_router as build_dataset_router
from .materializations import build_router as build_materialization_router
from .projects import build_router as build_project_router
from .reviews import build_router as build_review_router
from .system import build_router as build_system_router


def include_api_routes(app: FastAPI, context: ApiContext) -> None:
    for build_router in (
        build_system_router,
        build_dataset_router,
        build_project_router,
        build_review_router,
        build_materialization_router,
    ):
        app.include_router(build_router(context))
