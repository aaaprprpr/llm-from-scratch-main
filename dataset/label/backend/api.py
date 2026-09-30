from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from .api_context import ApiContext
from .routes import include_api_routes


def create_app(data_root: str | Path | None = None, *, start_local_model: bool | None = None):
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install project requirements first."
        ) from exc

    root = Path(
        data_root
        or os.environ.get(
            "LABEL_DATA_ROOT",
            str(Path(__file__).resolve().parents[1] / "data"),
        )
    )
    context = ApiContext(root)

    if start_local_model is None:
        start_local_model = False

    @asynccontextmanager
    async def lifespan(_app):
        if start_local_model:
            context.local_model.ensure_running()
        try:
            yield
        finally:
            context.local_model.close()

    app = FastAPI(title="LLM Dataset Curation", version="0.1.0", lifespan=lifespan)
    app.state.dataset_repository = context.repository
    app.state.api_context = context
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    include_api_routes(app, context)

    frontend_distribution = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if frontend_distribution.is_dir():
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(frontend_distribution / "index.html", headers={"Cache-Control": "no-store"})

        app.mount("/", StaticFiles(directory=frontend_distribution, html=True))

    return app
