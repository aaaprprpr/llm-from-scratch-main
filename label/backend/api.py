from __future__ import annotations

import os
from pathlib import Path

from .api_context import ApiContext
from .routes import include_api_routes


def create_app(data_root: str | Path | None = None):
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
    tokenizer_path = Path(
        os.environ.get(
            "LABEL_TOKENIZER_PATH",
            str(Path(__file__).resolve().parents[2] / "bpe" / "tokenizer_24576"),
        )
    )
    context = ApiContext(root, tokenizer_path)

    app = FastAPI(title="LLM Dataset Curation", version="0.1.0")
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
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=frontend_distribution, html=True))

    return app
