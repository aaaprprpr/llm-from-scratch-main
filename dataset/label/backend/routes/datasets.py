from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from ..api_context import ApiContext, automatic_source_id
from ..serialization import jsonable
from ..api_models import ImportRequest, InspectRequest, PrepareRequest, PreviewRequest
from ..import_service import ImportService
from ..importers import SourceSpec, get_adapter
from ..prepare import PrepareConfig, PrepareService
from ..source_catalog import list_download_sources


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/catalog")
    def source_catalog():
        catalog = []
        sources_root = context.root / "sources"
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
                for prepare_manifest_path in sorted(review_root.glob("*/manifest.json")):
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

    @router.get("/api/download-sources")
    def download_sources():
        try:
            imported = [item["manifest"]["original_location"] for item in source_catalog()]
            return list_download_sources(imported_locations=imported)
        except Exception as exc:
            context.raise_http(exc)

    @router.post("/api/imports/inspect")
    def inspect_source(request: InspectRequest):
        try:
            adapter = get_adapter(request.adapter)
            return jsonable(
                adapter.inspect(SourceSpec(Path(request.path), request.options))
            )
        except Exception as exc:
            context.raise_http(exc)

    @router.post("/api/imports/preview")
    def preview_source(request: PreviewRequest):
        try:
            adapter = get_adapter(request.adapter)
            return jsonable(
                adapter.preview(
                    SourceSpec(Path(request.path), request.options),
                    request.mapping.to_core(),
                    limit=request.limit,
                )
            )
        except Exception as exc:
            context.raise_http(exc)

    @router.post("/api/imports")
    def import_source(request: ImportRequest):
        try:
            return jsonable(
                ImportService(context.root).import_source(
                    source_id=request.source_id or automatic_source_id(request.path),
                    source_license=request.license or "unknown",
                    adapter=get_adapter(request.adapter),
                    spec=SourceSpec(Path(request.path), request.options),
                    mapping=request.mapping.to_core(),
                    max_shard_size=request.max_shard_size,
                )
            )
        except Exception as exc:
            context.raise_http(exc)

    @router.get("/api/prepares/progress/{progress_id}")
    def prepare_progress(progress_id: str):
        return context.get_prepare_progress(progress_id)

    @router.post("/api/prepares")
    def prepare_source(request: PrepareRequest):
        progress_id = request.progress_id
        if progress_id:
            context.set_prepare_progress(progress_id, 0, 0, "opening")
        try:
            result = PrepareService().prepare_source(
                request.source_revision_directory,
                config=PrepareConfig(**request.config),
                max_shard_size=request.max_shard_size,
                read_batch_size=request.read_batch_size,
                progress=(lambda count, total: context.set_prepare_progress(progress_id, count, total))
                if progress_id else None,
            )
            if progress_id:
                context.set_prepare_progress(progress_id, result.manifest.output_records,
                                             result.manifest.output_records, "done")
            return jsonable(result)
        except Exception as exc:
            if progress_id:
                context.set_prepare_progress(progress_id, 0, 0, "failed")
            context.raise_http(exc)

    return router
