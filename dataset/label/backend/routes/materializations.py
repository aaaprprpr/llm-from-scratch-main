from fastapi import APIRouter

from ..api_context import ApiContext, jsonable
from ..api_models import MaterializeRequest
from ..materialize import MaterializeService


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.post("/api/materializations")
    def materialize(request: MaterializeRequest):
        try:
            with context.open_database() as database:
                return jsonable(
                    MaterializeService(context.root / "exports").materialize(
                        database,
                        project_id=request.project_id,
                        policy=request.policy,
                        snapshot_event_seq=request.snapshot_event_seq,
                        max_shard_size=request.max_shard_size,
                        read_batch_size=request.read_batch_size,
                    )
                )
        except Exception as exc:
            context.raise_http(exc)

    @router.get("/api/materializations/{materialization_id}")
    def get_materialization(materialization_id: str):
        try:
            with context.open_database() as database:
                return database.get_materialization(materialization_id)
        except Exception as exc:
            context.raise_http(exc)

    return router
