from fastapi import APIRouter

from ..api_context import ApiContext


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health():
        return {"status": "ok", "data_root": str(context.root)}

    return router
