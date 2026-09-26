from fastapi import APIRouter

from ..api_context import ApiContext
from ..api_models import PathPickerRequest, SimplifyTextRequest
from ..path_picker import select_local_path


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health():
        return {"status": "ok", "data_root": str(context.root)}

    @router.post("/api/text/simplify")
    def simplify_text(request: SimplifyTextRequest):
        return context.simplify_texts(request.texts)

    @router.post("/api/system/select-path")
    def choose_local_path(request: PathPickerRequest):
        try:
            selected = select_local_path(
                kind=request.kind,
                adapter=request.adapter,
                initial_directory=request.initial_directory,
            )
            return {"path": selected}
        except Exception as exc:
            context.raise_http(exc)

    return router
