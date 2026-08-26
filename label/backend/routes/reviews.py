from __future__ import annotations

from fastapi import APIRouter

from ..api_context import ApiContext
from ..api_models import BlockReviewRequest, DocumentReviewRequest, UndoRequest
from ..database import BlockReviewInput, DocumentReviewInput
from ..documents import DocumentService


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/queues/{queue_id}/items/{ordinal}")
    def get_queue_document(queue_id: str, ordinal: int):
        try:
            with context.open_database() as database:
                return context.add_token_counts(
                    DocumentService(database, context.repository).queue_document(
                        queue_id, ordinal
                    )
                )
        except Exception as exc:
            context.raise_http(exc)

    @router.put("/api/reviews/documents/{doc_id}")
    def review_document(doc_id: str, request: DocumentReviewRequest):
        try:
            with context.open_database() as database:
                value = DocumentService(database, context.repository).queue_document(
                    request.queue_id, request.ordinal
                )
                row = value["document"]
                if row["doc_id"] != doc_id:
                    raise ValueError("request doc_id does not match queue item")
                project_id = value["item"]["project_id"]
                edited_text = request.edited_text
                if edited_text is None and value["document_review"] is not None:
                    edited_text = value["document_review"].get("edited_text")
                if edited_text == value["block_materialized_text"]:
                    edited_text = None
                state, event_seq = database.set_document_review(
                    project_id=project_id,
                    review=DocumentReviewInput(
                        doc_id=doc_id,
                        source_row=int(row["source_row"]),
                        content_sha256=row["content_sha256"],
                        decision=request.decision,
                        quality=request.quality,
                        primary_category=request.primary_category,
                        flags=tuple(request.flags),
                        notes=request.notes,
                        edited_text=edited_text,
                        guideline_version=database.get_project(project_id)[
                            "guideline_version"
                        ],
                    ),
                    expected_revision=request.expected_revision,
                    actor=request.actor,
                )
                return {"review": state, "event_seq": event_seq}
        except Exception as exc:
            context.raise_http(exc)

    @router.put("/api/reviews/blocks/{block_id}")
    def review_block(block_id: str, request: BlockReviewRequest):
        try:
            with context.open_database() as database:
                value = DocumentService(database, context.repository).queue_document(
                    request.queue_id, request.ordinal
                )
                matches = [
                    block for block in value["blocks"] if block["block_id"] == block_id
                ]
                if len(matches) != 1:
                    raise ValueError("block_id does not belong to queue document")
                block = matches[0]
                state, event_seq = database.set_block_review(
                    project_id=value["item"]["project_id"],
                    review=BlockReviewInput(
                        doc_id=block["doc_id"],
                        block_id=block_id,
                        base_block_hash=block["content_sha256"],
                        decision=request.decision,
                        reason=request.reason,
                    ),
                    expected_revision=request.expected_revision,
                    actor=request.actor,
                )
                return {"review": state, "event_seq": event_seq}
        except Exception as exc:
            context.raise_http(exc)

    @router.post("/api/events/{event_seq}/undo")
    def undo(event_seq: int, request: UndoRequest):
        try:
            with context.open_database() as database:
                undo_event_seq = database.undo_event(
                    project_id=request.project_id,
                    event_seq=event_seq,
                    actor=request.actor,
                )
                return {"event_seq": undo_event_seq}
        except Exception as exc:
            context.raise_http(exc)

    return router
