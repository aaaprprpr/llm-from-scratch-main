from __future__ import annotations

import json
import threading
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .blocks import parse_blocks, render_blocks
from .database import CurationDatabase
from .dataset_store import load_dataset
from .identity import sha256_text


class DatasetRepository:
    """Small process-local cache for memory-mapped Dataset handles."""

    def __init__(self):
        self._datasets: dict[Path, Any] = {}
        self._lock = threading.Lock()

    def open(self, path: str | Path):
        resolved = Path(path).resolve()
        with self._lock:
            dataset = self._datasets.get(resolved)
            if dataset is None:
                if not resolved.exists():
                    raise FileNotFoundError(resolved)
                dataset = load_dataset(resolved)
                self._datasets[resolved] = dataset
            return dataset

    def clear(self) -> None:
        """Release memory maps before a data directory is moved or deleted."""

        with self._lock:
            self._datasets.clear()
        gc.collect()


class DocumentService:
    def __init__(
        self,
        database: CurationDatabase,
        repository: DatasetRepository | None = None,
    ):
        self.database = database
        self.repository = repository or DatasetRepository()

    def queue_document(self, queue_id: str, ordinal: int) -> dict[str, Any]:
        item = self.database.get_queue_item(queue_id, ordinal)
        project_id = item["project_id"]
        attached = [
            value
            for value in self.database.list_project_sources(project_id)
            if value["source_id"] == item["source_id"]
            and value["source_revision"] == item["source_revision"]
        ]
        if len(attached) != 1:
            raise RuntimeError(
                "queue item does not resolve to exactly one attached source"
            )
        source = attached[0]
        review_dataset = self.repository.open(source["dataset_path"])
        source_row = int(item["source_row"])
        if not 0 <= source_row < len(review_dataset):
            raise IndexError(f"queue source_row is outside Dataset: {source_row}")
        row = review_dataset[source_row]
        if item["doc_id"] is None:
            item["doc_id"] = row["doc_id"]
        self._validate_item_row(item, row)

        prepare_manifest = json.loads(
            Path(source["manifest_path"]).read_text(encoding="utf-8")
        )
        raw_dataset_path = Path(prepare_manifest["input_dataset_path"])
        raw_dataset = self.repository.open(raw_dataset_path)
        raw_row = raw_dataset[source_row]
        if raw_row["doc_id"] != row["doc_id"]:
            raise ValueError("raw/review doc_id mismatch")
        if sha256_text(raw_row["text"]) != raw_row["content_sha256"]:
            raise ValueError("raw text content hash mismatch")

        source_manifest_path = Path(prepare_manifest["source_manifest_path"])
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        blocks = parse_blocks(row["text"], row["doc_id"])
        block_by_id = {block.block_id: block for block in blocks}
        block_reviews = {
            review["block_id"]: review
            for review in self.database.list_block_reviews(project_id, row["doc_id"])
        }
        unknown_reviews = set(block_reviews) - set(block_by_id)
        if unknown_reviews:
            raise ValueError(f"stale block reviews: {sorted(unknown_reviews)}")
        dropped = set()
        rendered_blocks = []
        for block in blocks:
            review = block_reviews.get(block.block_id)
            if review is not None:
                if review["base_block_hash"] != block.content_sha256:
                    raise ValueError(f"stale block content hash: {block.block_id}")
                if review["decision"] == "drop":
                    dropped.add(block.block_id)
            rendered_blocks.append({**asdict(block), "review": review})

        document_review = self.database.get_document_review(
            project_id,
            row["doc_id"],
        )
        item["state"] = "done" if document_review is not None else "pending"
        if document_review is not None:
            if document_review["content_sha256"] != row["content_sha256"]:
                raise ValueError("stale document review content hash")
            if int(document_review["source_row"]) != source_row:
                raise ValueError("stale document review source_row")

        block_materialized_text = render_blocks(row["text"], blocks, dropped)
        materialized_text = (
            document_review["edited_text"]
            if document_review is not None
            and document_review.get("edited_text") is not None
            else block_materialized_text
        )

        return {
            "queue": self.database.get_queue(queue_id),
            "item": item,
            "document": {
                key: value
                for key, value in row.items()
                if key not in {"text", "metadata_json"}
            },
            "metadata": json.loads(row["metadata_json"]),
            "raw_text": raw_row["text"],
            "review_text": row["text"],
            "block_materialized_text": block_materialized_text,
            "materialized_text": materialized_text,
            "blocks": rendered_blocks,
            "document_review": document_review,
            "provenance": {
                "source_id": row["source_id"],
                "source_revision": row["source_revision"],
                "source_row": source_row,
                "source_local_id": row["source_local_id"],
                "title": row["title"],
                "url": row["url"],
                "license": source_manifest["license"],
                "original_location": source_manifest["original_location"],
                "source_manifest_path": str(source_manifest_path),
                "prepare_manifest_path": source["manifest_path"],
            },
        }

    @staticmethod
    def _validate_item_row(item: dict[str, Any], row: dict[str, Any]) -> None:
        if row["doc_id"] != item["doc_id"]:
            raise ValueError("queue doc_id does not match Dataset row")
        if row["source_id"] != item["source_id"]:
            raise ValueError("queue source_id does not match Dataset row")
        if row["source_revision"] != item["source_revision"]:
            raise ValueError("queue source_revision does not match Dataset row")
        if int(row["source_row"]) != int(item["source_row"]):
            raise ValueError("queue source_row does not match Dataset row")
        if sha256_text(row["text"]) != row["content_sha256"]:
            raise ValueError("review text content hash mismatch")
