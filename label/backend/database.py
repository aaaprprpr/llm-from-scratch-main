from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .identity import stable_json
from .schema import Decision, PrimaryCategory, REVIEW_FLAGS


DATABASE_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class QueueCandidate:
    source_id: str
    source_revision: str
    source_row: int
    doc_id: str
    priority: float = 0.0


@dataclass(frozen=True)
class DocumentReviewInput:
    doc_id: str
    source_row: int
    content_sha256: str
    decision: str
    quality: int | None
    primary_category: str | None
    flags: tuple[str, ...] = ()
    notes: str = ""
    guideline_version: str = "1"


@dataclass(frozen=True)
class BlockReviewInput:
    doc_id: str
    block_id: str
    base_block_hash: str
    decision: str
    reason: str | None = None


class RevisionConflictError(RuntimeError):
    pass


class UndoConflictError(RuntimeError):
    pass


class CurationDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CurationDatabase:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.connection.in_transaction:
            raise RuntimeError("nested CurationDatabase transactions are unsupported")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _migrate(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        row = self.connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        current_version = int(row["version"])
        if current_version > DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema {current_version} is newer than supported "
                f"version {DATABASE_SCHEMA_VERSION}"
            )
        if current_version == 0:
            self._apply_migration_1()
            current_version = 1
        if current_version != DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database migration stopped at version {current_version}"
            )

    def _apply_migration_1(self) -> None:
        connection = self.connection
        try:
            connection.executescript(
                """
                BEGIN IMMEDIATE;

                CREATE TABLE projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    guideline_version TEXT NOT NULL,
                    current_revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE project_sources (
                    project_id TEXT NOT NULL REFERENCES projects(project_id)
                        ON DELETE CASCADE,
                    source_id TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    dataset_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK(row_count >= 0),
                    attached_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, source_id, source_revision)
                );

                CREATE TABLE review_queues (
                    queue_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id)
                        ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    sampling_policy_json TEXT NOT NULL,
                    sampling_seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE queue_items (
                    queue_id TEXT NOT NULL REFERENCES review_queues(queue_id)
                        ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                    source_id TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    source_row INTEGER NOT NULL CHECK(source_row >= 0),
                    doc_id TEXT NOT NULL,
                    priority REAL NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'pending'
                        CHECK(state IN ('pending', 'done', 'skipped')),
                    PRIMARY KEY(queue_id, ordinal),
                    UNIQUE(queue_id, doc_id)
                );

                CREATE INDEX queue_items_state_index
                    ON queue_items(queue_id, state, ordinal);

                CREATE TABLE document_reviews (
                    project_id TEXT NOT NULL REFERENCES projects(project_id)
                        ON DELETE CASCADE,
                    doc_id TEXT NOT NULL,
                    source_row INTEGER NOT NULL CHECK(source_row >= 0),
                    content_sha256 TEXT NOT NULL,
                    decision TEXT NOT NULL
                        CHECK(decision IN ('keep', 'drop', 'unsure')),
                    quality INTEGER CHECK(quality BETWEEN 0 AND 3),
                    primary_category TEXT,
                    flags_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    guideline_version TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, doc_id)
                );

                CREATE INDEX document_reviews_decision_index
                    ON document_reviews(project_id, decision);

                CREATE TABLE block_reviews (
                    project_id TEXT NOT NULL REFERENCES projects(project_id)
                        ON DELETE CASCADE,
                    doc_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    base_block_hash TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('keep', 'drop')),
                    reason TEXT,
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, block_id)
                );

                CREATE INDEX block_reviews_document_index
                    ON block_reviews(project_id, doc_id);

                CREATE TABLE events (
                    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL REFERENCES projects(project_id)
                        ON DELETE CASCADE,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    undo_of_event_seq INTEGER REFERENCES events(event_seq)
                );

                CREATE INDEX events_project_sequence_index
                    ON events(project_id, event_seq);
                CREATE INDEX events_entity_index
                    ON events(project_id, entity_type, entity_id, event_seq);

                CREATE TABLE materializations (
                    materialization_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id)
                        ON DELETE CASCADE,
                    snapshot_event_seq INTEGER NOT NULL,
                    policy_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (1, utc_now()),
            )
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    def create_project(
        self,
        name: str,
        *,
        guideline_version: str = "1",
        project_id: str | None = None,
    ) -> str:
        name = name.strip()
        if not name:
            raise ValueError("project name cannot be empty")
        if not guideline_version:
            raise ValueError("guideline_version cannot be empty")
        project_id = project_id or uuid.uuid4().hex
        timestamp = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO projects(
                    project_id, name, guideline_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, name, guideline_version, timestamp, timestamp),
            )
        return project_id

    def get_project(self, project_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        return dict(row)

    def list_projects(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM projects ORDER BY created_at, project_id"
            )
        ]

    def attach_source(
        self,
        *,
        project_id: str,
        source_id: str,
        source_revision: str,
        dataset_path: str | Path,
        manifest_path: str | Path,
        row_count: int,
    ) -> None:
        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        self.get_project(project_id)
        resolved_dataset_path = str(Path(dataset_path).resolve())
        resolved_manifest_path = str(Path(manifest_path).resolve())
        existing = self.connection.execute(
            """
            SELECT dataset_path, manifest_path, row_count FROM project_sources
            WHERE project_id = ? AND source_id = ? AND source_revision = ?
            """,
            (project_id, source_id, source_revision),
        ).fetchone()
        if existing is not None:
            expected = (
                resolved_dataset_path,
                resolved_manifest_path,
                row_count,
            )
            actual = (
                existing["dataset_path"],
                existing["manifest_path"],
                int(existing["row_count"]),
            )
            if actual != expected:
                raise ValueError(
                    "the same source revision is already attached with different "
                    "paths or row count"
                )
            return
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO project_sources(
                    project_id, source_id, source_revision, dataset_path,
                    manifest_path, row_count, attached_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    source_id,
                    source_revision,
                    resolved_dataset_path,
                    resolved_manifest_path,
                    row_count,
                    utc_now(),
                ),
            )

    def list_project_sources(self, project_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT * FROM project_sources
                WHERE project_id = ?
                ORDER BY source_id, source_revision
                """,
                (project_id,),
            )
        ]

    def create_queue(
        self,
        *,
        project_id: str,
        name: str,
        sampling_policy: Mapping[str, Any],
        sampling_seed: int,
        candidates: Iterable[QueueCandidate],
        queue_id: str | None = None,
    ) -> str:
        self.get_project(project_id)
        queue_id = queue_id or uuid.uuid4().hex
        rows = []
        seen_doc_ids = set()
        for ordinal, candidate in enumerate(candidates):
            if candidate.doc_id in seen_doc_ids:
                raise ValueError(f"Duplicate queue doc_id: {candidate.doc_id}")
            seen_doc_ids.add(candidate.doc_id)
            rows.append(
                (
                    queue_id,
                    ordinal,
                    candidate.source_id,
                    candidate.source_revision,
                    candidate.source_row,
                    candidate.doc_id,
                    candidate.priority,
                )
            )
        if not rows:
            raise ValueError("review queue cannot be empty")

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO review_queues(
                    queue_id, project_id, name, sampling_policy_json,
                    sampling_seed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    queue_id,
                    project_id,
                    name,
                    stable_json(dict(sampling_policy)),
                    sampling_seed,
                    utc_now(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO queue_items(
                    queue_id, ordinal, source_id, source_revision, source_row,
                    doc_id, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return queue_id

    def get_queue_item(self, queue_id: str, ordinal: int) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT qi.*, rq.project_id
            FROM queue_items AS qi
            JOIN review_queues AS rq ON rq.queue_id = qi.queue_id
            WHERE qi.queue_id = ? AND qi.ordinal = ?
            """,
            (queue_id, ordinal),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown queue item: {queue_id}/{ordinal}")
        return dict(row)

    def get_queue(self, queue_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM review_queues WHERE queue_id = ?",
            (queue_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown queue_id: {queue_id}")
        result = dict(row)
        result["sampling_policy"] = json.loads(result.pop("sampling_policy_json"))
        counts = self.connection.execute(
            """
            SELECT state, COUNT(*) AS count FROM queue_items
            WHERE queue_id = ? GROUP BY state
            """,
            (queue_id,),
        )
        result["state_counts"] = {row["state"]: row["count"] for row in counts}
        source_counts = self.connection.execute(
            """
            SELECT source_id, COUNT(*) AS count FROM queue_items
            WHERE queue_id = ? GROUP BY source_id ORDER BY source_id
            """,
            (queue_id,),
        )
        result["source_counts"] = {
            row["source_id"]: row["count"] for row in source_counts
        }
        return result

    def list_queues(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [
            self.get_queue(row["queue_id"])
            for row in self.connection.execute(
                """
                SELECT queue_id FROM review_queues
                WHERE project_id = ? ORDER BY created_at, queue_id
                """,
                (project_id,),
            )
        ]

    def record_materialization(
        self,
        *,
        materialization_id: str,
        project_id: str,
        snapshot_event_seq: int,
        policy: Mapping[str, Any],
        config_hash: str,
        output_path: str | Path,
        manifest_sha256: str,
    ) -> None:
        self.get_project(project_id)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO materializations(
                    materialization_id, project_id, snapshot_event_seq,
                    policy_json, config_hash, output_path, manifest_sha256,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    materialization_id,
                    project_id,
                    snapshot_event_seq,
                    stable_json(dict(policy)),
                    config_hash,
                    str(Path(output_path).resolve()),
                    manifest_sha256,
                    utc_now(),
                ),
            )

    def get_materialization(self, materialization_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM materializations WHERE materialization_id = ?",
            (materialization_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown materialization_id: {materialization_id}")
        result = dict(row)
        result["policy"] = json.loads(result.pop("policy_json"))
        return result

    @staticmethod
    def _validate_document_review(review: DocumentReviewInput) -> None:
        decisions = {Decision.KEEP.value, Decision.DROP.value, Decision.UNSURE.value}
        if review.decision not in decisions:
            raise ValueError(f"Unsupported document decision: {review.decision}")
        if review.quality is not None and not 0 <= review.quality <= 3:
            raise ValueError("quality must be None or an integer between 0 and 3")
        if review.primary_category is not None:
            categories = {value.value for value in PrimaryCategory}
            if review.primary_category not in categories:
                raise ValueError(
                    f"Unsupported primary_category: {review.primary_category}"
                )
        unknown_flags = set(review.flags) - REVIEW_FLAGS
        if unknown_flags:
            raise ValueError(f"Unsupported review flags: {sorted(unknown_flags)}")
        if review.source_row < 0:
            raise ValueError("source_row must be non-negative")
        if not review.content_sha256:
            raise ValueError("content_sha256 cannot be empty")

    @staticmethod
    def _document_row_to_state(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "doc_id": row["doc_id"],
            "source_row": row["source_row"],
            "content_sha256": row["content_sha256"],
            "decision": row["decision"],
            "quality": row["quality"],
            "primary_category": row["primary_category"],
            "flags": json.loads(row["flags_json"]),
            "notes": row["notes"],
            "guideline_version": row["guideline_version"],
            "revision": row["revision"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _block_row_to_state(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "doc_id": row["doc_id"],
            "block_id": row["block_id"],
            "base_block_hash": row["base_block_hash"],
            "decision": row["decision"],
            "reason": row["reason"],
            "revision": row["revision"],
            "updated_at": row["updated_at"],
        }

    def get_document_review(
        self,
        project_id: str,
        doc_id: str,
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM document_reviews
            WHERE project_id = ? AND doc_id = ?
            """,
            (project_id, doc_id),
        ).fetchone()
        return self._document_row_to_state(row)

    def set_document_review(
        self,
        *,
        project_id: str,
        review: DocumentReviewInput,
        expected_revision: int,
        actor: str = "local",
    ) -> tuple[dict[str, Any], int]:
        self._validate_document_review(review)
        timestamp = utc_now()
        with self.transaction() as connection:
            current_row = connection.execute(
                """
                SELECT * FROM document_reviews
                WHERE project_id = ? AND doc_id = ?
                """,
                (project_id, review.doc_id),
            ).fetchone()
            current_revision = 0 if current_row is None else int(current_row["revision"])
            if expected_revision != current_revision:
                raise RevisionConflictError(
                    f"document {review.doc_id} revision is {current_revision}, "
                    f"not {expected_revision}"
                )
            before = self._document_row_to_state(current_row)
            revision = current_revision + 1
            flags = sorted(set(review.flags))
            connection.execute(
                """
                INSERT INTO document_reviews(
                    project_id, doc_id, source_row, content_sha256, decision,
                    quality, primary_category, flags_json, notes,
                    guideline_version, revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, doc_id) DO UPDATE SET
                    source_row = excluded.source_row,
                    content_sha256 = excluded.content_sha256,
                    decision = excluded.decision,
                    quality = excluded.quality,
                    primary_category = excluded.primary_category,
                    flags_json = excluded.flags_json,
                    notes = excluded.notes,
                    guideline_version = excluded.guideline_version,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    review.doc_id,
                    review.source_row,
                    review.content_sha256,
                    review.decision,
                    review.quality,
                    review.primary_category,
                    stable_json(flags),
                    review.notes,
                    review.guideline_version,
                    revision,
                    timestamp,
                ),
            )
            after_row = connection.execute(
                """
                SELECT * FROM document_reviews
                WHERE project_id = ? AND doc_id = ?
                """,
                (project_id, review.doc_id),
            ).fetchone()
            after = self._document_row_to_state(after_row)
            event_seq = self._append_event(
                connection,
                project_id=project_id,
                entity_type="document_review",
                entity_id=review.doc_id,
                operation_type="set",
                before=before,
                after=after,
                actor=actor,
            )
            connection.execute(
                """
                UPDATE queue_items SET state = 'done'
                WHERE doc_id = ? AND queue_id IN (
                    SELECT queue_id FROM review_queues WHERE project_id = ?
                )
                """,
                (review.doc_id, project_id),
            )
        assert after is not None
        return after, event_seq

    def get_block_review(
        self,
        project_id: str,
        block_id: str,
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM block_reviews
            WHERE project_id = ? AND block_id = ?
            """,
            (project_id, block_id),
        ).fetchone()
        return self._block_row_to_state(row)

    def list_block_reviews(
        self,
        project_id: str,
        doc_id: str,
    ) -> list[dict[str, Any]]:
        return [
            self._block_row_to_state(row)
            for row in self.connection.execute(
                """
                SELECT * FROM block_reviews
                WHERE project_id = ? AND doc_id = ?
                ORDER BY block_id
                """,
                (project_id, doc_id),
            )
        ]

    def set_block_review(
        self,
        *,
        project_id: str,
        review: BlockReviewInput,
        expected_revision: int,
        actor: str = "local",
    ) -> tuple[dict[str, Any], int]:
        if review.decision not in {Decision.KEEP.value, Decision.DROP.value}:
            raise ValueError("block decision must be 'keep' or 'drop'")
        timestamp = utc_now()
        with self.transaction() as connection:
            current_row = connection.execute(
                """
                SELECT * FROM block_reviews
                WHERE project_id = ? AND block_id = ?
                """,
                (project_id, review.block_id),
            ).fetchone()
            current_revision = 0 if current_row is None else int(current_row["revision"])
            if expected_revision != current_revision:
                raise RevisionConflictError(
                    f"block {review.block_id} revision is {current_revision}, "
                    f"not {expected_revision}"
                )
            before = self._block_row_to_state(current_row)
            revision = current_revision + 1
            connection.execute(
                """
                INSERT INTO block_reviews(
                    project_id, doc_id, block_id, base_block_hash,
                    decision, reason, revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, block_id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    base_block_hash = excluded.base_block_hash,
                    decision = excluded.decision,
                    reason = excluded.reason,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    review.doc_id,
                    review.block_id,
                    review.base_block_hash,
                    review.decision,
                    review.reason,
                    revision,
                    timestamp,
                ),
            )
            after_row = connection.execute(
                """
                SELECT * FROM block_reviews
                WHERE project_id = ? AND block_id = ?
                """,
                (project_id, review.block_id),
            ).fetchone()
            after = self._block_row_to_state(after_row)
            event_seq = self._append_event(
                connection,
                project_id=project_id,
                entity_type="block_review",
                entity_id=review.block_id,
                operation_type="set",
                before=before,
                after=after,
                actor=actor,
            )
        assert after is not None
        return after, event_seq

    def _append_event(
        self,
        connection: sqlite3.Connection,
        *,
        project_id: str,
        entity_type: str,
        entity_id: str,
        operation_type: str,
        before: Mapping[str, Any] | None,
        after: Mapping[str, Any] | None,
        actor: str,
        undo_of_event_seq: int | None = None,
    ) -> int:
        timestamp = utc_now()
        cursor = connection.execute(
            """
            INSERT INTO events(
                project_id, entity_type, entity_id, operation_type,
                payload_json, actor, created_at, undo_of_event_seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                entity_type,
                entity_id,
                operation_type,
                stable_json({"before": before, "after": after}),
                actor,
                timestamp,
                undo_of_event_seq,
            ),
        )
        event_seq = int(cursor.lastrowid)
        connection.execute(
            """
            UPDATE projects
            SET current_revision = ?, updated_at = ?
            WHERE project_id = ?
            """,
            (event_seq, timestamp, project_id),
        )
        return event_seq

    def latest_event_seq(self, project_id: str) -> int:
        return int(self.get_project(project_id)["current_revision"])

    def undo_event(
        self,
        *,
        project_id: str,
        event_seq: int,
        actor: str = "local",
    ) -> int:
        with self.transaction() as connection:
            event = connection.execute(
                """
                SELECT * FROM events
                WHERE project_id = ? AND event_seq = ?
                """,
                (project_id, event_seq),
            ).fetchone()
            if event is None:
                raise KeyError(f"Unknown event_seq: {event_seq}")
            if event["operation_type"] == "undo":
                raise UndoConflictError("an undo event cannot be undone in V0.1")
            already_undone = connection.execute(
                "SELECT 1 FROM events WHERE undo_of_event_seq = ?",
                (event_seq,),
            ).fetchone()
            if already_undone is not None:
                raise UndoConflictError(f"event {event_seq} is already undone")
            latest_entity_event = connection.execute(
                """
                SELECT MAX(event_seq) AS event_seq FROM events
                WHERE project_id = ? AND entity_type = ? AND entity_id = ?
                """,
                (project_id, event["entity_type"], event["entity_id"]),
            ).fetchone()["event_seq"]
            if int(latest_entity_event) != event_seq:
                raise UndoConflictError(
                    "only the latest event for an entity can be undone"
                )

            payload = json.loads(event["payload_json"])
            before = payload["before"]
            current = payload["after"]
            if event["entity_type"] == "document_review":
                self._restore_document_state(
                    connection,
                    project_id,
                    event["entity_id"],
                    before,
                )
                restored_row = connection.execute(
                    """
                    SELECT * FROM document_reviews
                    WHERE project_id = ? AND doc_id = ?
                    """,
                    (project_id, event["entity_id"]),
                ).fetchone()
                restored = self._document_row_to_state(restored_row)
                connection.execute(
                    """
                    UPDATE queue_items SET state = ?
                    WHERE doc_id = ? AND queue_id IN (
                        SELECT queue_id FROM review_queues WHERE project_id = ?
                    )
                    """,
                    (
                        "pending" if restored is None else "done",
                        event["entity_id"],
                        project_id,
                    ),
                )
            elif event["entity_type"] == "block_review":
                self._restore_block_state(
                    connection,
                    project_id,
                    event["entity_id"],
                    before,
                )
                restored_row = connection.execute(
                    """
                    SELECT * FROM block_reviews
                    WHERE project_id = ? AND block_id = ?
                    """,
                    (project_id, event["entity_id"]),
                ).fetchone()
                restored = self._block_row_to_state(restored_row)
            else:
                raise UndoConflictError(
                    f"unsupported entity_type: {event['entity_type']}"
                )

            return self._append_event(
                connection,
                project_id=project_id,
                entity_type=event["entity_type"],
                entity_id=event["entity_id"],
                operation_type="undo",
                before=current,
                after=restored,
                actor=actor,
                undo_of_event_seq=event_seq,
            )

    def _restore_document_state(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        doc_id: str,
        state: Mapping[str, Any] | None,
    ) -> None:
        if state is None:
            connection.execute(
                "DELETE FROM document_reviews WHERE project_id = ? AND doc_id = ?",
                (project_id, doc_id),
            )
            return
        current = connection.execute(
            """
            SELECT revision FROM document_reviews
            WHERE project_id = ? AND doc_id = ?
            """,
            (project_id, doc_id),
        ).fetchone()
        revision = (0 if current is None else int(current["revision"])) + 1
        connection.execute(
            """
            UPDATE document_reviews SET
                source_row = ?, content_sha256 = ?, decision = ?, quality = ?,
                primary_category = ?, flags_json = ?, notes = ?,
                guideline_version = ?, revision = ?, updated_at = ?
            WHERE project_id = ? AND doc_id = ?
            """,
            (
                state["source_row"],
                state["content_sha256"],
                state["decision"],
                state["quality"],
                state["primary_category"],
                stable_json(state["flags"]),
                state["notes"],
                state["guideline_version"],
                revision,
                utc_now(),
                project_id,
                doc_id,
            ),
        )

    def _restore_block_state(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        block_id: str,
        state: Mapping[str, Any] | None,
    ) -> None:
        if state is None:
            connection.execute(
                "DELETE FROM block_reviews WHERE project_id = ? AND block_id = ?",
                (project_id, block_id),
            )
            return
        current = connection.execute(
            """
            SELECT revision FROM block_reviews
            WHERE project_id = ? AND block_id = ?
            """,
            (project_id, block_id),
        ).fetchone()
        revision = (0 if current is None else int(current["revision"])) + 1
        connection.execute(
            """
            UPDATE block_reviews SET
                doc_id = ?, base_block_hash = ?, decision = ?, reason = ?,
                revision = ?, updated_at = ?
            WHERE project_id = ? AND block_id = ?
            """,
            (
                state["doc_id"],
                state["base_block_hash"],
                state["decision"],
                state["reason"],
                revision,
                utc_now(),
                project_id,
                block_id,
            ),
        )

    def review_state_at(
        self,
        project_id: str,
        event_seq: int | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if event_seq is None:
            event_seq = self.latest_event_seq(project_id)
        document_states: dict[str, dict[str, Any]] = {}
        block_states: dict[str, dict[str, Any]] = {}
        rows = self.connection.execute(
            """
            SELECT entity_type, entity_id, payload_json
            FROM events
            WHERE project_id = ? AND event_seq <= ?
            ORDER BY event_seq
            """,
            (project_id, event_seq),
        )
        for row in rows:
            after = json.loads(row["payload_json"])["after"]
            if row["entity_type"] == "document_review":
                target = document_states
            elif row["entity_type"] == "block_review":
                target = block_states
            else:
                continue
            if after is None:
                target.pop(row["entity_id"], None)
            else:
                target[row["entity_id"]] = after
        return document_states, block_states
