from __future__ import annotations

import tempfile
import unittest
import gc
import sqlite3
from pathlib import Path

from label.backend.blocks import BLOCK_PARSER_VERSION, parse_blocks, render_blocks
from label.backend.database import (
    BlockReviewInput,
    CurationDatabase,
    DocumentReviewInput,
    QueueCandidate,
    RevisionConflictError,
    UndoConflictError,
)
from label.backend.dataset_store import load_dataset, write_arrow_dataset
from label.backend.identity import (
    make_doc_id,
    make_source_revision,
    sha256_text,
)
from label.backend.schema import FieldMapping, ImportedDocument, canonical_row
from data_pipeline.build_bin import load_input_dataset


class MigrationTests(unittest.TestCase):
    def test_version_one_database_adds_sparse_edited_text_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "version-one.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                INSERT INTO schema_migrations VALUES (1, 'old');
                CREATE TABLE document_reviews (
                    project_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_row INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    quality INTEGER,
                    primary_category TEXT,
                    flags_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    guideline_version TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, doc_id)
                );
                """
            )
            connection.commit()
            connection.close()

            database = CurationDatabase(path)
            try:
                columns = {
                    row["name"]
                    for row in database.connection.execute(
                        "PRAGMA table_info(document_reviews)"
                    )
                }
                version = database.connection.execute(
                    "SELECT MAX(version) AS version FROM schema_migrations"
                ).fetchone()["version"]
                self.assertIn("edited_text", columns)
                self.assertEqual(version, 2)
            finally:
                database.close()


class IdentityAndSchemaTests(unittest.TestCase):
    def test_source_revision_is_mapping_order_independent(self):
        first = make_source_revision(
            source_id="source-a",
            adapter_name="jsonl",
            adapter_version="1",
            input_fingerprint="abc",
            mapping={"text": "body", "title": "title"},
            schema_version="1",
        )
        second = make_source_revision(
            source_id="source-a",
            adapter_name="jsonl",
            adapter_version="1",
            input_fingerprint="abc",
            mapping={"title": "title", "text": "body"},
            schema_version="1",
        )
        self.assertEqual(first, second)

    def test_doc_id_uses_source_locator_not_only_text(self):
        first = make_doc_id("source-a", "revision", "line:1")
        second = make_doc_id("source-a", "revision", "line:2")
        self.assertNotEqual(first, second)

        document = ImportedDocument(stable_locator="line:1", text="same text")
        row = canonical_row(
            doc_id=first,
            source_id="source-a",
            source_revision="revision",
            source_row=0,
            document=document,
        )
        self.assertEqual(row["content_sha256"], sha256_text("same text"))
        self.assertEqual(row["metadata_json"], "{}")

    def test_field_mapping_requires_text(self):
        with self.assertRaises(ValueError):
            FieldMapping(text_fields=())

    def test_one_pass_arrow_store_shards_and_round_trips(self):
        from datasets import Dataset, Features, Value

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset"
            result = write_arrow_dataset(
                ({"text": str(index) * 200} for index in range(8)),
                path,
                features=Features({"text": Value("string")}),
                fingerprint="test-fingerprint",
                max_shard_size=300,
                writer_batch_size=1,
            )
            self.assertGreater(len(result.files), 1)
            dataset = load_dataset(path)
            self.assertEqual(len(dataset), 8)
            self.assertEqual(dataset._fingerprint, "test-fingerprint")
            self.assertEqual(dataset[3]["text"], "3" * 200)
            label_input = load_input_dataset(path)
            self.assertEqual(label_input[4]["text"], "4" * 200)
            del label_input
            del dataset
            gc.collect()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native-hf"
            Dataset.from_dict({"text": ["兼容原流程"]}).save_to_disk(str(path))
            native = load_dataset(path)
            self.assertEqual(native.column_names, ["text"])
            self.assertEqual(native[0]["text"], "兼容原流程")
            del native
            gc.collect()
            legacy = load_input_dataset(path)
            self.assertEqual(legacy[0]["text"], "兼容原流程")
            del legacy
            gc.collect()


class BlockParserTests(unittest.TestCase):
    def test_lossless_without_edits_and_deterministic_drop(self):
        text = "标题🙂\n\n第一段。\n\n\n第二段。"
        blocks = parse_blocks(text, "doc-1")
        self.assertEqual(render_blocks(text, blocks), text)
        self.assertEqual(
            render_blocks(text, blocks, {blocks[1].block_id}),
            "标题🙂\n\n第二段。",
        )
        self.assertEqual(
            [block.block_id for block in blocks],
            [block.block_id for block in parse_blocks(text, "doc-1")],
        )
        self.assertEqual(blocks[0].parser_version, BLOCK_PARSER_VERSION)
        self.assertEqual(blocks[0].end_cp, 3)

    def test_unknown_block_is_rejected(self):
        text = "一\n\n二"
        blocks = parse_blocks(text, "doc-1")
        with self.assertRaises(ValueError):
            render_blocks(text, blocks, {"not-a-block"})


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "curation.sqlite3"
        self.database = CurationDatabase(self.database_path)
        self.project_id = self.database.create_project(
            "test project",
            project_id="project-1",
        )
        candidates = [
            QueueCandidate("source", "revision", 1, "doc-1"),
            QueueCandidate("source", "revision", 2, "doc-2"),
        ]
        self.queue_id = self.database.create_queue(
            project_id=self.project_id,
            name="materialized test queue",
            sampling_policy={"type": "materialized_test"},
            sampling_seed=42,
            candidates=candidates,
            queue_id="queue-1",
        )

    def tearDown(self):
        self.database.close()
        self.temporary.cleanup()

    @staticmethod
    def review(doc_id: str, decision: str) -> DocumentReviewInput:
        return DocumentReviewInput(
            doc_id=doc_id,
            source_row=1 if doc_id == "doc-1" else 2,
            content_sha256=sha256_text(doc_id),
            decision=decision,
            quality=2,
            primary_category="news",
            flags=("boilerplate",),
            notes="note",
        )

    def test_full_dataset_queue_maps_ordinals_without_materialized_items(self):
        queue_id = self.database.create_virtual_queue(
            project_id=self.project_id,
            name="full",
            policy={
                "type": "full_dataset",
                "sources": [
                    {
                        "source_id": "source-a",
                        "source_revision": "revision-a",
                        "row_count": 2,
                    },
                    {
                        "source_id": "source-b",
                        "source_revision": "revision-b",
                        "row_count": 3,
                    },
                ],
            },
        )
        self.assertEqual(self.database.get_queue(queue_id)["state_counts"], {"pending": 5, "done": 0})
        self.assertEqual(self.database.get_queue_item(queue_id, 1)["source_row"], 1)
        crossed = self.database.get_queue_item(queue_id, 2)
        self.assertEqual(crossed["source_id"], "source-b")
        self.assertEqual(crossed["source_row"], 0)
        self.assertEqual(self.database.get_queue_item(queue_id, 4)["source_row"], 2)
        with self.assertRaises(KeyError):
            self.database.get_queue_item(queue_id, 5)
        stored = self.database.connection.execute(
            "SELECT COUNT(*) AS count FROM queue_items WHERE queue_id = ?",
            (queue_id,),
        ).fetchone()["count"]
        self.assertEqual(stored, 0)

    def test_review_revision_snapshot_and_undo(self):
        first, first_event = self.database.set_document_review(
            project_id=self.project_id,
            review=self.review("doc-1", "keep"),
            expected_revision=0,
        )
        self.assertEqual(first["revision"], 1)
        self.assertEqual(
            self.database.get_queue_item(self.queue_id, 0)["state"],
            "done",
        )
        with self.assertRaises(RevisionConflictError):
            self.database.set_document_review(
                project_id=self.project_id,
                review=self.review("doc-1", "drop"),
                expected_revision=0,
            )

        second, second_event = self.database.set_document_review(
            project_id=self.project_id,
            review=self.review("doc-1", "drop"),
            expected_revision=1,
        )
        self.assertEqual(second["revision"], 2)
        at_first, _ = self.database.review_state_at(
            self.project_id,
            first_event,
        )
        at_second, _ = self.database.review_state_at(
            self.project_id,
            second_event,
        )
        self.assertEqual(at_first["doc-1"]["decision"], "keep")
        self.assertEqual(at_second["doc-1"]["decision"], "drop")

        undo_event = self.database.undo_event(
            project_id=self.project_id,
            event_seq=second_event,
        )
        restored = self.database.get_document_review(self.project_id, "doc-1")
        self.assertIsNotNone(restored)
        self.assertEqual(restored["decision"], "keep")
        self.assertEqual(restored["revision"], 3)
        at_undo, _ = self.database.review_state_at(
            self.project_id,
            undo_event,
        )
        self.assertEqual(at_undo["doc-1"]["decision"], "keep")
        with self.assertRaises(UndoConflictError):
            self.database.undo_event(
                project_id=self.project_id,
                event_seq=first_event,
            )

    def test_undo_initial_review_returns_queue_to_pending(self):
        _, event_seq = self.database.set_document_review(
            project_id=self.project_id,
            review=self.review("doc-2", "drop"),
            expected_revision=0,
        )
        self.database.undo_event(
            project_id=self.project_id,
            event_seq=event_seq,
        )
        self.assertIsNone(
            self.database.get_document_review(self.project_id, "doc-2")
        )
        self.assertEqual(
            self.database.get_queue_item(self.queue_id, 1)["state"],
            "pending",
        )

    def test_block_review_and_undo(self):
        review, event_seq = self.database.set_block_review(
            project_id=self.project_id,
            review=BlockReviewInput(
                doc_id="doc-1",
                block_id="block-1",
                base_block_hash="hash",
                decision="drop",
                reason="boilerplate",
            ),
            expected_revision=0,
        )
        self.assertEqual(review["decision"], "drop")
        self.database.undo_event(
            project_id=self.project_id,
            event_seq=event_seq,
        )
        self.assertIsNone(
            self.database.get_block_review(self.project_id, "block-1")
        )


if __name__ == "__main__":
    unittest.main()
