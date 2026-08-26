from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from label.backend.dataset_store import load_dataset
from label.backend.blocks import parse_blocks
from label.backend.database import (
    BlockReviewInput,
    CurationDatabase,
    DocumentReviewInput,
)
from label.backend.documents import DocumentService
from label.backend.import_service import ImportService
from label.backend.importers.base import SourceSpec
from label.backend.importers.jsonl import JsonlAdapter
from label.backend.materialize import MaterializeService
from label.backend.prepare import PrepareConfig, PrepareService
from label.backend.queueing import (
    create_uniform_random_queue,
    open_project_sources,
    uniform_random_candidates,
)
from label.backend.schema import FieldMapping
from data_pipeline.build_bin import build_bins, tokenizer_fingerprint
from tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        input_path = self.root / "input.jsonl"
        input_path.write_text(
            '{"id":"0","text":"第一段\\r\\n\\r\\n第二段"}\n'
            '{"id":"1","text":"应当丢弃"}\n'
            '{"id":"2","text":"尚不确定"}\n'
            '{"id":"3","text":"还没有人工审核"}\n',
            encoding="utf-8",
        )
        imported = ImportService(self.root / "managed").import_source(
            source_id="source-a",
            source_license="test",
            adapter=JsonlAdapter(),
            spec=SourceSpec(input_path),
            mapping=FieldMapping(text_fields=("text",), local_id_field="id"),
            max_shard_size="1MB",
        )
        self.imported = imported
        self.prepared = PrepareService().prepare_source(
            imported.revision_directory,
            config=PrepareConfig(),
            max_shard_size="1MB",
        )
        self.database_path = self.root / "project.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def _open_database(self):
        database = CurationDatabase(self.database_path)
        project_id = database.create_project("test project", project_id="project-a")
        database.attach_source(
            project_id=project_id,
            source_id=self.prepared.manifest.source_id,
            source_revision=self.prepared.manifest.source_revision,
            dataset_path=(
                self.prepared.revision_directory / self.prepared.manifest.dataset_path
            ),
            manifest_path=self.prepared.revision_directory / "manifest.json",
            row_count=self.prepared.manifest.output_records,
        )
        return database, project_id

    def test_prepare_is_one_to_one_normalized_and_idempotent(self):
        dataset = load_dataset(
            self.prepared.revision_directory / self.prepared.manifest.dataset_path
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset[0]["text"], "第一段\n\n第二段")
        self.assertEqual([row for row in dataset["source_row"]], [0, 1, 2, 3])
        reused = PrepareService().prepare_source(self.imported.revision_directory)
        self.assertTrue(reused.reused_existing)
        self.assertEqual(
            reused.manifest.content_sequence_sha256,
            self.prepared.manifest.content_sequence_sha256,
        )

    def test_seeded_queue_review_and_materialization_policies(self):
        database, project_id = self._open_database()
        try:
            sources = open_project_sources(database, project_id)
            first = uniform_random_candidates(sources, sample_size=4, seed=42)
            second = uniform_random_candidates(sources, sample_size=4, seed=42)
            self.assertEqual(first, second)
            queue_id = create_uniform_random_queue(
                database,
                project_id=project_id,
                name="random audit",
                sample_size=4,
                seed=42,
                queue_id="queue-a",
            )
            self.assertEqual(sum(database.get_queue(queue_id)["state_counts"].values()), 4)

            dataset = sources[0].dataset
            event_sequences = []
            for row_index, decision in ((0, "keep"), (1, "drop"), (2, "unsure")):
                row = dataset[row_index]
                _, event_seq = database.set_document_review(
                    project_id=project_id,
                    review=DocumentReviewInput(
                        doc_id=row["doc_id"],
                        source_row=row_index,
                        content_sha256=row["content_sha256"],
                        decision=decision,
                        quality=3 if decision == "keep" else 1,
                        primary_category="encyclopedia",
                        flags=("bad_format",) if decision == "unsure" else (),
                    ),
                    expected_revision=0,
                )
                event_sequences.append(event_seq)

            kept_row = dataset[0]
            blocks = parse_blocks(kept_row["text"], kept_row["doc_id"])
            _, block_event = database.set_block_review(
                project_id=project_id,
                review=BlockReviewInput(
                    doc_id=kept_row["doc_id"],
                    block_id=blocks[1].block_id,
                    base_block_hash=blocks[1].content_sha256,
                    decision="drop",
                    reason="test",
                ),
                expected_revision=0,
            )
            self.assertGreater(block_event, event_sequences[-1])

            queue_ordinal = next(
                ordinal
                for ordinal in range(4)
                if database.get_queue_item(queue_id, ordinal)["doc_id"]
                == kept_row["doc_id"]
            )
            document = DocumentService(database).queue_document(
                queue_id,
                queue_ordinal,
            )
            self.assertEqual(document["raw_text"], "第一段\r\n\r\n第二段")
            self.assertEqual(document["review_text"], "第一段\n\n第二段")
            self.assertEqual(document["materialized_text"], "第一段")
            self.assertEqual(document["provenance"]["license"], "test")

            service = MaterializeService(self.root / "exports")
            keep_only = service.materialize(
                database,
                project_id=project_id,
                policy="keep_only",
                snapshot_event_seq=block_event,
                max_shard_size="1MB",
            )
            kept_dataset = load_dataset(
                keep_only.output_directory / keep_only.manifest.dataset_path
            )
            self.assertEqual(len(kept_dataset), 1)
            self.assertEqual(kept_dataset[0]["text"], "第一段")
            self.assertEqual(keep_only.manifest.block_drop_count, 1)

            dropped = service.materialize(
                database,
                project_id=project_id,
                policy="drop_rejected",
                snapshot_event_seq=block_event,
                max_shard_size="1MB",
            )
            dropped_dataset = load_dataset(
                dropped.output_directory / dropped.manifest.dataset_path
            )
            self.assertEqual(len(dropped_dataset), 3)
            self.assertEqual(
                dropped.manifest.decision_counts,
                {"drop": 1, "keep": 1, "unreviewed": 1, "unsure": 1},
            )

            reused = service.materialize(
                database,
                project_id=project_id,
                policy="keep_only",
                snapshot_event_seq=block_event,
            )
            self.assertTrue(reused.reused_existing)
            self.assertEqual(
                reused.manifest.content_sequence_sha256,
                keep_only.manifest.content_sequence_sha256,
            )

            tokenizer_path = PROJECT_ROOT / "bpe" / "tokenizer_24576"
            tokenizer = Tokenizer(str(tokenizer_path))
            eos_id = tokenizer.special_token_to_id["<|endoftext|>"]
            train_bin = self.root / "train.bin"
            validation_bin = self.root / "val.bin"
            train_metadata, validation_metadata = build_bins(
                dataset=dropped_dataset,
                input_dataset=dropped.output_directory / dropped.manifest.dataset_path,
                train_bin=train_bin,
                validation_bin=validation_bin,
                tokenizer_path=tokenizer_path,
                tokenizer_size=len(tokenizer.tokenizer),
                tokenizer_sha256=tokenizer_fingerprint(tokenizer_path),
                eos_token="<|endoftext|>",
                eos_id=eos_id,
                dtype=np.dtype("uint16"),
                train_ratio=2 / 3,
                seed=42,
                shuffle_block_records=2,
                batch_records=2,
                workers=1,
                tokenizer_threads=1,
                overwrite=True,
            )
            train_tokens = np.fromfile(train_bin, dtype=np.uint16)
            validation_tokens = np.fromfile(validation_bin, dtype=np.uint16)
            eos_count = int(np.count_nonzero(train_tokens == eos_id)) + int(
                np.count_nonzero(validation_tokens == eos_id)
            )
            self.assertEqual(eos_count, 3)
            self.assertEqual(
                train_metadata["records"] + validation_metadata["records"],
                3,
            )
        finally:
            database.close()


if __name__ == "__main__":
    unittest.main()
