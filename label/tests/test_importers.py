from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from label.backend.dataset_store import load_dataset
from label.backend.import_service import ImportService
from label.backend.importers.base import SourceAdapter, SourceSpec
from label.backend.importers.base import json_compatible
from label.backend.importers.huggingface import HuggingFaceLocalAdapter
from label.backend.importers.jsonl import JsonlAdapter
from label.backend.importers.text import TextAdapter
from label.backend.schema import FieldMapping, ImportedDocument, Inspection


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_jsonl_inspect_preview_and_nested_mapping(self):
        path = self.root / "records.jsonl"
        records = [
            {
                "id": "a",
                "title": "标题一",
                "payload": {"body": "正文一"},
                "url": "https://example.com/1",
            },
            {
                "id": "b",
                "title": "标题二",
                "payload": {"body": "正文二"},
                "url": "https://example.com/2",
            },
        ]
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
            encoding="utf-8",
        )
        adapter = JsonlAdapter()
        spec = SourceSpec(path)
        inspection = adapter.inspect(spec)
        self.assertEqual(inspection.record_count, 2)
        self.assertIn("payload.body", inspection.fields)

        mapping = FieldMapping(
            text_fields=("title", "payload.body"),
            title_field="title",
            url_field="url",
            local_id_field="id",
            metadata_fields=("payload",),
        )
        documents = adapter.preview(spec, mapping, limit=2)
        self.assertEqual(documents[0].text, "标题一\n\n正文一")
        self.assertEqual(documents[0].source_local_id, "a")
        self.assertEqual(documents[0].stable_locator, "line:1:byte:0")
        self.assertEqual(documents[0].metadata["payload"], {"body": "正文一"})
        self.assertIsNone(json_compatible(float("nan")))

    def test_jsonl_skip_policy(self):
        path = self.root / "invalid.jsonl"
        path.write_text(
            '{"text":"valid"}\nnot-json\n{"missing":"text"}\n',
            encoding="utf-8",
        )
        adapter = JsonlAdapter()
        mapping = FieldMapping(text_fields=("text",))
        documents = list(
            adapter.iter_documents(
                SourceSpec(path, {"invalid_record_policy": "skip"}),
                mapping,
            )
        )
        self.assertEqual([document.text for document in documents], ["valid"])
        with self.assertRaises(ValueError):
            list(adapter.iter_documents(SourceSpec(path), mapping))

    def test_text_blank_line_and_line_strategies(self):
        path = self.root / "records.txt"
        path.write_text("第一行\n第二行\n\n第三段\n\n\n第四段", encoding="utf-8")
        adapter = TextAdapter()
        mapping = FieldMapping(text_fields=("text",))
        grouped = list(adapter.iter_documents(SourceSpec(path), mapping))
        self.assertEqual(
            [document.text for document in grouped],
            ["第一行\n第二行", "第三段", "第四段"],
        )
        lines = list(
            adapter.iter_documents(
                SourceSpec(path, {"document_strategy": "one_line_per_document"}),
                mapping,
            )
        )
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[1].stable_locator.startswith("line:2:byte:"))

    def test_import_service_and_hf_adapter_round_trip(self):
        path = self.root / "records.jsonl"
        path.write_text(
            '{"id":"1","title":"标题","text":"正文"}\n'
            '{"id":"2","title":"标题二","text":"正文二"}\n',
            encoding="utf-8",
        )
        mapping = FieldMapping(
            text_fields=("title", "text"),
            title_field="title",
            local_id_field="id",
        )
        service = ImportService(self.root / "managed")
        result = service.import_source(
            source_id="jsonl-source",
            source_license="test-only",
            adapter=JsonlAdapter(),
            spec=SourceSpec(path),
            mapping=mapping,
            max_shard_size="1MB",
        )
        self.assertFalse(result.reused_existing)
        self.assertEqual(result.source_manifest.record_count, 2)
        dataset_path = result.revision_directory / "raw"
        dataset = load_dataset(dataset_path)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["text"], "标题\n\n正文")
        self.assertEqual(dataset[0]["source_row"], 0)
        self.assertNotEqual(dataset[0]["doc_id"], dataset[1]["doc_id"])

        reused = service.import_source(
            source_id="jsonl-source",
            source_license="test-only",
            adapter=JsonlAdapter(),
            spec=SourceSpec(path),
            mapping=mapping,
        )
        self.assertTrue(reused.reused_existing)
        self.assertEqual(
            reused.source_manifest.source_revision,
            result.source_manifest.source_revision,
        )

        hf_adapter = HuggingFaceLocalAdapter()
        hf_spec = SourceSpec(dataset_path, {"read_batch_size": 1})
        inspection = hf_adapter.inspect(hf_spec)
        self.assertEqual(inspection.record_count, 2)
        preview = hf_adapter.preview(
            hf_spec,
            FieldMapping(
                text_fields=("text",),
                title_field="title",
                local_id_field="doc_id",
            ),
            limit=1,
        )
        self.assertEqual(preview[0].text, "标题\n\n正文")

    def test_failed_import_is_atomic(self):
        class FailingAdapter(SourceAdapter):
            adapter_name = "failing"
            adapter_version = "1"

            def inspect(self, spec):
                return Inspection(
                    source_type="failing",
                    input_location=str(spec.path),
                    input_fingerprint="fingerprint",
                    fields=("text",),
                    record_count=2,
                )

            def iter_documents(self, spec, mapping):
                yield ImportedDocument("row:0", "valid")
                raise RuntimeError("intentional failure")

        input_path = self.root / "placeholder"
        input_path.write_text("x", encoding="utf-8")
        storage = self.root / "managed"
        with self.assertRaises(RuntimeError):
            ImportService(storage).import_source(
                source_id="failed-source",
                source_license="unknown",
                adapter=FailingAdapter(),
                spec=SourceSpec(input_path),
                mapping=FieldMapping(text_fields=("text",)),
            )
        source_directory = storage / "sources" / "failed-source"
        self.assertEqual(list(source_directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
