from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.label.backend.source_catalog import infer_mapping, list_download_sources


class SourceCatalogTests(unittest.TestCase):
    def test_current_downloads_have_usable_mapping(self):
        sources = {item["source_id"]: item for item in list_download_sources()}
        for source_id in (
            "minimind", "fineweb_edu_zh_4_5", "finewiki_zh",
            "wiki_zh_20231101", "tigerresearch_pretrain_zh",
        ):
            with self.subTest(source=source_id):
                source = sources[source_id]
                self.assertTrue(source["available"])
                self.assertTrue(source["ready"])
                self.assertTrue(source["mapping"]["text_fields"] or source["mapping"]["record_adapter"])
        self.assertEqual(sources["wiki_zh_20231101"]["mapping"]["text_fields"], ["text"])
        self.assertEqual(sources["tigerresearch_pretrain_zh"]["mapping"]["text_fields"], ["title", "content"])

    def test_new_jsonl_source_uses_project_path_and_infers_text(self):
        with tempfile.TemporaryDirectory() as root:
            project = Path(root)
            source = project / "dataset" / "downloads" / "example.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text('{"id":"1","text":"正文一"}\n{"id":"2","text":"正文二"}\n', encoding="utf-8")
            config = project / "config.json"
            config.write_text(json.dumps({"sources": {"example": {
                "kind": "jsonl", "path": "dataset/downloads/example.jsonl",
            }}}), encoding="utf-8")
            catalog = list_download_sources(config, project, project / "missing-samples")
            self.assertTrue(catalog[0]["ready"])
            self.assertEqual(catalog[0]["mapping"]["text_fields"], ["text"])
            self.assertEqual(catalog[0]["mapping"]["local_id_field"], "id")
            hidden = list_download_sources(
                config, project, project / "missing-samples",
                imported_locations=[source.parent / "." / source.name],
            )
            self.assertEqual(hidden, [])

    def test_structured_record_adapter_inference(self):
        examples = [
            (["instruction", "input", "output"], "instruction_input_output"),
            (["question", "answer"], "question_answer"),
            (["query", "response", "reasoning"], "question_answer_optional_think"),
            (["conversations", "conversations.0.value"], "sharegpt_conversations"),
            (["messages", "messages.0.content"], "openai_role_content_conversation"),
            (["标题", "楼主内容", "回复列表"], "tieba_thread"),
        ]
        for fields, adapter in examples:
            with self.subTest(fields=fields):
                self.assertEqual(infer_mapping(fields).record_adapter, adapter)
