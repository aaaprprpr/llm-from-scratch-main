from __future__ import annotations

import tempfile
import unittest
import sqlite3
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from label.backend.api import create_app


class ApiTests(unittest.TestCase):
    def test_simplification_preserves_phrases_whitespace_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            client = TestClient(create_app(root))
            texts = [
                "乾坤、乾燥、乾隆、頭髮、發展、皇后、後來、重複、覆蓋",
                "  繁體\n\n數學\r\nx² + y² = 1\t🙂 https://example.com/a?q=1  ",
                "臺灣軟體與滑鼠", "已经是简体", "",
            ]
            converted = client.post("/api/text/simplify", json={"texts": texts}).json()
            self.assertEqual(converted["texts"], [
                "乾坤、干燥、乾隆、头发、发展、皇后、后来、重复、覆盖",
                "  繁体\n\n数学\r\nx² + y² = 1\t🙂 https://example.com/a?q=1  ",
                "台湾软体与滑鼠", "已经是简体", "",
            ])
            second = client.post("/api/text/simplify", json={"texts": converted["texts"]}).json()
            self.assertEqual(second["texts"], converted["texts"])
            self.assertEqual(second["changed_characters"], 0)

    def test_document_load_uses_simplified_baseline_without_overwriting_reviews(self):
        with tempfile.TemporaryDirectory() as root:
            app = create_app(root)
            client = TestClient(app)
            source = Path(root) / "traditional.jsonl"
            raw_text = "數學研究數量。\n\n點擊廣告。\n\n頭髮與乾坤。"
            source.write_text(json.dumps({"text": raw_text}), encoding="utf-8")
            imported = client.post("/api/imports", json={
                "adapter": "jsonl", "path": str(source), "mapping": {"text_fields": ["text"]},
            }).json()
            prepared = client.post("/api/prepares", json={"source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "简体基准"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={"prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "全部"}).json()["queue_id"]
            url = f"/api/queues/{queue}/items/0"
            revision_before = client.get(f"/api/projects/{project}").json()["project"]["current_revision"]
            initial = client.get(url).json()
            simplified_raw = "数学研究数量。\n\n点击广告。\n\n头发与乾坤。"
            self.assertEqual(initial["simplified"]["raw_text"], simplified_raw)
            self.assertEqual(initial["simplified"]["materialized_text"], simplified_raw)
            self.assertEqual(initial["simplified"]["block_texts"], ["数学研究数量。", "点击广告。", "头发与乾坤。"])
            self.assertEqual(initial["raw_text"], raw_text)
            self.assertIsNone(initial["document_review"])
            self.assertEqual(client.get(f"/api/projects/{project}").json()["project"]["current_revision"], revision_before)
            # Existing reviews may still contain traditional text; loading must
            # preserve their deletions and revision while simplifying the view.
            saved = client.put(f"/api/reviews/documents/{initial['document']['doc_id']}", json={
                "queue_id": queue, "ordinal": 0, "expected_revision": 0, "decision": "keep",
                "edited_text": "數學研究數量。\n\n頭髮與乾坤。",
            }).json()
            reloaded = client.get(url).json()
            self.assertEqual(reloaded["simplified"]["materialized_text"], "数学研究数量。\n\n头发与乾坤。")
            self.assertEqual(reloaded["simplified"]["raw_text"], simplified_raw)
            self.assertEqual(reloaded["document_review"], saved["review"])
            self.assertEqual(reloaded["document"]["content_sha256"], initial["document"]["content_sha256"])
            saved_simplified = client.put(f"/api/reviews/documents/{initial['document']['doc_id']}", json={
                "queue_id": queue, "ordinal": 0, "expected_revision": saved["review"]["revision"], "decision": "keep",
                "edited_text": reloaded["simplified"]["materialized_text"],
            })
            self.assertEqual(saved_simplified.status_code, 200, saved_simplified.text)
            final = client.get(url).json()
            self.assertEqual(final["materialized_text"], reloaded["simplified"]["materialized_text"])
            self.assertEqual(final["simplified"], reloaded["simplified"])
            client.close()
            app.state.dataset_repository.clear()

    def test_structured_import_mapping_reaches_preview_and_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            client = TestClient(create_app(root))
            path = Path(root) / "structured.jsonl"
            path.write_text('{"instruction":"问题","input":"上下文","output":"答案"}\n', encoding="utf-8")
            request = {"adapter": "jsonl", "path": str(path), "mapping": {"record_adapter": "instruction_input_output"}}
            preview = client.post("/api/imports/preview", json={**request, "limit": 3})
            self.assertEqual(preview.status_code, 200, preview.text)
            self.assertEqual(preview.json()[0]["text"], "问题\n\n上下文\n\n答案")
            imported = client.post("/api/imports", json=request)
            self.assertEqual(imported.status_code, 200, imported.text)
            self.assertEqual(imported.json()["source_manifest"]["mapping"]["record_adapter"], "instruction_input_output")
            request["mapping"]["record_adapter"] = "missing"
            self.assertEqual(client.post("/api/imports/preview", json=request).status_code, 422)

    def test_health_project_and_error_contract(self):
        with tempfile.TemporaryDirectory() as root:
            app = create_app(root)
            client = TestClient(app)
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")
            simplified = client.post(
                "/api/text/simplify",
                json={"texts": ["數學與軟體", "已经是简体"]},
            )
            self.assertEqual(simplified.status_code, 200)
            self.assertEqual(simplified.json()["texts"], ["数学与软体", "已经是简体"])
            self.assertGreater(simplified.json()["changed_characters"], 0)
            created = client.post(
                "/api/projects",
                json={"name": "API test", "project_id": "api-project"},
            )
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["project_id"], "api-project")
            listing = client.get("/api/projects")
            self.assertEqual(len(listing.json()), 1)

            project = client.get("/api/projects/api-project")
            self.assertEqual(project.status_code, 200)
            self.assertEqual(project.json()["sources"], [])
            self.assertEqual(project.json()["queues"], [])

            duplicate = client.post(
                "/api/projects",
                json={"name": "duplicate", "project_id": "api-project"},
            )
            self.assertEqual(duplicate.status_code, 409)
            missing = client.get("/api/projects/missing")
            self.assertEqual(missing.status_code, 404)

            source_path = Path(root) / "source.jsonl"
            source_path.write_text(
                '{"id":"1","title":"标题","text":"正文"}\n'
                '{"id":"2","title":"标题二","text":"正文二"}\n',
                encoding="utf-8",
            )
            with patch(
                "label.backend.routes.system.select_local_path",
                return_value=str(source_path.parent),
            ):
                selected = client.post(
                    "/api/system/select-path",
                    json={"kind": "directory", "adapter": "huggingface_local"},
                )
            self.assertEqual(selected.status_code, 200)
            self.assertEqual(selected.json()["path"], str(source_path.parent))

            source_request = {
                "adapter": "jsonl",
                "path": str(source_path),
                "options": {},
            }
            inspection = client.post("/api/imports/inspect", json=source_request)
            self.assertEqual(inspection.status_code, 200)
            self.assertIn("text", inspection.json()["fields"])
            mapping = {
                "text_fields": ["title", "text"],
                "title_field": "title",
                "local_id_field": "id",
            }
            preview = client.post(
                "/api/imports/preview",
                json={**source_request, "mapping": mapping, "limit": 2},
            )
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.json()[0]["text"], "标题\n\n正文")
            imported = client.post(
                "/api/imports",
                json={
                    **source_request,
                    "mapping": mapping,
                },
            )
            self.assertEqual(imported.status_code, 200)
            self.assertRegex(
                imported.json()["source_manifest"]["source_id"],
                r"^source-[0-9a-f]{8}$",
            )
            self.assertEqual(imported.json()["source_manifest"]["license"], "unknown")
            prepared = client.post(
                "/api/prepares",
                json={
                    "source_revision_directory": imported.json()[
                        "revision_directory"
                    ],
                },
            )
            self.assertEqual(prepared.status_code, 200)
            catalog = client.get("/api/catalog")
            self.assertEqual(len(catalog.json()), 1)
            self.assertEqual(len(catalog.json()[0]["prepares"]), 1)

            attached = client.post(
                "/api/projects/api-project/sources",
                json={
                    "prepare_revision_directory": prepared.json()[
                        "revision_directory"
                    ]
                },
            )
            self.assertEqual(attached.status_code, 200)
            queue = client.post(
                "/api/projects/api-project/queues",
                json={
                    "name": "Full cleaning",
                    "policy": "full_dataset",
                },
            )
            self.assertEqual(queue.status_code, 200)
            self.assertEqual(queue.json()["state_counts"]["pending"], 2)
            self.assertEqual(queue.json()["sampling_policy"]["type"], "full_dataset")
            connection = sqlite3.connect(Path(root) / "curation.sqlite3")
            try:
                stored_items = connection.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE queue_id = ?",
                    (queue.json()["queue_id"],),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(stored_items, 0)
            document = client.get(
                f"/api/queues/{queue.json()['queue_id']}/items/0"
            )
            self.assertEqual(document.status_code, 200)
            self.assertIsNotNone(document.json()["token_counts"])
            cleaning_request = {
                "queue_id": queue.json()["queue_id"], "ordinal": 0,
                "expected_revision": 0,
                "content_sha256": document.json()["document"]["content_sha256"],
                "blocks": [{"id": "draft-1", "text": "尚未保存的编辑正文", "separator_after": ""}],
            }
            cleaning_url = f"/api/reviews/documents/{document.json()['document']['doc_id']}/llm-clean"
            with patch.object(app.state.api_context.llm_cleaner, "clean", return_value={"suggestion_id": "test"}) as cleaner:
                suggestion = client.post(cleaning_url, json=cleaning_request)
                self.assertEqual(suggestion.status_code, 200, suggestion.text)
                self.assertEqual(cleaner.call_args.args[0], cleaning_request["blocks"])
                self.assertEqual(client.post(cleaning_url, json={**cleaning_request, "expected_revision": 7}).status_code, 422)
                self.assertEqual(client.post(cleaning_url, json={**cleaning_request, "content_sha256": "stale"}).status_code, 422)
                self.assertEqual(cleaner.call_count, 1)
            after_suggestion = client.get(f"/api/queues/{queue.json()['queue_id']}/items/0").json()
            self.assertIsNone(after_suggestion["document_review"])
            self.assertEqual(after_suggestion["materialized_text"], document.json()["materialized_text"])
            self.assertEqual(after_suggestion["queue"]["state_counts"]["pending"], 2)
            saved = client.put(
                f"/api/reviews/documents/{document.json()['document']['doc_id']}",
                json={
                    "queue_id": queue.json()["queue_id"],
                    "ordinal": 0,
                    "expected_revision": 0,
                    "decision": "keep",
                    "edited_text": "人工修改正文",
                },
            )
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json()["review"]["edited_text"], "人工修改正文")
            edited_document = client.get(
                f"/api/queues/{queue.json()['queue_id']}/items/0"
            )
            self.assertEqual(
                edited_document.json()["materialized_text"],
                "人工修改正文",
            )
            refreshed_queue = client.get("/api/projects/api-project").json()["queues"][0]
            self.assertEqual(refreshed_queue["state_counts"]["done"], 1)
            self.assertEqual(refreshed_queue["state_counts"]["pending"], 1)
            client.close()
            app.state.dataset_repository.clear()


if __name__ == "__main__":
    unittest.main()
