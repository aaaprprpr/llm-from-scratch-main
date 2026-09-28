from __future__ import annotations

import tempfile
import unittest
import sqlite3
import json
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from dataset.label.backend.api import create_app
from dataset.label.backend.llm_cleaning import LlmCleaningError


class ApiTests(unittest.TestCase):
    def test_default_prepare_simplifies_before_review_without_changing_raw(self):
        with tempfile.TemporaryDirectory() as root:
            app = create_app(root)
            client = TestClient(app)
            source = Path(root) / "traditional.jsonl"
            raw_text = "數學研究數量。\n\n點擊廣告。\n\n頭髮與乾坤。"
            source.write_text(json.dumps({"text": raw_text}), encoding="utf-8")
            imported = client.post("/api/imports", json={
                "adapter": "jsonl", "path": str(source), "mapping": {"text_fields": ["text"]},
            }).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"],
                "progress_id": "test-conversion",
            }).json()
            self.assertTrue(prepared["manifest"]["config"]["simplify_chinese"])
            progress = client.get("/api/prepares/progress/test-conversion").json()
            self.assertEqual(progress, {"processed": 1, "total": 1, "stage": "done"})
            project = client.post("/api/projects", json={"name": "简体基准"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={"prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "全部"}).json()["queue_id"]
            url = f"/api/queues/{queue}/items/0"
            initial = client.get(url).json()
            simplified = "数学研究数量。\n\n点击广告。\n\n头发与乾坤。"
            self.assertEqual(initial["raw_text"], raw_text)
            self.assertEqual(initial["review_text"], simplified)
            self.assertEqual(initial["materialized_text"], simplified)
            self.assertEqual([block["text"] for block in initial["blocks"]],
                             ["数学研究数量。", "点击广告。", "头发与乾坤。"])
            self.assertNotIn("simplified", initial)
            self.assertEqual(client.post("/api/text/simplify", json={"texts": [raw_text]}).status_code, 405)
            client.close()
            app.state.dataset_repository.clear()

    def test_prepared_simplified_document_is_not_converted_twice(self):
        with tempfile.TemporaryDirectory() as root:
            app = create_app(root)
            client = TestClient(app)
            source = Path(root) / "traditional.jsonl"
            source.write_text(
                json.dumps({"title": "學堂", "text": "學堂隸屬於倫敦傳道會"}) + "\n",
                encoding="utf-8",
            )
            imported = client.post("/api/imports", json={
                "adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"], "title_field": "title"},
            }).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"],
                "config": {"simplify_chinese": True},
            }).json()
            project = client.post("/api/projects", json={"name": "简体快照"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"],
            })
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "全部"}).json()["queue_id"]
            document = client.get(f"/api/queues/{queue}/items/0").json()
            once = "学堂隶属於伦敦传道会"
            self.assertEqual(document["review_text"], once)
            self.assertEqual(document["materialized_text"], once)
            self.assertEqual([block["text"] for block in document["blocks"]], [once])
            self.assertEqual(document["provenance"]["title"], "学堂")
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
            self.assertNotIn("token_counts", document.json())
            cleaning_request = {
                "queue_id": queue.json()["queue_id"], "ordinal": 0,
                "expected_revision": 0,
                "content_sha256": document.json()["document"]["content_sha256"],
                "blocks": [{"id": "draft-1", "text": "尚未保存的编辑正文", "separator_after": ""}],
            }
            cleaning_url = f"/api/reviews/documents/{document.json()['document']['doc_id']}/llm-clean"
            with patch.object(app.state.api_context.llm_cleaner, "clean", return_value={
                "suggestion_id": "test", "decision": "keep", "quality": 2,
                "category": "encyclopedia", "edited_text": "清理后的正文",
                "assessments": [{"chunk": 1, "decision": "keep"}],
            }) as cleaner:
                cleaned = client.post(cleaning_url, json=cleaning_request)
                self.assertEqual(cleaned.status_code, 200, cleaned.text)
                self.assertTrue(cleaned.json()["saved"])
                self.assertEqual(cleaner.call_args.args[0], cleaning_request["blocks"])
                self.assertEqual(client.post(cleaning_url, json={**cleaning_request, "expected_revision": 7}).status_code, 422)
                self.assertEqual(client.post(cleaning_url, json={**cleaning_request, "content_sha256": "stale"}).status_code, 422)
                self.assertEqual(cleaner.call_count, 1)
            after_clean = client.get(f"/api/queues/{queue.json()['queue_id']}/items/0").json()
            self.assertEqual(after_clean["document_review"]["edited_text"], "清理后的正文")
            self.assertEqual(after_clean["materialized_text"], "清理后的正文")
            self.assertEqual(after_clean["queue"]["state_counts"]["done"], 1)
            exported = client.post("/api/materializations", json={
                "project_id": "api-project", "policy": "keep_only",
            })
            self.assertEqual(exported.status_code, 200, exported.text)
            from dataset.label.backend.dataset_store import load_dataset
            rows = load_dataset(Path(exported.json()["output_directory"]) / "dataset")
            self.assertEqual(list(rows["text"]), ["清理后的正文"])
            with patch.object(app.state.api_context.llm_cleaner, "clean", return_value={
                "decision": "unsure", "assessments": [{"chunk": 1, "fallback": True}],
            }):
                incomplete = client.post(cleaning_url, json={**cleaning_request, "expected_revision": 1})
            self.assertFalse(incomplete.json()["saved"])
            self.assertEqual(client.get(f"/api/queues/{queue.json()['queue_id']}/items/0").json()["document_review"]["revision"], 1)
            saved = client.put(
                f"/api/reviews/documents/{document.json()['document']['doc_id']}",
                json={
                    "queue_id": queue.json()["queue_id"],
                    "ordinal": 0,
                    "expected_revision": 1,
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
            with patch.object(app.state.api_context.llm_cleaner, "clean", return_value={
                "decision": "drop", "quality": 0, "category": "other", "edited_text": "",
                "assessments": [{"chunk": 1, "decision": "drop"}],
            }):
                dropped = client.post(cleaning_url, json={**cleaning_request, "expected_revision": 2})
            self.assertEqual(dropped.status_code, 200, dropped.text)
            self.assertTrue(dropped.json()["saved"])
            after_drop = client.get(f"/api/queues/{queue.json()['queue_id']}/items/0").json()
            self.assertEqual(after_drop["document_review"]["decision"], "drop")
            self.assertEqual(after_drop["materialized_text"], "")
            second = client.get(f"/api/queues/{queue.json()['queue_id']}/items/1").json()
            second_id = second["document"]["doc_id"]
            second_request = {
                "queue_id": queue.json()["queue_id"], "ordinal": 1,
                "expected_revision": 0,
                "content_sha256": second["document"]["content_sha256"],
                "blocks": [{"id": "second", "text": second["review_text"], "separator_after": ""}],
            }
            with patch.object(app.state.api_context.llm_cleaner, "clean",
                              side_effect=LlmCleaningError("Content Exists Risk")):
                rejected = client.post(f"/api/reviews/documents/{second_id}/llm-clean",
                                       json=second_request)
            self.assertEqual(rejected.status_code, 502)
            risk_items = client.get(f"/api/queues/{queue.json()['queue_id']}/statuses",
                                    params={"status": "risk"}).json()["items"]
            self.assertEqual([item["ordinal"] for item in risk_items], [1])
            client.put(f"/api/reviews/documents/{second_id}", json={
                "queue_id": queue.json()["queue_id"], "ordinal": 1,
                "expected_revision": 0, "decision": "drop", "edited_text": None,
            })
            self.assertEqual(client.get(f"/api/queues/{queue.json()['queue_id']}/statuses",
                                        params={"status": "risk"}).json()["items"], [])
            client.close()
            app.state.dataset_repository.clear()


if __name__ == "__main__":
    unittest.main()
