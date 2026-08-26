from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from label.backend.api import create_app


class ApiTests(unittest.TestCase):
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
