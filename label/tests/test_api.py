from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
                    "source_id": "api-source",
                    "license": "test",
                    "mapping": mapping,
                },
            )
            self.assertEqual(imported.status_code, 200)
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
                    "name": "API queue",
                    "policy": "uniform_random",
                    "seed": 42,
                    "sample_size": 2,
                },
            )
            self.assertEqual(queue.status_code, 200)
            document = client.get(
                f"/api/queues/{queue.json()['queue_id']}/items/0"
            )
            self.assertEqual(document.status_code, 200)
            self.assertIsNotNone(document.json()["token_counts"])
            client.close()
            app.state.dataset_repository.clear()


if __name__ == "__main__":
    unittest.main()
