from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from dataset.label.backend.api import create_app
from dataset.label.backend.database import CurationDatabase


class ReviewPositionTests(unittest.TestCase):
    def test_saved_queue_position_survives_reopening_and_rejects_invalid_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with CurationDatabase(root / "curation.sqlite3") as database:
                project_id = database.create_project("test")
                queue_id = database.create_virtual_queue(
                    project_id=project_id, name="test queue",
                    policy={"type": "full_dataset", "sources": [{
                        "source_id": "source", "source_revision": "revision", "row_count": 3,
                    }]},
                )
            with TestClient(create_app(root)) as client:
                url = f"/api/queues/{queue_id}/position"
                self.assertEqual(client.get(url).json(), {"ordinal": None})
                self.assertEqual(client.put(url, json={"ordinal": 2}).json(), {"ordinal": 2})
                self.assertEqual(client.put(url, json={"ordinal": 3}).status_code, 404)
                self.assertEqual(client.get(url).json(), {"ordinal": 2})
            with TestClient(create_app(root)) as client:
                self.assertEqual(client.get(url).json(), {"ordinal": 2})
