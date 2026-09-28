from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from dataset.label.backend.api import create_app
from dataset.label.backend.batch_clean import clean_queue, output_path
from dataset.label.backend.model_settings import ModelSelection, ModelSettings


class ModelSettingsTests(unittest.TestCase):
    def test_single_and_batch_choices_persist_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            initial = client.get("/api/settings/models")
            self.assertEqual(initial.status_code, 200)
            self.assertEqual((initial.json()["single"], initial.json()["batch"]), ("deepseek", "local"))
            saved = client.put("/api/settings/models", json={"single": "qwen_api", "batch": "local"})
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertEqual(app.state.api_context.llm_cleaner.config.provider, "dashscope")
            self.assertEqual(app.state.api_context.batch_jobs.config.provider, "llamacpp")
            self.assertEqual(ModelSettings(root).read(), ModelSelection("qwen_api", "local"))
            self.assertEqual(create_app(root).state.api_context.llm_cleaner.config.provider, "dashscope")
            self.assertNotIn("api_key", saved.text)
            self.assertEqual(client.put("/api/settings/models", json={"single": "invalid", "batch": "local"}).status_code, 422)
            client.close()

    def test_cannot_switch_batch_model_while_job_is_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            manager = app.state.api_context.batch_jobs
            active = SimpleNamespace(is_alive=lambda: True)
            with patch.object(manager, "_thread", active):
                result = TestClient(app).put("/api/settings/models", json={"single": "local", "batch": "qwen_api"})
            self.assertEqual(result.status_code, 409)
            self.assertFalse((root / "model_settings.json").exists())
            self.assertEqual(app.state.api_context.llm_cleaner.config.provider, "deepseek")

    def test_existing_progress_path_survives_model_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = ModelSettings(root)
            original = output_path(root, "queue", settings.config("deepseek"))
            original.mkdir(parents=True)
            (original / "progress.jsonl").write_text('{"ordinal":0}\n', encoding="utf-8")
            (original / "manifest.json").write_text(json.dumps({"processed": 1}), encoding="utf-8")
            self.assertEqual(output_path(root, "queue", settings.config("local")), original)

    def test_batch_resume_uses_new_model_without_reprocessing_completed_items(self):
        class FakeCleaner:
            def __init__(self, config):
                self.config = config

            @staticmethod
            def is_complete(_result):
                return True

            def clean(self, blocks, *, title, provenance):
                return {"decision": "keep", "edited_text": f"{self.config.model}:" + blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = TestClient(create_app(root))
            source = root / "source.jsonl"
            source.write_text('{"text":"第一条。"}\n{"text":"第二条。"}\n', encoding="utf-8")
            imported = client.post("/api/imports", json={
                "adapter": "jsonl", "path": str(source), "mapping": {"text_fields": ["text"]},
            }).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"],
            }).json()
            project = client.post("/api/projects", json={"name": "切换模型"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"],
            })
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "全部"}).json()["queue_id"]
            settings = ModelSettings(root)
            deepseek = FakeCleaner(settings.config("deepseek"))
            output = output_path(root, queue, deepseek.config)
            first = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                output_directory=output, cleaner=deepseek, limit=1)
            self.assertEqual(first["processed"], 1)
            local = FakeCleaner(settings.config("local"))
            self.assertEqual(output_path(root, queue, local.config), output)
            second = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=output, cleaner=local, limit=1)
            self.assertEqual(second["processed"], 2)
            rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([row["ordinal"] for row in rows], [0, 1])
            self.assertEqual([row["model"] for row in rows], [deepseek.config.model, "qwen-local"])
            self.assertEqual(len(json.loads((output / "job.json").read_text())["model_history"]), 2)
            client.close()

    def test_tool_startup_checks_local_model_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("dataset.label.backend.local_model.LocalModelService.ensure_running") as ensure, \
                 patch("dataset.label.backend.local_model.LocalModelService.close") as close:
                with TestClient(create_app(temporary, start_local_model=True)) as client:
                    self.assertEqual(client.get("/api/health").status_code, 200)
                ensure.assert_called_once_with()
                close.assert_called_once_with()
