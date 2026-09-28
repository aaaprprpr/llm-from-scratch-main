from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dataset.label.backend.api import create_app
from dataset.label.backend.batch_clean import _process, clean_queue, output_path
from dataset.label.backend.database import CurationDatabase, DocumentReviewInput
from dataset.label.backend.cleaning_progress import CleaningProgressIndex
from dataset.label.backend.llm_cleaning import LlmCleaningError


class FakeCleaner:
    def __init__(self):
        self.config = SimpleNamespace(provider="llamacpp", model="fake")
        self.calls = []
        self.recover = False

    @staticmethod
    def is_complete(result):
        return result["decision"] != "unsure"

    def clean(self, blocks, *, title, provenance):
        self.calls.append(provenance["ordinal"])
        if provenance["ordinal"] == 0:
            return {"decision": "keep", "edited_text": "清理后的正文。"}
        if self.recover:
            return {"decision": "keep", "edited_text": "第二条清理后的正文。"}
        return {"decision": "unsure", "edited_text": "仍有垃圾。"}


class BatchCleaningTests(unittest.TestCase):
    def test_batch_writes_only_complete_text_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps({"text": "正文。广告。"}, ensure_ascii=False) + "\n"
                + json.dumps({"text": "仍有垃圾。"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            imported = client.post("/api/imports", json={
                "adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]},
            }).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"],
            }).json()
            project = client.post("/api/projects", json={"name": "batch"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"],
            })
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            cleaner = FakeCleaner()
            output = root / "batch"
            first = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                output_directory=output, limit=1, cleaner=cleaner)
            self.assertEqual(first["counts"], {"keep": 1})
            job_path = output / "job.json"
            old_job = json.loads(job_path.read_text())
            old_job["prompt_version"] = "older_prompt"
            old_job.pop("prompt_versions")
            job_path.write_text(json.dumps(old_job))
            second = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=output, limit=1, cleaner=cleaner)
            self.assertEqual(second["counts"], {"keep": 1, "incomplete": 1})
            self.assertIn("older_prompt", json.loads(job_path.read_text())["prompt_versions"])
            self.assertTrue(second["queue_exhausted"])
            self.assertFalse(second["complete"])
            self.assertEqual(cleaner.calls, [0, 1])
            cleaned = [json.loads(line) for line in (output / "cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(cleaned, [{"text": "清理后的正文。"}])
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([item["ordinal"] for item in progress], [0, 1])
            cleaner.recover = True
            third = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                output_directory=output, limit=1, cleaner=cleaner)
            self.assertEqual(third["counts"], {"keep": 2})
            self.assertTrue(third["complete"])
            self.assertEqual(third["status"], "completed")
            self.assertEqual(third["attempts"], 3)
            overview = client.get("/api/auto-clean").json()
            self.assertEqual(len(overview), 1)
            self.assertEqual(overview[0]["source_records"], 2)
            self.assertEqual(cleaner.calls, [0, 1, 1])
            cleaned = [json.loads(line) for line in (output / "cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(cleaned, [{"text": "清理后的正文。"}, {"text": "第二条清理后的正文。"}])
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([item["ordinal"] for item in progress], [0, 1, 1])
            client.close()
            app.state.dataset_repository.clear()


class ProgressIndexFailureTests(unittest.TestCase):
    def test_existing_and_new_rejections_have_distinct_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            progress = output / "progress.jsonl"
            rows = [
                {"ordinal": 0, "doc_id": "old", "status": "incomplete", "cleaned_offset": 0,
                 "error": "模型接口返回 HTTP 400"},
                {"ordinal": 1, "doc_id": "risk", "status": "incomplete", "cleaned_offset": 0,
                 "error": "Content Exists Risk"},
                {"ordinal": 2, "doc_id": "uncertain", "status": "incomplete", "cleaned_offset": 0},
            ]
            progress.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with CleaningProgressIndex(output) as index:
                index.sync()
                self.assertEqual([index.details([i])[i]["failure_reason"] for i in range(3)],
                                 ["rejected", "content_risk", "uncertain"])


class SharedProgressTests(unittest.TestCase):
    def test_batch_drop_and_incomplete_are_visible_in_review_page(self):
        class DecisionCleaner:
            def __init__(self, config):
                self.config = config

            @staticmethod
            def is_complete(result):
                return result["decision"] != "unsure"

            def clean(self, blocks, *, title, provenance):
                if provenance["ordinal"] == 0:
                    return {"decision": "drop", "edited_text": ""}
                raise LlmCleaningError("Content Exists Risk")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text('{"text":"整条应删"}\n{"text":"继续核查"}\n', encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "decisions"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            cleaner = DecisionCleaner(app.state.api_context.llm_cleaner.config)
            clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                        output_directory=output_path(root, queue, cleaner.config),
                        cleaner=cleaner, workers=1, max_requests=1)
            dropped = client.get(f"/api/queues/{queue}/items/0").json()
            self.assertEqual(dropped["effective_source"], "batch")
            self.assertEqual(dropped["batch_clean"]["status"], "drop")
            self.assertEqual(dropped["materialized_text"], "")
            self.assertEqual(dropped["review_text"], "整条应删")
            incomplete = client.get(f"/api/queues/{queue}/items/1").json()
            self.assertEqual(incomplete["batch_clean"]["status"], "incomplete")
            self.assertEqual(incomplete["effective_source"], "original")
            self.assertEqual(incomplete["materialized_text"], "继续核查")
            self.assertEqual(client.get(f"/api/queues/{queue}/statuses",
                                        params={"start": 1, "limit": 1}).json()["items"][0]["status"], "risk")
            saved_draft = client.put(
                f"/api/reviews/documents/{incomplete['document']['doc_id']}",
                json={
                    "queue_id": queue, "ordinal": 1, "expected_revision": 0,
                    "decision": "unsure", "edited_text": "人工改过的正文",
                },
            )
            self.assertEqual(saved_draft.status_code, 200, saved_draft.text)
            self.assertEqual(client.get(f"/api/queues/{queue}/statuses",
                                        params={"start": 1, "limit": 1}).json()["items"][0]["status"], "review")
            confirmed = client.put(
                f"/api/reviews/documents/{dropped['document']['doc_id']}",
                json={
                    "queue_id": queue, "ordinal": 0, "expected_revision": 0,
                    "decision": "drop", "edited_text": None,
                },
            )
            self.assertEqual(confirmed.status_code, 200, confirmed.text)
            manual_view = client.get(f"/api/queues/{queue}/items/0").json()
            self.assertEqual(manual_view["effective_source"], "manual")
            self.assertEqual(manual_view["materialized_text"], "")
            client.close()
            app.state.dataset_repository.clear()

    def test_virtual_queue_counts_only_its_own_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            project = client.post("/api/projects", json={"name": "two sources"}).json()["project_id"]
            queues = []
            for number in (1, 2):
                source = root / f"source-{number}.jsonl"
                source.write_text(
                    json.dumps({"text": f"来源 {number} 的正文"}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                imported = client.post("/api/imports", json={
                    "adapter": "jsonl", "path": str(source),
                    "mapping": {"text_fields": ["text"]},
                }).json()
                prepared = client.post("/api/prepares", json={
                    "source_revision_directory": imported["revision_directory"],
                }).json()
                attached = client.post(f"/api/projects/{project}/sources", json={
                    "prepare_revision_directory": prepared["revision_directory"],
                })
                self.assertEqual(attached.status_code, 200, attached.text)
                queues.append(client.post(f"/api/projects/{project}/queues", json={
                    "name": f"queue-{number}",
                }).json()["queue_id"])
            second = client.get(f"/api/queues/{queues[1]}/items/1").json()
            saved = client.put(
                f"/api/reviews/documents/{second['document']['doc_id']}",
                json={
                    "queue_id": queues[1], "ordinal": 1, "expected_revision": 0,
                    "decision": "keep", "edited_text": second["review_text"],
                },
            )
            self.assertEqual(saved.status_code, 200, saved.text)
            first_queue = client.get(f"/api/projects/{project}").json()["queues"][0]
            self.assertEqual(first_queue["state_counts"], {"pending": 1, "done": 0})
            self.assertEqual(
                client.get(f"/api/auto-clean/{queues[0]}").json()["manual_completed"], 0,
            )
            second_queue = client.get(f"/api/projects/{project}").json()["queues"][1]
            self.assertEqual(second_queue["state_counts"], {"pending": 1, "done": 1})
            client.close()
            app.state.dataset_repository.clear()

    def test_incomplete_document_uses_one_whole_document_attempt(self):
        cleaner = FakeCleaner()
        result = _process(1, {
            "document": {"doc_id": "pause"}, "materialized_text": "未完成正文",
            "document_review": None, "provenance": {"title": None},
        }, "queue", cleaner)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(cleaner.calls, [1])

    def test_confirmed_keep_without_edit_skips_model(self):
        cleaner = FakeCleaner()
        result = _process(0, {
            "document": {"doc_id": "manual"}, "materialized_text": "人工确认正文",
            "document_review": {"decision": "keep", "edited_text": None},
            "provenance": {"title": None},
        }, "queue", cleaner)
        self.assertEqual(result["status"], "keep")
        self.assertEqual(result["text"], "人工确认正文")
        self.assertEqual(cleaner.calls, [])

    def test_manual_decisions_override_batch_counts_and_export(self):
        class KeepCleaner:
            def __init__(self, config):
                self.config = config
                self.calls = []

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                ordinal = provenance["ordinal"]
                self.calls.append(ordinal)
                return {"decision": "keep", "edited_text": f"批量正文 {ordinal}"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"text": f"原始正文 {i}"}, ensure_ascii=False) + "\n"
                                      for i in range(3)), encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "shared"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            cleaner = KeepCleaner(app.state.api_context.llm_cleaner.config)
            output = output_path(root, queue, cleaner.config)
            first = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                output_directory=output, cleaner=cleaner, workers=1,
                                max_requests=1, limit=2)
            self.assertEqual(first["processed"], 2)
            views = [client.get(f"/api/queues/{queue}/items/{i}").json() for i in range(3)]
            for ordinal in (0, 1):
                self.assertEqual(views[ordinal]["batch_clean"]["status"], "keep")
                self.assertEqual(views[ordinal]["materialized_text"], f"批量正文 {ordinal}")
                self.assertEqual(views[ordinal]["effective_source"], "batch")
                self.assertEqual(views[ordinal]["item"]["state"], "done")
                self.assertIsNone(views[ordinal]["document_review"])
            self.assertIsNone(views[2]["batch_clean"])
            self.assertEqual(views[2]["effective_source"], "original")
            documents = [view["document"] for view in views]
            with CurationDatabase(root / "curation.sqlite3") as database:
                for ordinal, decision, edited, actor in ((1, "drop", None, "local-web"),
                                                        (2, "keep", "人工正文 2", "llm-clean")):
                    document = documents[ordinal]
                    database.set_document_review(
                        project_id=project,
                        review=DocumentReviewInput(
                            doc_id=document["doc_id"], source_row=ordinal,
                            content_sha256=document["content_sha256"], decision=decision,
                            quality=2, primary_category=None, edited_text=edited,
                        ), expected_revision=0, actor=actor)
            job = client.get(f"/api/auto-clean/{queue}").json()
            self.assertEqual(job["batch_counts"], {"keep": 2})
            self.assertEqual(job["counts"], {"keep": 2, "drop": 1})
            self.assertEqual(job["manual_completed"], 2)
            self.assertEqual(job["manual_llm_saved"], 1)
            self.assertEqual(job["completed"], 3)
            manual_drop = client.get(f"/api/queues/{queue}/items/1").json()
            self.assertEqual(manual_drop["effective_source"], "manual")
            self.assertEqual(manual_drop["document_review"]["decision"], "drop")
            manual_keep = client.get(f"/api/queues/{queue}/items/2").json()
            self.assertEqual(manual_keep["effective_source"], "manual")
            self.assertEqual(manual_keep["materialized_text"], "人工正文 2")
            statuses = client.get(f"/api/queues/{queue}/statuses", params={"limit": 2}).json()
            self.assertEqual([(item["ordinal"], item["status"]) for item in statuses["items"]],
                             [(0, "batch"), (1, "manual")])
            self.assertEqual(statuses["next_start"], 2)
            self.assertEqual(client.get(f"/api/queues/{queue}/statuses",
                                        params={"status": "single_llm"}).json()["items"][0]["ordinal"], 2)
            self.assertEqual(client.get(f"/api/queues/{queue}/statuses",
                                        params={"status": "manual"}).json()["items"][0]["ordinal"], 1)
            exported = client.post(f"/api/auto-clean/{queue}/export").json()
            self.assertEqual(exported["exported_records"], 2)
            lines = [json.loads(line)["text"] for line in
                     (output / "effective_cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(lines, ["批量正文 0", "人工正文 2"])
            self.assertFalse(client.get(f"/api/auto-clean/{queue}").json()["export_stale"])
            with CurationDatabase(root / "curation.sqlite3") as database:
                document = documents[0]
                database.set_document_review(
                    project_id=project,
                    review=DocumentReviewInput(
                        doc_id=document["doc_id"], source_row=0,
                        content_sha256=document["content_sha256"], decision="keep",
                        quality=2, primary_category=None, edited_text="更新的人工正文 0",
                    ), expected_revision=0, actor="local-web")
            self.assertTrue(client.get(f"/api/auto-clean/{queue}").json()["export_stale"])
            client.post(f"/api/auto-clean/{queue}/export")
            lines = [json.loads(line)["text"] for line in
                     (output / "effective_cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(lines, ["更新的人工正文 0", "人工正文 2"])
            client.close()
            app.state.dataset_repository.clear()


class ParallelBatchTests(unittest.TestCase):
    def test_pause_drains_only_prefetched_work_and_resumes(self):
        stop = threading.Event()

        class PausingCleaner:
            def __init__(self):
                self.config = SimpleNamespace(provider="llamacpp", model="pause-fake")

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                if provenance["ordinal"] == 0:
                    stop.set()
                time.sleep(0.02)
                return {"decision": "keep", "edited_text": f"正文 {provenance['ordinal']}"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"text": f"原文 {i}"}, ensure_ascii=False) + "\n"
                                      for i in range(12)), encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "pause"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            output = root / "batch"
            first = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                output_directory=output, cleaner=PausingCleaner(),
                                workers=2, stop_event=stop)
            self.assertEqual(first["status"], "paused")
            self.assertLess(first["processed"], 12)
            self.assertGreater(first["processed"], 0)
            second = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=output, cleaner=PausingCleaner(), workers=2)
            self.assertEqual(second["status"], "completed")
            lines = [json.loads(line)["text"] for line in (output / "cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(lines, [f"正文 {i}" for i in range(12)])
            client.close()
            app.state.dataset_repository.clear()

    def test_parallel_requests_commit_in_ordinal_order_and_resume(self):
        class SlowCleaner:
            def __init__(self):
                self.config = SimpleNamespace(provider="llamacpp", model="parallel-fake")
                self.lock = threading.Lock()
                self.active = 0
                self.peak = 0
                self.finished = []

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.06 if provenance["ordinal"] == 0 else 0.01)
                with self.lock:
                    self.active -= 1
                    self.finished.append(provenance["ordinal"])
                return {"decision": "keep", "edited_text": f"清洗 {provenance['ordinal']}"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"text": f"正文 {i}"}, ensure_ascii=False) + "\n"
                                      for i in range(5)), encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "parallel"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            cleaner = SlowCleaner()
            output = root / "batch"
            first = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                output_directory=output, limit=4, cleaner=cleaner, workers=3)
            self.assertGreater(cleaner.peak, 1)
            self.assertLess(cleaner.finished.index(3), cleaner.finished.index(0))
            self.assertEqual(first["status"], "partial")
            second = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=output, cleaner=cleaner, workers=3)
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["counts"], {"keep": 5})
            lines = [json.loads(line)["text"] for line in (output / "cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(lines, [f"清洗 {i}" for i in range(5)])
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([row["ordinal"] for row in progress], list(range(5)))
            self.assertTrue(all(row["elapsed_seconds"] >= 0 for row in progress))
            client.close()
            app.state.dataset_repository.clear()

class RequestLimitTests(unittest.TestCase):
    def test_all_worker_cleaners_share_the_api_request_limit(self):
        from concurrent.futures import ThreadPoolExecutor
        from dataset.label.backend.llm_cleaning import CleaningConfig, LlmCleaner

        class Response:
            def __init__(self, opener):
                self.opener = opener

            def __enter__(self):
                with self.opener.lock:
                    self.opener.active += 1
                    self.opener.peak = max(self.opener.peak, self.opener.active)
                return self

            def __exit__(self, *_):
                with self.opener.lock:
                    self.opener.active -= 1

            def read(self):
                time.sleep(0.02)
                return b"{}"

        class Opener:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = self.peak = 0

            def open(self, *_args, **_kwargs):
                return Response(self)

        config = CleaningConfig(provider="deepseek", base_url="https://api.deepseek.com",
                                model="fake", api_key="test")
        shared = threading.BoundedSemaphore(2)
        opener = Opener()
        with tempfile.TemporaryDirectory() as temporary:
            cleaners = [LlmCleaner(config, Path(temporary), shared) for _ in range(6)]
            for cleaner in cleaners:
                cleaner._opener = opener
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(lambda cleaner: cleaner._request("/chat/completions", {}), cleaners))
        self.assertEqual(results, [{}] * 6)
        self.assertEqual(opener.peak, 2)

class ProgressIndexTests(unittest.TestCase):
    def test_incomplete_log_line_is_replayed_when_finished(self):
        from dataset.label.backend.cleaning_progress import CleaningProgressIndex

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            progress = output / "progress.jsonl"
            first = json.dumps({"ordinal": 0, "doc_id": "a", "status": "keep", "cleaned_offset": 10}) + "\n"
            second = json.dumps({"ordinal": 1, "doc_id": "b", "status": "drop", "cleaned_offset": 10}) + "\n"
            progress.write_bytes((first + second[:10]).encode())
            with CleaningProgressIndex(output) as index:
                index.sync()
                self.assertEqual(index.summary()[0], {"keep": 1})
                with progress.open("ab") as stream:
                    stream.write(second[10:].encode())
                index.sync()
                self.assertEqual(index.summary()[0], {"keep": 1, "drop": 1})
                progress.write_bytes(first.encode())
                index.sync()
                self.assertEqual(index.summary()[0], {"keep": 1})
