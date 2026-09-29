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


def _make_batch_queue(client, root: Path, texts: list[str]) -> str:
    source = root / "source.jsonl"
    source.write_text("".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
                      encoding="utf-8")
    imported = client.post("/api/imports", json={
        "adapter": "jsonl", "path": str(source), "mapping": {"text_fields": ["text"]},
    }).json()
    prepared = client.post("/api/prepares", json={
        "source_revision_directory": imported["revision_directory"],
    }).json()
    project = client.post("/api/projects", json={"name": "fallback"}).json()["project_id"]
    client.post(f"/api/projects/{project}/sources", json={
        "prepare_revision_directory": prepared["revision_directory"],
    })
    return client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]


class BatchCleaningTests(unittest.TestCase):
    def test_non_risk_failure_uses_paid_api_without_sending_fresh_rows_to_it(self):
        class WebCleaner:
            config = SimpleNamespace(provider="deepseek_web", model="web")

            def __init__(self):
                self.calls = []

            @staticmethod
            def is_complete(result):
                return result["decision"] != "unsure"

            def clean(self, blocks, *, title, provenance):
                self.calls.append(provenance["ordinal"])
                if provenance["ordinal"] == 0:
                    return {"decision": "unsure", "edited_text": blocks[0]["text"]}
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        class PaidCleaner:
            config = SimpleNamespace(provider="deepseek", model="paid")

            def __init__(self):
                self.calls = []

            @staticmethod
            def is_complete(_result):
                return True

            def clean(self, blocks, *, title, provenance):
                self.calls.append(provenance["ordinal"])
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["第一条正文。", "第二条正文。"])
                web, paid = WebCleaner(), PaidCleaner()
                output = root / "batch"
                manifest = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                       output_directory=output, cleaner=web, cleaners={"web": web},
                                       failure_fallback=paid, workers=1)
                rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
                self.assertEqual([(row["ordinal"], row["status"], row["source"]) for row in rows],
                                 [(0, "incomplete", "web"), (0, "keep", "deepseek_api_retry"),
                                  (1, "keep", "web")])
                self.assertEqual(web.calls, [0, 1])
                self.assertEqual(paid.calls, [0])
                self.assertEqual(manifest["counts"], {"keep": 2})
                self.assertEqual(manifest["source_activity"]["deepseek_api_retry"]["capacity"], 1)
                self.assertEqual(manifest["source_outcomes"], {
                    "web": {"incomplete": 1, "keep": 1},
                    "deepseek_api_retry": {"keep": 1},
                })

    def test_content_risk_does_not_use_paid_api(self):
        class RiskCleaner:
            config = SimpleNamespace(provider="deepseek_web", model="web")

            @staticmethod
            def is_complete(_result):
                return False

            def clean(self, blocks, *, title, provenance):
                if provenance["ordinal"] == 0:
                    raise LlmCleaningError("Content Exists Risk")
                return {"decision": "unsure", "edited_text": blocks[0]["text"],
                        "warnings": ["第 1 块 Content Exists Risk"]}

        class PaidCleaner:
            config = SimpleNamespace(provider="deepseek", model="paid")
            calls = 0

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                raise AssertionError("敏感条目不应发送到付费 API")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["第一条正文。", "第二条正文。"])
                risk, paid = RiskCleaner(), PaidCleaner()
                output = root / "batch"
                manifest = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                       output_directory=output, cleaner=risk, cleaners={"web": risk},
                                       failure_fallback=paid, workers=1)
                rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
                self.assertEqual([row["failure_reason"] for row in rows], ["content_risk", "content_risk"])
                self.assertEqual(paid.calls, 0)
                self.assertEqual(manifest["counts"], {"incomplete": 2})

    def test_other_selected_model_recovers_without_paid_call(self):
        class Cleaner:
            def __init__(self, provider):
                self.config = SimpleNamespace(provider=provider, model=provider)
                self.calls = 0

            @staticmethod
            def is_complete(result):
                return result["decision"] == "keep"

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                return {"decision": "unsure" if self.config.provider == "deepseek_web" else "keep",
                        "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["待清洗正文。"])
                web, local, paid = (Cleaner(source) for source in
                                    ("deepseek_web", "llamacpp", "deepseek"))
                result = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                     output_directory=root / "batch", cleaner=web,
                                     cleaners={"web": web, "local": local},
                                     failure_fallback=paid, workers=1)
                self.assertEqual((web.calls, local.calls, paid.calls), (1, 1, 0))
                self.assertEqual(result["counts"], {"keep": 1})

    def test_non_risk_failure_after_source_reassignment_reaches_paid_api(self):
        class Cleaner:
            def __init__(self, provider):
                self.config = SimpleNamespace(provider=provider, model=provider)
                self.calls = 0

            @staticmethod
            def is_complete(result):
                return result["decision"] == "keep"

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                if self.config.provider == "deepseek_web":
                    raise LlmCleaningError("HTTP 429")
                return {"decision": "keep" if self.config.provider == "deepseek" else "unsure",
                        "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["待清洗正文。"])
                web, local, paid = (Cleaner(source) for source in
                                    ("deepseek_web", "llamacpp", "deepseek"))
                output = root / "batch"
                manifest = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                       output_directory=output, cleaner=web,
                                       cleaners={"web": web, "local": local},
                                       failure_fallback=paid, workers=1)
                self.assertEqual((web.calls, local.calls, paid.calls), (1, 1, 1))
                self.assertEqual(manifest["counts"], {"keep": 1})

    def test_second_source_outage_after_document_failure_uses_paid_api(self):
        class Cleaner:
            def __init__(self, provider):
                self.config = SimpleNamespace(provider=provider, model=provider)
                self.calls = 0

            @staticmethod
            def is_complete(result):
                return result["decision"] == "keep"

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                if self.config.provider == "llamacpp":
                    raise LlmCleaningError("HTTP 429")
                return {"decision": "keep" if self.config.provider == "deepseek" else "unsure",
                        "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["待清洗正文。"])
                web, local, paid = (Cleaner(source) for source in
                                    ("deepseek_web", "llamacpp", "deepseek"))
                output = root / "batch"
                result = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                     output_directory=output, cleaner=web,
                                     cleaners={"web": web, "local": local},
                                     failure_fallback=paid, workers=1)
                self.assertEqual((web.calls, local.calls, paid.calls), (1, 1, 1))
                self.assertEqual(result["counts"], {"keep": 1})
                self.assertEqual(result["source_errors"].keys(), {"local"})

    def test_earlier_content_risk_blocks_paid_retry_even_after_other_failure(self):
        class Cleaner:
            def __init__(self, provider):
                self.config = SimpleNamespace(provider=provider, model=provider)
                self.calls = 0

            @staticmethod
            def is_complete(_result):
                return False

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                if self.config.provider == "deepseek_web":
                    raise LlmCleaningError("Content Exists Risk")
                return {"decision": "unsure", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["待清洗正文。"])
                web, local, paid = (Cleaner(source) for source in
                                    ("deepseek_web", "llamacpp", "deepseek"))
                output = root / "batch"
                options = dict(database_path=root / "curation.sqlite3", queue_id=queue,
                               output_directory=output, cleaner=web,
                               cleaners={"web": web, "local": local},
                               failure_fallback=paid, workers=1)
                clean_queue(**options)
                clean_queue(**options)
                self.assertEqual(paid.calls, 0)
                rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
                self.assertEqual(rows[0]["failure_reason"], "content_risk")
                self.assertEqual(rows[1]["source"], "local")
                self.assertEqual(rows[1]["status"], "incomplete")

    def test_qwen_validation_rejection_is_retried_without_paid_fallback(self):
        class Cleaner:
            def __init__(self, provider):
                self.config = SimpleNamespace(provider=provider, model=provider)
                self.calls = 0

            @staticmethod
            def is_complete(result):
                return result["decision"] == "keep"

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                if self.config.provider == "qwen_web":
                    raise LlmCleaningError("千问聊天请求失败：HTTP 200, FAIL_SYS_USER_VALIDATE")
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["待清洗正文。"])
                qwen, local, paid = (Cleaner(source) for source in
                                     ("qwen_web", "llamacpp", "deepseek"))
                output = root / "batch"
                first = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                    output_directory=output, cleaner=qwen,
                                    cleaners={"qwen": qwen}, workers=1)
                self.assertEqual(first["counts"], {"incomplete": 1})
                second = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                     output_directory=output, cleaner=qwen,
                                     cleaners={"qwen": qwen, "local": local},
                                     failure_fallback=paid, workers=1)
                self.assertEqual(second["counts"], {"keep": 1})
                self.assertEqual((qwen.calls, local.calls, paid.calls), (2, 1, 0))

    def test_prior_failure_goes_directly_to_paid_api_only_once(self):
        class IncompleteCleaner:
            def __init__(self, provider):
                self.config = SimpleNamespace(provider=provider, model=provider)
                self.calls = 0

            @staticmethod
            def is_complete(_result):
                return False

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                return {"decision": "unsure", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["待清洗正文。"])
                web, paid = IncompleteCleaner("deepseek_web"), IncompleteCleaner("deepseek")
                output = root / "batch"
                options = dict(database_path=root / "curation.sqlite3", queue_id=queue,
                               output_directory=output, cleaner=web, cleaners={"web": web},
                               workers=1, limit=1)
                clean_queue(**options)
                clean_queue(**options, failure_fallback=paid)
                self.assertEqual((web.calls, paid.calls), (1, 1))
                third = clean_queue(**options, failure_fallback=paid)
                self.assertEqual((web.calls, paid.calls), (2, 1))
                self.assertEqual(third["source_outcomes"], {
                    "web": {"incomplete": 2}, "deepseek_api_retry": {"incomplete": 1},
                })
                rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
                self.assertEqual([row["source"] for row in rows],
                                 ["web", "deepseek_api_retry", "web"])

    def test_fresh_only_run_skips_prior_incomplete_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"text": f"正文 {i}"}, ensure_ascii=False) + "\n"
                                      for i in range(3)), encoding="utf-8")
            imported = client.post("/api/imports", json={
                "adapter": "jsonl", "path": str(source), "mapping": {"text_fields": ["text"]},
            }).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"],
            }).json()
            project = client.post("/api/projects", json={"name": "fresh"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"],
            })
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            cleaner = FakeCleaner()
            output = root / "batch"
            clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                        output_directory=output, cleaner=cleaner, limit=2, workers=1)
            result = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=output, cleaner=cleaner, limit=1,
                                 workers=1, retry_incomplete=False)
            self.assertEqual(cleaner.calls, [0, 1, 2])
            self.assertEqual(result["processed"], 3)
            self.assertEqual(result["counts"]["incomplete"], 2)
            client.close()

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


class OutOfOrderResumeTests(unittest.TestCase):
    def test_resume_fills_holes_without_reprocessing_later_finished_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text('{"text":"正文 0"}\n{"text":"正文 1"}\n{"text":"正文 2"}\n', encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "resume holes"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            doc2 = client.get(f"/api/queues/{queue}/items/2").json()["document"]["doc_id"]
            output = root / "batch"
            output.mkdir()
            saved = (json.dumps({"text": "已完成的第三条"}, ensure_ascii=False) + "\n").encode()
            (output / "cleaned.jsonl").write_bytes(saved)
            (output / "progress.jsonl").write_text(json.dumps({
                "ordinal": 2, "doc_id": doc2, "status": "keep", "cleaned_offset": len(saved),
            }) + "\n", encoding="utf-8")
            cleaner = FakeCleaner()
            cleaner.recover = True

            class PaidCleaner:
                config = SimpleNamespace(provider="deepseek", model="paid")
                calls = 0

                def clean(self, blocks, *, title, provenance):
                    self.calls += 1
                    raise AssertionError("未处理的进度空洞不应交给付费 API")

            paid = PaidCleaner()
            result = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=output, cleaner=cleaner,
                                 failure_fallback=paid, workers=1)
            self.assertEqual(paid.calls, 0)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["processed"], 3)
            self.assertEqual(cleaner.calls, [0, 1])
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([row["ordinal"] for row in progress], [2, 0, 1])
            from dataset.label.backend.cleaning_export import export_effective
            export_effective(root / "curation.sqlite3", queue, output, {})
            exported = [json.loads(line)["text"] for line in (output / "effective_cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(exported, ["清理后的正文。", "第二条清理后的正文。", "已完成的第三条"])
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

    def test_doubao_non_sse_response_stops_source(self):
        class RejectedCleaner:
            def clean(self, blocks, *, title, provenance):
                raise LlmCleaningError("豆包网页请求失败：豆包没有返回 SSE 聊天流：HTTP 200，Content-Type text/plain，响应为空")

        with self.assertRaisesRegex(LlmCleaningError, "豆包没有返回 SSE"):
            _process(0, {
                "document": {"doc_id": "rejected"}, "materialized_text": "正文",
                "document_review": None, "provenance": {"title": None},
            }, "queue", RejectedCleaner())

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

            # A save after the exporter snapshots manual decisions must make the
            # resulting file stale, even if the save precedes file generation.
            manager = app.state.api_context.batch_jobs
            original_reviews = manager._manual_reviews
            def save_during_export(database, selected_queue):
                snapshot = original_reviews(database, selected_queue)
                document = documents[0]
                database.set_document_review(
                    project_id=project,
                    review=DocumentReviewInput(
                        doc_id=document["doc_id"], source_row=0,
                        content_sha256=document["content_sha256"], decision="keep",
                        quality=None, primary_category=None, edited_text="并发更新的人工正文 0",
                    ), expected_revision=1, actor="local-web")
                return snapshot
            manager._manual_reviews = save_during_export
            try:
                manager.export(queue)
            finally:
                manager._manual_reviews = original_reviews
            self.assertTrue(client.get(f"/api/auto-clean/{queue}").json()["export_stale"])
            client.close()
            app.state.dataset_repository.clear()


    def test_single_llm_save_wins_while_batch_request_is_in_flight(self):
        ready = threading.Event()
        release = threading.Event()

        class BlockingCleaner:
            config = SimpleNamespace(provider="deepseek", model="fake-batch")

            @staticmethod
            def is_complete(_result):
                return True

            def clean(self, _blocks, *, title, provenance):
                ready.set()
                if not release.wait(5):
                    raise TimeoutError("batch test was not released")
                return {"decision": "keep", "edited_text": "批量正文"}

        class SingleCleaner:
            config = SimpleNamespace(provider="deepseek", model="fake-single")

            @staticmethod
            def is_complete(_result):
                return True

            def clean(self, _blocks, *, title, provenance):
                return {"decision": "keep", "edited_text": "单条 AI 正文", "removals": [], "edits": []}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"text": "需要清洗的原始正文。"}, ensure_ascii=False) + "\n", encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "overlap"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            manager = app.state.api_context.batch_jobs
            output = manager._output(queue)
            outcomes = []
            def run_batch():
                try:
                    outcomes.append(clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                                 output_directory=output, cleaner=BlockingCleaner(),
                                                 workers=1, max_requests=1))
                except BaseException as error:
                    outcomes.append(error)
            batch_thread = threading.Thread(target=run_batch)
            batch_thread.start()
            try:
                self.assertTrue(ready.wait(5), "batch request did not start")
                view = client.get(f"/api/queues/{queue}/items/0").json()
                app.state.api_context.llm_cleaner = SingleCleaner()
                response = client.post(f"/api/reviews/documents/{view['document']['doc_id']}/llm-clean",
                    json={"queue_id": queue, "ordinal": 0, "expected_revision": 0,
                          "content_sha256": view["document"]["content_sha256"],
                          "blocks": [{"id": block["block_id"], "text": block["text"],
                                      "separator_after": block["separator_after"]}
                                     for block in view["blocks"]]})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()["saved"])
            finally:
                release.set()
                batch_thread.join(timeout=5)
            self.assertFalse(batch_thread.is_alive(), "batch thread did not finish")
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], dict)
            view = client.get(f"/api/queues/{queue}/items/0").json()
            self.assertEqual(view["batch_clean"]["status"], "keep")
            self.assertEqual(view["effective_source"], "manual")
            self.assertEqual(view["materialized_text"], "单条 AI 正文")
            exported = manager.export(queue)
            self.assertEqual(exported["exported_records"], 1)
            self.assertEqual(json.loads((output / "effective_cleaned.jsonl").read_text().strip()),
                             {"text": "单条 AI 正文"})
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

    def test_incomplete_model_result_gets_one_other_source(self):
        class IncompleteCleaner:
            def __init__(self):
                self.config = SimpleNamespace(provider="deepseek_web", model="web")

            @staticmethod
            def is_complete(result):
                return False

            def clean(self, blocks, *, title, provenance):
                return {"decision": "unsure", "edited_text": blocks[0]["text"]}

        class CompleteCleaner:
            def __init__(self):
                self.config = SimpleNamespace(provider="llamacpp", model="local")

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = TestClient(create_app(root))
            source = root / "source.jsonl"
            source.write_text('{"text":"待清洗正文。"}\n', encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "fallback"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            bad, good = IncompleteCleaner(), CompleteCleaner()
            output = root / "batch"
            manifest = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                   output_directory=output, cleaner=bad,
                                   cleaners={"bad": bad, "good": good}, workers=1, limit=1)
            rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([(row["ordinal"], row["status"], row["source"]) for row in rows],
                             [(0, "incomplete", "bad"), (0, "keep", "good")])
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["source_attempts"], {"bad": 1, "good": 1})
            self.assertEqual(manifest["source_activity"]["bad"]["incomplete"], 1)
            self.assertEqual(manifest["source_activity"]["good"]["completed"], 1)
            client.close()

    def test_waiting_retry_does_not_idle_fast_source(self):
        local_finished = threading.Event()

        class FastIncompleteCleaner:
            config = SimpleNamespace(provider="deepseek_web", model="web")
            before_local_finished = 0

            @staticmethod
            def is_complete(result):
                return False

            def clean(self, blocks, *, title, provenance):
                if not local_finished.is_set():
                    self.before_local_finished += 1
                return {"decision": "unsure", "edited_text": blocks[0]["text"]}

        class SlowCompleteCleaner:
            config = SimpleNamespace(provider="llamacpp", model="local")

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                time.sleep(0.15)
                local_finished.set()
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"text": f"正文 {i}"}, ensure_ascii=False) + "\n"
                                      for i in range(8)), encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "retry scheduling"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            fast, slow = FastIncompleteCleaner(), SlowCompleteCleaner()
            result = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=root / "batch", cleaner=slow,
                                 cleaners={"web": fast, "local": slow}, workers=3)
            self.assertGreater(fast.before_local_finished, 2)
            self.assertEqual(result["source_activity"]["web"]["capacity"], 2)
            self.assertEqual(result["status"], "completed")
            client.close()
            app.state.dataset_repository.clear()

    def test_rate_limited_source_is_disabled_and_document_reassigned(self):
        class RateLimitedCleaner:
            def __init__(self):
                self.config = SimpleNamespace(provider="deepseek", model="limited")

            def clean(self, blocks, *, title, provenance):
                raise LlmCleaningError("模型接口返回 HTTP 429")

        class HealthyCleaner:
            def __init__(self):
                self.config = SimpleNamespace(provider="llamacpp", model="healthy")

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text('{"text":"第一条。"}\n{"text":"第二条。"}\n', encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "fallback"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            limited, healthy = RateLimitedCleaner(), HealthyCleaner()
            output = root / "batch"
            manifest = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                   output_directory=output, cleaner=limited,
                                   cleaners={"limited": limited, "healthy": healthy}, workers=2)
            self.assertEqual(manifest["status"], "completed")
            self.assertIn("limited", manifest["source_errors"])
            rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(row["ordinal"] for row in rows), [0, 1])
            self.assertEqual({row["source"] for row in rows}, {"healthy"})
            client.close()
            app.state.dataset_repository.clear()

    def test_qwen_refresh_failure_leaves_work_for_other_sources(self):
        class Cleaner:
            def __init__(self, provider, api_key=""):
                self.config = SimpleNamespace(provider=provider, model=provider, api_key=api_key)
                self.calls = 0

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                self.calls += 1
                if self.config.provider == "qwen_web":
                    raise LlmCleaningError("签名材料已用完或过期，自动补充失败")
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text('{"text":"正文。"}\n', encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "qwen exhausted"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            qwen, local = Cleaner("qwen_web"), Cleaner("llamacpp")
            result = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                 output_directory=root / "batch", cleaner=local,
                                 cleaners={"qwen_web": qwen, "local": local}, workers=2)
            self.assertEqual(qwen.calls, 1)
            self.assertEqual(local.calls, 1)
            self.assertIn("签名材料已用完", result["source_errors"]["qwen_web"])
            self.assertEqual(result["source_activity"]["qwen_web"]["active"], 0)
            self.assertEqual(result["source_activity"]["qwen_web"]["capacity"], 1)
            client.close()
            app.state.dataset_repository.clear()

    def test_multiple_sources_share_queue_and_fast_source_takes_more_work(self):
        class LaneCleaner:
            def __init__(self, provider, delay):
                self.config = SimpleNamespace(provider=provider, model=provider)
                self.delay = delay
                self.calls = []

            @staticmethod
            def is_complete(result):
                return True

            def clean(self, blocks, *, title, provenance):
                time.sleep(self.delay)
                self.calls.append(provenance["ordinal"])
                return {"decision": "keep", "edited_text": blocks[0]["text"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = TestClient(app)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"text": f"正文 {i}"}, ensure_ascii=False) + "\n"
                                      for i in range(10)), encoding="utf-8")
            imported = client.post("/api/imports", json={"adapter": "jsonl", "path": str(source),
                "mapping": {"text_fields": ["text"]}}).json()
            prepared = client.post("/api/prepares", json={
                "source_revision_directory": imported["revision_directory"]}).json()
            project = client.post("/api/projects", json={"name": "mixed"}).json()["project_id"]
            client.post(f"/api/projects/{project}/sources", json={
                "prepare_revision_directory": prepared["revision_directory"]})
            queue = client.post(f"/api/projects/{project}/queues", json={"name": "all"}).json()["queue_id"]
            fast = LaneCleaner("deepseek", 0.005)
            slow = LaneCleaner("llamacpp", 0.06)
            output = root / "batch"
            manifest = clean_queue(database_path=root / "curation.sqlite3", queue_id=queue,
                                   output_directory=output, cleaner=fast,
                                   cleaners={"fast": fast, "slow": slow}, workers=3)
            self.assertEqual(manifest["counts"], {"keep": 10})
            self.assertGreater(len(fast.calls), len(slow.calls))
            self.assertGreater(len(slow.calls), 0)
            self.assertEqual(sum(manifest["source_attempts"].values()), 10)
            self.assertEqual(sum(item["completed"] for item in manifest["source_activity"].values()), 10)
            self.assertTrue(all(item["active"] == 0 for item in manifest["source_activity"].values()))
            self.assertEqual(manifest["source_activity"]["fast"]["capacity"], 6)
            self.assertEqual(manifest["source_activity"]["slow"]["capacity"], 1)
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(row["ordinal"] for row in progress), list(range(10)))
            self.assertEqual({row["source"] for row in progress}, {"fast", "slow"})
            self.assertEqual(len(json.loads((output / "job.json").read_text())["model_history"]), 2)
            client.close()
            app.state.dataset_repository.clear()

    def test_parallel_requests_commit_in_ordinal_order_and_resume(self):
        class SlowCleaner:
            def __init__(self):
                self.config = SimpleNamespace(provider="deepseek", model="parallel-fake")
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
            self.assertEqual(sorted(lines), [f"清洗 {i}" for i in range(5)])
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(row["ordinal"] for row in progress), list(range(5)))
            self.assertNotEqual(progress[0]["ordinal"], 0)
            from dataset.label.backend.cleaning_export import export_effective
            export_effective(root / "curation.sqlite3", queue, output, {})
            effective = [json.loads(line)["text"] for line in (output / "effective_cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(effective, [f"清洗 {i}" for i in range(5)])
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
