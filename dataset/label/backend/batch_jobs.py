"""Background control and catalog statistics for automatic cleaning jobs."""
from __future__ import annotations

import json
import threading
from heapq import merge
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping

from .batch_clean import clean_queue, output_path
from .cleaning_export import export_effective
from .cleaning_progress import CleaningProgressIndex, write_snapshot
from .llm_cleaning import LlmCleaner
from .local_model import LocalModelService
from .model_settings import ModelSettings
from .database import CurationDatabase
from .documents import DatasetRepository
from .llm_cleaning import CleaningConfig


class BatchJobManager:
    def __init__(self, root: Path, database_path: Path,
                 configs: Mapping[str, CleaningConfig] | CleaningConfig,
                 local_model: LocalModelService | None = None):
        self.root = root
        self.database_path = database_path
        self.configs = (dict(configs) if isinstance(configs, Mapping) else {"default": configs})
        self.local_model = local_model
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._queue_id: str | None = None
        self._stop = threading.Event()
        self._error: str | None = None
        self._repository = DatasetRepository()
        self._manual_cache: dict[str, tuple[int, dict[int, dict]]] = {}

    @property
    def config(self) -> CleaningConfig:
        return next(iter(self.configs.values()))

    def update_config(self, configs: Mapping[str, CleaningConfig] | CleaningConfig,
                      persist: Callable[[], None]) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("批量清洗运行中，请先暂停再切换模型")
            persist()
            self.configs = (dict(configs) if isinstance(configs, Mapping) else {"default": configs})

    def _output(self, queue_id: str) -> Path:
        return output_path(self.root, queue_id, self.config)

    def document_result(self, queue_id: str, ordinal: int, doc_id: str) -> dict | None:
        """Return the latest durable batch decision for one queue item."""
        output = self._output(queue_id)
        with CleaningProgressIndex(output) as index:
            index.sync()
            return index.result(ordinal, doc_id)

    def document_view(self, value: dict) -> dict:
        """Combine batch and human work for the review UI without rewriting either."""
        batch = self.document_result(
            value["item"]["queue_id"], value["item"]["ordinal"], value["document"]["doc_id"],
        )
        value["batch_clean"] = {"status": batch["status"]} if batch is not None else None
        review = value["document_review"]
        manual_wins = review is not None and (
            review["decision"] in {"keep", "drop"} or review.get("edited_text") is not None
        )
        value["effective_source"] = "manual" if manual_wins else "original"
        if batch is not None and not manual_wins and batch["status"] in {"keep", "drop"}:
            value["materialized_text"] = batch["text"] if batch["status"] == "keep" else ""
            value["item"]["state"] = "done"
            value["effective_source"] = "batch"
        return value

    def _manual_reviews(self, database: CurationDatabase, queue: dict) -> dict[int, dict]:
        queue_id = queue["queue_id"]
        revision = database.get_project(queue["project_id"])["current_revision"]
        with self._lock:
            cached = self._manual_cache.get(queue_id)
        if cached and cached[0] == revision:
            return cached[1]
        llm_docs = {row["entity_id"] for row in database.connection.execute("""
            SELECT events.entity_id FROM events
            JOIN (
                SELECT entity_id, MAX(event_seq) AS event_seq FROM events
                WHERE project_id=? AND entity_type='document_review' GROUP BY entity_id
            ) AS latest ON events.event_seq=latest.event_seq
            WHERE events.actor='llm-clean'
        """, (queue["project_id"],))}
        if queue["sampling_policy"].get("type") == "full_dataset":
            positions = database.full_queue_review_positions(queue, self._repository)
        else:
            positions = {}
            for review in database.connection.execute(
                "SELECT doc_id, decision, edited_text IS NOT NULL AS has_edit "
                "FROM document_reviews WHERE project_id=?",
                (queue["project_id"],),
            ):
                row = database.connection.execute(
                    "SELECT ordinal FROM queue_items WHERE queue_id=? AND doc_id=?",
                    (queue_id, review["doc_id"]),
                ).fetchone()
                if row is not None:
                    positions[row["ordinal"]] = {
                        "doc_id": review["doc_id"], "decision": review["decision"],
                        "has_edit": bool(review["has_edit"]),
                    }
        for item in positions.values():
            item["llm_saved"] = item["doc_id"] in llm_docs
        with self._lock:
            self._manual_cache[queue_id] = revision, positions
        return positions

    def status(self, queue_id: str) -> dict:
        with CurationDatabase(self.database_path) as database:
            queue = database.get_queue(queue_id)
            return self._status(database, queue)

    def _status(self, database: CurationDatabase, queue: dict) -> dict:
        queue_id = queue["queue_id"]
        path = self._output(queue_id) / "manifest.json"
        if path.exists():
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = {}
        else:
            result = {}
        with self._lock:
            active = self._queue_id == queue_id and self._thread is not None and self._thread.is_alive()
            error = self._error if self._queue_id == queue_id else None
            stopping = self._stop.is_set() if active else False
        if active:
            result["status"] = "stopping" if stopping else "running"
            if result.get("updated_at"):
                age = (datetime.now(UTC) - datetime.fromisoformat(result["updated_at"])).total_seconds()
                result["elapsed_seconds"] = round(result.get("elapsed_seconds", 0) + max(0, age), 2)
        elif error:
            result = {**result, "status": "failed", "error": error}
        elif result.get("status") == "running":
            result["status"] = "interrupted"
        manual = self._manual_reviews(database, queue)
        with CleaningProgressIndex(self._output(queue_id)) as index:
            index.sync()
            batch_counts, scanned, attempts = index.summary()
            conclusive = {ordinal: item for ordinal, item in manual.items()
                          if item["decision"] in {"keep", "drop"}}
            overridden = index.statuses(list(conclusive))
        effective = batch_counts.copy()
        for ordinal, item in conclusive.items():
            prior = overridden.get(ordinal)
            if prior:
                effective[prior] -= 1
            effective[item["decision"]] += 1
        effective = {key: value for key, value in effective.items() if value > 0}
        total = sum(queue["state_counts"].values())
        completed = effective.get("keep", 0) + effective.get("drop", 0)
        export_path = self._output(queue_id) / "effective_manifest.json"
        exported = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else None
        export_stale = bool(exported and (
            exported["project_revision"] != database.get_project(queue["project_id"])["current_revision"]
            or exported["batch_attempts"] != attempts))
        result.update(total=total, processed=scanned, attempts=attempts,
                      batch_counts=dict(batch_counts), counts=effective,
                      completed=completed, remaining=total - completed,
                      manual_completed=len(conclusive),
                      manual_unsure=sum(item["decision"] == "unsure" for item in manual.values()),
                      manual_llm_saved=sum(item["llm_saved"] for item in manual.values()),
                      export=exported, export_stale=export_stale,
                      complete=completed == total,
                      eta_seconds=(round((total - completed) * 60 / result["rate_per_minute"])
                                   if result.get("rate_per_minute") else None))
        return result

    def item_status_page(self, queue_id: str, start: int, limit: int, status_filter: str) -> dict:
        """A small ordered window of effective queue states; never load the full queue."""
        allowed = {"all", "uncleaned", "manual", "single_llm", "batch", "risk", "rejected", "failed", "review"}
        if status_filter not in allowed:
            raise ValueError("未知的清洗状态筛选")
        if start < 0 or not 1 <= limit <= 100:
            raise ValueError("条目列表范围无效")
        with CurationDatabase(self.database_path) as database:
            queue = database.get_queue(queue_id)
            total = sum(queue["state_counts"].values())
            manual = self._manual_reviews(database, queue)
            failures = database.list_cleaning_failures(queue_id)
        with CleaningProgressIndex(self._output(queue_id)) as index:
            index.sync()

            def effective(ordinal: int, batch: dict | None) -> dict:
                review = manual.get(ordinal)
                if review and review["decision"] in {"keep", "drop"}:
                    state = "single_llm" if review["llm_saved"] else "manual"
                    decision = review["decision"]
                elif review and review["has_edit"]:
                    state, decision = "review", None
                elif batch and batch["status"] in {"keep", "drop"}:
                    state, decision = "batch", batch["status"]
                elif ordinal in failures:
                    state, decision = failures[ordinal], None
                elif batch and batch["status"] == "incomplete":
                    reason = batch["failure_reason"]
                    state = ("risk" if reason == "content_risk" else "rejected" if reason == "rejected"
                             else "failed" if reason == "error" else "review")
                    decision = None
                elif review:
                    state, decision = "review", None
                else:
                    state, decision = "uncleaned", None
                return {"ordinal": ordinal, "status": state, "decision": decision}

            items = []
            def accept(ordinal: int, batch: dict | None) -> bool:
                item = effective(ordinal, batch)
                if status_filter == "all" or item["status"] == status_filter:
                    items.append(item)
                return len(items) > limit

            if status_filter in {"all", "uncleaned"}:
                cursor = start
                while cursor < total and len(items) <= limit:
                    end = min(total, cursor + 500)
                    details = index.details(list(range(cursor, end)))
                    for ordinal in range(cursor, end):
                        if accept(ordinal, details.get(ordinal)):
                            break
                    cursor = end
            elif status_filter in {"manual", "single_llm"}:
                for ordinal in sorted(number for number, review in manual.items() if number >= start
                                      and review["decision"] in {"keep", "drop"}):
                    if accept(ordinal, None):
                        break
            elif status_filter == "batch":
                for row in index.filtered(start, ("keep", "drop")):
                    if accept(row["ordinal"], row):
                        break
            else:
                single_pending = sorted(number for number, reason in failures.items()
                                        if number >= start and reason == status_filter)
                manual_pending = (sorted(number for number, review in manual.items()
                                         if number >= start and review["decision"] == "unsure")
                                  if status_filter == "review" else [])
                batch_pending = (row["ordinal"] for row in index.filtered(start, ("incomplete",)))
                previous = None
                for ordinal in merge(single_pending, manual_pending, batch_pending):
                    if ordinal == previous:
                        continue
                    previous = ordinal
                    if accept(ordinal, index.details([ordinal]).get(ordinal)):
                        break
        next_start = items[limit]["ordinal"] if len(items) > limit else None
        return {"items": items[:limit], "next_start": next_start, "total": total}

    def overview(self) -> list[dict]:
        rows = []
        with CurationDatabase(self.database_path) as database:
            for project in database.list_projects():
                attached = {(item["source_id"], item["source_revision"]): item
                            for item in database.list_project_sources(project["project_id"])}
                for queue in database.list_queues(project["project_id"]):
                    job = self._status(database, queue)
                    for source in queue["sampling_policy"].get("sources", []):
                        rows.append({
                            "project_id": project["project_id"], "project_name": project["name"],
                            "queue_id": queue["queue_id"], "queue_name": queue["name"],
                            "source_id": source["source_id"],
                            "source_revision": source["source_revision"],
                            "source_records": source["row_count"],
                            "prepared_directory": str(Path(attached[(source["source_id"], source["source_revision"])]["manifest_path"]).parent),
                            "queue_records": sum(queue["state_counts"].values()),
                            "job": job,
                        })
        return rows

    def start(self, queue_id: str, *, workers: int | None = None, max_requests: int | None = None,
              limit: int | None = None, retry_incomplete: bool = True) -> dict:
        with CurationDatabase(self.database_path) as database:
            database.get_queue(queue_id)
        if workers is not None and not 1 <= workers <= 16:
            raise ValueError("文档并发需为 1～16")
        if max_requests is not None and not 1 <= max_requests <= 24:
            raise ValueError("API 并发需为 1～24")
        if limit is not None and limit < 1:
            raise ValueError("本次处理条数必须大于零")
        for config in self.configs.values():
            if config.provider in {"deepseek", "dashscope", "deepseek_web", "qwen_web", "kimi_web", "doubao_web"} and not config.api_key:
                raise ValueError(f"{config.provider} 凭据未配置")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("已有自动清洗任务运行中，请先暂停")
            self._queue_id = queue_id
            self._error = None
            self._stop = threading.Event()
            stop = self._stop
            configs = self.configs.copy()
            output = self._output(queue_id)

            def run():
                try:
                    if any(config.provider == "llamacpp" for config in configs.values()) and self.local_model is not None:
                        self.local_model.wait_ready()
                    cleaners = {source: LlmCleaner(config, self.root / "llm_suggestions")
                                for source, config in configs.items()}
                    settings = ModelSettings(self.root)
                    failure_fallback = None
                    if settings.read().failure_fallback:
                        paid = settings.config("deepseek")
                        if paid.provider == "deepseek" and paid.api_key:
                            failure_fallback = LlmCleaner(paid, self.root / "llm_suggestions")
                    result = clean_queue(database_path=self.database_path, queue_id=queue_id,
                                         output_directory=output, limit=limit,
                                         cleaner=next(iter(cleaners.values())), cleaners=cleaners,
                                         failure_fallback=failure_fallback,
                                         workers=workers, max_requests=max_requests, stop_event=stop,
                                         retry_incomplete=retry_incomplete)
                    if result["complete"]:
                        self.export(queue_id)
                except Exception as exc:
                    with self._lock:
                        self._error = str(exc)
                finally:
                    # Batch web sessions belong to this run; the interactive
                    # single-document chat is separate.
                    from .deepseek_web import cleanup_batch_sessions
                    errors = cleanup_batch_sessions(self.root, configs)
                    manifest_path = output / "manifest.json"
                    if manifest_path.exists():
                        try:
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            manifest["web_session_cleanup_errors"] = errors
                            write_snapshot(manifest_path, manifest)
                        except (OSError, ValueError):
                            pass

            self._thread = threading.Thread(target=run, name=f"auto-clean-{queue_id}", daemon=True)
            self._thread.start()
        return self.status(queue_id)

    def export(self, queue_id: str, output_directory: Path | None = None) -> dict:
        with CurationDatabase(self.database_path) as database:
            queue = database.get_queue(queue_id)
            # Bind the export to the revision observed before collecting manual
            # decisions. A concurrent save then marks this output stale instead
            # of falsely reporting that it contains the newest review.
            manual_revision = database.get_project(queue["project_id"])["current_revision"]
            manual = self._manual_reviews(database, queue)
        return export_effective(self.database_path, queue_id,
                                output_directory or self._output(queue_id), manual,
                                manual_revision=manual_revision)

    def stop(self, queue_id: str) -> dict:
        with self._lock:
            if self._queue_id != queue_id or self._thread is None or not self._thread.is_alive():
                raise ValueError("该队列没有正在运行的自动清洗任务")
            self._stop.set()
        return self.status(queue_id)
