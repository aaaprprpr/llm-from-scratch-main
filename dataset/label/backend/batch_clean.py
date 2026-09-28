"""Resumable, bounded-concurrency cleaning of a review queue into JSONL."""
from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from itertools import chain, islice
from pathlib import Path

from .blocks import parse_blocks
from .database import CurationDatabase
from .documents import QueueTextReader
from .cleaning_progress import write_snapshot
from .llm_cleaning import CleaningConfig, LlmCleaner, LlmCleaningError, PROMPT_VERSION


def output_path(root: Path, queue_id: str, config: CleaningConfig) -> Path:
    """Keep using the queue's durable log when only the cleaning prompt changes."""
    model = re.sub(r"[^A-Za-z0-9._-]", "-", config.model)
    directory = root / "batch_cleaned" / queue_id
    current = directory / f"{config.provider}-{model}-{PROMPT_VERSION}"
    # One queue keeps its durable progress when the selected model changes.
    previous = [path for path in directory.iterdir() if path.is_dir()
                and (path / "progress.jsonl").is_file()
                and (path / "progress.jsonl").stat().st_size > 0] if directory.exists() else []
    if previous:
        def progress_key(path: Path) -> tuple[int, int]:
            manifest = path / "manifest.json"
            try:
                processed = json.loads(manifest.read_text(encoding="utf-8")).get("processed", 0)
            except (OSError, ValueError):
                processed = 0
            return processed, (path / "progress.jsonl").stat().st_mtime_ns
        return max(previous, key=progress_key)
    return current


def _read_progress(stream) -> tuple[int, int, Counter, list[int], Counter, int]:
    """Restore latest decisions and the last durable cleaned-file offset."""
    next_ordinal = clean_offset = attempts = 0
    counts: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    incomplete: set[int] = set()
    stream.seek(0)
    while True:
        start = stream.tell()
        line = stream.readline()
        if not line:
            break
        try:
            if not line.endswith(b"\n"):
                raise ValueError("partial progress line")
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            stream.truncate(start)
            break
        ordinal = item["ordinal"]
        if ordinal == next_ordinal:
            next_ordinal += 1
        elif ordinal >= next_ordinal or ordinal not in incomplete:
            raise ValueError("批量清洗进度文件的条目序号不连续")
        if item["cleaned_offset"] < clean_offset:
            raise ValueError("批量清洗数据文件偏移量倒退")
        if ordinal in incomplete:
            counts["incomplete"] -= 1
            if not counts["incomplete"]:
                del counts["incomplete"]
        status = item["status"]
        if status == "incomplete":
            incomplete.add(ordinal)
        else:
            incomplete.discard(ordinal)
        counts[status] += 1
        clean_offset = item["cleaned_offset"]
        attempts += 1
        for key in ("requests", "input_tokens", "output_tokens", "input_characters", "output_characters"):
            usage[key] += item.get(key, 0)
    stream.seek(0, os.SEEK_END)
    return next_ordinal, clean_offset, counts, sorted(incomplete), usage, attempts


def _process(ordinal: int, value: dict, queue_id: str, cleaner: LlmCleaner) -> dict:
    started = time.monotonic()
    doc_id = value["document"]["doc_id"]
    text = value["materialized_text"]
    review = value["document_review"] or {}
    record = {"ordinal": ordinal, "doc_id": doc_id, "status": "incomplete", "text": None,
              "requests": 0, "input_tokens": 0, "output_tokens": 0,
              "input_characters": len(text), "output_characters": 0}
    if review.get("decision") == "drop":
        record["status"] = "drop"
    elif review.get("decision") == "keep":
        record["status"], record["text"] = "keep", text.strip()
    elif not text.strip():
        record["status"] = "drop"
    else:
        parsed = parse_blocks(text, doc_id)
        blocks = [{"id": block.block_id, "text": block.text,
                   "separator_after": block.separator_after} for block in parsed]
        provenance = {**value["provenance"], "doc_id": doc_id,
                      "queue_id": queue_id, "ordinal": ordinal}
        try:
            result = cleaner.clean(blocks, title=value["provenance"]["title"],
                                   provenance=provenance)
            record["prompt_sha256"] = result.get("prompt_sha256")
            record["model"] = result.get("model", cleaner.config.model)
            cached = "已载入上次完成" in " ".join(result.get("warnings", []))
            if not cached:
                record["requests"] += sum(item.get("requests", 0) for item in result.get("chunk_timings", []))
                record["input_tokens"] += result.get("input_tokens", 0)
                record["output_tokens"] += result.get("output_tokens", 0)
            if cleaner.is_complete(result):
                record["status"] = result["decision"]
                if record["status"] == "keep":
                    record["text"] = result["edited_text"]
                    if not record["text"].strip():
                        raise ValueError("模型保留了空正文")
        except (LlmCleaningError, ValueError) as exc:
            if "HTTP 401" in str(exc) or "HTTP 403" in str(exc) or "HTTP 429" in str(exc):
                raise
            record["error"] = str(exc)[:300]
            if "Content Exists Risk" in str(exc):
                record["failure_reason"] = "content_risk"
    record["output_characters"] = len(record["text"] or "")
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def clean_queue(
    *, database_path: str | Path, queue_id: str, output_directory: str | Path,
    limit: int | None = None, cleaner: LlmCleaner | None = None,
    workers: int | None = None, max_requests: int | None = None,
    stop_event: threading.Event | None = None,
) -> dict:
    """Clean bounded batches concurrently, committing results in queue order."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cleaner = cleaner or LlmCleaner(CleaningConfig.from_file(), Path(database_path).parent / "llm_suggestions")
    remote = cleaner.config.provider in {"deepseek", "dashscope"}
    workers = workers if workers is not None else (6 if remote else 1)
    max_requests = max_requests if max_requests is not None else (8 if remote else 1)
    if not 1 <= workers <= 16 or not 1 <= max_requests <= 24:
        raise ValueError("文档并发需为 1～16，API 并发需为 1～24")
    stop_event = stop_event or threading.Event()
    semaphore = threading.BoundedSemaphore(max_requests)
    local = threading.local()

    def worker_cleaner():
        if not hasattr(local, "cleaner"):
            local.cleaner = (LlmCleaner(cleaner.config, cleaner.report_directory, semaphore)
                             if isinstance(cleaner, LlmCleaner) else cleaner)
        return local.cleaner

    def run_item(ordinal: int, value: dict) -> dict:
        return _process(ordinal, value, queue_id, worker_cleaner())

    lock_path = output / ".lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该队列的批量清洗任务已经在运行") from exc
        with CurationDatabase(database_path) as database:
            queue = database.get_queue(queue_id)
            total = sum(queue["state_counts"].values())
            job = {"queue_id": queue_id, "total": total,
                   "provider": cleaner.config.provider, "model": cleaner.config.model,
                   "prompt_version": PROMPT_VERSION}
            job_path = output / "job.json"
            if job_path.exists():
                previous_job = json.loads(job_path.read_text(encoding="utf-8"))
                if any(previous_job.get(key) != job[key] for key in ("queue_id", "total")):
                    raise ValueError("输出目录属于不同的清洗队列")
                versions = previous_job.get("prompt_versions", [previous_job["prompt_version"]])
                if PROMPT_VERSION not in versions:
                    versions.append(PROMPT_VERSION)
                job["prompt_versions"] = versions
                models = previous_job.get("model_history", [{"provider": previous_job["provider"],
                                                              "model": previous_job["model"]}])
                current_model = {"provider": job["provider"], "model": job["model"]}
                if current_model not in models:
                    models.append(current_model)
                job["model_history"] = models
                if previous_job != job:
                    write_snapshot(job_path, job)
            else:
                job["prompt_versions"] = [PROMPT_VERSION]
                job["model_history"] = [{"provider": job["provider"], "model": job["model"]}]
                write_snapshot(job_path, job)
            reader = QueueTextReader(database, queue)
            progress_path = output / "progress.jsonl"
            cleaned_path = output / "cleaned.jsonl"
            with progress_path.open("a+b") as progress, cleaned_path.open("a+b") as cleaned:
                next_ordinal, clean_offset, counts, incomplete, usage, attempts = _read_progress(progress)
                if cleaned.seek(0, os.SEEK_END) < clean_offset:
                    raise ValueError("清洗数据文件短于进度记录")
                cleaned.truncate(clean_offset)
                cleaned.seek(0, os.SEEK_END)
                previous = {}
                manifest_path = output / "manifest.json"
                if manifest_path.exists():
                    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                started = time.monotonic()
                elapsed_before = float(previous.get("elapsed_seconds", 0))
                session_attempts = 0
                pending_records: list[dict] = []
                last_snapshot = 0.0

                def snapshot(status: str, error: str | None = None) -> dict:
                    elapsed = elapsed_before + time.monotonic() - started
                    completed = counts["keep"] + counts["drop"]
                    rate = session_attempts / max(time.monotonic() - started, 0.001)
                    remaining = total - completed
                    manifest = {**job, "status": status,
                                "started_at": previous.get("started_at", datetime.now(UTC).isoformat()),
                                "processed": next_ordinal, "last_ordinal": next_ordinal - 1,
                                "completed": completed, "remaining": remaining,
                                "counts": dict(counts), "attempts": attempts,
                                "requests": usage["requests"],
                                "input_tokens": usage["input_tokens"],
                                "output_tokens": usage["output_tokens"],
                                "input_characters": usage["input_characters"],
                                "output_characters": usage["output_characters"],
                                "elapsed_seconds": round(elapsed, 2),
                                "rate_per_minute": round(rate * 60, 2),
                                "eta_seconds": round(remaining / rate) if rate and remaining else None,
                                "workers": workers, "max_requests": max_requests,
                                "updated_at": datetime.now(UTC).isoformat(),
                                "cleaned_dataset": str(cleaned_path.resolve()),
                                "queue_exhausted": next_ordinal == total,
                                "complete": next_ordinal == total and not counts["incomplete"]}
                    if error:
                        manifest["error"] = error[:500]
                    write_snapshot(manifest_path, manifest)
                    return manifest

                def commit():
                    nonlocal next_ordinal, attempts, session_attempts, last_snapshot
                    if not pending_records:
                        return
                    progress_rows = []
                    for result in pending_records:
                        if result["text"] is not None:
                            cleaned.write((json.dumps({"text": result["text"]}, ensure_ascii=False) + "\n").encode("utf-8"))
                        progress_rows.append({key: value for key, value in result.items() if key != "text"})
                        progress_rows[-1]["cleaned_offset"] = cleaned.tell()
                    cleaned.flush()
                    os.fsync(cleaned.fileno())
                    for row in progress_rows:
                        progress.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                        ordinal, status = row["ordinal"], row["status"]
                        if ordinal < next_ordinal:
                            counts["incomplete"] -= 1
                            if not counts["incomplete"]:
                                del counts["incomplete"]
                        else:
                            next_ordinal += 1
                        counts[status] += 1
                        attempts += 1
                        session_attempts += 1
                        for key in ("requests", "input_tokens", "output_tokens", "input_characters", "output_characters"):
                            usage[key] += row.get(key, 0)
                    progress.flush()
                    os.fsync(progress.fileno())
                    pending_records.clear()
                    if time.monotonic() - last_snapshot >= 1:
                        snapshot("running")
                        last_snapshot = time.monotonic()

                budget = limit if limit is not None else total + len(incomplete)
                new_ordinals = range(next_ordinal, min(total, next_ordinal + max(0, budget - len(incomplete))))
                ordinals = islice(chain(incomplete, new_ordinals), budget)
                snapshot("running")
                try:
                    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batch-clean") as pool:
                        tasks = iter(ordinals)
                        in_flight = {}
                        ready = {}
                        submission_order = deque()
                        exhausted = False
                        buffer_limit = min(workers * 16, 128)
                        while True:
                            while (not exhausted and not stop_event.is_set()
                                   and len(in_flight) < workers
                                   and len(submission_order) < buffer_limit):
                                try:
                                    ordinal = next(tasks)
                                except StopIteration:
                                    exhausted = True
                                    break
                                value = reader.read(ordinal)
                                in_flight[pool.submit(run_item, ordinal, value)] = ordinal
                                submission_order.append(ordinal)
                            if not in_flight:
                                break
                            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                            for future in finished:
                                ready[in_flight.pop(future)] = future.result()
                            while submission_order and submission_order[0] in ready:
                                pending_records.append(ready.pop(submission_order.popleft()))
                                if len(pending_records) >= 8:
                                    commit()
                    commit()
                except Exception as exc:
                    commit()
                    snapshot("failed", str(exc))
                    raise
                status = ("completed" if next_ordinal == total and not counts["incomplete"] else "paused" if stop_event.is_set()
                          else "needs_retry" if next_ordinal == total else "partial")
                return snapshot(status)
