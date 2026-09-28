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
from itertools import islice
from typing import Mapping
from pathlib import Path

from .blocks import parse_blocks
from .database import CurationDatabase
from .documents import QueueTextReader
from .deepseek_web import QWEN_WEB_BATCH_SLOTS, DEEPSEEK_WEB_BATCH_SLOTS
from .cleaning_progress import CleaningProgressIndex, write_snapshot
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


def _read_progress(stream, total: int):
    """Restore completed ordinals, retry holes, and the last durable text offset."""
    next_ordinal = clean_offset = attempts = processed = 0
    seen = bytearray(total)
    counts: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    source_attempts: Counter[str] = Counter()
    source_outcomes: dict[str, Counter[str]] = {}
    incomplete: set[int] = set()
    paid_attempted: set[int] = set()
    risk_seen: set[int] = set()
    source_rejected: set[int] = set()
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
        if not 0 <= ordinal < total or seen[ordinal] and ordinal not in incomplete:
            raise ValueError("批量清洗进度文件的条目序号无效或重复")
        if item["cleaned_offset"] < clean_offset:
            raise ValueError("批量清洗数据文件偏移量倒退")
        if seen[ordinal]:
            counts["incomplete"] -= 1
            if not counts["incomplete"]:
                del counts["incomplete"]
        else:
            seen[ordinal] = 1
            processed += 1
        next_ordinal = max(next_ordinal, ordinal + 1)
        status = item["status"]
        if item.get("source") == "deepseek_api_retry":
            paid_attempted.add(ordinal)
        if "FAIL_SYS_USER_VALIDATE" in item.get("error", ""):
            source_rejected.add(ordinal)
        if status == "incomplete":
            incomplete.add(ordinal)
            reason = CleaningProgressIndex.failure_reason(item) or "uncertain"
            if reason == "content_risk":
                risk_seen.add(ordinal)
        else:
            incomplete.discard(ordinal)
            paid_attempted.discard(ordinal)
        counts[status] += 1
        clean_offset = item["cleaned_offset"]
        attempts += 1
        if item.get("source"):
            source = item["source"]
            source_attempts[source] += 1
            source_outcomes.setdefault(source, Counter())[status] += 1
        for key in ("requests", "input_tokens", "output_tokens", "input_characters", "output_characters"):
            usage[key] += item.get(key, 0)
    holes = []
    cursor = seen.find(0, 0, next_ordinal)
    while cursor >= 0:
        holes.append(cursor)
        cursor = seen.find(0, cursor + 1, next_ordinal)
    stream.seek(0, os.SEEK_END)
    pending = sorted(incomplete.union(holes))
    return (next_ordinal, clean_offset, counts, pending, incomplete,
            paid_attempted, risk_seen, source_rejected, usage, attempts,
            source_attempts, source_outcomes, seen, processed)


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
            elif any("Content Exists Risk" in warning for warning in result.get("warnings", [])):
                record["failure_reason"] = "content_risk"
        except (LlmCleaningError, ValueError) as exc:
            if ("HTTP 401" in str(exc) or "HTTP 403" in str(exc) or "HTTP 429" in str(exc)
                    or "签名材料已过期" in str(exc) or "签名材料已用完" in str(exc)
                    or "缺少千问凭据" in str(exc)):
                raise
            record["error"] = str(exc)[:300]
            record["failure_reason"] = "content_risk" if "Content Exists Risk" in str(exc) else "error"
    record["output_characters"] = len(record["text"] or "")
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def clean_queue(
    *, database_path: str | Path, queue_id: str, output_directory: str | Path,
    limit: int | None = None, cleaner: LlmCleaner | None = None,
    cleaners: Mapping[str, LlmCleaner] | None = None,
    failure_fallback: LlmCleaner | None = None,
    workers: int | None = None, max_requests: int | None = None,
    stop_event: threading.Event | None = None,
    retry_incomplete: bool = True,
) -> dict:
    """Clean bounded batches concurrently, committing completed results durably."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cleaner = cleaner or LlmCleaner(CleaningConfig.from_file(), Path(database_path).parent / "llm_suggestions")
    primary_sources = dict(cleaners) if cleaners is not None else {"default": cleaner}
    if not primary_sources:
        raise ValueError("至少选择一个批量模型")
    sources = primary_sources.copy()
    fallback_source = next((source for source, selected in sources.items()
                            if selected.config.provider == "deepseek"), None)
    retry_only: set[str] = set()
    if failure_fallback is not None and fallback_source is None:
        if failure_fallback.config.provider != "deepseek":
            raise ValueError("失败备用模型必须是 DeepSeek API")
        fallback_source = "deepseek_api_retry"
        sources[fallback_source] = failure_fallback
        retry_only.add(fallback_source)
    # A model gets another document whenever one of its slots finishes. Faster
    # sources naturally process more rows; a slow local model never queues up a
    # fixed fraction of the corpus and blocks all remote workers.
    capacities = {source: (1 if source in retry_only or selected.config.provider == "llamacpp" else
                           DEEPSEEK_WEB_BATCH_SLOTS if selected.config.provider == "deepseek_web" else
                           QWEN_WEB_BATCH_SLOTS if selected.config.provider == "qwen_web" else 6)
                  for source, selected in sources.items()}
    total_capacity = sum(capacities.values())
    workers = min(workers if workers is not None else min(total_capacity, 16), total_capacity)
    max_requests = max_requests if max_requests is not None else workers
    if not 1 <= workers <= 16 or not 1 <= max_requests <= 24:
        raise ValueError("文档并发需为 1～16，API 并发需为 1～24")
    stop_event = stop_event or threading.Event()
    semaphore = threading.BoundedSemaphore(max_requests)
    local = threading.local()

    def worker_cleaner(source: str, slot: int):
        if not hasattr(local, "cleaners"):
            local.cleaners = {}
        key = (source, slot)
        if key not in local.cleaners:
            selected = sources[source]
            local.cleaners[key] = (LlmCleaner(
                selected.config, selected.report_directory, semaphore,
                web_session_key=f"batch_{slot}" if selected.config.provider.endswith("_web") else "single",
            ) if isinstance(selected, LlmCleaner) else selected)
        return local.cleaners[key]

    def run_item(ordinal: int, value: dict, source: str, slot: int) -> dict:
        record = _process(ordinal, value, queue_id, worker_cleaner(source, slot))
        if "model" in record or "error" in record:
            record["source"] = source
        return record

    lock_path = output / ".lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该队列的批量清洗任务已经在运行") from exc
        with CurationDatabase(database_path) as database:
            queue = database.get_queue(queue_id)
            total = sum(queue["state_counts"].values())
            selected_models = [{"source": source, "provider": item.config.provider,
                                "model": item.config.model, "retry_only": source in retry_only}
                               for source, item in sources.items()]
            job = {"queue_id": queue_id, "total": total,
                   "provider": cleaner.config.provider, "model": cleaner.config.model,
                   "sources": selected_models, "prompt_version": PROMPT_VERSION}
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
                for selected in selected_models:
                    current_model = {"provider": selected["provider"], "model": selected["model"]}
                    if current_model not in models:
                        models.append(current_model)
                job["model_history"] = models
                if previous_job != job:
                    write_snapshot(job_path, job)
            else:
                job["prompt_versions"] = [PROMPT_VERSION]
                job["model_history"] = [{"provider": item["provider"], "model": item["model"]}
                                        for item in selected_models]
                write_snapshot(job_path, job)
            reader = QueueTextReader(database, queue)
            progress_path = output / "progress.jsonl"
            cleaned_path = output / "cleaned.jsonl"
            with progress_path.open("a+b") as progress, cleaned_path.open("a+b") as cleaned:
                (next_ordinal, clean_offset, counts, incomplete, failed_ordinals,
                 paid_attempted, risk_seen, source_rejected, usage, attempts,
                 source_attempts, source_outcomes, seen, processed_count) = _read_progress(progress, total)
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
                session_completed = 0
                pending_records: list[dict] = []
                source_errors: dict[str, str] = {}
                source_activity = {source: {"capacity": capacity, "active": 0,
                                            "completed": 0, "incomplete": 0,
                                            "last_seconds": None}
                                   for source, capacity in capacities.items()}
                source_recent = {source: deque() for source in sources}
                last_snapshot = 0.0

                def snapshot(status: str, error: str | None = None) -> dict:
                    elapsed = elapsed_before + time.monotonic() - started
                    completed = counts["keep"] + counts["drop"]
                    now = time.monotonic()
                    rate = session_completed / max(now - started, 0.001)
                    remaining = total - completed
                    activity = {}
                    for source, details in source_activity.items():
                        recent = source_recent[source]
                        while recent and recent[0] < now - 60:
                            recent.popleft()
                        activity[source] = {**details,
                            "rate_per_minute": round(len(recent) * 60 / min(60, max(now - started, 1)), 1)}
                    manifest = {**job, "status": status,
                                "started_at": previous.get("started_at", datetime.now(UTC).isoformat()),
                                "processed": processed_count, "last_ordinal": next_ordinal - 1,
                                "completed": completed, "remaining": remaining,
                                "counts": dict(counts), "attempts": attempts,
                                "source_attempts": dict(source_attempts),
                                "source_outcomes": {source: dict(outcomes)
                                                    for source, outcomes in source_outcomes.items()},
                                "source_activity": activity,
                                "source_errors": source_errors.copy(),
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
                                "queue_exhausted": processed_count == total,
                                "complete": processed_count == total and not counts["incomplete"]}
                    if error:
                        manifest["error"] = error[:500]
                    write_snapshot(manifest_path, manifest)
                    return manifest

                def commit():
                    nonlocal next_ordinal, processed_count, attempts, session_completed, last_snapshot
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
                        if seen[ordinal]:
                            counts["incomplete"] -= 1
                            if not counts["incomplete"]:
                                del counts["incomplete"]
                        else:
                            seen[ordinal] = 1
                            processed_count += 1
                            next_ordinal = max(next_ordinal, ordinal + 1)
                        counts[status] += 1
                        attempts += 1
                        if status in {"keep", "drop"}:
                            session_completed += 1
                        if row.get("source"):
                            source = row["source"]
                            source_attempts[source] += 1
                            source_outcomes.setdefault(source, Counter())[status] += 1
                            activity = source_activity[source]
                            activity["last_seconds"] = row.get("elapsed_seconds")
                            if status in {"keep", "drop"}:
                                activity["completed"] += 1
                                source_recent[source].append(time.monotonic())
                            else:
                                activity["incomplete"] += 1
                        for key in ("requests", "input_tokens", "output_tokens", "input_characters", "output_characters"):
                            usage[key] += row.get(key, 0)
                    progress.flush()
                    os.fsync(progress.fileno())
                    pending_records.clear()
                    if time.monotonic() - last_snapshot >= 1:
                        snapshot("running")
                        last_snapshot = time.monotonic()

                budget = limit if limit is not None else total + len(incomplete)
                selected_incomplete = list(islice(incomplete, budget)) if retry_incomplete else []
                new_ordinals = range(next_ordinal, min(total, next_ordinal + max(0, budget - len(selected_incomplete))))
                snapshot("running")
                try:
                    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batch-clean") as pool:
                        tasks = iter(new_ordinals)
                        in_flight = {}
                        retry_ordinals = deque(selected_incomplete)
                        retry_exclude: dict[int, str] = {}
                        retry_preferred = {ordinal: fallback_source for ordinal in selected_incomplete
                                           if fallback_source and ordinal in failed_ordinals
                                           and ordinal not in paid_attempted and ordinal not in risk_seen
                                           and ordinal not in source_rejected}
                        retried: set[int] = set(failed_ordinals - source_rejected)
                        nonrisk_failed: set[int] = set()
                        available_slots = {source: deque(range(capacity))
                                           for source, capacity in capacities.items()}
                        source_turn = deque(primary_sources)
                        exhausted = False

                        def take_slot(excluded: str | None = None,
                                      preferred: str | None = None) -> tuple[str, int] | None:
                            if preferred and preferred != excluded and preferred not in source_errors:
                                if available_slots[preferred]:
                                    return preferred, available_slots[preferred].popleft()
                                return None
                            for _ in range(len(source_turn)):
                                source = source_turn[0]
                                source_turn.rotate(-1)
                                if source != excluded and source not in source_errors and available_slots[source]:
                                    return source, available_slots[source].popleft()
                            return None

                        while True:
                            while (not stop_event.is_set() and len(in_flight) < workers
                                   and (retry_ordinals or not exhausted)):
                                assignment = None
                                if retry_ordinals:
                                    retry_ordinal = retry_ordinals[0]
                                    assignment = take_slot(retry_exclude.get(retry_ordinal),
                                                           retry_preferred.get(retry_ordinal))
                                    if assignment is not None:
                                        ordinal = retry_ordinals.popleft()
                                        retry_exclude.pop(ordinal, None)
                                        retry_preferred.pop(ordinal, None)
                                if assignment is None:
                                    # A retry waiting for a different model must not idle
                                    # free slots that can keep cleaning fresh documents.
                                    if exhausted:
                                        break
                                    assignment = take_slot()
                                    if assignment is None:
                                        break
                                    try:
                                        ordinal = next(tasks)
                                    except StopIteration:
                                        exhausted = True
                                        available_slots[assignment[0]].appendleft(assignment[1])
                                        break
                                source, slot = assignment
                                value = reader.read(ordinal)
                                in_flight[pool.submit(run_item, ordinal, value, source, slot)] = (ordinal, source, slot)
                                source_activity[source]["active"] += 1
                            if not in_flight:
                                if (not stop_event.is_set() and (retry_ordinals or not exhausted)
                                        and all(source in source_errors for source in primary_sources)):
                                    raise LlmCleaningError("所选批量模型均不可用：" + "; ".join(
                                        f"{source}: {error}" for source, error in source_errors.items()))
                                break
                            finished, _ = wait(in_flight, timeout=1, return_when=FIRST_COMPLETED)
                            if not finished and time.monotonic() - last_snapshot >= 2:
                                snapshot("running")
                                last_snapshot = time.monotonic()
                            for future in finished:
                                ordinal, source, slot = in_flight.pop(future)
                                available_slots[source].append(slot)
                                source_activity[source]["active"] -= 1
                                try:
                                    record = future.result()
                                    pending_records.append(record)
                                    if record["status"] == "incomplete" and not stop_event.is_set():
                                        if record.get("failure_reason") == "content_risk":
                                            risk_seen.add(ordinal)
                                        else:
                                            nonrisk_failed.add(ordinal)
                                        if source == fallback_source:
                                            paid_attempted.add(ordinal)
                                        other = (ordinal not in retried and any(
                                            item != source and item not in source_errors
                                            for item in primary_sources))
                                        paid = (fallback_source if not other and ordinal not in risk_seen
                                                and ordinal not in source_rejected
                                                and fallback_source != source and ordinal not in paid_attempted
                                                and fallback_source not in source_errors else None)
                                        if other or paid:
                                            retried.add(ordinal)
                                            retry_exclude[ordinal] = source
                                            if paid:
                                                paid_attempted.add(ordinal)
                                                retry_preferred[ordinal] = paid
                                            retry_ordinals.append(ordinal)
                                except LlmCleaningError as exc:
                                    source_errors[source] = str(exc)[:200]
                                    if ordinal not in retried:
                                        retried.add(ordinal)
                                        retry_exclude[ordinal] = source
                                        retry_ordinals.append(ordinal)
                                    elif (ordinal in nonrisk_failed and ordinal not in risk_seen
                                          and ordinal not in source_rejected
                                          and fallback_source and fallback_source != source
                                          and ordinal not in paid_attempted
                                          and fallback_source not in source_errors):
                                        paid_attempted.add(ordinal)
                                        retry_exclude[ordinal] = source
                                        retry_preferred[ordinal] = fallback_source
                                        retry_ordinals.append(ordinal)
                            if len(pending_records) >= 8 or pending_records and time.monotonic() - last_snapshot >= 1:
                                commit()
                    commit()
                except Exception as exc:
                    commit()
                    snapshot("failed", str(exc))
                    raise
                status = ("completed" if processed_count == total and not counts["incomplete"] else "paused" if stop_event.is_set()
                          else "needs_retry" if processed_count == total else "partial")
                return snapshot(status)
