"""Export one current clean corpus, with saved human decisions taking precedence."""
from __future__ import annotations

import fcntl
import json
import os
from datetime import UTC, datetime
from heapq import merge
from pathlib import Path

from .documents import QueueTextReader
from .cleaning_progress import CleaningProgressIndex, write_snapshot
from .database import CurationDatabase


def export_effective(database_path: Path, queue_id: str, output: Path,
                     manual: dict[int, dict], *, manual_revision: int | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".export.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with CurationDatabase(database_path) as database, CleaningProgressIndex(output) as index:
            queue = database.get_queue(queue_id)
            index.sync()
            reader = QueueTextReader(database, queue)
            project_revision = (manual_revision if manual_revision is not None else
                                database.get_project(queue["project_id"])["current_revision"])
            manual_ordinals = sorted(ordinal for ordinal, item in manual.items()
                                     if item["decision"] in {"keep", "drop"} or item["has_edit"])
            rows = index.db.execute(
                "SELECT ordinal, doc_id, status, cleaned_start, cleaned_end FROM results ORDER BY ordinal"
            )
            batch = ((row["ordinal"], dict(row)) for row in rows)
            human = ((ordinal, None) for ordinal in manual_ordinals)
            combined = merge(batch, human, key=lambda item: item[0])
            destination = output / "effective_cleaned.jsonl"
            temporary = output / ".effective_cleaned.jsonl.tmp"
            counts = {"keep": 0, "drop": 0, "incomplete": 0}
            (output / "cleaned.jsonl").touch(exist_ok=True)
            with (output / "cleaned.jsonl").open("rb") as cleaned, temporary.open("wb") as exported:
                previous_ordinal = -1
                batch_row = None
                for ordinal, item in combined:
                    if ordinal != previous_ordinal and previous_ordinal >= 0:
                        _write_effective(previous_ordinal, batch_row, manual, reader, cleaned, exported, counts)
                        batch_row = None
                    if item is not None:
                        batch_row = item
                    previous_ordinal = ordinal
                if previous_ordinal >= 0:
                    _write_effective(previous_ordinal, batch_row, manual, reader, cleaned, exported, counts)
                exported.flush()
                os.fsync(exported.fileno())
            temporary.replace(destination)
            indexed_counts, scanned, attempts = index.summary()
            result = {"queue_id": queue_id, "updated_at": datetime.now(UTC).isoformat(),
                      "project_revision": project_revision,
                      "batch_attempts": attempts, "batch_scanned": scanned,
                      "batch_counts": dict(indexed_counts), "counts": counts,
                      "total": sum(queue["state_counts"].values()),
                      "exported_records": counts["keep"],
                      "dataset": str(destination.resolve())}
            write_snapshot(output / "effective_manifest.json", result)
            return result


def _write_effective(ordinal: int, batch: dict | None, manual: dict[int, dict],
                     reader: QueueTextReader, cleaned, exported, counts: dict) -> None:
    decision = manual.get(ordinal, {}).get("decision")
    if decision in {"keep", "drop"}:
        if decision == "drop":
            counts["drop"] += 1
            return
        text = reader.read(ordinal)["materialized_text"].strip()
    elif decision == "unsure" and manual[ordinal]["has_edit"]:
        counts["incomplete"] += 1
        return
    elif batch is not None:
        if batch["status"] == "drop":
            counts["drop"] += 1
            return
        if batch["status"] != "keep":
            counts["incomplete"] += 1
            return
        text = CleaningProgressIndex.read_text(cleaned, batch["cleaned_start"], batch["cleaned_end"])
    else:
        return
    if not text:
        raise ValueError(f"已保留的正文为空：{ordinal}")
    exported.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
    counts["keep"] += 1
