"""Small SQLite index of the append-only batch log for shared progress views."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path


def write_snapshot(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class CleaningProgressIndex:
    def __init__(self, output: Path):
        self.output = output
        self.path = output / "progress_index.sqlite3"
        output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA busy_timeout = 10000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS results (
                ordinal INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL,
                status TEXT NOT NULL,
                failure_reason TEXT,
                cleaned_start INTEGER NOT NULL,
                cleaned_end INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS results_doc_id ON results(doc_id);
            CREATE TABLE IF NOT EXISTS cursor (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                progress_bytes INTEGER NOT NULL,
                cleaned_end INTEGER NOT NULL,
                attempts INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO cursor VALUES (1, 0, 0, 0);
        """)
        # Existing progress indexes predate the failure reason column.
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(results)")}
        if "failure_reason" not in columns:
            self.db.execute("ALTER TABLE results ADD COLUMN failure_reason TEXT")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS results_filter ON results(status, failure_reason, ordinal)"
        )
        self.db.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def sync(self) -> None:
        progress = self.output / "progress.jsonl"
        row = self.db.execute("SELECT * FROM cursor WHERE id = 1").fetchone()
        position, previous_end, attempts = row["progress_bytes"], row["cleaned_end"], row["attempts"]
        size = progress.stat().st_size if progress.exists() else 0
        if size < position:
            with self.db:
                self.db.execute("DELETE FROM results")
                self.db.execute("UPDATE cursor SET progress_bytes=0, cleaned_end=0, attempts=0 WHERE id=1")
            position = previous_end = attempts = 0
        if size == position:
            self._backfill_failure_reasons(progress)
            return
        with progress.open("rb") as stream:
            stream.seek(position)
            while True:
                batch = []
                while len(batch) < 1000:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    item = json.loads(line)
                    end = int(item["cleaned_offset"])
                    if end < previous_end:
                        raise ValueError("清洗日志中的正文偏移量倒退")
                    failure_reason = self.failure_reason(item)
                    batch.append((int(item["ordinal"]), item["doc_id"], item["status"],
                                  failure_reason, previous_end, end))
                    previous_end = end
                    position = stream.tell()
                if not batch:
                    break
                attempts += len(batch)
                with self.db:
                    self.db.executemany("""
                        INSERT INTO results(ordinal, doc_id, status, failure_reason, cleaned_start, cleaned_end)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ordinal) DO UPDATE SET
                          doc_id=excluded.doc_id, status=excluded.status,
                          failure_reason=excluded.failure_reason,
                          cleaned_start=excluded.cleaned_start, cleaned_end=excluded.cleaned_end
                    """, batch)
                    self.db.execute("""UPDATE cursor SET progress_bytes=?, cleaned_end=?, attempts=? WHERE id=1""",
                                    (position, previous_end, attempts))
                if len(batch) < 1000:
                    break
        self._backfill_failure_reasons(progress)

    @staticmethod
    def failure_reason(item: dict) -> str | None:
        if item["status"] != "incomplete":
            return None
        if item.get("failure_reason"):
            return item["failure_reason"]
        error = item.get("error", "")
        if "Content Exists Risk" in error:
            return "content_risk"
        if "HTTP 400" in error:
            return "rejected"
        return "error" if error else "uncertain"

    def _backfill_failure_reasons(self, progress: Path) -> None:
        missing = {row["ordinal"] for row in self.db.execute(
            "SELECT ordinal FROM results WHERE status='incomplete' AND failure_reason IS NULL"
        )}
        if not missing or not progress.is_file():
            return
        reasons = {}
        with progress.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                item = json.loads(line)
                if item["ordinal"] in missing:
                    reasons[item["ordinal"]] = self.failure_reason(item)
        with self.db:
            self.db.executemany(
                "UPDATE results SET failure_reason=? WHERE ordinal=? AND status='incomplete' AND failure_reason IS NULL",
                ((reason, ordinal) for ordinal, reason in reasons.items()),
            )

    def result(self, ordinal: int, doc_id: str | None = None) -> dict | None:
        """Read one committed result; incomplete and dropped rows have no text."""
        row = self.db.execute(
            "SELECT doc_id, status, cleaned_start, cleaned_end FROM results WHERE ordinal=?",
            (ordinal,),
        ).fetchone()
        if row is None:
            return None
        if doc_id is not None and row["doc_id"] != doc_id:
            raise ValueError(f"批量清洗结果与队列文档不一致：{ordinal}")
        text = None
        if row["status"] == "keep":
            with (self.output / "cleaned.jsonl").open("rb") as stream:
                text = self.read_text(stream, row["cleaned_start"], row["cleaned_end"])
        return {"status": row["status"], "text": text}

    @staticmethod
    def read_text(stream, start: int, end: int) -> str:
        size = end - start
        if size <= 0:
            raise ValueError("批量清洗正文偏移量无效")
        stream.seek(start)
        text = json.loads(stream.read(size))["text"]
        if not isinstance(text, str) or not text:
            raise ValueError("批量清洗正文为空")
        return text

    def summary(self) -> tuple[Counter, int, int]:
        counts = Counter({row["status"]: row["n"] for row in self.db.execute(
            "SELECT status, COUNT(*) AS n FROM results GROUP BY status")})
        scanned = self.db.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 FROM results").fetchone()[0]
        attempts = self.db.execute("SELECT attempts FROM cursor WHERE id=1").fetchone()[0]
        return counts, scanned, attempts

    def statuses(self, ordinals: list[int]) -> dict[int, str]:
        result = {}
        for start in range(0, len(ordinals), 800):
            group = ordinals[start:start + 800]
            placeholders = ",".join("?" for _ in group)
            for row in self.db.execute(
                f"SELECT ordinal, status FROM results WHERE ordinal IN ({placeholders})", group
            ):
                result[row["ordinal"]] = row["status"]
        return result


    def details(self, ordinals: list[int]) -> dict[int, dict]:
        """Read just the requested queue positions for the sidebar."""
        found = {}
        for start in range(0, len(ordinals), 800):
            group = ordinals[start:start + 800]
            if not group:
                continue
            placeholders = ",".join("?" for _ in group)
            for row in self.db.execute(
                f"SELECT ordinal, status, failure_reason FROM results WHERE ordinal IN ({placeholders})", group
            ):
                found[row["ordinal"]] = {"status": row["status"],
                                          "failure_reason": row["failure_reason"]}
        return found

    def filtered(self, start: int, statuses: tuple[str, ...]):
        """Stream matching batch rows using the progress index, in queue order."""
        placeholders = ",".join("?" for _ in statuses)
        return self.db.execute(
            f"SELECT ordinal, status, failure_reason FROM results "
            f"WHERE ordinal >= ? AND status IN ({placeholders}) ORDER BY ordinal",
            (start, *statuses),
        )
