from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CliEndToEndTests(unittest.TestCase):
    def run_cli(self, *arguments: str):
        completed = subprocess.run(
            [sys.executable, "-m", "label.cli", *map(str, arguments)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            timeout=30,
        )
        if completed.returncode:
            self.fail(
                f"CLI failed ({completed.returncode}): {' '.join(map(str, arguments))}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return json.loads(completed.stdout)

    def test_cli_import_to_materialize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(
                '{"id":"1","text":"第一条"}\n'
                '{"id":"2","text":"第二条"}\n',
                encoding="utf-8",
            )
            mapping = root / "mapping.json"
            mapping.write_text(
                json.dumps(
                    {"text_fields": ["text"], "local_id_field": "id"},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            imported = self.run_cli(
                "import",
                "--adapter",
                "jsonl",
                "--path",
                source,
                "--mapping",
                mapping,
                "--storage",
                root / "managed",
                "--source-id",
                "cli-source",
                "--license",
                "test",
            )
            prepared = self.run_cli(
                "prepare",
                "--source-revision-directory",
                imported["revision_directory"],
            )
            database = root / "curation.sqlite3"
            self.run_cli(
                "project-create",
                "--database",
                database,
                "--name",
                "CLI project",
                "--project-id",
                "cli-project",
            )
            self.run_cli(
                "source-attach",
                "--database",
                database,
                "--project-id",
                "cli-project",
                "--prepare-revision-directory",
                prepared["revision_directory"],
            )
            self.run_cli(
                "queue-full",
                "--database",
                database,
                "--project-id",
                "cli-project",
                "--name",
                "CLI queue",
                "--queue-id",
                "cli-queue",
            )
            shown = self.run_cli(
                "show-item",
                "--database",
                database,
                "--queue-id",
                "cli-queue",
                "--ordinal",
                "0",
            )
            self.assertEqual(shown["review_text"], "第一条")
            review = self.run_cli(
                "review-document",
                "--database",
                database,
                "--queue-id",
                "cli-queue",
                "--ordinal",
                "0",
                "--decision",
                "keep",
                "--quality",
                "2",
            )
            materialized = self.run_cli(
                "materialize",
                "--database",
                database,
                "--project-id",
                "cli-project",
                "--policy",
                "keep_only",
                "--snapshot-event-seq",
                str(review["event_seq"]),
                "--output-root",
                root / "exports",
            )
            self.assertEqual(materialized["manifest"]["output_records"], 1)


if __name__ == "__main__":
    unittest.main()
