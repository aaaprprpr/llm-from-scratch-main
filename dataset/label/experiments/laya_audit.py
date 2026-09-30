"""Materialize the fixed, manually reviewed block audit from a frozen test file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dataset.label.experiments.laya_dataset import write_jsonl

LABELS = Path(__file__).with_name("laya_audit_labels.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    teacher = {}
    for line in args.teacher_test.open(encoding="utf-8"):
        row = json.loads(line)
        teacher[row["doc_id"], row["block_index"]] = row
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = []
    for item in labels["items"]:
        row = teacher[item["doc_id"], item["block_index"]]
        if (row["label"] != item["teacher_label"] or
                hashlib.sha256(row["text"].encode()).hexdigest() != item["text_sha256"]):
            raise ValueError("Audit label refers to changed teacher text")
        if item["reviewed_label"] != "unsure":
            rows.append({**row, "teacher_label": row["label"],
                         "label": item["reviewed_label"]})
    count, digest = write_jsonl(args.output, rows)
    print(json.dumps({"reviewer": labels["reviewer"], "rows": count,
                      "sha256": digest, "teacher_test": str(args.teacher_test)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
