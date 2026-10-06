"""Prepare a three-way Laya experiment: retain, delete, or send to an editor.

This remains an offline experiment. 'edit' is a teacher suggestion, not human gold.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from dataset.label.backend.identity import sha256_text, stable_json
from dataset.label.laya.training.dataset import (
    DATABASE, REPORTS, human_rows, partition, row_for, usable, write_jsonl,
)

QUESTION_TERNARY = {
    "body": {
        "type": "choice",
        "instructions": "判断当前文字块应原样保留、整块删除，还是需要局部修剪。",
        "criteria": {
            "keep": "整块是可阅读的事实、解释、定义、事件或观点，可原样保留。",
            "drop": "整块没有可用正文，是导航、标题、名单、表格值、书目或其他残片。",
            "edit": "同一块里既有值得保留的正文又有应删的杂质，需要局部修剪。",
        },
    }
}
VERSION = "laya_ternary_block_dataset_v1"


def extract_rows(excluded_docs: set[str]):
    latest = {}
    for path in sorted(glob.glob(str(REPORTS / "*" / "*.json"))):
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            if report.get("provider") != "deepseek_web":
                continue
            result = report["result"]
            if (not result.get("complete") or result.get("decision") not in {"keep", "drop"}
                    or result.get("input_sha256") != sha256_text(stable_json(report["input_blocks"]))
                    or result.get("risk_fallback_chunks")):
                continue
            doc_id = report["provenance"]["doc_id"]
            if doc_id in excluded_docs:
                continue
            prior = latest.get(doc_id)
            if prior is None or (report.get("created_at", ""), path) > prior[:2]:
                latest[doc_id] = (report.get("created_at", ""), path, report)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    splits = defaultdict(list)
    counts = Counter()
    for doc_id, (_, _, report) in sorted(latest.items()):
        blocks = report["input_blocks"]
        removals = defaultdict(list)
        edited = {item["block_id"] for item in report["result"].get("edits", [])}
        for item in report["result"].get("removals", []):
            removals[item["block_id"]].append((item["start"], item["end"]))
        split = partition(doc_id)
        counts[f"{split}_docs"] += 1
        for index, block in enumerate(blocks):
            text = block["text"]
            if not usable(text):
                counts["skip_length"] += 1
                continue
            spans = removals[block["id"]]
            if block["id"] in edited:
                label = "edit"
            elif not spans:
                label = "keep"
            else:
                covered = bytearray(len(text))
                for start, end in spans:
                    if not 0 <= start <= end <= len(text):
                        raise ValueError(f"invalid removal span in {doc_id}")
                    covered[start:end] = b"\1" * (end - start)
                label = ("edit" if any(not covered[pos] and not char.isspace()
                                       for pos, char in enumerate(text)) else "drop")
            splits[split].append(row_for(blocks, index, report.get("title") or "",
                                       doc_id, label, "deepseek_web"))
            counts[f"{split}_{label}"] += 1
    return splits, counts


def train_case(row: dict) -> dict:
    label = row["label"]
    probabilities = {key: (.96 if key == label else .02)
                     for key in ("keep", "drop", "edit")}
    return {"state": row["text"], "questions": QUESTION_TERNARY,
            "gold": {"body": {"label": label, "probabilities": probabilities}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=4000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    connection = sqlite3.connect(f"file:{DATABASE.resolve()}?mode=ro", uri=True)
    human, human_ids, human_counts = human_rows(connection)
    splits, counts = extract_rows(human_ids)
    for split in ("train", "dev", "test"):
        if {row["doc_id"] for row in splits[split]} & human_ids:
            raise ValueError("human document leaked into teacher split")
    train = []
    for label in ("keep", "drop", "edit"):
        rows = [row for row in splits["train"] if row["label"] == label]
        rows.sort(key=lambda row: hashlib.sha256(
            f"{row['doc_id']}:{row['block_index']}:{VERSION}".encode()).digest())
        train.extend(rows[:args.train_per_class])
    train.sort(key=lambda row: hashlib.sha256(
        f"{row['doc_id']}:{row['block_index']}:order".encode()).digest())
    args.output.mkdir(parents=True)
    manifest = {"version": VERSION, "teacher": dict(counts), "human": dict(human_counts),
                "train_selected": dict(Counter(row["label"] for row in train)), "files": {}}
    files = {
        "train.jsonl": (train_case(row) for row in train),
        "teacher_dev.jsonl": splits["dev"],
        "teacher_test.jsonl": splits["test"],
    }
    for name, rows in files.items():
        count, digest = write_jsonl(args.output / name, rows)
        manifest["files"][name] = {"rows": count, "sha256": digest}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
