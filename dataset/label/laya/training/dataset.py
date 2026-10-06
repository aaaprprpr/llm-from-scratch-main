"""Snapshot block decisions for Laya training and document-disjoint evaluation.

Teacher labels are DeepSeek Web suggestions, not human ground truth. Human labels
come only from exact block alignment against saved local-web document edits.
The script reads the live project but writes exclusively to --output.
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import glob
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from dataset.label.backend.blocks import parse_blocks
from dataset.label.backend.dataset_store import load_dataset
from dataset.label.backend.identity import sha256_text, stable_json
from dataset.label.backend.laya_cleaning import QUESTION
from dataset.label.laya import LABEL_ROOT

REPORTS = LABEL_ROOT / "data/llm_suggestions/cache"
DATABASE = LABEL_ROOT / "data/curation.sqlite3"
VERSION = "laya_block_dataset_v1"


def partition(doc_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(doc_id.encode()).digest()[:4], "big") / 2**32
    return "train" if value < .70 else "dev" if value < .85 else "test"


def row_for(blocks: list[dict], index: int, title: str, doc_id: str,
            label: str, source: str) -> dict:
    body = blocks[index]["text"]
    return {
        "doc_id": doc_id, "block_id": blocks[index]["id"], "block_index": index,
        "source": source, "label": label, "title": (title or "")[:100],
        "text": body, "previous": blocks[index - 1]["text"][-160:] if index else "",
        "next": blocks[index + 1]["text"][:160] if index + 1 < len(blocks) else "",
    }


def usable(text: str) -> bool:
    return bool(text.strip()) and len(text) <= 1200


def teacher_rows(manual_doc_ids: set[str]) -> tuple[dict[str, list[dict]], Counter]:
    latest = {}
    for path in sorted(glob.glob(str(REPORTS / "*" / "*.json"))):
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            if report.get("provider") != "deepseek_web":
                continue
            result = report["result"]
            if not result.get("complete") or result.get("decision") not in {"keep", "drop"}:
                continue
            if (result.get("input_sha256") != sha256_text(stable_json(report["input_blocks"]))
                    or result.get("risk_fallback_chunks")):
                continue
            doc_id = report["provenance"]["doc_id"]
            if doc_id in manual_doc_ids:
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
        for index, block in enumerate(blocks):
            text = block["text"]
            if not usable(text) or block["id"] in edited:
                counts["skip_length_or_edit"] += 1
                continue
            spans = removals[block["id"]]
            if not spans:
                label = "keep"
            else:
                covered = bytearray(len(text))
                for start, end in spans:
                    if not 0 <= start <= end <= len(text):
                        raise ValueError(f"invalid removal span in {doc_id}")
                    covered[start:end] = b"\1" * (end - start)
                if any(not covered[pos] and not char.isspace() for pos, char in enumerate(text)):
                    counts["skip_mixed"] += 1
                    continue
                label = "drop"
            split = partition(doc_id)
            splits[split].append(row_for(blocks, index, report.get("title") or "",
                                       doc_id, label, "deepseek_web"))
            counts[f"{split}_{label}"] += 1
        counts[f"{partition(doc_id)}_docs"] += 1
    return splits, counts


def human_rows(connection: sqlite3.Connection) -> tuple[list[dict], set[str], Counter]:
    latest_actors = {
        doc_id: actor for doc_id, actor in connection.execute("""
            SELECT entity_id, actor FROM events
            WHERE entity_type='document_review' AND event_seq IN (
                SELECT MAX(event_seq) FROM events
                WHERE entity_type='document_review' GROUP BY entity_id)
        """)
    }
    source = connection.execute("SELECT dataset_path FROM project_sources LIMIT 1").fetchone()
    if source is None:
        raise RuntimeError("No prepared dataset attached")
    dataset = load_dataset(source[0])
    results = []
    doc_ids = set()
    counts = Counter()
    for doc_id, source_row, content_hash, decision, edited_text in connection.execute(
        "SELECT doc_id, source_row, content_sha256, decision, edited_text FROM document_reviews"
    ):
        if latest_actors.get(doc_id) != "local-web" or decision not in {"keep", "drop"}:
            continue
        original = dataset[source_row]
        if original["doc_id"] != doc_id or sha256_text(original["text"]) != content_hash:
            raise ValueError(f"manual review source mismatch: {doc_id}")
        blocks = [
            {"id": item.block_id, "text": item.text}
            for item in parse_blocks(original["text"], doc_id)
        ]
        labels = {}
        if decision == "drop":
            labels = {index: "drop" for index in range(len(blocks))}
        elif edited_text is not None:
            edited = [item.text for item in parse_blocks(edited_text.strip(), doc_id)]
            original_lines = [item["text"] for item in blocks]
            matcher = difflib.SequenceMatcher(None, original_lines, edited, autojunk=False)
            for action, first, last, _, _ in matcher.get_opcodes():
                if action in {"equal", "delete"}:
                    for index in range(first, last):
                        labels[index] = "keep" if action == "equal" else "drop"
                else:
                    counts[f"skip_{action}"] += last - first
        if labels:
            doc_ids.add(doc_id)
        for index, label in labels.items():
            text = blocks[index]["text"]
            if usable(text):
                results.append(row_for(blocks, index, original.get("title") or "",
                                           doc_id, label, "human"))
                counts[label] += 1
            else:
                counts["skip_length"] += 1
    counts["docs"] = len(doc_ids)
    return results, doc_ids, counts



def paired_web_human_rows(connection: sqlite3.Connection, human: list[dict]) -> list[dict]:
    """Pair independent human and DeepSeek Web labels on the same original lines."""
    by_doc = defaultdict(list)
    for row in human:
        by_doc[row["doc_id"]].append(row)
    if not by_doc:
        return []
    source = connection.execute("SELECT dataset_path FROM project_sources LIMIT 1").fetchone()
    dataset = load_dataset(source[0])
    positions = dict(connection.execute("SELECT doc_id, source_row FROM document_reviews"))
    latest = {}
    for path in sorted(glob.glob(str(REPORTS / "*" / "*.json"))):
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            doc_id = report["provenance"]["doc_id"]
            if (doc_id not in by_doc or report.get("provider") != "deepseek_web"
                    or not report["result"].get("complete")):
                continue
            prior = latest.get(doc_id)
            if prior is None or (report.get("created_at", ""), path) > prior[:2]:
                latest[doc_id] = (report.get("created_at", ""), path, report)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    paired = []
    for doc_id, (_, _, report) in sorted(latest.items()):
        original = dataset[positions[doc_id]]["text"]
        report_source = "".join(block["text"] + block["separator_after"]
                                for block in report["input_blocks"])
        if original != report_source:
            continue
        original_lines = original.splitlines()
        edited_lines = report["result"]["edited_text"].splitlines()
        web_labels = {}
        for action, first, last, _, _ in difflib.SequenceMatcher(
            None, original_lines, edited_lines, autojunk=False
        ).get_opcodes():
            if action in {"equal", "delete"}:
                for index in range(first, last):
                    web_labels[index] = "keep" if action == "equal" else "drop"
        blocks = parse_blocks(original, doc_id)
        line_starts = [0] + [position + 1 for position, char in enumerate(original) if char == "\n"]
        for row in by_doc[doc_id]:
            index = row["block_index"]
            if index < len(blocks):
                line_index = bisect.bisect_right(line_starts, blocks[index].start_cp) - 1
                if line_index in web_labels:
                    paired.append({**row, "deepseek_web_label": web_labels[line_index]})
    return paired


def train_case(row: dict, *, context: bool) -> dict:
    state = (
        {"title": row["title"], "previous": row["previous"],
         "target": row["text"], "next": row["next"]}
        if context else row["text"]
    )
    label = row["label"]
    probabilities = {"keep": .98, "drop": .02} if label == "keep" else {"keep": .02, "drop": .98}
    return {"state": state, "questions": QUESTION,
            "gold": {"body": {"label": label, "probabilities": probabilities}}}


def write_jsonl(path: Path, rows) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as output:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
            output.write(line)
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=6000)
    args = parser.parse_args()
    if args.train_per_class <= 0:
        parser.error("--train-per-class must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    connection = sqlite3.connect(f"file:{DATABASE.resolve()}?mode=ro", uri=True)
    human, manual_ids, human_counts = human_rows(connection)
    teacher, teacher_counts = teacher_rows(manual_ids)
    paired = paired_web_human_rows(connection, human)
    human_dev = [row for row in human if hashlib.sha256(row["doc_id"].encode()).digest()[0] < 102]
    human_test = [row for row in human if hashlib.sha256(row["doc_id"].encode()).digest()[0] >= 102]
    training = []
    for label in ("keep", "drop"):
        candidates = [row for row in teacher["train"] if row["label"] == label]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{row['doc_id']}:{row['block_index']}:{VERSION}".encode()).digest())
        training.extend(candidates[:args.train_per_class])
    training.sort(key=lambda row: hashlib.sha256(
        f"{row['doc_id']}:{row['block_index']}:order".encode()).digest())
    doc_sets = {
        "train": {row["doc_id"] for row in teacher["train"]},
        "dev": {row["doc_id"] for row in teacher["dev"]},
        "test": {row["doc_id"] for row in teacher["test"]},
        "human": {row["doc_id"] for row in human},
    }
    for name, docs in doc_sets.items():
        for other, other_docs in doc_sets.items():
            if name < other and docs & other_docs:
                raise ValueError(f"document leakage between {name} and {other}")
    if {row["doc_id"] for row in human_dev} & {row["doc_id"] for row in human_test}:
        raise ValueError("human calibration/test document leakage")
    args.output.mkdir(parents=True)
    manifest = {"version": VERSION, "partition": "SHA256(doc_id), 70/15/15",
                "teacher": dict(teacher_counts), "human": dict(human_counts),
                "train_selected": dict(Counter(row["label"] for row in training)),
                "train_doc_ids": len({row["doc_id"] for row in training}),
                "paired_human_web_docs": len({row["doc_id"] for row in paired}),
                "human_dev_docs": len({row["doc_id"] for row in human_dev}),
                "human_test_docs": len({row["doc_id"] for row in human_test}),
                "files": {}}
    files = {
        "train_plain.jsonl": (train_case(row, context=False) for row in training),
        "train_context.jsonl": (train_case(row, context=True) for row in training),
        "teacher_dev.jsonl": teacher["dev"],
        "teacher_test.jsonl": teacher["test"],
        "human_gold.jsonl": human,
        "human_dev.jsonl": human_dev,
        "human_test.jsonl": human_test,
        "paired_human_web.jsonl": paired,
    }
    for filename, rows in files.items():
        count, digest = write_jsonl(args.output / filename, rows)
        manifest["files"][filename] = {"rows": count, "sha256": digest}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
