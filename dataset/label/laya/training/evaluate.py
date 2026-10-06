"""Evaluate a Laya checkpoint against frozen block labels without changing reviews."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from dataset.label.backend.laya_cleaning import QUESTION


def load_rows(path: Path, max_docs: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if max_docs:
        docs = sorted({row["doc_id"] for row in rows},
                      key=lambda doc: hashlib.sha256(doc.encode()).digest())[:max_docs]
        selected = set(docs)
        rows = [row for row in rows if row["doc_id"] in selected]
    return rows


def state(row: dict, context: bool):
    if not context:
        return row["text"]
    return {"title": row["title"], "previous": row["previous"],
            "target": row["text"], "next": row["next"]}


def evaluate(model, rows: list[dict], context: bool, drop_threshold: float | None = None) -> tuple[dict, list[dict]]:
    predictions = []
    started = time.monotonic()
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        answers = model.predict_batch(
            [state(row, context) for row in batch], QUESTION,
            batch_size=16, sort_by_length=True, max_len=2048,
        )
        if len(answers) != len(batch):
            raise RuntimeError("Laya returned the wrong number of decisions")
        predictions.extend((
            answer["answers"]["body"]["choice"],
            float(answer["answers"]["body"]["probabilities"]["drop"]),
        ) for answer in answers)
    elapsed = time.monotonic() - started
    counts = Counter()
    per_doc = defaultdict(list)
    false_delete_docs = set()
    false_deleted_chars = missed_deleted_chars = reference_keep_chars = reference_drop_chars = 0
    false_deletes = []
    missed_deletes = []
    prediction_rows = []
    for row, (raw_choice, drop_probability) in zip(rows, predictions):
        predicted = ("drop" if drop_probability >= drop_threshold else "keep") if drop_threshold is not None else raw_choice
        prediction_rows.append({
            "doc_id": row["doc_id"], "block_index": row["block_index"],
            "label": row["label"], "raw_choice": raw_choice,
            "drop_probability": drop_probability, "choice": predicted,
        })
        label = row["label"]
        if predicted not in {"keep", "drop"}:
            raise RuntimeError(f"Unexpected choice: {predicted}")
        counts[f"{label}_{predicted}"] += 1
        per_doc[row["doc_id"]].append(label == predicted)
        if label == "keep":
            reference_keep_chars += len(row["text"])
        else:
            reference_drop_chars += len(row["text"])
        if label == "keep" and predicted == "drop":
            false_deletes.append(row)
            false_delete_docs.add(row["doc_id"])
            false_deleted_chars += len(row["text"])
        elif label == "drop" and predicted == "keep":
            missed_deletes.append(row)
            missed_deleted_chars += len(row["text"])
    reference_keep = counts["keep_keep"] + counts["keep_drop"]
    reference_drop = counts["drop_drop"] + counts["drop_keep"]
    return {
        "drop_threshold": drop_threshold,
        "rows": len(rows), "documents": len(per_doc), "elapsed_seconds": round(elapsed, 2),
        "blocks_per_second": round(len(rows) / elapsed, 2) if elapsed else 0,
        "counts": dict(counts),
        "false_deletion_rate": round(counts["keep_drop"] / reference_keep, 4) if reference_keep else None,
        "junk_removal_recall": round(counts["drop_drop"] / reference_drop, 4) if reference_drop else None,
        "exact_block_decision_documents": sum(all(matches) for matches in per_doc.values()),
        "documents_without_false_deletions": len(per_doc) - len(false_delete_docs),
        "false_deleted_characters": false_deleted_chars,
        "reference_keep_characters": reference_keep_chars,
        "false_deleted_character_rate": round(false_deleted_chars / reference_keep_chars, 4) if reference_keep_chars else None,
        "missed_deleted_characters": missed_deleted_chars,
        "reference_drop_characters": reference_drop_chars,
        "missed_deleted_character_rate": round(missed_deleted_chars / reference_drop_chars, 4) if reference_drop_chars else None,
        "false_deletion_examples": [
            {"doc_id": row["doc_id"], "block_index": row["block_index"],
             "title": row["title"], "text": row["text"][:350]}
            for row in sorted(false_deletes, key=lambda item: hashlib.sha256(
                f"{item['doc_id']}:{item['block_index']}".encode()).digest())[:12]
        ],
        "missed_deletion_examples": [
            {"doc_id": row["doc_id"], "block_index": row["block_index"],
             "title": row["title"], "text": row["text"][:350]}
            for row in sorted(missed_deletes, key=lambda item: hashlib.sha256(
                f"{item['doc_id']}:{item['block_index']}".encode()).digest())[:12]
        ],
    }, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--drop-threshold", type=float, default=None)
    parser.add_argument("--prediction-output", type=Path, default=None)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Select this many documents by stable hash; 0 uses all")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    from dataset.label.laya.runtime import load
    import torch
    rows = load_rows(args.dataset, args.max_docs)
    model = load(str(args.model_dir.resolve()),
                 device="cuda" if torch.cuda.is_available() else "cpu")
    metrics, prediction_rows = evaluate(model, rows, args.context, args.drop_threshold)
    metrics.update({"model_dir": str(args.model_dir.resolve()),
                    "dataset": str(args.dataset.resolve()),
                    "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    "context": args.context, "max_docs": args.max_docs})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prediction_output:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        with args.prediction_output.open("w", encoding="utf-8") as output:
            for prediction in prediction_rows:
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in metrics.items()
                      if not key.endswith("_examples")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
