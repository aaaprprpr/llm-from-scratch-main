"""Measure a three-way Laya checkpoint and its hypothetical editor workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from dataset.label.laya.training.evaluate import load_rows
from dataset.label.laya.experiments.ternary_dataset import QUESTION_TERNARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--prediction-output", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    import torch
    from dataset.label.laya.runtime import load
    rows = load_rows(args.dataset, args.max_docs)
    model = load(str(args.model_dir.resolve()), device="cuda" if torch.cuda.is_available() else "cpu")
    confusion = Counter()
    document_edits = set()
    document_drop_errors = set()
    examples = defaultdict(list)
    prediction_rows = []
    started = time.monotonic()
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        answers = model.predict_batch([row["text"] for row in batch], QUESTION_TERNARY,
                                      batch_size=16, sort_by_length=True, max_len=2048)
        for row, answer in zip(batch, answers):
            label = row["label"]
            response = answer["answers"]["body"]
            predicted = response["choice"]
            prediction_rows.append({"doc_id": row["doc_id"], "block_index": row["block_index"],
                                    "label": label, "choice": predicted,
                                    "probabilities": response["probabilities"]})
            confusion[f"{label}_{predicted}"] += 1
            if predicted == "edit":
                document_edits.add(row["doc_id"])
            if label == "keep" and predicted == "drop":
                document_drop_errors.add(row["doc_id"])
            key = f"{label}_{predicted}"
            if label != predicted and len(examples[key]) < 8:
                examples[key].append({"title": row["title"], "doc_id": row["doc_id"],
                                      "block_index": row["block_index"],
                                      "text": row["text"][:350]})
    elapsed = time.monotonic() - started
    docs = {row["doc_id"] for row in rows}
    keep = sum(confusion[f"keep_{choice}"] for choice in ("keep", "drop", "edit"))
    drop = sum(confusion[f"drop_{choice}"] for choice in ("keep", "drop", "edit"))
    edit = sum(confusion[f"edit_{choice}"] for choice in ("keep", "drop", "edit"))
    metrics = {
        "model_dir": str(args.model_dir.resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "rows": len(rows), "documents": len(docs),
        "seconds": round(elapsed, 2), "blocks_per_second": round(len(rows) / elapsed, 2),
        "confusion": dict(confusion),
        "false_drop_on_keep": round(confusion["keep_drop"] / keep, 4) if keep else None,
        "direct_drop_recall": round(confusion["drop_drop"] / drop, 4) if drop else None,
        "edit_recall": round(confusion["edit_edit"] / edit, 4) if edit else None,
        "editor_fallback_blocks": sum(confusion[f"{label}_edit"] for label in ("keep", "drop", "edit")),
        "editor_fallback_documents": len(document_edits),
        "documents_without_false_drop": len(docs - document_drop_errors),
        "examples": dict(examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prediction_output:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        with args.prediction_output.open("w", encoding="utf-8") as output:
            for prediction in prediction_rows:
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in metrics.items() if key != "examples"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
