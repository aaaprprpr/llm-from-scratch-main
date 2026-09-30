"""Render full Laya-cleaned documents beside saved human and DeepSeek Web text.

Read-only evaluation: the only writes are under --output. Line-match scores are
mechanical comparisons, and the emitted text/diffs are for qualitative review.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from dataset.label.backend.blocks import parse_blocks
from dataset.label.backend.dataset_store import load_dataset
from dataset.label.backend.laya_cleaning import LayaCleaner, QUESTION

ROOT = Path(__file__).resolve().parents[3]
DATABASE = ROOT / "dataset/label/data/curation.sqlite3"
REPORTS = ROOT / "dataset/label/data/llm_suggestions/cache"


def line_score(candidate: str, human: str) -> dict:
    candidate_lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    human_lines = [line.strip() for line in human.splitlines() if line.strip()]
    matching = sum(block.size for block in difflib.SequenceMatcher(
        None, candidate_lines, human_lines, autojunk=False
    ).get_matching_blocks())
    precision = matching / len(candidate_lines) if candidate_lines else 0
    recall = matching / len(human_lines) if human_lines else 0
    return {"matched_lines": matching, "candidate_lines": len(candidate_lines),
            "human_lines": len(human_lines), "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision + recall else 0}


def latest_web_reports(doc_ids: set[str]) -> dict[str, dict]:
    latest = {}
    for path in glob.glob(str(REPORTS / "*" / "*.json")):
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            doc_id = report["provenance"]["doc_id"]
            if (doc_id not in doc_ids or report.get("provider") != "deepseek_web"
                    or not report["result"].get("complete")):
                continue
            old = latest.get(doc_id)
            if old is None or (report.get("created_at", ""), path) > old[:2]:
                latest[doc_id] = (report.get("created_at", ""), path, report)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {doc_id: value[2] for doc_id, value in latest.items()}


def clean_document(model, original: str, doc_id: str, title: str, context: bool) -> tuple[str, int]:
    blocks = [
        {"id": block.block_id, "text": block.text, "separator_after": block.separator_after}
        for block in parse_blocks(original, doc_id)
    ]
    units = LayaCleaner._split_units(blocks)
    positions = {block["id"]: index for index, block in enumerate(blocks)}
    states = []
    for unit in units:
        if not context:
            states.append(unit.text.strip())
            continue
        index = positions[unit.block_id]
        states.append({"title": title[:100],
                       "previous": blocks[index - 1]["text"][-160:] if index else "",
                       "target": unit.text.strip(),
                       "next": blocks[index + 1]["text"][:160] if index + 1 < len(blocks) else ""})
    decisions = []
    for start in range(0, len(states), 256):
        answers = model.predict_batch(states[start:start + 256], QUESTION,
                                      batch_size=16, sort_by_length=True, max_len=2048)
        decisions.extend(answer["answers"]["body"]["choice"] for answer in answers)
    by_block = defaultdict(list)
    for unit, choice in zip(units, decisions):
        if choice == "keep":
            by_block[unit.block_id].append(unit.text)
        elif choice != "drop":
            raise ValueError(f"unknown choice: {choice}")
    kept = []
    for block in blocks:
        text = "".join(by_block[block["id"]])
        if text.strip():
            kept.append({"text": text, "separator_after": block["separator_after"]})
    cleaned = "".join(block["text"] + (block["separator_after"] if index < len(kept) - 1 else "")
                      for index, block in enumerate(kept))
    return cleaned, len(units)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--paired-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    doc_ids = {json.loads(line)["doc_id"] for line in args.paired_dataset.open(encoding="utf-8")}
    reports = latest_web_reports(doc_ids)
    if set(reports) != doc_ids:
        raise ValueError("missing DeepSeek Web report for a paired document")
    connection = sqlite3.connect(f"file:{DATABASE.resolve()}?mode=ro", uri=True)
    source = connection.execute("SELECT dataset_path FROM project_sources LIMIT 1").fetchone()
    dataset = load_dataset(source[0])
    reviews = {doc_id: (row, edited) for doc_id, row, edited in connection.execute(
        "SELECT doc_id, source_row, edited_text FROM document_reviews"
    ) if doc_id in doc_ids}
    import torch
    from laya import load
    model = load(str(args.model_dir.resolve()), device="cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True)
    summary = []
    for doc_id in sorted(doc_ids):
        row, human = reviews[doc_id]
        original = dataset[row]["text"]
        report = reports[doc_id]
        report_original = "".join(block["text"] + block["separator_after"]
                                  for block in report["input_blocks"])
        if report_original != original:
            raise ValueError(f"source differs from DeepSeek report: {doc_id}")
        started = time.monotonic()
        cleaned, units = clean_document(model, original, doc_id, dataset[row].get("title") or "",
                                        args.context)
        web = report["result"]["edited_text"]
        result = {"doc_id": doc_id, "source_row": row, "title": dataset[row].get("title"),
                  "units": units, "laya_seconds": round(time.monotonic() - started, 2),
                  "original_chars": len(original), "human_chars": len(human),
                  "laya_chars": len(cleaned), "deepseek_web_chars": len(web),
                  "laya_vs_human_lines": line_score(cleaned, human),
                  "deepseek_web_vs_human_lines": line_score(web, human)}
        summary.append(result)
        destination = args.output / doc_id
        destination.mkdir()
        for name, text in {"original.txt": original, "human.txt": human,
                           "laya.txt": cleaned, "deepseek_web.txt": web}.items():
            (destination / name).write_text(text, encoding="utf-8")
        (destination / "laya_vs_human.diff").write_text("".join(difflib.unified_diff(
            human.splitlines(keepends=True), cleaned.splitlines(keepends=True),
            fromfile="human", tofile="laya")), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    for row in summary:
        print(row["source_row"], row["title"], "Laya F1", row["laya_vs_human_lines"]["f1"],
              "DeepSeek F1", row["deepseek_web_vs_human_lines"]["f1"],
              "Laya s", row["laya_seconds"], flush=True)


if __name__ == "__main__":
    main()
