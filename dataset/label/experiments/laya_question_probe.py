"""Compare zero-shot Laya question formats on existing complete review suggestions.

Reference labels come from saved LLM suggestions, not human gold. This script only reads
reports and runs local inference; it does not alter queue or review state.
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import time
from pathlib import Path

from dataset.label.backend.laya_cleaning import LayaCleaner, QUESTION
from dataset.label.backend.llm_cleaning import split_units

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "dataset/label/data/llm_suggestions/cache"


def sample(per_class: int = 160):
    paths = sorted(glob.glob(str(REPORTS / "*" / "*.json")))
    random.Random(23).shuffle(paths)
    partitions = {name: {"keep": [], "drop": []} for name in ("dev", "holdout")}
    seen_docs = set()
    for path in paths:
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            result = report["result"]
            doc_id = report["provenance"]["doc_id"]
            if (doc_id in seen_docs or result.get("decision") not in {"keep", "drop"}
                    or not result.get("complete")):
                continue
            seen_docs.add(doc_id)
            removed = {(item["block_id"], item["start"], item["end"])
                       for item in result.get("removals", [])}
            edited = {(item["block_id"], item["start"], item["end"])
                      for item in result.get("edits", [])}
            units = split_units(report["input_blocks"])
            candidates = {"keep": [], "drop": []}
            for index, unit in enumerate(units):
                key = unit.block_id, unit.start, unit.end
                body = unit.text.strip()
                if key in edited or not 6 <= len(body) <= 350:
                    continue
                state = {
                    "title": (report.get("title") or "")[:80],
                    "previous": units[index - 1].text.strip()[:110] if index else "",
                    "target": body,
                    "next": units[index + 1].text.strip()[:110] if index + 1 < len(units) else "",
                }
                label = "drop" if key in removed else "keep"
                candidates[label].append((body, state, label))
            rng = random.Random(doc_id)
            partition = "dev" if hashlib.sha256(doc_id.encode()).digest()[0] < 128 else "holdout"
            for label in ("keep", "drop"):
                rng.shuffle(candidates[label])
                partitions[partition][label].extend(candidates[label][:2])
            if all(len(partitions[part][label]) >= per_class
                   for part in partitions for label in ("keep", "drop")):
                break
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not all(len(partitions[part][label]) >= per_class
               for part in partitions for label in ("keep", "drop")):
        raise RuntimeError(f"not enough reference units: {partitions.keys()}")
    dev = partitions["dev"]["keep"][:per_class] + partitions["dev"]["drop"][:per_class]
    holdout = partitions["holdout"]["keep"][:per_class] + partitions["holdout"]["drop"][:per_class]
    return dev, holdout


def questions():
    current = QUESTION["body"]
    descriptions = current["criteria"]
    return {
        "current": (current, {"keep": "keep", "drop": "drop"}),
        "reverse": ({**current, "criteria": {"drop": descriptions["drop"], "keep": descriptions["keep"]}},
                    {"keep": "keep", "drop": "drop"}),
        "opaque": ({**current, "criteria": {"A": descriptions["keep"], "B": descriptions["drop"]}},
                   {"A": "keep", "B": "drop"}),
        "chinese": ({**current, "criteria": {"正文": descriptions["keep"], "页面残片": descriptions["drop"]}},
                    {"正文": "keep", "页面残片": "drop"}),
        "literal": ({"type": "choice", "instructions": "判断当前片段的语义作用", "criteria": {
            "正文叙述": "当前片段本身表达事实、定义、观点、经过或解释",
            "结构标记": "当前片段只是标题、索引、名单、日期或表格值"}},
                    {"正文叙述": "keep", "结构标记": "drop"}),
        "noul": ({"type": "noul", "instructions": "当前片段本身是否表达一个可阅读的事实、定义、观点、过程或解释？"}, None),
    }


def evaluate(agent, rows, *, context: bool, name: str, question: dict, mapping: dict | None):
    states = [row[1] if context else row[0] for row in rows]
    if context:
        question = {**question, "instructions": "仅判断 state.target；title、previous、next 只作为局部上下文。" + question["instructions"]}
    start = time.monotonic()
    answers = agent.predict_batch(states, {"body": question}, batch_size=16,
                                  sort_by_length=True, max_len=2048)
    if mapping is None:
        predicted = ["keep" if answer["answers"]["body"]["noul"] >= 0.5 else "drop"
                     for answer in answers]
    else:
        predicted = [mapping[answer["answers"]["body"]["choice"]] for answer in answers]
    actual = [row[2] for row in rows]
    keep = sum(a == p == "keep" for a, p in zip(actual, predicted))
    drop = sum(a == p == "drop" for a, p in zip(actual, predicted))
    bad_delete = sum(a == "keep" and p == "drop" for a, p in zip(actual, predicted))
    print(f"{name:20s} {keep + drop:3d}/{len(rows)} keep {keep:3d} drop {drop:3d} "
          f"bad_delete {bad_delete:3d} {time.monotonic()-start:.2f}s", flush=True)
    return keep + drop, bad_delete


def main():
    dev, holdout = sample()
    agent = LayaCleaner._model()
    variants = questions()
    print("development: 160 reference keep + 160 reference drop", flush=True)
    results = []
    for name, (question, mapping) in variants.items():
        for context in (False, True):
            score, bad_delete = evaluate(agent, dev, context=context,
                                         name=name + ("+context" if context else ""),
                                         question=question, mapping=mapping)
            results.append((score, -bad_delete, name, context))
    _, _, best_name, best_context = max(results)
    print("holdout: 160 reference keep + 160 reference drop", flush=True)
    for name, context in [("current", False), (best_name, best_context)]:
        question, mapping = variants[name]
        evaluate(agent, holdout, context=context, name=name + ("+context" if context else ""),
                 question=question, mapping=mapping)


def sample_blocks(per_class: int = 160):
    """Block-level labels only when every unit in the line has the same reference label."""
    paths = sorted(glob.glob(str(REPORTS / "*" / "*.json")))
    random.Random(31).shuffle(paths)
    partitions = {name: {"keep": [], "drop": []} for name in ("dev", "holdout")}
    seen_docs = set()
    for path in paths:
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            result = report["result"]
            doc_id = report["provenance"]["doc_id"]
            if (doc_id in seen_docs or result.get("decision") not in {"keep", "drop"}
                    or not result.get("complete")):
                continue
            seen_docs.add(doc_id)
            removed = {(item["block_id"], item["start"], item["end"])
                       for item in result.get("removals", [])}
            edited_blocks = {item["block_id"] for item in result.get("edits", [])}
            blocks = report["input_blocks"]
            candidates = {"keep": [], "drop": []}
            for index, block in enumerate(blocks):
                body = block["text"].strip()
                if block["id"] in edited_blocks or not 6 <= len(body) <= 500:
                    continue
                units = split_units([block])
                labels = {(unit.block_id, unit.start, unit.end) in removed
                          for unit in units if unit.text.strip()}
                if len(labels) != 1:
                    continue
                label = "drop" if True in labels else "keep"
                state = {"title": (report.get("title") or "")[:80],
                         "previous": blocks[index - 1]["text"].strip()[:110] if index else "",
                         "target": body,
                         "next": blocks[index + 1]["text"].strip()[:110] if index + 1 < len(blocks) else ""}
                candidates[label].append((body, state, label))
            rng = random.Random(doc_id)
            partition = "dev" if hashlib.sha256(doc_id.encode()).digest()[0] < 128 else "holdout"
            for label in ("keep", "drop"):
                rng.shuffle(candidates[label])
                partitions[partition][label].extend(candidates[label][:2])
            if all(len(partitions[part][label]) >= per_class
                   for part in partitions for label in ("keep", "drop")):
                break
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not all(len(partitions[part][label]) >= per_class
               for part in partitions for label in ("keep", "drop")):
        raise RuntimeError("not enough block references")
    return tuple(partitions[part]["keep"][:per_class] + partitions[part]["drop"][:per_class]
                 for part in ("dev", "holdout"))


def compare_granularity(agent):
    """Use the same held-out blocks for whole-block and inherited sentence splitting."""
    rows = sample_blocks()[1]
    question, mapping = questions()["current"]
    print("same held-out blocks: 160 reference keep + 160 reference drop", flush=True)
    evaluate(agent, rows, context=False, name="whole block", question=question, mapping=mapping)
    lengths = []
    units = []
    for body, _, _ in rows:
        parts = split_units([{"id": "sample", "text": body}])
        lengths.append(len(parts))
        units.extend(part.text for part in parts)
    answers = agent.predict_batch(units, {"body": question}, batch_size=16,
                                  sort_by_length=True, max_len=2048)
    choices = [answer["answers"]["body"]["choice"] for answer in answers]
    cursor = kept = dropped = false_deleted = partial_drop = 0
    for (_, _, label), count in zip(rows, lengths):
        predictions = choices[cursor:cursor + count]
        cursor += count
        if label == "keep":
            if all(value == "keep" for value in predictions):
                kept += 1
            else:
                false_deleted += 1
        elif all(value == "drop" for value in predictions):
            dropped += 1
        elif any(value == "drop" for value in predictions):
            partial_drop += 1
    print(f"sentence split        correct={kept + dropped}/320 keep={kept}/160 "
          f"fully_drop={dropped}/160 false_delete_blocks={false_deleted}/160 "
          f"partial_drop={partial_drop}/160 units={len(units)}", flush=True)


if __name__ == "__main__":
    main()
    compare_granularity(LayaCleaner._model())
