"""Compare local cleaning with the first two completed reviews, without writing reviews."""
import difflib
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from opencc import OpenCC
from label.backend.dataset_store import load_dataset
from label.backend.llm_cleaning import CleaningConfig, LlmCleaner, PROMPT_VERSION

out = ROOT / "label/experiments" / PROMPT_VERSION
out.mkdir(exist_ok=True)
db = sqlite3.connect((ROOT / "label/data/curation.sqlite3").as_uri() + "?mode=ro", uri=True)
db.row_factory = sqlite3.Row
source = db.execute("SELECT * FROM project_sources LIMIT 1").fetchone()
dataset = load_dataset(source["dataset_path"])
reviews = [dict(db.execute("SELECT * FROM document_reviews WHERE project_id=? AND source_row=?",
                          (source["project_id"], row)).fetchone()) for row in range(2)]
cleaner = LlmCleaner(CleaningConfig.from_file(), out)
request = cleaner._request
def logged_request(path, payload=None):
    result = request(path, payload)
    if path == "/v1/chat/completions":
        print("chunk", result["choices"][0]["message"]["content"], flush=True)
    return result
cleaner._request = logged_request
convert = OpenCC("t2s")
def comparable(text):
    return re.sub(r"\s+", "", convert.convert(text))
def deleted_positions(original, edited):
    return {i for tag, a, b, c, d in difflib.SequenceMatcher(None, original, edited, autojunk=False).get_opcodes()
            if tag in {"delete", "replace"} for i in range(a, b)}
for review in reviews:
    row = dataset[review["source_row"]]
    original = convert.convert(row["text"])
    # Match the editor's paragraph boundaries and separator preservation.
    pieces = re.split(r"(\n[ \t]*\n(?:[ \t]*\n)*)", original)
    blocks = [{"id": str(i), "text": pieces[i], "separator_after": pieces[i+1] if i+1<len(pieces) else ""}
              for i in range(0,len(pieces),2) if pieces[i]]
    print("START", row["title"], len(original), flush=True)
    try:
        result = cleaner.clean(blocks, title=row["title"], provenance={"doc_id": review["doc_id"], "evaluation_only": True})
    except Exception as exc:
        print("FAILED", row["title"], str(exc), flush=True)
        continue
    raw = comparable(original)
    human = deleted_positions(raw, comparable(review["edited_text"]))
    predicted = deleted_positions(raw, comparable(result["edited_text"]))
    stats = {"title": row["title"], "original_chars":len(original), "human_chars":len(review["edited_text"]),
             "llm_chars":len(result["edited_text"]), "human_deleted":len(human), "llm_deleted":len(predicted),
             "matched_deletions":len(human & predicted), "extra_deletions":len(predicted-human),
             "missed_deletions":len(human-predicted), "chunks":result["chunks"], "elapsed_seconds":result["elapsed_seconds"]}
    report={"stats":stats,"original":original,"human":review["edited_text"],"llm":result["edited_text"],"result":result}
    (out / f"row-{review['source_row']}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (out / f"row-{review['source_row']}.diff.txt").write_text("".join(difflib.unified_diff(
        review["edited_text"].splitlines(keepends=True),result["edited_text"].splitlines(keepends=True),fromfile="human",tofile="llm")),encoding="utf-8")
    print("RESULT",json.dumps(stats,ensure_ascii=False),flush=True)
assert all(dict(db.execute("SELECT * FROM document_reviews WHERE project_id=? AND doc_id=?",
                          (r["project_id"],r["doc_id"])).fetchone()) == r for r in reviews), "Reviews changed during evaluation"
