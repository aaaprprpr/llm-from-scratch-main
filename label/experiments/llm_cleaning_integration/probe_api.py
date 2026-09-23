"""Live localhost smoke test; creates suggestions, never saves human reviews."""
import json
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8000"
QUEUE = "122160dd14054e5c92bed6cf5e89fd46"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, data=None):
    req = urllib.request.Request(BASE + path,
        data=None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=240) as response:
        return json.load(response)


results = []
for ordinal in [500000, 1000, 1384747]:
    path = f"/api/queues/{QUEUE}/items/{ordinal}"
    before = request(path)
    assert before["document_review"] is None
    blocks = [{"id": block["block_id"], "text": block["text"]} for block in before["blocks"]]
    result = request(f"/api/reviews/documents/{before['document']['doc_id']}/llm-clean", {
        "queue_id": QUEUE, "ordinal": ordinal, "expected_revision": 0,
        "content_sha256": before["document"]["content_sha256"], "blocks": blocks,
    })
    after = request(path)
    assert before["document_review"] == after["document_review"]
    assert before["materialized_text"] == after["materialized_text"]
    assert before["queue"]["state_counts"] == after["queue"]["state_counts"]
    for block in blocks:
        text = block["text"]
        spans = sorted([s for s in result["removals"] if s["block_id"] == block["id"]], key=lambda s:s["start"], reverse=True)
        for span in spans:
            assert text[span["start"]:span["end"]] == span["text"]
            text = text[:span["start"]] + text[span["end"]:]
        assert text == next(b["text"] for b in result["blocks"] if b["id"] == block["id"])
    results.append({"ordinal": ordinal, "title": before["provenance"]["title"],
                    "characters": len(before["materialized_text"]), "result": result})
    print(json.dumps({key: value for key, value in results[-1].items() if key != "result"}, ensure_ascii=False), flush=True)
    print(json.dumps({key: result[key] for key in ["decision", "quality", "category", "chunks", "elapsed_seconds", "removals"]}, ensure_ascii=False), flush=True)

Path(__file__).with_name("live_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("PASS: three real records, exact extraction, human reviews unchanged", flush=True)
