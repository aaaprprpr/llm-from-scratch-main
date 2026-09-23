import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from label.backend.llm_cleaning import CleaningConfig, LlmCleaner

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = "http://127.0.0.1:8000/api/queues/122160dd14054e5c92bed6cf5e89fd46/items/3"
with opener.open(url, timeout=30) as response:
    document = json.load(response)
text = document["simplified"]["materialized_text"]
pieces = re.split(r"(\n[ \t]*\n(?:[ \t]*\n)*)", text)
blocks = [{"id": f"{document['document']['doc_id']}:edited:{i//2}", "text": pieces[i],
           "separator_after": pieces[i + 1] if i + 1 < len(pieces) else ""}
          for i in range(0, len(pieces), 2) if pieces[i]]
folder = ROOT / "label/experiments/history-cleaning-debug"
folder.mkdir(exist_ok=True)
cleaner = LlmCleaner(CleaningConfig.from_file(), folder)
original_request = cleaner._request
calls = []
def request(path, payload=None):
    response = original_request(path, payload)
    if path == "/v1/chat/completions":
        calls.append({"request": payload, "response": response})
        (folder / "calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8")
        choice = response["choices"][0]
        print("CHUNK", len(calls), choice["finish_reason"], choice["message"]["content"], flush=True)
    return response
cleaner._request = request
print("START", document["provenance"]["title"], "chars", len(text), "blocks", len(blocks), flush=True)
try:
    result = cleaner.clean(blocks, title=document["provenance"]["title"], provenance={
        "doc_id": document["document"]["doc_id"], "debug_only": True,
    })
    print("SUCCESS", {k: result[k] for k in ["chunks", "elapsed_seconds", "text_changed"]}, flush=True)
except Exception as exc:
    print("FAILED", type(exc).__name__, str(exc), flush=True)
    raise
