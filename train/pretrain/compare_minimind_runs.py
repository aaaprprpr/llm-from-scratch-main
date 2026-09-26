"""Compare the final 24K/4K and 8K/32K MiniMind pretraining runs.

Run from the repository root:
    .venv/bin/python -m train.pretrain.compare_minimind_runs

The output is a diagnostic report, not a benchmark of instruction following.
"""

from __future__ import annotations

import gc
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from checkpoint_io import load_checkpoint
from config_loader import resolve_recorded_path
from models.model import Transformer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "minimind_24k_vs_8k_20260927.json"
RUNS = {
    "24k_4k": ("run_20260923_130232", 13248),
    "8k_32k": ("run_20260924_005318", 14137),
}
PROMPTS = [
    ("capital", "中国的首都是", ["北京"]),
    ("boiling", "水在标准大气压下多少摄氏度沸腾？答案：", ["100", "一百"]),
    ("addition", "2加3等于多少？答案：", ["5", "五"]),
    ("red_key", "小明把红色钥匙放进左边抽屉，把蓝色钥匙放进右边抽屉。红色钥匙在", ["左边", "左"]),
    ("person", "张华去了上海，李明留在北京。张华所在的城市是", ["上海"]),
    ("seasons", "一年有四个季节，依次是春、夏、秋、", ["冬"]),
    ("assistant", "你好，我是", []),
    ("rain", "因为外面下着很大的雨，所以我", []),
    ("story", "从前有一个住在山里的小男孩，他每天都会", []),
    ("comfort", "我今天心情不太好，能陪我聊聊吗？答案：", []),
    ("weekend", "这个周末我想在家放松，可以做些什么？答案：", []),
]


def load_run(name: str, device: torch.device):
    run, step = RUNS[name]
    run_dir = ROOT / "output" / "train_logs" / run
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(resolve_recorded_path(config["paths"]["tokenizer_vocab"]) / "tokenizer.json"))
    checkpoint = load_checkpoint(run_dir / f"ckpt_step_{step}.pt", mmap=True)
    args = checkpoint["model_args"]
    assert args["vocab_size"] == tokenizer.get_vocab_size(with_added_tokens=True)
    assert checkpoint["iteration"] == step
    model = Transformer(**args)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    return model, tokenizer, config


def shared_validation_texts() -> tuple[list[dict], list[dict]]:
    """Select evenly spread, identical records from the two validation bins."""
    arrays = []
    endings = []
    tokenizers = []
    for name in RUNS:
        run, _ = RUNS[name]
        config = json.loads((ROOT / "output" / "train_logs" / run / "config.json").read_text(encoding="utf-8"))
        arrays.append(np.memmap(resolve_recorded_path(config["paths"]["val_data"]), dtype=np.uint16))
        tokenizers.append(Tokenizer.from_file(str(resolve_recorded_path(config["paths"]["tokenizer_vocab"]) / "tokenizer.json")))
        endings.append(np.flatnonzero(arrays[-1] == 0))
    assert len(endings[0]) == len(endings[1]) == 423441
    candidates = []
    for record in np.linspace(100, len(endings[0]) - 101, 240, dtype=int):
        texts = []
        lengths = []
        for a, e, tokenizer in zip(arrays, endings, tokenizers, strict=True):
            ids = a[e[record - 1] + 1 : e[record]]
            texts.append(tokenizer.decode(ids.tolist()))
            lengths.append(len(ids))
        if texts[0] != texts[1]:
            raise ValueError(f"Validation text mismatch at record {record}")
        byte_count = len(texts[0].encode("utf-8"))
        if 150 <= byte_count <= 3000 and max(lengths) <= 2048:
            candidates.append({"record": int(record), "text": texts[0], "utf8_bytes": byte_count})
    if len(candidates) < 64:
        raise RuntimeError(f"Only {len(candidates)} eligible validation records")
    selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, 64, dtype=int)]
    return selected, candidates


def score_text(model, tokenizer, text: str, device):
    eos = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
    if len(ids) > model.context_length:
        raise ValueError(f"Record has {len(ids)} tokens, beyond model context")
    input_ids = torch.tensor([[eos] + ids[:-1]], device=device)
    target_ids = torch.tensor(ids, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, _ = model(input_ids)
        nll = F.cross_entropy(logits[0].float(), target_ids, reduction="sum")
    return {"nll": float(nll), "tokens": len(ids)}


def repetition(ids):
    if len(ids) < 4:
        return 0.0
    ngrams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(Counter(ngrams)) / len(ngrams)


def generate(model, tokenizer, prompt: str, device, seed: int, temperature: float, max_new_tokens=80):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eos = tokenizer.token_to_id("<|endoftext|>")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_p=0.9, eos_id=eos)
    completion_ids = output[0, len(ids) :].tolist()
    return {"text": tokenizer.decode(completion_ids, skip_special_tokens=False),
            "tokens": len(completion_ids), "eos": eos in completion_ids,
            "repeated_4gram_fraction": repetition(completion_ids)}


def long_context(model, tokenizer, device, filler_records):
    fact = "档案记录：代号X17的物品存放在青松柜。\n"
    query = "\n根据档案记录，代号X17的物品存放在"
    filler = [r["text"] + "\n" for r in filler_records
              if "青松柜" not in r["text"] and "X17" not in r["text"]]
    outcome = []
    for target in (0, 3000, 8000, 16000, 30000):
        pieces = []
        if target:
            for chunk in filler:
                pieces.append(chunk)
                if len(tokenizer.encode(fact + "".join(pieces) + query, add_special_tokens=False).ids) >= target:
                    break
        prompt = fact + "".join(pieces) + query
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        if token_count < target:
            raise RuntimeError(f"Only {token_count} tokens of distinct filler for {target}")
        if token_count + 24 > model.context_length:
            outcome.append({"target_tokens": target, "prompt_tokens": token_count, "status": "exceeds_context"})
            continue
        result = generate(model, tokenizer, prompt, device, seed=20260927,
                          temperature=0.3, max_new_tokens=24)
        item = {"target_tokens": target, "prompt_tokens": token_count,
                "completion": result["text"],
                "exact_start": result["text"].lstrip(' \n\t"“').startswith("青松柜")}
        if target:
            recent_prompt = "".join(pieces) + fact + query
            recent_count = len(tokenizer.encode(recent_prompt, add_special_tokens=False).ids)
            if recent_count + 24 <= model.context_length:
                recent = generate(model, tokenizer, recent_prompt, device,
                                  seed=20260927, temperature=0.3, max_new_tokens=24)
                item.update(recent_prompt_tokens=recent_count,
                            recent_completion=recent["text"],
                            recent_exact_start=recent["text"].lstrip(' \n\t"“').startswith("青松柜"))
        outcome.append(item)
        print(f"  context {target}: early={item['completion'][:60]!r}, recent={item.get('recent_completion', '')[:60]!r}", flush=True)
    return outcome


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    records, filler_records = shared_validation_texts()
    report = {"method": "64 matching validation records, per-record EOS prefix and target; cross-vocabulary bits per UTF-8 byte",
              "validation_records": [{"record": r["record"], "utf8_bytes": r["utf8_bytes"]} for r in records],
              "runs": {}}
    for name in RUNS:
        print(f"loading {name}", flush=True)
        model, tokenizer, config = load_run(name, device)
        scores = []
        for i, record in enumerate(records):
            scores.append(score_text(model, tokenizer, record["text"], device))
            if (i + 1) % 16 == 0:
                print(f"  scored {i + 1}/64", flush=True)
        nll = sum(item["nll"] for item in scores)
        bytes_total = sum(item["utf8_bytes"] for item in records)
        generations = []
        for i, (prompt_name, prompt, expected) in enumerate(PROMPTS):
            result = generate(model, tokenizer, prompt, device, seed=20260927 + i, temperature=0.3)
            clean = result["text"].lstrip(' \n\t"“：:')
            result.update(name=prompt_name, prompt=prompt, expected=expected,
                          exact_start=any(clean.startswith(x) for x in expected) if expected else None)
            generations.append(result)
            print(f"  {prompt_name}: {result['text'][:85]!r}", flush=True)
        stochastic = []
        for i in (0, 1, 2, 3, 4, 6, 9):
            prompt_name, prompt, expected = PROMPTS[i]
            for seed in (11, 22, 33):
                result = generate(model, tokenizer, prompt, device, seed=seed, temperature=0.7)
                clean = result["text"].lstrip(' \n\t"“：:')
                result.update(name=prompt_name, seed=seed, expected=expected,
                              exact_start=any(clean.startswith(x) for x in expected) if expected else None)
                stochastic.append(result)
        contexts = long_context(model, tokenizer, device, filler_records)
        report["runs"][name] = {"step": RUNS[name][1], "model": config["model"],
                                "train_tokens": config["runtime"]["planned_train_tokens"],
                                "validation": {"bits_per_byte": nll / (bytes_total * math.log(2)),
                                               "total_nll": nll, "total_bytes": bytes_total,
                                               "total_tokens": sum(x["tokens"] for x in scores),
                                               "record_scores": scores},
                                "low_temperature": generations, "stochastic": stochastic,
                                "long_context": contexts}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
