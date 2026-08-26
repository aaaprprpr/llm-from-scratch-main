"""横向评估预训练 checkpoint 的续写、复读和 KV-cache 一致性。

不使用命令行参数。修改下方配置后直接运行：

    python pretrain/evaluate_checkpoints.py

报告会写到对应 run 目录下的 ``checkpoint_evaluation.json``。
"""

import csv
import gc
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer as BackendTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import Transformer


# ============================== 可修改配置 ==============================

RUN_DIR = PROJECT_ROOT / "output/train_logs/run_20260823_173959"

# None 表示评估目录里的全部 checkpoint；也可以只写关心的步数。
CHECKPOINT_STEPS = [10000, 30000, 50000, 80000]

# 留空时从 run/config.json 的 paths.tokenizer_vocab 自动解析。
TOKENIZER_PATH = None

MAX_NEW_TOKENS = 64
COMPARISON_TEMPERATURE = 0.3
COMPARISON_TOP_P = 0.9

# 最新 checkpoint 额外用正常温度跑多个随机种子，避免单样本误判。
MULTI_SEED_TEMPERATURE = 0.7
MULTI_SEED_TOP_P = 0.9
MULTI_SEED_MAX_NEW_TOKENS = 80
MULTI_SEEDS = [11, 22, 33]

# FP32 下逐 token 比较 cache 增量推理与完整上下文重算。
CACHE_PROBE_STEPS = 16
CACHE_PROBE_PROMPT = (
    "张华今天去了上海，李明留在北京。晚上打电话时，"
    "张华说他所在的城市是"
)

PROMPTS = [
    {
        "name": "capital_cloze",
        "text": "中国的首都是",
        "expected_near_start": ["北京"],
    },
    {
        "name": "boiling_point",
        "text": "问题：水在标准大气压下多少摄氏度沸腾？答案：",
        "expected_near_start": ["100", "一百"],
    },
    {
        "name": "key_location",
        "text": (
            "小明把红色钥匙放进左边抽屉，把蓝色钥匙放进右边抽屉。"
            "后来他拿出红色钥匙。红色钥匙原来在"
        ),
        "expected_near_start": ["左边抽屉", "左边的抽屉"],
    },
    {
        "name": "person_location",
        "text": (
            "张华今天去了上海，李明留在北京。晚上打电话时，"
            "张华说他所在的城市是"
        ),
        "expected_near_start": ["上海"],
    },
    {
        "name": "causal_continuation",
        "text": "因为外面下着很大的雨，所以我",
        "expected_near_start": [],
    },
    {
        "name": "story_continuation",
        "text": "从前有一个住在山里的小男孩，他每天都会",
        "expected_near_start": [],
    },
]

REPORT_NAME = "checkpoint_evaluation.json"


def resolve_tokenizer_file(run_config: dict) -> Path:
    if TOKENIZER_PATH is not None:
        path = Path(TOKENIZER_PATH)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = PROJECT_ROOT / run_config["paths"]["tokenizer_vocab"]

    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file does not exist: {path}")
    return path


def discover_checkpoints() -> list[tuple[int, Path]]:
    checkpoints = []
    for path in RUN_DIR.glob("ckpt_step_*.pt"):
        try:
            step = int(path.stem.removeprefix("ckpt_step_"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    checkpoints.sort()

    if CHECKPOINT_STEPS is not None:
        by_step = dict(checkpoints)
        missing = [step for step in CHECKPOINT_STEPS if step not in by_step]
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoint steps in {RUN_DIR}: {missing}"
            )
        checkpoints = [(step, by_step[step]) for step in CHECKPOINT_STEPS]

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {RUN_DIR}")
    return checkpoints


def load_validation_metrics() -> dict[int, dict]:
    path = RUN_DIR / "metrics.csv"
    if not path.is_file():
        return {}

    metrics = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            metrics[step] = {
                "tokens_seen": int(row["tokens_seen"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "adamw_lr": float(row["lr"]),
            }
    return metrics


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    tokenizer_size: int,
) -> tuple[Transformer, dict, dict]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model_args = dict(checkpoint["model_args"])
    if model_args["vocab_size"] != tokenizer_size:
        raise ValueError(
            f"Checkpoint vocab_size={model_args['vocab_size']} but "
            f"tokenizer size={tokenizer_size}"
        )

    checkpoint_info = {
        "iteration": int(checkpoint["iteration"]),
        "tokens_seen": checkpoint.get("tokens_seen"),
    }
    model = Transformer(**model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    return model, model_args, checkpoint_info


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def repetition_metrics(token_ids: list[int], ngram_size: int = 4) -> dict:
    if not token_ids:
        return {
            "repeated_ngram_fraction": 0.0,
            "max_ngram_occurrences": 0,
            "longest_identical_token_run": 0,
        }

    longest_run = 1
    current_run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        if current == previous:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    if len(token_ids) < ngram_size:
        repeated_fraction = 0.0
        max_occurrences = 1
    else:
        ngrams = [
            tuple(token_ids[index : index + ngram_size])
            for index in range(len(token_ids) - ngram_size + 1)
        ]
        counts = Counter(ngrams)
        repeated_fraction = 1.0 - len(counts) / len(ngrams)
        max_occurrences = max(counts.values())

    return {
        "repeated_ngram_fraction": repeated_fraction,
        "max_ngram_occurrences": max_occurrences,
        "longest_identical_token_run": longest_run,
    }


def generate_once(
    model: Transformer,
    model_args: dict,
    tokenizer: BackendTokenizer,
    device: torch.device,
    eos_id: int | None,
    prompt: dict,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    set_seed(seed, device)
    prompt_ids = tokenizer.encode(
        prompt["text"],
        add_special_tokens=False,
    ).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode(), autocast_context(device):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_id=eos_id,
            context_length=model_args["context_length"],
        )

    completion_ids = output_ids[0, len(prompt_ids) :].tolist()
    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
    )
    normalized_start = completion.lstrip(" \t\r\n\"'“”‘’「」『』（([【")
    expected = prompt["expected_near_start"]
    expected_match = (
        None
        if not expected
        else any(normalized_start.startswith(candidate) for candidate in expected)
    )

    return {
        "name": prompt["name"],
        "prompt": prompt["text"],
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "completion": completion,
        "completion_tokens": len(completion_ids),
        "stopped_on_eos": eos_id is not None and eos_id in completion_ids,
        "expected_near_start": expected,
        "expected_match": expected_match,
        **repetition_metrics(completion_ids),
    }


def summarize_generations(generations: list[dict]) -> dict:
    repetition_values = [
        item["repeated_ngram_fraction"] for item in generations
    ]
    scored = [
        item["expected_match"]
        for item in generations
        if item["expected_match"] is not None
    ]
    return {
        "samples": len(generations),
        "mean_repeated_4gram_fraction": (
            sum(repetition_values) / len(repetition_values)
            if repetition_values
            else 0.0
        ),
        "high_repetition_samples": sum(
            value >= 0.25 for value in repetition_values
        ),
        "expected_match_accuracy": (
            sum(scored) / len(scored) if scored else None
        ),
        "eos_samples": sum(item["stopped_on_eos"] for item in generations),
    }


def cache_parity_probe(
    model: Transformer,
    model_args: dict,
    tokenizer: BackendTokenizer,
    device: torch.device,
) -> dict:
    """用 FP32 排除 autocast/kernel 数值差异，探测 KV-cache 逻辑。"""
    prompt_ids = tokenizer.encode(
        CACHE_PROBE_PROMPT,
        add_special_tokens=False,
    ).ids
    sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = None
    maximum_differences = []
    top1_matches = []

    with torch.inference_mode():
        cached_logits, cache = model(
            sequence,
            past_key_values=cache,
            use_cache=True,
        )
        full_logits, _ = model(sequence, use_cache=False)

        for _ in range(CACHE_PROBE_STEPS):
            cached_last = cached_logits[:, -1, :]
            full_last = full_logits[:, -1, :]
            maximum_differences.append(
                (cached_last - full_last).abs().max().item()
            )
            top1_matches.append(
                cached_last.argmax(dim=-1).item()
                == full_last.argmax(dim=-1).item()
            )

            next_id = full_last.argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_id), dim=1)
            cached_logits, cache = model(
                next_id,
                past_key_values=cache,
                use_cache=True,
            )
            full_logits, _ = model(sequence, use_cache=False)

    continuation_ids = sequence[0, len(prompt_ids) :].tolist()
    return {
        "prompt": CACHE_PROBE_PROMPT,
        "steps": CACHE_PROBE_STEPS,
        "top1_matches": sum(top1_matches),
        "max_absolute_logit_difference": max(maximum_differences),
        "mean_max_logit_difference": (
            sum(maximum_differences) / len(maximum_differences)
        ),
        "greedy_continuation": tokenizer.decode(
            continuation_ids,
            skip_special_tokens=False,
        ),
    }


def print_generation(result: dict) -> None:
    expected_text = ""
    if result["expected_match"] is not None:
        expected_text = f", expected={result['expected_match']}"
    print(f"[{result['name']}] {result['prompt']}")
    print(f"→ {result['completion']}")
    print(
        f"tokens={result['completion_tokens']}, "
        f"rep4={result['repeated_ngram_fraction']:.3f}, "
        f"max4={result['max_ngram_occurrences']}"
        f"{expected_text}"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run_config_path = RUN_DIR / "config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    tokenizer_file = resolve_tokenizer_file(run_config)
    tokenizer = BackendTokenizer.from_file(str(tokenizer_file))
    tokenizer_size = tokenizer.get_vocab_size(with_added_tokens=True)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    checkpoints = discover_checkpoints()
    validation_metrics = load_validation_metrics()
    latest_step = checkpoints[-1][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"tokenizer: {tokenizer_file}")
    print(f"checkpoints: {[step for step, _ in checkpoints]}")

    report = {
        "run_dir": str(RUN_DIR),
        "tokenizer": str(tokenizer_file),
        "device": str(device),
        "comparison": {
            "temperature": COMPARISON_TEMPERATURE,
            "top_p": COMPARISON_TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "checkpoints": [],
    }

    for step, checkpoint_path in checkpoints:
        model, model_args, checkpoint_info = load_model(
            checkpoint_path,
            device,
            tokenizer_size,
        )
        if checkpoint_info["iteration"] != step:
            raise ValueError(
                f"Filename says step {step}, checkpoint says "
                f"{checkpoint_info['iteration']}"
            )

        print(f"\n{'#' * 24} checkpoint {step} {'#' * 24}")
        generations = []
        for prompt_index, prompt in enumerate(PROMPTS):
            result = generate_once(
                model,
                model_args,
                tokenizer,
                device,
                eos_id,
                prompt,
                seed=20260825 + prompt_index,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=COMPARISON_TEMPERATURE,
                top_p=COMPARISON_TOP_P,
            )
            generations.append(result)
            print_generation(result)

        checkpoint_report = {
            "step": step,
            **checkpoint_info,
            "validation": validation_metrics.get(step),
            "generations": generations,
            "summary": summarize_generations(generations),
        }

        if step == latest_step:
            multi_seed_generations = []
            print(f"\n{'-' * 20} latest checkpoint multi-seed {'-' * 20}")
            for prompt in PROMPTS:
                for seed in MULTI_SEEDS:
                    result = generate_once(
                        model,
                        model_args,
                        tokenizer,
                        device,
                        eos_id,
                        prompt,
                        seed=seed,
                        max_new_tokens=MULTI_SEED_MAX_NEW_TOKENS,
                        temperature=MULTI_SEED_TEMPERATURE,
                        top_p=MULTI_SEED_TOP_P,
                    )
                    multi_seed_generations.append(result)
                    print_generation(result)

            checkpoint_report["multi_seed"] = {
                "temperature": MULTI_SEED_TEMPERATURE,
                "top_p": MULTI_SEED_TOP_P,
                "generations": multi_seed_generations,
                "summary": summarize_generations(multi_seed_generations),
            }

            print(f"\n{'-' * 24} KV-cache parity {'-' * 24}")
            cache_probe = cache_parity_probe(
                model,
                model_args,
                tokenizer,
                device,
            )
            checkpoint_report["cache_parity"] = cache_probe
            print(json.dumps(cache_probe, ensure_ascii=False, indent=2))

        report["checkpoints"].append(checkpoint_report)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report_path = RUN_DIR / REPORT_NAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
