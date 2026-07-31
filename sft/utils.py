import re
from pathlib import Path

from config_loader import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "sft.json"


def load_config() -> Config:
    return Config(CONFIG_PATH)


def load_tokenizer(config: Config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.resolve_path("paths", "tokenizer")
    )
    tokenizer.chat_template = config.resolve_path(
        "paths", "chat_template"
    ).read_text(encoding="utf-8")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return enc["input_ids"]


def clean_answer(text: str) -> str:
    text = text.split("<|im_end|>", 1)[0]
    text = text.split("<|endoftext|>", 1)[0]
    return text.strip()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"ckpt_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(logs_path: Path) -> Path:
    checkpoints = list(logs_path.glob("run_*/ckpt_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"没有找到 SFT checkpoint：{logs_path}")
    return max(
        checkpoints,
        key=lambda path: (checkpoint_step(path), path.stat().st_mtime),
    )
