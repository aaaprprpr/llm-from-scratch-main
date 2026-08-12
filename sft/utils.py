import re
from pathlib import Path

from config_loader import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "sft.json"


def load_config() -> Config:
    return Config(CONFIG_PATH)


def load_tokenizer_from_paths(
    tokenizer_path: str | Path,
    chat_template_path: str | Path,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.chat_template = Path(chat_template_path).read_text(
        encoding="utf-8"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_tokenizer(config: Config):
    return load_tokenizer_from_paths(
        config.resolve_path("paths", "tokenizer"),
        config.resolve_path("paths", "chat_template"),
    )


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


def find_latest_checkpoint(logs_path: Path, stage_name: str = "SFT") -> Path:
    runs_with_checkpoints = []
    for run_dir in logs_path.glob("run_*"):
        checkpoints = list(run_dir.glob("ckpt_step_*.pt"))
        if checkpoints:
            runs_with_checkpoints.append((run_dir, checkpoints))

    if not runs_with_checkpoints:
        raise FileNotFoundError(f"没有找到 {stage_name} checkpoint：{logs_path}")

    _, checkpoints = max(
        runs_with_checkpoints,
        key=lambda item: (item[0].name, item[0].stat().st_mtime),
    )
    return max(
        checkpoints,
        key=lambda path: checkpoint_step(path),
    )
