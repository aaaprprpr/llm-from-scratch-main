import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from config_loader import Config
from pretrain.model import Transformer

CONFIG_PATH = PROJECT_ROOT / "configs" / "dpo.json"


def load_config() -> Config:
    return Config(CONFIG_PATH)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"ckpt_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(config: Config) -> Path:
    dpo_logs_path = config.resolve_path("paths", "dpo_logs")
    checkpoints = list(dpo_logs_path.glob("run_*/ckpt_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"没有找到 DPO checkpoint：{dpo_logs_path}")
    return max(
        checkpoints, key=lambda path: (checkpoint_step(path), path.stat().st_mtime)
    )


def load_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.resolve_path("paths", "tokenizer"))
    tokenizer.chat_template = config.resolve_path("paths", "chat_template").read_text(
        encoding="utf-8"
    )
    return tokenizer


def load_model(config: Config, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    model_args = checkpoint.get("model_args", config.require("model"))
    model = Transformer(**model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, model_args


def build_prompt(tokenizer, prompt: str) -> torch.Tensor:
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


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    context_length: int,
    play_config: dict,
):
    input_ids = build_prompt(tokenizer, prompt).to(device)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=play_config["max_new_tokens"],
        temperature=play_config["temperature"],
        top_p=play_config["top_p"],
        eos_id=im_end_id,
        context_length=context_length,
    )

    new_ids = output_ids[0, input_ids.size(1) :].tolist()
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return clean_answer(text)


def main():
    config = load_config()
    play_config = config.require("play")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = find_latest_checkpoint(config)

    print(f"使用设备：{device}")
    print(f"加载 DPO checkpoint：{checkpoint_path}")

    tokenizer = load_tokenizer(config)
    model, model_args = load_model(config, checkpoint_path, device)
    context_length = model_args["context_length"]

    print("-" * 50)
    for prompt in play_config["prompts"]:
        answer = generate_answer(
            model, tokenizer, prompt, device, context_length, play_config
        )
        print(f"用户：{prompt}")
        print(f"助手：{answer}")
        print("=" * 80)

    if not sys.stdin.isatty():
        return

    print("输入问题继续试玩，直接回车退出。")
    while True:
        prompt = input("用户：").strip()
        if not prompt:
            break
        answer = generate_answer(
            model, tokenizer, prompt, device, context_length, play_config
        )
        print(f"助手：{answer}")
        print("=" * 80)


if __name__ == "__main__":
    main()
