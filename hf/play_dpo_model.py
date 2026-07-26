import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "hf" / "dpo_model"
MODULE_CACHE = PROJECT_ROOT / ".hf_modules_cache"

os.environ.setdefault("HF_MODULES_CACHE", str(MODULE_CACHE))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "中国的首都是哪里？",
    "自然语言处理是什么？",
    "请用三句话介绍深度学习。",
    "北京是一座什么样的城市？",
    "我今天吃了",
    "上路被三人越塔，应该怎么办？",
    "写一个 Python 函数，计算列表里所有数字的平均值。",
]


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
def generate_answer(model, tokenizer, prompt: str, device: torch.device):
    input_ids = build_prompt(tokenizer, prompt).to(device)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=im_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_ids = output_ids[0, input_ids.size(1) :].tolist()
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return clean_answer(text)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"加载 HF 模型：{MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model.to(device)
    model.eval()

    print("-" * 50)
    for prompt in PROMPTS:
        answer = generate_answer(model, tokenizer, prompt, device)
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
        answer = generate_answer(model, tokenizer, prompt, device)
        print(f"助手：{answer}")
        print("=" * 80)


if __name__ == "__main__":
    main()
