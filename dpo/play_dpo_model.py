import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dpo.utils import load_config
from models.model import Transformer
from pretrain.train_model import autocast_context, get_device, resolve_amp_dtype
from sft.utils import (
    build_prompt,
    clean_answer,
    find_latest_checkpoint,
    load_tokenizer,
)


def load_model(config, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    model_args = checkpoint.get("model_args", config.require("model"))
    model = Transformer(**model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, model_args


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    context_length: int,
    play_config: dict,
    amp_dtype=None,
):
    input_ids = build_prompt(tokenizer, prompt).to(device)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    with autocast_context(device, amp_dtype):
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
    device = get_device(config)
    amp_dtype = resolve_amp_dtype(
        config.get("train", "precision", default="bfloat16"),
        device,
    )
    checkpoint_path = find_latest_checkpoint(
        config.resolve_path("paths", "dpo_logs"),
        stage_name="DPO",
    )

    print(f"使用设备：{device}")
    print(f"加载 DPO checkpoint：{checkpoint_path}")

    tokenizer = load_tokenizer(config)
    model, model_args = load_model(config, checkpoint_path, device)
    context_length = model_args["context_length"]

    print("-" * 50)
    for prompt in play_config["prompts"]:
        answer = generate_answer(
            model,
            tokenizer,
            prompt,
            device,
            context_length,
            play_config,
            amp_dtype,
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
            model,
            tokenizer,
            prompt,
            device,
            context_length,
            play_config,
            amp_dtype,
        )
        print(f"助手：{answer}")
        print("=" * 80)


if __name__ == "__main__":
    main()
