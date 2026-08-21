import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config_loader import Config
from models.model import Transformer
from tokenizer import Tokenizer

CONFIG_PATH = PROJECT_ROOT / "configs" / "pretrain.json"

# 指定文件时加载指定 checkpoint；保持 None 就玩最近保存的预训练模型。
CHECKPOINT_PATH = None

MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0
TOP_P = 0.95

PROMPTS = [
    "中国的首都是",
    "自然语言处理",
    "北京是一座",
    "深度学习是",
    "hello ,i am",
    "今天天气",
    "我今天吃了",
    '今天早上起床以后，我',
    '因为外面下着很大的雨，所以我',
    '一年有十二个月，分别是',
    '从前有一个住在山里的小男孩，他每天都会',
    '近年来，随着计算机技术的发展，人工智能在'
]


def find_checkpoint(config: Config) -> Path:
    if CHECKPOINT_PATH is not None:
        path = Path(CHECKPOINT_PATH)
        return path if path.is_absolute() else PROJECT_ROOT / path

    out_root = config.resolve_path("paths", "out_root")
    checkpoints = list(out_root.glob("run_*/ckpt_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"没有找到预训练 checkpoint：{out_root}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_model(
    config: Config,
    checkpoint_path: Path,
    device: torch.device,
    tokenizer_size: int,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        model_args = checkpoint.get("model_args")
        if model_args is None:
            model_args = config.require("model")
    else:
        state_dict = checkpoint
        model_args = config.require("model")

    if model_args["vocab_size"] != tokenizer_size:
        raise ValueError(
            f"Checkpoint vocabulary size {model_args['vocab_size']} does not "
            f"match tokenizer size {tokenizer_size}. Use the tokenizer that "
            "was used to build this checkpoint."
        )

    model = Transformer(**model_args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, model_args


def main():
    config = Config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = find_checkpoint(config)

    print(f"使用设备：{device}")
    print(f"加载 checkpoint：{checkpoint_path}")

    tokenizer = Tokenizer(
        str(config.resolve_path("paths", "tokenizer_vocab"))
    )
    model, model_args = load_model(
        config,
        checkpoint_path,
        device,
        len(tokenizer.tokenizer),
    )
    context_length = model_args["context_length"]
    eos_id = tokenizer.special_token_to_id.get("<|endoftext|>")
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    print(f"模型上下文：{context_length}")
    print("-" * 50)

    with torch.inference_mode():
        for prompt in PROMPTS:
            input_ids = tokenizer.idx(prompt, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    eos_id=eos_id,
                    context_length=context_length,
                )

            print(f"输入：{prompt}")
            print(f"输出：{tokenizer.text(output_ids, device=device)}")
            print("=" * 80)


if __name__ == "__main__":
    main()
