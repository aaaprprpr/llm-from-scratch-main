import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from checkpoint_io import load_checkpoint
from train.pretrain.train_model import tokenizer_fingerprint
from train.sft.utils import find_latest_checkpoint, load_config, load_tokenizer

OUTPUT_PATH = PROJECT_ROOT / "output" / "sft_weights" / "model.pt"


def export_checkpoint(config, checkpoint_path: Path, output_path: Path = OUTPUT_PATH):
    print(f"加载 SFT checkpoint：{checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, mmap=True)
    tokenizer = load_tokenizer(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_path.parent / "tokenizer"
    template_path = output_path.parent / "chat_template.jinja"
    tokenizer.save_pretrained(tokenizer_path)
    template_path.write_text(tokenizer.chat_template, encoding="utf-8")
    export_config = deepcopy(checkpoint.get("config", config.data))
    export_config["paths"]["tokenizer"] = tokenizer_path.relative_to(
        PROJECT_ROOT
    ).as_posix()
    export_config["paths"]["chat_template"] = template_path.relative_to(
        PROJECT_ROOT
    ).as_posix()
    torch.save(
        {
            "model": checkpoint["model"],
            "model_args": checkpoint["model_args"],
            "config": export_config,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer_path),
            "stage": "sft",
        },
        output_path,
    )
    print(f"模型权重与架构已保存到：{output_path}")
    print(f"文件大小：{output_path.stat().st_size / 1024**2:.2f} MiB")


def main():
    config = load_config()
    checkpoint_path = find_latest_checkpoint(
        config.resolve_path("paths", "sft_logs")
    )
    export_checkpoint(config, checkpoint_path)


if __name__ == "__main__":
    main()
