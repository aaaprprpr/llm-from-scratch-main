import copy
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from checkpoint_io import load_checkpoint
from train.dpo.utils import load_config
from train.sft.utils import find_latest_checkpoint, load_tokenizer


def export_checkpoint(config, checkpoint_path: Path, output_path: Path):
    print(f"加载 DPO checkpoint：{checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, mmap=True)
    tokenizer = load_tokenizer(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_path.parent / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    template_path = output_path.parent / "chat_template.jinja"
    template_path.write_text(tokenizer.chat_template, encoding="utf-8")
    exported_config = copy.deepcopy(checkpoint.get("config", config.data))
    exported_config["paths"]["tokenizer"] = (
        tokenizer_dir.relative_to(PROJECT_ROOT) if tokenizer_dir.is_relative_to(PROJECT_ROOT) else tokenizer_dir
    ).as_posix()
    exported_config["paths"]["chat_template"] = (
        template_path.relative_to(PROJECT_ROOT) if template_path.is_relative_to(PROJECT_ROOT) else template_path
    ).as_posix()
    torch.save({
        "model": checkpoint["model"],
        "model_args": checkpoint["model_args"],
        "config": exported_config,
        "tokenizer_sha256": hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "stage": "dpo",
    }, output_path)

    print(f"模型权重、结构与 tokenizer 已保存到：{output_path}")
    print(f"文件大小：{output_path.stat().st_size / 1024**2:.2f} MiB")


def main():
    config = load_config()
    checkpoint_path = config.optional_path("paths", "dpo_checkpoint") or find_latest_checkpoint(
        config.resolve_path("paths", "dpo_logs"), stage_name="DPO"
    )
    export_checkpoint(config, checkpoint_path, config.resolve_path("paths", "clean_weights"))


if __name__ == "__main__":
    main()
