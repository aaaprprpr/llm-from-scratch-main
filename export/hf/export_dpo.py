import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint_io import load_checkpoint
from export.hf.convert_llama import build_llama_model
from train.dpo.utils import load_config
from train.sft.utils import load_tokenizer, load_tokenizer_from_paths


def export_checkpoint(checkpoint_path: Path, output_dir: Path, config):
    checkpoint = load_checkpoint(checkpoint_path, mmap=True)
    bundled_tokenizer = checkpoint_path.parent / "tokenizer"
    bundled_template = checkpoint_path.parent / "chat_template.jinja"
    if bundled_tokenizer.is_dir() and bundled_template.is_file():
        tokenizer = load_tokenizer_from_paths(bundled_tokenizer, bundled_template)
    else:
        tokenizer = load_tokenizer(config)
    model = build_llama_model(checkpoint, tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "stage": checkpoint.get("stage", "dpo"),
        "model_args": checkpoint["model_args"],
        "tokenizer_sha256": checkpoint.get("tokenizer_sha256"),
        "weight_mapping": "dense-to-llama; q/k interleaved-to-half RoPE rows",
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return model, tokenizer


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="导出标准 Llama 格式，供 Transformers 和 vLLM 原生加载")
    parser.add_argument("--checkpoint", type=Path, default=config.resolve_path("paths", "clean_weights"))
    parser.add_argument("--output", type=Path, default=config.resolve_path("paths", "hf_export"))
    args = parser.parse_args()
    print(f"加载模型：{args.checkpoint}")
    export_checkpoint(args.checkpoint, args.output, config)
    print(f"标准 Llama 模型和 tokenizer 已保存到：{args.output}")


if __name__ == "__main__":
    main()
