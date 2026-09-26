from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
QWEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if "bpe" not in config:
        raise ValueError(f"Missing 'bpe' section in {CONFIG_PATH}")
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def resolve_input_files(patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        absolute_pattern = str(resolve_path(pattern))
        for match in glob.glob(absolute_pattern, recursive=True):
            path = Path(match)
            if path.is_file():
                files.add(path.resolve())
    if not files:
        raise FileNotFoundError(
            "No tokenizer training files matched bpe.input_files in "
            f"{CONFIG_PATH}"
        )
    return sorted(files)


def build_tokenizer():
    from tokenizers import Regex, Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFC
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    from tokenizers.processors import ByteLevel as ByteLevelProcessor

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(pattern=Regex(QWEN_PATTERN), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False)
    tokenizer.post_processor = ByteLevelProcessor(
        add_prefix_space=False, use_regex=False
    )
    return tokenizer


def train_tokenizer(config: dict[str, Any]):
    settings = config["bpe"]
    threads = settings.get("threads", "auto")
    if threads != "auto":
        threads = int(threads)
        if threads <= 0:
            raise ValueError("bpe.threads must be 'auto' or a positive integer")
        os.environ["RAYON_NUM_THREADS"] = str(threads)

    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    input_files = resolve_input_files(list(settings["input_files"]))
    output_dir = resolve_path(settings["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not settings.get(
        "overwrite_output", False
    ):
        raise FileExistsError(
            f"Tokenizer output directory is not empty: {output_dir}. "
            "Set bpe.overwrite_output=true to replace its tokenizer files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=int(settings["vocab_size"]),
        min_frequency=int(settings["min_frequency"]),
        special_tokens=list(settings["special_tokens"]),
        show_progress=bool(settings.get("show_progress", True)),
        initial_alphabet=ByteLevel.alphabet(),
        continuing_subword_prefix=str(
            settings.get("continuing_subword_prefix", "")
        ),
        max_token_length=int(settings["max_token_length"]),
    )

    total_size = sum(path.stat().st_size for path in input_files)
    print(
        f"Training BPE from {len(input_files)} file(s), "
        f"{total_size / (1024 ** 2):.2f} MiB"
    )
    for path in input_files:
        print(f"  - {path}")
    tokenizer.train(files=[str(path) for path in input_files], trainer=trainer)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=settings["bos_token"],
        eos_token=settings["eos_token"],
        unk_token=None,
        pad_token=settings["pad_token"],
        additional_special_tokens=list(settings["additional_special_tokens"]),
        model_input_names=["input_ids", "attention_mask"],
    )
    wrapped.save_pretrained(output_dir)
    print(f"Saved tokenizer to {output_dir}")
    print(f"Vocabulary size: {wrapped.vocab_size}")

    reloaded = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    for text in settings.get("test_texts", []):
        token_ids = reloaded.encode(text, add_special_tokens=False)
        decoded = reloaded.decode(token_ids, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise RuntimeError(
                f"Tokenizer round-trip failed: input={text!r}, decoded={decoded!r}"
            )
        print(f"Test: {text!r} -> {len(token_ids)} tokens -> round-trip OK")
    return wrapped


def main() -> None:
    train_tokenizer(load_config())


if __name__ == "__main__":
    main()
