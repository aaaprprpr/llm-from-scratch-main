import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config

# 当前 Windows 环境里先 import torch 再 import datasets/pyarrow 会崩。
from sft.datasets.alpaca_zh import AlpacaZhDataset, load_alpaca_zh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, random_split
from transformers import AutoTokenizer

from pretrain.model import Transformer
from sft.collator import SFTCollator
from sft.tokenize_sft import IGNORE_INDEX, tokenize_chat_example

CONFIG_PATH = PROJECT_ROOT / "configs" / "sft.json"


class LengthBucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int,
        bucket_multiplier: int,
        shuffle: bool,
        epoch: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_multiplier = max(1, bucket_multiplier)
        self.shuffle = shuffle
        self.epoch = epoch
        self.lengths = [len(dataset[i]["input_ids"]) for i in range(len(dataset))]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        if self.shuffle:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]

        bucket_size = self.batch_size * self.bucket_multiplier
        batches = []
        for start in range(0, len(indices), bucket_size):
            bucket = indices[start : start + bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            for batch_start in range(0, len(bucket), self.batch_size):
                batches.append(bucket[batch_start : batch_start + self.batch_size])

        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]

        yield from batches

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def load_config() -> Config:
    return Config(CONFIG_PATH)


def get_device(config: Config):
    device_name = config.get("train", "device", default="auto")
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.resolve_path("paths", "tokenizer"))
    tokenizer.chat_template = config.resolve_path("paths", "chat_template").read_text(
        encoding="utf-8"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_fingerprint(path: Path):
    if path.is_file():
        stat = path.stat()
        return {
            "path": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    files = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            stat = child.stat()
            files.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return files


def build_cache_key(config: Config) -> dict:
    dataset_path = config.resolve_path("paths", "dataset")
    tokenizer_path = config.resolve_path("paths", "tokenizer")
    return {
        "dataset": str(dataset_path),
        "dataset_fingerprint": path_fingerprint(dataset_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_fingerprint": path_fingerprint(tokenizer_path),
        "chat_template_sha256": file_sha256(
            config.resolve_path("paths", "chat_template")
        ),
        "max_seq_len": config.require("train", "max_seq_len"),
    }


def load_or_build_dataset(config: Config, tokenizer):
    cache_path = config.resolve_path("paths", "tokenized_cache")
    cache_key = build_cache_key(config)

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("cache_key") == cache_key:
            dataset = cached["dataset"]
            skipped = cached.get("skipped", 0)
            print(f"使用 tokenized cache：{cache_path}")
            print(f"数据处理完成：保留 {len(dataset)} 条，跳过 {skipped} 条")
            return dataset

    raw_dataset = load_alpaca_zh(config.resolve_path("paths", "dataset"))
    chat_dataset = AlpacaZhDataset(raw_dataset)
    max_seq_len = config.require("train", "max_seq_len")

    tokenized = []
    skipped = 0
    for example in chat_dataset:
        item = tokenize_chat_example(example, tokenizer, max_seq_len=max_seq_len)
        if item is None:
            skipped += 1
            continue
        tokenized.append(item)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_key": cache_key,
            "dataset": tokenized,
            "skipped": skipped,
        },
        cache_path,
    )

    print(f"数据处理完成：保留 {len(tokenized)} 条，跳过 {skipped} 条")
    print(f"已保存 tokenized cache：{cache_path}")
    return tokenized


def split_dataset(config: Config, dataset):
    val_size = min(config.require("train", "val_size"), max(1, len(dataset) // 20))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.require("train", "seed"))
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_dataloader(config: Config, dataset, tokenizer, shuffle: bool, epoch: int = 0):
    train_config = config.require("train")
    batch_sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=train_config["micro_batch_size"],
        seed=train_config["seed"],
        bucket_multiplier=train_config["length_bucket_multiplier"],
        shuffle=shuffle,
        epoch=epoch,
    )
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collator)


def build_model(config: Config, device):
    model_args = config.require("model")
    model = Transformer(**model_args)
    state_dict = torch.load(
        config.resolve_path("paths", "pretrained_weights"),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model, model_args


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def causal_lm_loss(logits, labels):
    # logits[:, t] 预测 input_ids[:, t + 1]，所以 labels 也要右移一格来对齐。
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def sft_lr_schedule(step, max_lr, min_lr, warmup_steps, decay_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if decay_steps <= warmup_steps:
        return max_lr

    if step >= decay_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + cosine * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(config: Config, model, dataloader, device):
    model.eval()
    losses = []
    eval_batches = config.require("train", "eval_batches")

    for step, batch in enumerate(dataloader):
        if step >= eval_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(input_ids, attention_mask=attention_mask)
            loss = causal_lm_loss(logits, labels)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


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
def write_samples(config: Config, model, tokenizer, device, model_args, run_dir, step):
    play_config = config.require("play")
    path = run_dir / f"samples_step_{step}.txt"
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    was_training = model.training

    model.eval()
    try:
        with path.open("w", encoding="utf-8") as f:
            for prompt in play_config["prompts"]:
                input_ids = build_prompt(tokenizer, prompt).to(device)
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=play_config["max_new_tokens"],
                    temperature=play_config["temperature"],
                    top_p=play_config["top_p"],
                    eos_id=im_end_id,
                    context_length=model_args["context_length"],
                )
                new_ids = output_ids[0, input_ids.size(1) :].tolist()
                text = tokenizer.decode(
                    new_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                f.write(f"用户：{prompt}\n")
                f.write(f"助手：{clean_answer(text)}\n")
                f.write("=" * 80 + "\n")
    finally:
        model.train(was_training)


def prepare_run_dir(config: Config):
    resume_path = config.optional_path("paths", "resume")
    if resume_path is not None:
        return resume_path.parent, resume_path

    run_dir = config.resolve_path("paths", "sft_logs") / time.strftime(
        "run_%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.config_path, run_dir / "config.json")
    shutil.copy2(config.resolve_path("paths", "chat_template"), run_dir / "chat_template.jinja")
    return run_dir, None


def save_checkpoint(
    model,
    optimizer,
    global_step,
    epoch,
    next_batch_index,
    output_dir,
    model_args,
    config: Config,
):
    path = output_dir / f"ckpt_step_{global_step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "step": global_step,
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "model_args": model_args,
            "config": config.data,
        },
        path,
    )
    print(f"已保存 checkpoint：{path}")


def load_checkpoint_if_needed(resume_path, model, optimizer, device):
    if resume_path is None:
        return 0, 0, 0

    checkpoint = torch.load(
        resume_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_to_device(optimizer, device)

    global_step = checkpoint.get("global_step", checkpoint.get("step", 0))
    epoch = checkpoint.get("epoch", 0)
    next_batch_index = checkpoint.get("next_batch_index", 0)
    print(
        f"从 checkpoint 恢复：step {global_step}, "
        f"epoch {epoch}, next_batch_index {next_batch_index}"
    )
    return global_step, epoch, next_batch_index


def write_metric(metrics_file, step, train_loss, val_loss, lr):
    exists = metrics_file.exists()
    with metrics_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["step", "train_loss", "val_loss", "lr"])
        writer.writerow([step, train_loss, val_loss, lr])


def train(config: Config):
    train_config = config.require("train")
    torch.manual_seed(train_config["seed"])
    device = get_device(config)
    print(f"使用设备：{device}")

    tokenizer = load_tokenizer(config)
    dataset = load_or_build_dataset(config, tokenizer)
    train_dataset, val_dataset = split_dataset(config, dataset)
    val_loader = build_dataloader(config, val_dataset, tokenizer, shuffle=False)

    model, model_args = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["max_learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    run_dir, resume_path = prepare_run_dir(config)
    metrics_file = run_dir / "metrics.csv"
    global_step, start_epoch, start_batch_index = load_checkpoint_if_needed(
        resume_path, model, optimizer, device
    )

    train_loader_for_len = build_dataloader(
        config, train_dataset, tokenizer, shuffle=True, epoch=start_epoch
    )
    steps_per_epoch = math.ceil(
        len(train_loader_for_len) / train_config["gradient_accumulation_steps"]
    )
    total_steps = train_config["num_epochs"] * steps_per_epoch
    decay_steps = train_config["lr_decay_steps"] or total_steps

    model.train()
    for epoch in range(start_epoch, train_config["num_epochs"]):
        train_loader = build_dataloader(
            config, train_dataset, tokenizer, shuffle=True, epoch=epoch
        )
        accum_steps = 0
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue

            lr = sft_lr_schedule(
                global_step,
                train_config["max_learning_rate"],
                train_config["min_learning_rate"],
                train_config["warmup_steps"],
                decay_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(input_ids, attention_mask=attention_mask)
                loss = causal_lm_loss(logits, labels)

            (loss / train_config["gradient_accumulation_steps"]).backward()
            accum_steps += 1
            accum_loss += loss.item()

            last_micro_batch = batch_index + 1 == len(train_loader)
            should_step = (
                accum_steps == train_config["gradient_accumulation_steps"]
                or last_micro_batch
            )
            if not should_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config["grad_clip"]
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            train_loss = accum_loss / accum_steps
            next_batch_index = batch_index + 1
            last_step = (
                epoch + 1 == train_config["num_epochs"] and next_batch_index == len(train_loader)
            )

            if global_step % 10 == 0 or last_step:
                print(
                    f"epoch {epoch + 1} step {global_step}: "
                    f"train_loss={train_loss:.4f}, "
                    f"lr={lr:.2e}, "
                    f"grad_norm={grad_norm.item():.4f}"
                )

            if global_step % train_config["eval_every"] == 0 or last_step:
                val_loss = estimate_loss(config, model, val_loader, device)
                print(
                    f"epoch {epoch + 1} step {global_step}: "
                    f"val_loss={val_loss:.4f}"
                )
                write_metric(metrics_file, global_step, train_loss, val_loss, lr)

            if global_step % train_config["save_every"] == 0 or last_step:
                write_samples(
                    config, model, tokenizer, device, model_args, run_dir, global_step
                )
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    epoch,
                    next_batch_index,
                    run_dir,
                    model_args,
                    config,
                )

            accum_steps = 0
            accum_loss = 0.0

        start_batch_index = 0


def main():
    train(load_config())


if __name__ == "__main__":
    main()
