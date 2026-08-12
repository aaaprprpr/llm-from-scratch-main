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
from sft.datasets import load_chat_dataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, random_split

from models.model import Transformer
from pretrain.train_model import (
    attention_kernel_context,
    autocast_context,
    get_device,
    lr_cosine_schedule,
    resolve_amp_dtype,
    verify_flash_attention,
)
from sft.collator import SFTCollator
from sft.tokenize_sft import (
    IGNORE_INDEX,
    iter_tokenized_examples,
    resolve_tokenize_workers,
)
from sft.utils import build_prompt, clean_answer, load_config, load_tokenizer


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
        "adapter": config.require("data", "adapter"),
        "deduplicate": config.get("data", "deduplicate", default=True),
        "dataset": str(dataset_path),
        "dataset_fingerprint": path_fingerprint(dataset_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_fingerprint": path_fingerprint(tokenizer_path),
        "chat_template_sha256": file_sha256(
            config.resolve_path("paths", "chat_template")
        ),
        "max_seq_len": config.require("train", "max_seq_len"),
    }


def prepare_chat_examples(chat_dataset, deduplicate: bool):
    examples = []
    seen_answers = set()
    invalid = 0
    duplicates = 0

    for example in chat_dataset:
        if example is None:
            invalid += 1
            continue

        answers = tuple(
            " ".join(message["content"].split())
            for message in example["messages"]
            if message["role"] == "assistant"
        )
        if not answers:
            invalid += 1
            continue
        if deduplicate and answers in seen_answers:
            duplicates += 1
            continue

        seen_answers.add(answers)
        examples.append(example)

    return examples, invalid, duplicates


def print_dataset_stats(dataset, skip_stats: dict) -> None:
    skipped = sum(skip_stats.values())
    print(
        f"数据处理完成：保留 {len(dataset)} 条，跳过 {skipped} 条"
        f"（无效 {skip_stats['invalid']}，重复 {skip_stats['duplicates']}，"
        f"超长 {skip_stats['too_long']}）"
    )


def load_or_build_dataset(config: Config, tokenizer):
    cache_path = config.resolve_path("paths", "tokenized_cache")
    cache_key = build_cache_key(config)

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("cache_key") == cache_key:
            dataset = cached["dataset"]
            skip_stats = cached.get(
                "skip_stats",
                {
                    "invalid": 0,
                    "duplicates": 0,
                    "too_long": cached.get("skipped", 0),
                },
            )
            print(f"使用 tokenized cache：{cache_path}")
            print_dataset_stats(dataset, skip_stats)
            return dataset

    chat_dataset = load_chat_dataset(
        config.require("data", "adapter"),
        config.resolve_path("paths", "dataset"),
    )
    max_seq_len = config.require("train", "max_seq_len")
    chat_examples, invalid, duplicates = prepare_chat_examples(
        chat_dataset,
        deduplicate=config.get("data", "deduplicate", default=True),
    )
    tokenize_workers = resolve_tokenize_workers(
        config.get("data", "tokenize_workers", default="auto")
    )
    tokenize_chunksize = int(
        config.get("data", "tokenize_chunksize", default=32)
    )

    tokenized = []
    too_long = 0
    print(f"首次构建 tokenized cache，使用 {tokenize_workers} 个进程")
    items = iter_tokenized_examples(
        chat_examples,
        tokenizer,
        max_seq_len=max_seq_len,
        workers=tokenize_workers,
        tokenizer_path=config.resolve_path("paths", "tokenizer"),
        chat_template_path=config.resolve_path("paths", "chat_template"),
        chunksize=tokenize_chunksize,
    )
    for item in items:
        if item is None:
            too_long += 1
            continue
        tokenized.append(item)

    skip_stats = {
        "invalid": invalid,
        "duplicates": duplicates,
        "too_long": too_long,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_key": cache_key,
            "dataset": tokenized,
            "skip_stats": skip_stats,
        },
        cache_path,
    )

    print_dataset_stats(tokenized, skip_stats)
    print(f"已保存 tokenized cache：{cache_path}")
    return tokenized


def split_dataset(config: Config, dataset):
    val_size = min(config.require("train", "val_size"), max(1, len(dataset) // 20))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.require("train", "seed"))
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_dataloader(
    config: Config,
    dataset,
    tokenizer,
    shuffle: bool,
    epoch: int = 0,
    pin_memory: bool = False,
):
    train_config = config.require("train")
    num_workers = train_config.get("num_workers", 0)
    batch_sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=train_config["micro_batch_size"],
        seed=train_config["seed"],
        bucket_multiplier=train_config["length_bucket_multiplier"],
        shuffle=shuffle,
        epoch=epoch,
    )
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def build_model(config: Config, device):
    model_args = config.require("model")
    model = Transformer(**model_args)
    checkpoint = torch.load(
        config.resolve_path("paths", "pretrained_weights"),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model, model_args


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def causal_lm_loss(
    logits,
    labels,
    end_token_id: int,
    end_token_weight: float,
):
    # logits[:, t] 预测 input_ids[:, t + 1]，所以 labels 也要右移一格来对齐。
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    flat_labels = shift_labels.view(-1)
    loss_weights = flat_labels.ne(IGNORE_INDEX).to(token_losses.dtype)
    loss_weights.masked_fill_(
        flat_labels.eq(end_token_id),
        end_token_weight,
    )
    return (token_losses * loss_weights).sum(), loss_weights.sum()


def move_batch_to_device(batch, device):
    non_blocking = device.type == "cuda"
    attention_mask = batch["attention_mask"]
    if bool(attention_mask.all()):
        attention_mask = None
    else:
        attention_mask = attention_mask.to(device, non_blocking=non_blocking)
    return (
        batch["input_ids"].to(device, non_blocking=non_blocking),
        batch["labels"].to(device, non_blocking=non_blocking),
        attention_mask,
    )


@torch.inference_mode()
def estimate_loss(
    config: Config,
    model,
    dataloader,
    device,
    end_token_id: int,
    end_token_weight: float,
    amp_dtype=None,
    require_flash_attention=False,
):
    eval_batches = config.require("train", "eval_batches")
    was_training = model.training
    total_loss = torch.zeros((), device=device)
    total_loss_weight = torch.zeros((), device=device)
    evaluated_batches = 0

    model.eval()
    try:
        for step, batch in enumerate(dataloader):
            if step >= eval_batches:
                break

            input_ids, labels, attention_mask = move_batch_to_device(batch, device)

            with attention_kernel_context(device, require_flash_attention):
                with autocast_context(device, amp_dtype):
                    logits, _ = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    loss_sum, loss_weight = causal_lm_loss(
                        logits,
                        labels,
                        end_token_id=end_token_id,
                        end_token_weight=end_token_weight,
                    )

            total_loss += loss_sum
            total_loss_weight += loss_weight
            evaluated_batches += 1

        if evaluated_batches == 0:
            raise ValueError("验证集没有可用 batch")
        return (total_loss / total_loss_weight).item()
    finally:
        model.train(was_training)


@torch.inference_mode()
def write_samples(
    config: Config,
    model,
    tokenizer,
    device,
    model_args,
    run_dir,
    step,
    amp_dtype=None,
):
    play_config = config.require("play")
    path = run_dir / f"samples_step_{step}.txt"
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    was_training = model.training

    model.eval()
    try:
        with path.open("w", encoding="utf-8") as f:
            for prompt in play_config["prompts"]:
                input_ids = build_prompt(tokenizer, prompt).to(device)
                with autocast_context(device, amp_dtype):
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
        weights_only=True,
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
    tokenizer = load_tokenizer(config)
    dataset = load_or_build_dataset(config, tokenizer)
    train_dataset, val_dataset = split_dataset(config, dataset)
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end_token_weight = float(train_config.get("end_token_weight", 1.0))
    if end_token_weight <= 0:
        raise ValueError("train.end_token_weight 必须大于 0")

    torch.manual_seed(train_config["seed"])
    device = get_device(config)
    amp_dtype = resolve_amp_dtype(
        train_config.get("precision", "bfloat16"),
        device,
    )
    require_flash = train_config.get("require_flash_attention", False)
    if require_flash:
        verify_flash_attention(device, amp_dtype)
    pin_memory = device.type == "cuda"

    effective_precision = (
        "bfloat16 autocast" if amp_dtype == torch.bfloat16 else "float32"
    )
    print(f"使用设备：{device}，精度：{effective_precision}")

    val_loader = build_dataloader(
        config,
        val_dataset,
        tokenizer,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model, model_args = build_model(config, device)
    if train_config.get("activation_checkpointing", False):
        model.gradient_checkpointing_enable()
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

    micro_batches_per_epoch = math.ceil(
        len(train_dataset) / train_config["micro_batch_size"]
    )
    steps_per_epoch = math.ceil(
        micro_batches_per_epoch / train_config["gradient_accumulation_steps"]
    )
    total_steps = train_config["num_epochs"] * steps_per_epoch
    decay_steps = train_config["lr_decay_steps"] or total_steps

    model.train()
    for epoch in range(start_epoch, train_config["num_epochs"]):
        train_loader = build_dataloader(
            config,
            train_dataset,
            tokenizer,
            shuffle=True,
            epoch=epoch,
            pin_memory=pin_memory,
        )
        accum_steps = 0
        accum_loss_sum = torch.zeros((), device=device)
        accum_loss_weight = torch.zeros((), device=device)
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue

            input_ids, labels, attention_mask = move_batch_to_device(batch, device)

            with attention_kernel_context(device, require_flash):
                with autocast_context(device, amp_dtype):
                    logits, _ = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    loss_sum, loss_weight = causal_lm_loss(
                        logits,
                        labels,
                        end_token_id=end_token_id,
                        end_token_weight=end_token_weight,
                    )

                loss_sum.backward()
            accum_steps += 1
            accum_loss_sum += loss_sum.detach().float()
            accum_loss_weight += loss_weight.detach().float()

            last_micro_batch = batch_index + 1 == len(train_loader)
            should_step = (
                accum_steps == train_config["gradient_accumulation_steps"]
                or last_micro_batch
            )
            if not should_step:
                continue

            lr = lr_cosine_schedule(
                global_step,
                train_config["max_learning_rate"],
                train_config["min_learning_rate"],
                train_config["warmup_steps"],
                decay_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accum_loss_weight)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config["grad_clip"]
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            train_loss = (accum_loss_sum / accum_loss_weight).item()
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
                val_loss = estimate_loss(
                    config,
                    model,
                    val_loader,
                    device,
                    amp_dtype=amp_dtype,
                    require_flash_attention=require_flash,
                    end_token_id=end_token_id,
                    end_token_weight=end_token_weight,
                )
                print(
                    f"epoch {epoch + 1} step {global_step}: "
                    f"val_loss={val_loss:.4f}"
                )
                write_metric(metrics_file, global_step, train_loss, val_loss, lr)

            if global_step % train_config["save_every"] == 0 or last_step:
                write_samples(
                    config,
                    model,
                    tokenizer,
                    device,
                    model_args,
                    run_dir,
                    global_step,
                    amp_dtype=amp_dtype,
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
            accum_loss_sum.zero_()
            accum_loss_weight.zero_()

        start_batch_index = 0


def main():
    train(load_config())


if __name__ == "__main__":
    main()
