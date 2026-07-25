import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

# 当前 Windows 环境里先 import torch 再 import datasets/pyarrow 会崩。
from sft.datasets.alpaca_zh import AlpacaZhDataset, load_alpaca_zh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from config_loader import Config
from pretrain.model import Transformer
from sft.collator import SFTCollator
from sft.tokenize_sft import IGNORE_INDEX, tokenize_chat_example

CONFIG_PATH = PROJECT_ROOT / "configs" / "sft.json"


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


def build_dataset(config: Config, tokenizer):
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

    print(f"数据处理完成：保留 {len(tokenized)} 条，跳过 {skipped} 条")
    return tokenized


def split_dataset(config: Config, dataset):
    val_size = min(config.require("train", "val_size"), max(1, len(dataset) // 20))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.require("train", "seed"))
    return random_split(dataset, [train_size, val_size], generator=generator)


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


def causal_lm_loss(logits, labels):
    # logits[:, t] 预测 input_ids[:, t + 1]，所以 labels 也要右移一格来对齐。
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


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


def save_checkpoint(model, optimizer, step, output_dir, model_args):
    path = output_dir / f"ckpt_step_{step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_args": model_args,
        },
        path,
    )
    print(f"已保存 checkpoint：{path}")


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
    dataset = build_dataset(config, tokenizer)
    train_dataset, val_dataset = split_dataset(config, dataset)

    smoke_test = train_config["smoke_test"]
    if smoke_test:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(train_config["smoke_train_size"])
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(train_config["smoke_val_size"])
        )

    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    model, model_args = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    run_dir = config.resolve_path("paths", "sft_logs") / time.strftime(
        "run_%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "metrics.csv"

    model.train()
    global_step = 0

    for epoch in range(train_config["num_epochs"]):
        for batch in train_loader:
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(input_ids, attention_mask=attention_mask)
                loss = causal_lm_loss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config["grad_clip"]
            )
            optimizer.step()

            if smoke_test or global_step % 10 == 0:
                print(
                    f"epoch {epoch + 1} step {global_step}: "
                    f"train_loss={loss.item():.4f}"
                )

            if smoke_test or global_step % train_config["eval_every"] == 0:
                val_loss = estimate_loss(config, model, val_loader, device)
                print(
                    f"epoch {epoch + 1} step {global_step}: " f"val_loss={val_loss:.4f}"
                )
                write_metric(
                    metrics_file,
                    global_step,
                    loss.item(),
                    val_loss,
                    train_config["learning_rate"],
                )

            if not smoke_test and global_step % train_config["save_every"] == 0:
                save_checkpoint(model, optimizer, global_step, run_dir, model_args)

            if smoke_test and global_step >= train_config["smoke_max_steps"]:
                save_checkpoint(model, optimizer, global_step, run_dir, model_args)
                return

    save_checkpoint(model, optimizer, global_step, run_dir, model_args)


def main():
    train(load_config())


if __name__ == "__main__":
    main()
