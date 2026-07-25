import os, time, torch, csv, wandb, sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config
from pretrain.tokenizer_optimized import Tokenizer
from pretrain.train_model import (
    lr_cosine_schedule,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)
from pretrain.model import Transformer as Model

CONFIG_PATH = PROJECT_ROOT / "configs" / "pretrain.json"


def load_config() -> Config:
    return Config(CONFIG_PATH)


def get_device(config: Config):
    device_name = config.get("train", "device", default="auto")
    if device_name != "auto":
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_config(config: Config):
    print("=" * 20 + " Training Configurations " + "=" * 20)
    for section, values in config.data.items():
        print(f"[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"{key:20}: {value}")
        else:
            print(values)
    print("=" * 65)


@torch.no_grad()
def estimate_loss(
    model,
    data,
    batch_size,
    context_length,
    device,
    eval_iters,
    start_position=0,
):
    """在固定数据窗口上计算平均 loss，不修改训练游标。"""
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    was_training = model.training
    position = start_position
    total_loss = torch.zeros((), device=device)

    model.eval()
    try:
        for _ in range(eval_iters):
            x, y, position = get_batch(
                data,
                batch_size,
                context_length,
                device,
                position,
            )  # (B, T)
            logits, _ = model(x, use_cache=False)  # logits size (B, T, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )  # 等同于 (B*T, V) 以及 (B*T, )
            total_loss += loss
        return (total_loss / eval_iters).item()
    finally:
        model.train(was_training)


def main():
    config = load_config()
    model_config = config.require("model")
    train_config = config.require("train")
    optimizer_config = config.require("optimizer")
    lr_config = config.require("lr_schedule")
    logging_config = config.require("logging")
    sample_config = config.require("sample")

    device = get_device(config)
    print(f"using device: {device}")

    out_dir = config.resolve_path("paths", "out_root") / time.strftime(
        "run_%Y%m%d_%H%M%S"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 在日志开头记录训练配置
    print_config(config)
    print(f"{'out_dir':20}: {out_dir}")

    resume_path = config.optional_path("paths", "resume")
    metrics_path = os.path.join(out_dir, "metrics.csv")
    write_header = not (resume_path and os.path.exists(metrics_path))
    mode = "w" if write_header else "a"

    with open(metrics_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "train_loss", "val_loss", "lr"])

    tokenizer = Tokenizer(str(config.resolve_path("paths", "tokenizer_vocab")))

    # 使用np.memmap以高效内存的方式加载数据
    train_data = np.memmap(config.resolve_path("paths", "train_data"), dtype=np.uint16, mode="r")
    val_data = np.memmap(config.resolve_path("paths", "val_data"), dtype=np.uint16, mode="r")

    # model, optimizer  优化器手写换官方了
    model = Model(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_config["max_lr"],
        weight_decay=optimizer_config["weight_decay"],
    )  # 这个初始化的 lr 只是个占位。后面都会被 cosine 的强行覆盖.

    # 检查点恢复
    start_iter = 0
    train_position = 0
    if resume_path:
        start_iter, train_position = load_checkpoint(resume_path, model, optimizer)
        print(f"Resuming from iteration {start_iter}, train_position {train_position}")

    # initialize wandb
    if logging_config["use_wandb"]:
        wandb.init(project=logging_config["wandb_project"], config=config.data)

    # ==============================
    # 训练循环
    last_log_time = time.time()
    last_log_step = start_iter

    for it in range(start_iter, train_config["max_iters"]):

        # 更新学习率（余弦调度）
        lr = lr_cosine_schedule(
            it,
            lr_config["max_lr"],
            lr_config["min_lr"],
            lr_config["warmup_iters"],
            lr_config["lr_decay_iters"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 当前 iteration 的 batch 完整地在循环内部产生和消费
        x, y, train_position = get_batch(
            train_data,
            train_config["batch_size"],
            model_config["context_length"],
            device,
            train_position,
        )

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(x, use_cache=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), optimizer_config["max_norm"]
        )
        optimizer.step()
        # 到这里才算真正完成一次更新
        completed_steps = it + 1
        last_step = completed_steps == train_config["max_iters"]

        # 每隔一定步数（日志间隔）打印训练进度
        if completed_steps % train_config["log_interval"] == 0 or last_step:
            now = time.time()
            elapsed = now - last_log_time
            steps_since_log = completed_steps - last_log_step
            ms_per_step = elapsed * 1000 / steps_since_log

            print(
                f"step {completed_steps}: "
                f"loss {loss.item():.4f}, "
                f"time {ms_per_step:.2f}ms/step, "
                f"grad_norm {grad_norm.item():.4f}"
            )

            last_log_time = now
            last_log_step = completed_steps

        # 每隔一定步数（评估间隔）执行评估并记录日志
        if completed_steps % train_config["eval_interval"] == 0 or last_step:
            train_loss = estimate_loss(
                model,
                train_data,
                train_config["batch_size"],
                model_config["context_length"],
                device,
                train_config["eval_iters"],
            )
            val_loss = estimate_loss(
                model,
                val_data,
                train_config["batch_size"],
                model_config["context_length"],
                device,
                train_config["eval_iters"],
            )
            print(
                f"Step {completed_steps}: "
                f"train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}, "
                f"lr {lr:.2e}"
            )

            if logging_config["use_wandb"]:
                wandb.log(
                    {
                        "step": completed_steps,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "lr": lr,
                    }
                )

            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        completed_steps,
                        (
                            train_loss.item()
                            if torch.is_tensor(train_loss)
                            else train_loss
                        ),
                        val_loss.item() if torch.is_tensor(val_loss) else val_loss,
                        lr,
                    ]
                )

        # 每隔一定步数（检查点间隔）从模型生成文本并保存结果
        checkpoint_interval = (
            train_config["eval_interval"]
            * logging_config["checkpoint_interval_multiplier"]
        )
        if completed_steps % checkpoint_interval == 0 or last_step:

            context = sample_config["prompt"]
            temperature = sample_config["temperature"]
            top_p = sample_config["top_p"]
            idx = tokenizer.idx(context, device=device)
            with torch.inference_mode():
                full_sentence = model.generate(
                    idx,
                    max_new_tokens=sample_config["max_new_tokens"],
                    temperature=temperature,
                    top_p=top_p,
                    eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
                    context_length=model_config["context_length"],
                )
            full_sentence = tokenizer.text(full_sentence, device=device)
            print(
                f"[Generated at iter {completed_steps}, temperature {temperature}, top_p {top_p}]: {full_sentence}"
            )

            ckpt_path = os.path.join(out_dir, f"ckpt_step_{completed_steps}.pt")
            save_checkpoint(
                model,
                optimizer,
                completed_steps,
                ckpt_path,
                train_position=train_position,
            )


if __name__ == "__main__":
    main()
