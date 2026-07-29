import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    import wandb
except ImportError:
    wandb = None

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


def parse_args():
    parser = argparse.ArgumentParser(description="预训练 dense Transformer")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="训练配置路径",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=(2048, 4096),
        help="覆盖配置中的训练序列长度，用于切换 2K/4K 基线",
    )
    return parser.parse_args()


def load_config(config_path=CONFIG_PATH) -> Config:
    return Config(config_path)


def get_device(config: Config):
    device_name = config.get("train", "device", default="auto")
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(precision: str, device: torch.device):
    if precision == "float32":
        return None
    if precision != "bfloat16":
        raise ValueError(
            f"train.precision must be 'float32' or 'bfloat16', got {precision!r}"
        )
    if device.type != "cuda":
        return None
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "当前 CUDA 设备不支持 bfloat16；请把 train.precision 改为 float32"
        )
    return torch.bfloat16


def autocast_context(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def attention_kernel_context(device: torch.device, require_flash: bool):
    if device.type == "cuda" and require_flash:
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    return nullcontext()


def verify_flash_attention(device: torch.device, dtype):
    if device.type != "cuda":
        raise RuntimeError(
            "train.require_flash_attention=true，但当前训练设备不是 CUDA"
        )
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashAttention baseline requires float16 or bfloat16")

    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        1,
        8,
        128,
        64,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(q, q, q, is_causal=True)
            output.square().mean().backward()
            torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(
            "当前设备或 PyTorch 构建无法运行 FlashAttention SDPA"
        ) from exc
    print("FlashAttention SDPA forward/backward verification passed")


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
    amp_dtype=None,
    require_flash_attention=False,
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
            with attention_kernel_context(device, require_flash_attention):
                with autocast_context(device, amp_dtype):
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
    args = parse_args()
    config = load_config(args.config)
    if args.sequence_length is not None:
        config.require("train")["sequence_length"] = args.sequence_length

    model_config = dict(config.require("model"))
    train_config = config.require("train")
    optimizer_config = config.require("optimizer")
    lr_config = config.require("lr_schedule")
    logging_config = config.require("logging")
    sample_config = config.require("sample")

    device = get_device(config)
    seed = train_config.get("seed", 42)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"train.seed must be a non-negative integer, got {seed}")
    torch.manual_seed(seed)

    sequence_length = train_config.get(
        "sequence_length",
        model_config["context_length"],
    )
    if sequence_length <= 0:
        raise ValueError(
            f"train.sequence_length must be positive, got {sequence_length}"
        )
    if sequence_length > model_config["context_length"]:
        raise ValueError(
            f"train.sequence_length {sequence_length} exceeds model.context_length "
            f"{model_config['context_length']}"
        )

    batch_size = train_config["batch_size"]
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            f"train.batch_size must be a positive integer, got {batch_size}"
        )
    tokens_per_micro_batch = batch_size * sequence_length
    configured_tokens_per_update = train_config.get("tokens_per_update")
    if configured_tokens_per_update is not None:
        if configured_tokens_per_update < tokens_per_micro_batch:
            raise ValueError(
                "train.tokens_per_update cannot be smaller than one micro-batch "
                f"({tokens_per_micro_batch})"
            )
        if configured_tokens_per_update % tokens_per_micro_batch != 0:
            raise ValueError(
                "train.tokens_per_update must be divisible by "
                f"batch_size * sequence_length ({tokens_per_micro_batch})"
            )
        gradient_accumulation_steps = (
            configured_tokens_per_update // tokens_per_micro_batch
        )
    else:
        gradient_accumulation_steps = train_config.get(
            "gradient_accumulation_steps",
            1,
        )
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "train.gradient_accumulation_steps must be positive"
            )
    tokens_per_update = tokens_per_micro_batch * gradient_accumulation_steps

    eval_tokens = train_config.get("eval_tokens")
    if eval_tokens is None:
        eval_iters = train_config["eval_iters"]
    else:
        if eval_tokens <= 0:
            raise ValueError(f"train.eval_tokens must be positive, got {eval_tokens}")
        eval_iters = max(
            1,
            (eval_tokens + tokens_per_micro_batch - 1)
            // tokens_per_micro_batch,
        )

    precision = train_config.get("precision", "float32")
    amp_dtype = resolve_amp_dtype(precision, device)
    require_flash = train_config.get("require_flash_attention", False)
    if require_flash:
        verify_flash_attention(device, amp_dtype)

    print(f"using device: {device}")
    print(
        f"sequence_length: {sequence_length}, "
        f"micro_batch_size: {batch_size}, "
        f"gradient_accumulation_steps: {gradient_accumulation_steps}, "
        f"tokens_per_update: {tokens_per_update}"
    )
    print(
        f"precision: "
        f"{'bfloat16 autocast' if amp_dtype == torch.bfloat16 else 'float32'}, "
        f"eval_iters: {eval_iters}"
    )

    out_dir = config.resolve_path("paths", "out_root") / time.strftime(
        "run_%Y%m%d_%H%M%S"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 在日志开头记录训练配置
    print_config(config)
    print(f"{'out_dir':20}: {out_dir}")
    run_config = json.loads(json.dumps(config.data))
    run_config["runtime"] = {
        "sequence_length": sequence_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "tokens_per_update": tokens_per_update,
        "eval_iters": eval_iters,
        "actual_eval_tokens": eval_iters * tokens_per_micro_batch,
        "device": str(device),
        "effective_precision": (
            "bfloat16" if amp_dtype == torch.bfloat16 else "float32"
        ),
        "seed": seed,
    }
    (out_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resume_path = config.optional_path("paths", "resume")
    metrics_path = os.path.join(out_dir, "metrics.csv")
    write_header = not (resume_path and os.path.exists(metrics_path))
    mode = "w" if write_header else "a"

    with open(metrics_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["step", "tokens_seen", "train_loss", "val_loss", "lr"]
            )

    tokenizer = Tokenizer(str(config.resolve_path("paths", "tokenizer_vocab")))

    # 使用np.memmap以高效内存的方式加载数据
    train_data = np.memmap(
        config.resolve_path("paths", "train_data"),
        dtype=np.uint16,
        mode="r",
    )
    val_data = np.memmap(
        config.resolve_path("paths", "val_data"),
        dtype=np.uint16,
        mode="r",
    )

    # model, optimizer  优化器手写换官方了
    model = Model(**model_config).to(device)
    if train_config.get("activation_checkpointing", False):
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_config["max_lr"],
        weight_decay=optimizer_config["weight_decay"],
    )  # 这个初始化的 lr 只是个占位。后面都会被 cosine 的强行覆盖.

    # 检查点恢复
    start_iter = 0
    train_position = 0
    tokens_seen = 0
    if resume_path:
        start_iter, train_position, loaded_tokens_seen = load_checkpoint(
            resume_path,
            model,
            optimizer,
        )
        if loaded_tokens_seen is None:
            print(
                "Warning: legacy checkpoint has no tokens_seen; "
                "token counting restarts from zero"
            )
        else:
            tokens_seen = loaded_tokens_seen
        print(
            f"Resuming from iteration {start_iter}, "
            f"train_position {train_position}, tokens_seen {tokens_seen}"
        )

    # initialize wandb
    if logging_config["use_wandb"]:
        if wandb is None:
            raise RuntimeError(
                "logging.use_wandb=true，但当前环境没有安装 wandb"
            )
        wandb.init(project=logging_config["wandb_project"], config=config.data)

    # ==============================
    # 训练循环
    last_log_step = start_iter
    last_log_tokens = tokens_seen
    log_interval_start = None

    for it in range(start_iter, train_config["max_iters"]):
        if log_interval_start is None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            log_interval_start = time.perf_counter()

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

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)

        for _ in range(gradient_accumulation_steps):
            x, y, train_position = get_batch(
                train_data,
                batch_size,
                sequence_length,
                device,
                train_position,
            )

            with attention_kernel_context(device, require_flash):
                with autocast_context(device, amp_dtype):
                    logits, _ = model(x, use_cache=False)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )
                (loss / gradient_accumulation_steps).backward()

            accumulated_loss += loss.detach().float()
            tokens_seen += tokens_per_micro_batch

        step_loss = accumulated_loss / gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), optimizer_config["max_norm"]
        )
        optimizer.step()
        # 到这里才算真正完成一次更新
        completed_steps = it + 1
        last_step = completed_steps == train_config["max_iters"]

        # 每隔一定步数（日志间隔）打印训练进度
        if completed_steps % train_config["log_interval"] == 0 or last_step:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_time = time.perf_counter() - log_interval_start
            steps_since_log = completed_steps - last_log_step
            ms_per_step = training_time * 1000 / steps_since_log
            tokens_per_second = (
                (tokens_seen - last_log_tokens) / training_time
            )

            print(
                f"step {completed_steps}: "
                f"loss {step_loss.item():.4f}, "
                f"time {ms_per_step:.2f}ms/step, "
                f"tokens/s {tokens_per_second:.0f}, "
                f"grad_norm {grad_norm.item():.4f}"
            )

            last_log_step = completed_steps
            last_log_tokens = tokens_seen
            log_interval_start = None

        # 每隔一定步数（评估间隔）执行评估并记录日志
        if completed_steps % train_config["eval_interval"] == 0 or last_step:
            train_loss = estimate_loss(
                model,
                train_data,
                batch_size,
                sequence_length,
                device,
                eval_iters,
                amp_dtype=amp_dtype,
                require_flash_attention=require_flash,
            )
            val_loss = estimate_loss(
                model,
                val_data,
                batch_size,
                sequence_length,
                device,
                eval_iters,
                amp_dtype=amp_dtype,
                require_flash_attention=require_flash,
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
                        "tokens_seen": tokens_seen,
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
                        tokens_seen,
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
            with torch.inference_mode(), autocast_context(device, amp_dtype):
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
                tokens_seen=tokens_seen,
                model_args=model_config,
                config=run_config,
            )


if __name__ == "__main__":
    main()
