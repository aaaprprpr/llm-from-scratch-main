import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config
from tokenizer import Tokenizer
from pretrain.train_model import (
    attention_kernel_context,
    autocast_context,
    estimate_loss,
    get_device,
    lr_cosine_schedule,
    load_token_bin,
    load_checkpoint,
    resolve_amp_dtype,
    resolve_training_parameters,
    save_checkpoint,
    TokenBatchLoader,
    verify_flash_attention,
)
from pretrain.training_tracker import print_config, TrainingTracker
from models.model import Transformer as Model

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

    (
        sequence_length,
        batch_size,
        tokens_per_micro_batch,
        gradient_accumulation_steps,
        tokens_per_update,
        eval_iters,
    ) = resolve_training_parameters(model_config, train_config)

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
    out_dir.mkdir(parents=True, exist_ok=True)

    # 在日志开头记录训练配置
    print_config(config.data)
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

    tokenizer = Tokenizer(str(config.resolve_path("paths", "tokenizer_vocab")))

    train_data = load_token_bin(config.resolve_path("paths", "train_data"))
    val_data = load_token_bin(config.resolve_path("paths", "val_data"))

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
    train_batches = TokenBatchLoader(
        train_data,
        batch_size,
        sequence_length,
        device,
        position=train_position,
    )
    tracker = TrainingTracker(
        run_dir=out_dir,
        device=device,
        start_step=start_iter,
        train_position=train_position,
        tokens_seen=tokens_seen,
        use_wandb=logging_config["use_wandb"],
        wandb_project=logging_config["wandb_project"],
        wandb_config=config.data,
    )

    # ==============================
    # 训练循环
    for it in range(start_iter, train_config["max_iters"]):
        tracker.start_training_step()

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
            x, y, train_position = train_batches.next()

            with attention_kernel_context(device, require_flash):
                with autocast_context(device, amp_dtype):
                    logits, _ = model(x, use_cache=False)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )
                (loss / gradient_accumulation_steps).backward()

            accumulated_loss += loss.detach().float()
            tracker.record_batch(train_position, tokens_per_micro_batch)

        step_loss = accumulated_loss / gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), optimizer_config["max_norm"]
        )
        optimizer.step()
        tracker.end_training_step()

        # 到这里才算真正完成一次更新
        completed_steps = it + 1
        last_step = completed_steps == train_config["max_iters"]

        # 每隔一定步数（日志间隔）打印训练进度
        if completed_steps % train_config["log_interval"] == 0 or last_step:
            tracker.log_training_step(completed_steps, step_loss, grad_norm)

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
            tracker.log_evaluation(completed_steps, train_loss, val_loss, lr)

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

            ckpt_path = out_dir / f"ckpt_step_{completed_steps}.pt"
            save_checkpoint(
                model,
                optimizer,
                completed_steps,
                ckpt_path,
                train_position=tracker.train_position,
                tokens_seen=tracker.tokens_seen,
                model_args=model_config,
                config=run_config,
            )


if __name__ == "__main__":
    main()
