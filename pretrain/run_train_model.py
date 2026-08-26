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
    OptimizerBundle,
    resolve_amp_dtype,
    resolve_training_parameters,
    save_checkpoint,
    tokenizer_fingerprint,
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


def build_pretraining_optimizers(
    model: Model,
    optimizer_config: dict,
    lr_config: dict,
    use_fused_adamw: bool,
) -> tuple[OptimizerBundle, dict[str, int]]:
    """按参数角色构建全 AdamW 或 Muon+AdamW 优化器。"""
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    if optimizer_type not in {"adamw", "muon"}:
        raise ValueError(
            "optimizer.type must be 'adamw' or 'muon', "
            f"got {optimizer_type!r}"
        )

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    muon_parameters = []
    adamw_decay_parameters = []
    adamw_no_decay_parameters = []

    for name, parameter in trainable_parameters:
        # Muon 只处理 Transformer block 内的隐藏层矩阵。embedding 虽然
        # 也是二维参数，但同时承担 lm_head，必须留给 AdamW。
        use_muon = (
            optimizer_type == "muon"
            and name.startswith("layers.")
            and parameter.ndim == 2
        )
        if use_muon:
            muon_parameters.append(parameter)
        elif parameter.ndim >= 2:
            adamw_decay_parameters.append(parameter)
        else:
            adamw_no_decay_parameters.append(parameter)

    assigned_ids = {
        id(parameter)
        for parameter in (
            muon_parameters
            + adamw_decay_parameters
            + adamw_no_decay_parameters
        )
    }
    expected_ids = {id(parameter) for _, parameter in trainable_parameters}
    if assigned_ids != expected_ids:
        raise RuntimeError("Optimizer parameter partition is incomplete")

    if optimizer_type == "muon":
        if not muon_parameters:
            raise RuntimeError("No hidden 2D matrices were assigned to Muon")
        if any(parameter.ndim != 2 for parameter in muon_parameters):
            raise RuntimeError("Muon received a non-2D parameter")
        if any(
            parameter is model.embedding.weight
            for parameter in muon_parameters
        ):
            raise RuntimeError("The shared embedding/lm_head cannot use Muon")

    adamw = torch.optim.AdamW(
        [
            {
                "params": adamw_decay_parameters,
                "weight_decay": optimizer_config["weight_decay"],
            },
            {
                "params": adamw_no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=lr_config["max_lr"],
        betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
        eps=optimizer_config["eps"],
        weight_decay=0.0,
        fused=use_fused_adamw,
    )

    optimizers = {"adamw": adamw}
    if optimizer_type == "muon":
        muon_class = getattr(torch.optim, "Muon", None)
        if muon_class is None:
            raise RuntimeError(
                "optimizer.type='muon' requires torch.optim.Muon; "
                f"the current PyTorch is {torch.__version__}. "
                "Use PyTorch 2.9 or newer, or set optimizer.type='adamw'."
            )
        muon_config = optimizer_config.get("muon")
        if not isinstance(muon_config, dict):
            raise ValueError("optimizer.muon must be an object in Muon mode")
        muon = muon_class(
            muon_parameters,
            lr=muon_config["max_lr"],
            weight_decay=muon_config["weight_decay"],
            momentum=muon_config["momentum"],
            nesterov=muon_config["nesterov"],
            ns_steps=muon_config["ns_steps"],
            adjust_lr_fn=muon_config["adjust_lr_fn"],
        )
        # 两组参数互斥，更新先后不影响数值；Muon 放前面便于日志阅读。
        optimizers = {"muon": muon, "adamw": adamw}

    parameter_counts = {
        "muon": sum(parameter.numel() for parameter in muon_parameters),
        "adamw_decay": sum(
            parameter.numel() for parameter in adamw_decay_parameters
        ),
        "adamw_no_decay": sum(
            parameter.numel() for parameter in adamw_no_decay_parameters
        ),
    }
    return OptimizerBundle(optimizers), parameter_counts


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
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    if optimizer_type not in {"adamw", "muon"}:
        raise ValueError(
            "optimizer.type must be 'adamw' or 'muon', "
            f"got {optimizer_type!r}"
        )

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

    fused_optimizer_requested = optimizer_config.get("fused", False)
    if not isinstance(fused_optimizer_requested, bool):
        raise ValueError("optimizer.fused must be a boolean")
    use_fused_adamw = fused_optimizer_requested and device.type == "cuda"

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
        f"eval_iters: {eval_iters}, "
        f"optimizer: {optimizer_type}, "
        f"fused_adamw: {use_fused_adamw}"
    )

    out_dir = config.resolve_path("paths", "out_root") / time.strftime(
        "run_%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 在日志开头记录训练配置
    print_config(config.data)
    print(f"{'out_dir':20}: {out_dir}")
    resume_path = config.optional_path("paths", "resume")

    tokenizer_path = config.resolve_path("paths", "tokenizer_vocab")
    tokenizer = Tokenizer(str(tokenizer_path))
    tokenizer_size = len(tokenizer.tokenizer)
    if tokenizer_size != model_config["vocab_size"]:
        raise ValueError(
            f"Tokenizer size {tokenizer_size} does not match model.vocab_size "
            f"{model_config['vocab_size']}"
        )
    tokenizer_sha256 = tokenizer_fingerprint(tokenizer_path)

    train_data = load_token_bin(
        config.resolve_path("paths", "train_data"),
        expected_tokenizer_size=tokenizer_size,
        expected_tokenizer_sha256=tokenizer_sha256,
    )
    val_data = load_token_bin(
        config.resolve_path("paths", "val_data"),
        expected_tokenizer_size=tokenizer_size,
        expected_tokenizer_sha256=tokenizer_sha256,
    )

    # 固定形状的 micro-batch 无法只处理最后几个零头 token，因此向上取整；
    # 最后一次更新若越过数据末尾，TokenBatchLoader 会从开头接续。
    total_train_updates = (
        len(train_data) + tokens_per_update - 1
    ) // tokens_per_update
    planned_train_tokens = total_train_updates * tokens_per_update
    planned_dataset_passes = planned_train_tokens / len(train_data)
    warmup_iters = lr_config["warmup_iters"]
    if warmup_iters >= total_train_updates:
        raise ValueError(
            f"lr_schedule.warmup_iters ({warmup_iters}) must be smaller than "
            f"the computed total updates ({total_train_updates})"
        )

    print(
        f"train data: {len(train_data):,} tokens; "
        f"planned: {planned_train_tokens:,} tokens "
        f"({planned_dataset_passes:.6f} passes)"
    )
    print(
        f"computed training plan: {total_train_updates:,} updates; "
        f"cosine decay end: step {total_train_updates:,}; "
        f"warmup: {warmup_iters:,} updates"
    )
    print(
        f"validation data: {len(val_data):,} tokens; "
        f"sampled per evaluation: {eval_iters * tokens_per_micro_batch:,} tokens"
    )

    # model, optimizer  优化器手写换官方了
    model = Model(**model_config).to(device)
    embeddings_tied = model.embedding.weight is model.lm_head.weight
    if bool(model_config.get("tie_word_embeddings", False)) != embeddings_tied:
        raise RuntimeError(
            "Configured tie_word_embeddings does not match the constructed model"
        )
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"model parameters: {model_parameters:,}; "
        f"planned tokens/parameter: {planned_train_tokens / model_parameters:.2f}; "
        f"word embeddings tied: {embeddings_tied}"
    )

    optimizer, optimizer_parameter_counts = build_pretraining_optimizers(
        model,
        optimizer_config,
        lr_config,
        use_fused_adamw,
    )
    print(
        "optimizer parameters: "
        f"Muon {optimizer_parameter_counts['muon']:,}; "
        f"AdamW decay {optimizer_parameter_counts['adamw_decay']:,}; "
        f"AdamW no-decay {optimizer_parameter_counts['adamw_no_decay']:,}"
    )

    run_config = json.loads(json.dumps(config.data))
    run_config["runtime"] = {
        "sequence_length": sequence_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "tokens_per_update": tokens_per_update,
        "total_train_updates": total_train_updates,
        "lr_decay_end_step": total_train_updates,
        "eval_iters": eval_iters,
        "actual_eval_tokens": eval_iters * tokens_per_micro_batch,
        "train_dataset_tokens": len(train_data),
        "validation_dataset_tokens": len(val_data),
        "planned_train_tokens": planned_train_tokens,
        "planned_dataset_passes": planned_dataset_passes,
        "model_parameters": model_parameters,
        "planned_tokens_per_parameter": (
            planned_train_tokens / model_parameters
        ),
        "word_embeddings_tied": embeddings_tied,
        "optimizer_type": optimizer_type,
        "optimizer_parameter_counts": optimizer_parameter_counts,
        "torch_version": torch.__version__,
        "device": str(device),
        "effective_precision": (
            "bfloat16" if amp_dtype == torch.bfloat16 else "float32"
        ),
        "fused_adamw": use_fused_adamw,
        "seed": seed,
    }
    (out_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if train_config.get("activation_checkpointing", False):
        model.gradient_checkpointing_enable()

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
        wandb_config=run_config,
    )

    # ==============================
    # 训练循环
    for it in range(start_iter, total_train_updates):
        tracker.start_training_step()

        # 更新学习率（余弦调度）
        adamw_lr = lr_cosine_schedule(
            it,
            lr_config["max_lr"],
            lr_config["min_lr"],
            warmup_iters,
            total_train_updates,
        )
        for param_group in optimizer["adamw"].param_groups:
            param_group["lr"] = adamw_lr

        muon_lr = None
        if "muon" in optimizer:
            muon_config = optimizer_config["muon"]
            muon_lr = lr_cosine_schedule(
                it,
                muon_config["max_lr"],
                muon_config["min_lr"],
                warmup_iters,
                total_train_updates,
            )
            for param_group in optimizer["muon"].param_groups:
                param_group["lr"] = muon_lr

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
        last_step = completed_steps == total_train_updates

        # 每隔一定步数（日志间隔）打印训练进度
        if completed_steps % train_config["log_interval"] == 0 or last_step:
            tracker.log_training_step(
                completed_steps,
                step_loss,
                grad_norm,
                adamw_lr,
                muon_lr,
            )

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
            tracker.log_evaluation(
                completed_steps,
                train_loss,
                val_loss,
                adamw_lr,
                muon_lr,
            )

        # 采样不再和大体积检查点绑定：每次验证时生成，并单独记入 CSV。
        if completed_steps % train_config["eval_interval"] == 0 or last_step:
            context = sample_config["prompt"]
            temperature = sample_config["temperature"]
            top_p = sample_config["top_p"]
            idx = tokenizer.idx(context, device=device)
            prompt_token_count = idx.size(1)
            with torch.inference_mode(), autocast_context(device, amp_dtype):
                generated_ids = model.generate(
                    idx,
                    max_new_tokens=sample_config["max_new_tokens"],
                    temperature=temperature,
                    top_p=top_p,
                    eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
                    context_length=model_config["context_length"],
                )
            full_text = tokenizer.text(generated_ids, device=device)
            completion = tokenizer.text(
                generated_ids[:, prompt_token_count:],
                device=device,
            )
            print(
                f"[Generated at iter {completed_steps}, temperature "
                f"{temperature}, top_p {top_p}]: {full_text}"
            )
            tracker.log_sample(
                completed_steps,
                context,
                completion,
                full_text,
                temperature,
                top_p,
                sample_config["max_new_tokens"],
            )

        # 检查点仍按较稀疏的独立周期保存。
        checkpoint_interval = (
            train_config["eval_interval"]
            * logging_config["checkpoint_interval_multiplier"]
        )
        if completed_steps % checkpoint_interval == 0 or last_step:
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
