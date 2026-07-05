import os, time, argparse, torch, csv, wandb
import numpy as np
import torch.nn.functional as F
from tokenizer_optimized import Tokenizer

# -> 如果你不想使用我的BPE分词器，可以改用tiktoken
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# 注意1：确保分词器词汇表大小与模型的词汇表大小一致（GPT2为50257）- 请在`run.sh`文件中设置
# 注意2：如果使用tiktoken，你需要相应地修改`generate`函数
from train_model import lr_cosine_schedule, get_batch, save_checkpoint, load_checkpoint
from model import Transformer as Model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on user-provided data")

    # 数据路径与输出路径
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to train.bin (np.memmap)"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to val.bin (np.memmap)"
    )
    parser.add_argument(
        "--tokenizer_vocab",
        type=str,
        required=True,
        help="Path to tokenizer vocab file (json)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="out", help="Directory to save checkpoints"
    )

    # 训练超参数
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=5000,
        help="Total number of training iterations",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Evaluate the model every eval_interval steps",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of iters in ONE evaluation run",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Every log_interval steps, log the training loss",
    )

    # 模型超参数
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Size of models vocabulary, must align with tokenizer vocab size",
    )
    parser.add_argument(
        "--context_length", type=int, default=256, help="Context length for the model"
    )
    parser.add_argument(
        "--n_head", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--theta", type=float, default=10000, help="Theta parameter for RoPE"
    )
    parser.add_argument(
        "--n_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Dimensionality of the model wrt embd space",
    )
    parser.add_argument(
        "--d_ff", type=int, default=1344, help="Dimensionality of the feedforward layer"
    )

    # 优化器超参数
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Gradient clipping norm"
    )

    # 学习率调度器参数
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)

    # 日志记录
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights and Biases for logging"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    return parser.parse_args()


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 在日志开头记录训练配置
    print("=" * 20 + " Training Configurations " + "=" * 20)
    for arg in vars(args):
        print(f"{arg:20}: {getattr(args, arg)}")
    print("=" * 65)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    write_header = not (args.resume and os.path.exists(metrics_path))
    mode = "w" if write_header else "a"

    with open(metrics_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "train_loss", "val_loss", "lr"])

    tokenizer = Tokenizer(args.tokenizer_vocab)

    # 使用np.memmap以高效内存的方式加载数据
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    # model, optimizer  优化器手写换官方了
    model = Model(
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        theta=args.theta,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.n_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
    )  # 这个初始化的 lr 只是个占位。后面都会被 cosine 的强行覆盖.

    # 检查点恢复
    start_iter = 0
    train_position = 0
    if args.resume:
        start_iter, train_position = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from iteration {start_iter}, train_position {train_position}")

    # initialize wandb
    if args.use_wandb:
        wandb.init(project="training-260114-orig", config=args)

    # ==============================
    # 训练循环
    last_log_time = time.time()
    last_log_step = start_iter

    for it in range(start_iter, args.max_iters):

        # 更新学习率（余弦调度）
        lr = lr_cosine_schedule(
            it, args.max_lr, args.min_lr, args.warmup_iters, args.lr_decay_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 当前 iteration 的 batch 完整地在循环内部产生和消费
        x, y, train_position = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
            train_position,
        )

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(x, use_cache=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        # 到这里才算真正完成一次更新
        completed_steps = it + 1
        last_step = completed_steps == args.max_iters

        # 每隔一定步数（日志间隔）打印训练进度
        if completed_steps % args.log_interval == 0 or last_step:
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
        if completed_steps % args.eval_interval == 0 or last_step:
            train_loss = estimate_loss(
                model,
                train_data,
                args.batch_size,
                args.context_length,
                device,
                args.eval_iters,
            )
            val_loss = estimate_loss(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                device,
                args.eval_iters,
            )
            print(
                f"Step {completed_steps}: "
                f"train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}, "
                f"lr {lr:.2e}"
            )

            if args.use_wandb:
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
        checkpoint_interval = args.eval_interval * 10
        if completed_steps % checkpoint_interval == 0 or last_step:

            context, temperature, top_p = "你好，我是", 0.7, 0.95
            idx = tokenizer.idx(context, device=device)
            with torch.inference_mode():
                full_sentence = model.generate(
                    idx,
                    max_new_tokens=100,
                    temperature=temperature,
                    top_p=top_p,
                    eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
                    context_length=args.context_length,
                )
            full_sentence = tokenizer.text(full_sentence, device=device)
            print(
                f"[Generated at iter {completed_steps}, temperature {temperature}, top_p {top_p}]: {full_sentence}"
            )

            ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{completed_steps}.pt")
            save_checkpoint(
                model,
                optimizer,
                completed_steps,
                ckpt_path,
                train_position=train_position,
            )


if __name__ == "__main__":
    main()
