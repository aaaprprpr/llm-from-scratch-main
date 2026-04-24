"""Main script to train a transformer based language model on user-provided data.
"""

import os
import time
import math
import argparse
import torch
import numpy as np
import csv
import wandb

from tokenizer_optimized import Tokenizer
# -> If you do not want to use my BPE tokenizer, you can use tiktoken instead
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2") 
# Note 1: Make sure the tokenizer vocab size matches the model's vocab size (50257 for GPT2) - set it in the `run.sh`
# Note 2: If you use tiktoken, you need to modify the `generate` function accordingly

from train_model import (
    cross_entropy, AdamW, lr_cosine_schedule, 
    gradient_clipping, get_batch, save_checkpoint, load_checkpoint
)

from model import Transformer as Model
from model import softmax

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # macOS M-series GPU
    device = "mps"
print(f"using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on user-provided data")
    
    # data and output paths
    parser.add_argument('--train_data', type=str, required=True, help='Path to train.bin (np.memmap)')
    parser.add_argument('--val_data', type=str, required=True, help='Path to val.bin (np.memmap)')
    parser.add_argument('--tokenizer_vocab', type=str, required=True, help='Path to tokenizer vocab file (json)')
    parser.add_argument('--tokenizer_merges', type=str, required=True, help='Path to tokenizer merges file (txt)')
    parser.add_argument('--out_dir', type=str, default='out', help='Directory to save checkpoints')
    
    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_iters', type=int, default=5000, help='Total number of training iterations')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluate the model every eval_interval steps')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iters in ONE evaluation run')
    parser.add_argument('--log_interval', type=int, default=10, help='Every log_interval steps, log the training loss')

    # model hyperparameters
    parser.add_argument('--vocab_size', type=int, required=True, help='Size of models vocabulary, must align with tokenizer vocab size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length for the model')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--theta', type=float, default=10000, help='Theta parameter for RoPE')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=512, help='Dimensionality of the model wrt embd space')
    parser.add_argument('--d_ff', type=int, default=1344, help='Dimensionality of the feedforward layer')
    
    # Optimizer hyperparameters
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # Learning rate schedule parameters
    parser.add_argument('--max_lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--lr_decay_iters', type=int, default=5000)
    
    # Logging
    #parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    return parser.parse_args()

def init_tokenizer(vocab_file, merge_file, special_tokens=["<|endoftext|>"]):
    global tokenizer
    tokenizer = Tokenizer.from_files(vocab_file, merge_file, special_tokens)

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """评估模型在训练集或验证集上的平均 Loss"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device) # (B, T)
        logits = model(X) # logits size (B, T, V)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) # 等同于 (B*T, V) 以及 (B*T, )
        losses[k] = loss.item()
    model.train()
    return losses.mean()

@torch.no_grad()
def generate(model, tokenizer, context, max_new_tokens, temperature=1.0, top_p=0.9, eos_id=None, context_length=256, device=None):
    """
    Args:
        model, tokenizer: trained model and tokenizer
        context: input context string
        max_new_tokens: maximum generation length
        temperature: temperature scaling parameter (1.0 = no effect, <1.0 = more certain, >1.0 = more random)
        top_p: nucleus sampling threshold (between 0.0 and 1.0)
        eos_id: ID of the end-of-sequence token
        context_length: maximum context length supported by the model
    """
    model.eval()
    
    # Encode context
    idx = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    generated_tokens = []

    for _ in range(max_new_tokens):
        # If the current sequence exceeds the model's context_length, truncate the beginning
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] # (B, V)

        # Temperature Scaling
        logits = logits / max(temperature, 1e-5)

        # Top-p (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
            
            # Find indices where cumulative probability exceeds top_p (mask)
            # Keep the token where cumulative probability just exceeds top_p by shifting the mask to the right.
            # Force keep the first token with the highest probability to avoid removing all tokens if the first token's probability exceeds p.
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set logits of tokens to remove to negative infinity
            for b in range(logits.size(0)):
                indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                logits[b, indices_to_remove] = -float('Inf')

        probs = softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1). Random sampling based on probability distribution, not greedy search for max. B == 1
        generated_tokens.append(idx_next.item())
        
        # idx is the full sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        if eos_id is not None and (idx_next.item() == eos_id):
            break

    return tokenizer.decode(idx[0].tolist()), tokenizer.decode(generated_tokens)
    # (full_sentence, generated_new_tokens)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Record training configurations at the begining of the log
    print("="*20 + " Training Configurations " + "="*20)
    for arg in vars(args):
        print(f"{arg:20}: {getattr(args, arg)}")
    print("="*65)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "train_loss", "val_loss", "lr"])

    if tokenizer is None:
        init_tokenizer(args.tokenizer_vocab, args.tokenizer_merges)

    # use np.memmap to load data in a memory-efficient way
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r') 
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    # model, optimizer
    model = Model(d_model=args.d_model, n_head=args.n_head, d_ff=args.d_ff, theta=args.theta, vocab_size=args.vocab_size, context_length=args.context_length, num_layers=args.n_layers, ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay) # 这个初始化的 lr 只是个占位。后面都会被 cosine 的强行覆盖.

    # checkpoint recovery
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from iteration {start_iter}")

    # initialize wandb
    if args.use_wandb:
        wandb.init(project="training-260114-orig", config=args)

    # TRAIN LOOP
    X, Y = get_batch(train_data, args.batch_size, args.context_length, device) # initial batch
    t0 = time.time()

    for it in range(start_iter, args.max_iters):
        
        # update learning rate (Cosine Schedule)
        lr = lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.lr_decay_iters)
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr

        # once in a while (eval interval) eval and log
        last_step = (it == args.max_iters - 1)
        if (it % args.eval_interval == 0) or last_step:
            train_loss = estimate_loss(model, train_data, args.batch_size, args.context_length, device, args.eval_iters)
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"Iter {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {lr:.2e}")
            
            if args.use_wandb:
                wandb.log({
                    "iter": it,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": lr,
                })
        
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([it, train_loss.item() if torch.is_tensor(train_loss) else train_loss, 
                         val_loss.item() if torch.is_tensor(val_loss) else val_loss, 
                         lr])
            
        
        # once in a while (checkpoint interval) genrate from model and save it
        if (it % (args.eval_interval * 10) == 0 and it > 0) or last_step:
            # generate from model
            context, temperature, top_p = "Hello, I'm a language model, ", 1.0, 0.9
            full_sentence, new_tokens = generate(
                model, 
                tokenizer=tokenizer,
                context=context, 
                max_new_tokens=100, 
                temperature=temperature, 
                top_p=top_p, 
                eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
                context_length=args.context_length,
                device=device
            )
            print(f"[Generated at iter {it}, temperature {temperature}, top_p {top_p}]: {full_sentence}")


            ckpt_path = os.path.join(args.out_dir, f"ckpt_iter_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
        
        # --------------------------------------------
        # Train for one step
        logits = model(X) 
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) 
        optimizer.zero_grad(set_to_none=True)
        loss.backward() 
        grad_norm = gradient_clipping(model.parameters(), args.max_norm) 
        optimizer.step() 
        # --------------------------------------------

        # 获取下一个 batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)

        # once in a while (log interval) 打印训练进度
        if it % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {it}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, grad_norm {grad_norm:.4f}")

    # final save - no need. Because the last step in the loop already does it
    # save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "final_model.pt"))

if __name__ == "__main__":
    main()




















