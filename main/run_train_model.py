import os,time,argparse,torch,csv,wandb
import numpy as np
import torch.nn.functional as F
from tokenizer_optimized import Tokenizer
# -> 如果你不想使用我的BPE分词器，可以改用tiktoken
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2") 
# 注意1：确保分词器词汇表大小与模型的词汇表大小一致（GPT2为50257）- 请在`run.sh`文件中设置
# 注意2：如果使用tiktoken，你需要相应地修改`generate`函数
from train_model import (lr_cosine_schedule, gradient_clipping, get_batch, save_checkpoint, load_checkpoint)
from model import Transformer as Model


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on user-provided data")
    
    # 数据路径与输出路径
    parser.add_argument('--train_data', type=str, required=True, help='Path to train.bin (np.memmap)')
    parser.add_argument('--val_data', type=str, required=True, help='Path to val.bin (np.memmap)')
    parser.add_argument('--tokenizer_vocab', type=str, required=True, help='Path to tokenizer vocab file (json)')
    parser.add_argument('--tokenizer_merges', type=str, required=True, help='Path to tokenizer merges file (txt)')
    parser.add_argument('--out_dir', type=str, default='out', help='Directory to save checkpoints')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_iters', type=int, default=5000, help='Total number of training iterations')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluate the model every eval_interval steps')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iters in ONE evaluation run')
    parser.add_argument('--log_interval', type=int, default=10, help='Every log_interval steps, log the training loss')

    # 模型超参数
    parser.add_argument('--vocab_size', type=int, required=True, help='Size of models vocabulary, must align with tokenizer vocab size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length for the model')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--theta', type=float, default=10000, help='Theta parameter for RoPE')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=512, help='Dimensionality of the model wrt embd space')
    parser.add_argument('--d_ff', type=int, default=1344, help='Dimensionality of the feedforward layer')
    
    # 优化器超参数
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # 学习率调度器参数
    parser.add_argument('--max_lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--lr_decay_iters', type=int, default=5000)
    
    # 日志记录
    #parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    return parser.parse_args()

def init_tokenizer(vocab_file):
    global tokenizer
    tokenizer = Tokenizer(vocab_file)

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """评估模型在训练集或验证集上的平均 Loss"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device) # (B, T)
        logits = model(X) # logits size (B, T, V)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) # 等同于 (B*T, V) 以及 (B*T, )
        losses[k] = loss.item()
    model.train()
    return losses.mean()

@torch.no_grad()
def generate(model, tokenizer, context, max_new_tokens, temperature=1.0, top_p=0.9, eos_id=None, context_length=256, device=None):
    """
    参数说明：
        model, tokenizer: 已训练的模型与分词器
        context: 输入上下文文本
        max_new_tokens: 最大生成长度
        temperature: 温度缩放参数（1.0 = 无效果，<1.0 = 更确定性，>1.0 = 更随机）
        top_p: 核采样阈值（取值范围 0.0 ~ 1.0）
        eos_id: 序列结束标记的ID
        context_length: 模型支持的最大上下文长度
    """
    model.eval()
    
    # 编码上下文
    idx = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    generated_tokens = []

    for _ in range(max_new_tokens):
        # 如果当前序列长度超过模型的context_length，从开头截断
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] # (B, V)

        # 温度缩放
        logits = logits / max(temperature, 1e-5)

        # Top-p（核）过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 找到累积概率超过top_p的索引（掩码）
            # 将掩码右移一位，保留刚好超过top_p的那个标记
            # 强制保留概率最高的第一个标记，避免因第一个标记概率超过p而移除所有标记
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 将需要移除的标记的逻辑值设为负无穷
            for b in range(logits.size(0)):
                indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                logits[b, indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1). Random sampling based on probability distribution, not greedy search for max. B == 1
        generated_tokens.append(idx_next.item())
        
        # idx 代表完整的序列
        idx = torch.cat((idx, idx_next), dim=1)
        
        if eos_id is not None and (idx_next.item() == eos_id):
            break

    return tokenizer.decode(idx[0].tolist()), tokenizer.decode(generated_tokens)
    # (full_sentence, generated_new_tokens)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 在日志开头记录训练配置
    print("="*20 + " Training Configurations " + "="*20)
    for arg in vars(args):
        print(f"{arg:20}: {getattr(args, arg)}")
    print("="*65)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "train_loss", "val_loss", "lr"])

   
    init_tokenizer(args.tokenizer_vocab)

    # 使用np.memmap以高效内存的方式加载数据
    train_data = np.memmap(args.train_data, dtype=np.uint32, mode='r') 
    val_data = np.memmap(args.val_data, dtype=np.uint32, mode='r')

    # model, optimizer  优化器手写换官方了
    model = Model(d_model=args.d_model, n_head=args.n_head, d_ff=args.d_ff, theta=args.theta, vocab_size=args.vocab_size, context_length=args.context_length, num_layers=args.n_layers, ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay) # 这个初始化的 lr 只是个占位。后面都会被 cosine 的强行覆盖.

    # 检查点恢复
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from iteration {start_iter}")

    # initialize wandb
    if args.use_wandb:
        wandb.init(project="training-260114-orig", config=args)

    # 训练循环
    X, Y = get_batch(train_data, args.batch_size, args.context_length, device) # initial batch
    t0 = time.time()

    for it in range(start_iter, args.max_iters):
        
        # 更新学习率（余弦调度）
        lr = lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.lr_decay_iters)
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr

        # 每隔一定步数（评估间隔）执行评估并记录日志
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
            
        
        # 每隔一定步数（检查点间隔）从模型生成文本并保存结果
        if (it % (args.eval_interval * 10) == 0 and it > 0) or last_step:
            # generate from model
            context, temperature, top_p = "你好，我是", 0.7, 0.95
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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) 
        optimizer.zero_grad(set_to_none=True)
        loss.backward() 
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm) 
        optimizer.step() 
        # --------------------------------------------

        # 获取下一个 batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)

        # 每隔一定步数（日志间隔）打印训练进度
        if it % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {it}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, grad_norm {grad_norm:.4f}")

    # 最终保存 - 无需执行。因为循环中的最后一步已完成该操作
    # save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "final_model.pt"))

if __name__ == "__main__":
    main()

