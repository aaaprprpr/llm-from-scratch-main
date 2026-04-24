import torch
import numpy as np
import math
import os
from typing import Optional, Callable, Iterable, BinaryIO, IO

def cross_entropy(o_i, y_i):
    """
    o_i: predicted logits, shape (batch_like, vocab_size)
    y_i: targets, shape (batch_like,)
    Requirements:
    - Subtract the largest element for numerical stability.
    - Cancel out log and exp whenever possible. 
    - Handle any additional batch dimensions and return the average across the batch. we assume batch-like dimensions always come first, before the vocabulary size dimension.
    """
    o_i = o_i - o_i.max(dim=-1, keepdim=True).values # 算完之后，不要把这个维度删掉，留着当一个长度为 1 的维度。避免错误广播。
    # 如果一个 reduction 的结果还要参与广播运算 → 用 keepdim=True

    o_y = o_i.gather(dim=-1, index=y_i.unsqueeze(-1)).squeeze(-1) # gather: 从 o_i 的最后一维中，以 y_i 为索引，取出对应的元素。
    # 等价写法： o_y = o_i[torch.arange(o_i.shape[0]), y_i]
    logsumexp = torch.log(torch.exp(o_i).sum(dim=-1))
    
    ce = -o_y + logsumexp    # (batch_like,)
    ce = ce.mean()           # scalar
    return ce

class AdamW(torch.optim.AdamW):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    @torch.no_grad() # 根据梯度挪动参数一步，本身这一步不需要求导。
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): # 闭包通常需要重新计算梯度
                loss = closure()
        
        for group in self.param_groups:
            alpha = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lambda_ = group['weight_decay']
            
            for theta in group['params']:
                if theta.grad is None:
                    continue

                grad = theta.grad.data
                state = self.state[theta]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(theta.data) # torch.zeros_like(p.data) creates a new tensor filled with zeros that has the same shape, dtype and device as p.data.
                    state['v'] = torch.zeros_like(theta.data)

                m, v = state['m'], state['v']
                state['step'] += 1 # begins with 1
                t = state['step']

                # Update moments
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1) # mul_ 带下划线是 in-place 乘法，相比起 m = m * beta1 显著节省内存 (不产生新的张量)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute adjusted alpha for iteration t
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = alpha * math.sqrt(bias_correction2) / bias_correction1

                # update parameters
                denom = v.sqrt().add_(eps)
                theta.addcdiv_(m, denom, value=-alpha_t)

                # apply weight decay 这是 AdamW 的核心。
                if lambda_ != 0:
                    theta.mul_(1 - alpha * lambda_)
                    # theta = theta - alpha * lambda_ * theta   <- # 等价写法，但会多占用内存

        return loss

def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    Params:
        t: current step
        alpha_max: maximum learning rate
        alpha_min: minimum (final) learning rate
        T_w: the number of warm-up iterations
        T_c: the number of cosine annealing iterations

    returns: 
        learning rate alpha_t at step t
    """
    if t < T_w:
        alpha_t = alpha_max * t / T_w
    elif t >= T_w and t <= T_c:
        temp = math.pi * (t-T_w) / (T_c-T_w)
        alpha_t = alpha_min + 1/2 * (1 + math.cos(temp)) * (alpha_max - alpha_min)
        # 草。这么算的意义是什么？看下论文去
    elif t > T_c:
        alpha_t = alpha_min

    return alpha_t
    
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float=1e-6):
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    if len(grads) == 0:
        return
    
    # 将所有梯度拼接成一个长向量 g
    g = torch.cat(grads)

    g_norm = torch.linalg.norm(g)

    if g_norm >= max_norm:
        clip_coef = max_norm / (g_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef) # remember this mul_ (原地缩放)
    
    return g_norm # 通常返回 norm 以便监控

def get_batch(data, batch_size, context_length, device):
    """
    Params:
        data: the input sequence, a tensor of shape (context_length,)
        batch_size: the batch size
        context_length: the context length
        device: the device to put the data on e.g. 'cpu' or 'cuda:0' or 'mps'

    Returns:
        x: a tensor of shape (batch_size, context_length)
        y: a tensor of shape (batch_size, context_length)
    """

    # 随机产生 batch_size 个起始位置
    ix = torch.randint(0, len(data) - context_length, (batch_size,))

    # 根据起始位置提取 x，以及 x 的下一个位置 (若 data 是 np.memmap 则只有被切到的那部分数据才会被从磁盘加载到内存中)
    x_list = [data[i : i + context_length].astype(np.int64) for i in ix]
    y_list = [data[i + 1 : i + context_length + 1].astype(np.int64) for i in ix]

    # 为保障 memmap，最后再转换为 Tensor 并堆叠 
    x = torch.from_numpy(np.stack(x_list)).to(device) # or torch.tensor(data)
    y = torch.from_numpy(np.stack(y_list)).to(device)

    return x.to(device), y.to(device)

"""
在目前的 get_batch 中，虽然逻辑正确，但由于它是纯随机的，可能会导致在一个 Epoch 里有些数据被重复读，有些没读到。
进阶技巧：随机打乱索引 (Shuffled Indices). 可以先生成一组所有可能的索引 range(0, len(data) - context_length)，将其 Shuffle，然后按顺序取这些打乱后的索引。
这样既保证了随机性，又保证了在一个周期内遍历了所有数据。
"""

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, 
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    """dump all the state from the first three parameters into the file-like object out.
    """

    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(obj, out)
    
def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']



