import torch
import numpy as np
import math
import os
from typing import Optional, Callable, Iterable, BinaryIO, IO

# def cross_entropy(o_i, y_i):
#     """
#     o_i: 模型输出的原始预测值 logits，形状为 (批次维度, 词表大小)
#     y_i: 真实目标标签，形状为 (批次维度,)
#     要求:
#     - 减去最大值以保证数值计算稳定性（防止指数爆炸）
#     - 尽可能化简抵消 log 和 exp 运算
#     - 兼容任意批次维度，最终返回整个批次的平均损失
#       约定：批次维度永远在前面，词表维度在最后一维
#     """
#     # 避免计算爆炸，全变成负值，这样e之后绝对在0-1之间。LSE技巧
#     o_i = o_i - o_i.max(dim=-1, keepdim=True).values # 算完之后，不要把这个维度删掉，留着当一个长度为 1 的维度。避免错误广播。
#     # 如果一个 reduction 的结果还要参与广播运算 → 用 keepdim=True

#     # 取出标注的token对应位置的预测值，yi是正确token的索引
#     o_y = o_i.gather(dim=-1, index=y_i.unsqueeze(-1)).squeeze(-1) # gather: 从 o_i 的最后一维中，以 y_i 为索引，取出对应的元素。
#     # 等价写法： o_y = o_i[torch.arange(o_i.shape[0]), y_i]
#     logsumexp = torch.log(torch.exp(o_i).sum(dim=-1))
    
#     ce = -o_y + logsumexp    # (batch_like,)
#     ce = ce.mean()           # scalar批次的平均损失
#     return ce

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
            alpha = group['lr'] # 学习率
            beta1, beta2 = group['betas']# Adam 的两个动量衰减系数
            eps = group['eps']# Adam 的数值稳定项
            lambda_ = group['weight_decay']# AdamW 的权重衰减系数（与学习率一起衰减参数，而不是像 Adam 那样直接在梯度上加一个正则项）
            
            for theta in group['params']:# theta 是一个参数张量，遍历所有参数
                if theta.grad is None:
                    continue

                grad = theta.grad.data#梯度
                state = self.state[theta]#每个参数对应一个状态字典，存储动量等信息。

                # 第一次访问时是空的，所以需要初始化。
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(theta.data)
                    state['v'] = torch.zeros_like(theta.data)

                m, v = state['m'], state['v']
                state['step'] += 1 # 从 1 开始计数
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
    参数：
        t: 当前步数
        alpha_max: 最大学习率
        alpha_min: 最小（最终）学习率
        T_w: 预热迭代次数
        T_c: 余弦退火迭代次数

    返回：
        第 t 步的学习率 alpha_t
    """
    if t < T_w:# 预热阶段，线性增加
        alpha_t = alpha_max * t / T_w
    elif t >= T_w and t <= T_c:
        temp = math.pi * (t-T_w) / (T_c-T_w)
        alpha_t = alpha_min + 1/2 * (1 + math.cos(temp)) * (alpha_max - alpha_min)
        # 草。这么算的意义是什么？看下论文去
    elif t > T_c:
        alpha_t = alpha_min

    return alpha_t
    
# def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float=1e-6):
#     grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
#     if len(grads) == 0:
#         return
    
#     # 将所有梯度拼接成一个长向量 g
#     g = torch.cat(grads)

#     g_norm = torch.linalg.norm(g)

#     if g_norm >= max_norm:
#         clip_coef = max_norm / (g_norm + eps)
#         for p in params:
#             if p.grad is not None:
#                 p.grad.detach().mul_(clip_coef) # remember this mul_ (原地缩放)
    
#     return g_norm # 通常返回 norm 以便监控

def get_batch(data, batch_size, context_length, device):
    """
    参数：
        data: 输入序列，形状为 (context_length,) 的张量
        batch_size: 批次大小
        context_length: 上下文长度
        device: 数据存放的设备，例如 'cpu'、'cuda:0' 或 'mps'

    返回：
        x: 形状为 (batch_size, context_length) 的张量
        y: 形状为 (batch_size, context_length) 的张量
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
    """将前三个参数的所有状态转储到类文件对象 out 中。
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
