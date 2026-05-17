import torch
from torch.utils.data import Dataset
import numpy as np
import math
import os
from typing import  BinaryIO,IO


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
    

class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length - 1

    def __getitem__(self, i):
        x = torch.from_numpy(self.data[i:i+self.context_length].astype(np.int64))
        y = torch.from_numpy(self.data[i+1:i+self.context_length+1].astype(np.int64))
        return x, y


_current_step_pos = 0

def get_batch(data, batch_size, context_length, device):
    global _current_step_pos
    
    # 1. 计算总共有多少个合法的起始位置
    total_samples = len(data) - context_length - 1
    
    # 2. 生成 batch_size 个索引
    # 不再是 randint，而是从当前位置开始往后排
    # 比如：[pos, pos + context_length, pos + 2*context_length, ...]
    # 这样能保证数据被地毯式扫过
    ix = []
    for _ in range(batch_size):
        if _current_step_pos > total_samples:
            _current_step_pos = 0 # 扫完了，回到开头
        ix.append(_current_step_pos)
        _current_step_pos += context_length # 步长等于上下文长度，无缝衔接

    # 3. 提取数据（保持你原来的 memmap 逻辑）
    x_list = [data[i : i + context_length].astype(np.int64) for i in ix]
    y_list = [data[i + 1 : i + context_length + 1].astype(np.int64) for i in ix]

    # 4. 堆叠并转 Tensor
    x = torch.from_numpy(np.stack(x_list)).to(device)
    y = torch.from_numpy(np.stack(y_list)).to(device)

    return x, y

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    将前三个参数的所有状态转储到类文件对象 out 中。
    """

    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(obj, out)
    
def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']
