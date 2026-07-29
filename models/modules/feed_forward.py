import torch
from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    # 实现 SwiGLU 前馈网络，由 SiLU 激活函数与 GLU 门控单元组合构成。
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(
            d_model, d_ff, device=device, dtype=dtype, bias=False
        )  # d_model -> d_ff  gate preact生成“门”
        self.w3 = nn.Linear(
            d_model, d_ff, device=device, dtype=dtype, bias=False
        )  # d_model -> d_ff  value生成“内容”
        self.w2 = nn.Linear(
            d_ff, d_model, device=device, dtype=dtype, bias=False
        )  # d_ff -> d_model projection back把过滤后的内容还原回去

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        x1 = self.w1(x)
        x3 = self.w3(x)
        x = (
            F.silu(x1) * x3
        )  # gated silu. Hardamard product.门控SiLU激活，逐元素相乘（哈达玛积）
        x = self.w2(x)
        return x
