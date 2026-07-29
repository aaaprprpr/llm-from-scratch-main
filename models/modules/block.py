import torch
from torch import nn

from .attention import CausalSelfAttention_RoPE
from .cache import StaticKVCache
from .feed_forward import SwiGLU


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-6)  # 可学习权重的归一化
        self.attn = CausalSelfAttention_RoPE(
            d_model, n_head
        )  # 带旋转位置编码的自注意力，旋转注意力是相对位置，所以每次都要加。最简单那个位置编码是绝对位置，加一次就够了
        self.ffn_norm = nn.RMSNorm(
            d_model, eps=1e-6
        )  # 前置归一化设计，啥操作之前都带一个
        self.ffn = SwiGLU(d_model, d_ff)  # 激活

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        kv_cache: StaticKVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.attn_norm(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_index=layer_index,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x
