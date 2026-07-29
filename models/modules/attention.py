import torch
from torch import nn
import torch.nn.functional as F

from .cache import StaticKVCache
from .rope import RoPE


class CausalSelfAttention_RoPE(nn.Module):
    """
    Causal multi-head self-attention.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()


        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by " f"n_head ({n_head})"
            )

        head_dim = d_model // n_head

        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim  # 每个头的尺寸

        # qkv投影，对所有头，但是写成一次矩阵乘法
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        # output projection
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        kv_cache: StaticKVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq_len, d_model

        past_length = kv_cache.get_seq_length() if kv_cache is not None else 0

        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)  # each is (batch, seq_len, d_model)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (batch, n_head, seq_len, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (batch, n_head, seq_len, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (batch, n_head, seq_len, head_size)

        # 旋转qk# (B,H,T,hd)
        cos, sin = position_embeddings
        while cos.ndim < q.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        q = RoPE._apply_rope(q, cos, sin)
        k = RoPE._apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_index, k, v)

        key_length = past_length + T

        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    f"attention_mask must have shape (B, S), "
                    f"got {attention_mask.shape}"
                )

            if attention_mask.shape != (B, key_length):
                raise ValueError(
                    f"Expected attention_mask shape {(B, key_length)}, "
                    f"got {tuple(attention_mask.shape)}"
                )

            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)

            query_positions = past_length + torch.arange(T, device=x.device)

            key_positions = torch.arange(key_length, device=x.device)

            # (T, key_length)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

            # (B, 1, 1, key_length)
            padding_mask = attention_mask[:, None, None, :]

            # (B, 1, T, key_length)
            combined_mask = causal_mask[None, None, :, :] & padding_mask

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=combined_mask, is_causal=False
            )

        elif past_length == 0:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)

        elif T == 1:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        else:
            query_positions = past_length + torch.arange(T, device=x.device)
            key_positions = torch.arange(key_length, device=x.device)

            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, is_causal=False
            )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T, C)
        )  # 多头拼接(batch, seq_len, d_model) <- `concatenation` operation
        y = self.out_proj(attn_output)  # (batch, seq_len, d_model)

        return y
