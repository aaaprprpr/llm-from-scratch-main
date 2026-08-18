import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modules import Block, CausalSelfAttention_RoPE, RoPE, StaticKVCache, SwiGLU
from .utils import filter_top_p_logits

__all__ = [
    "RoPE",
    "SwiGLU",
    "CausalSelfAttention_RoPE",
    "Block",
    "StaticKVCache",
    "Transformer",
]


class Transformer(nn.Module):
    initializer_std = 0.02

    def __init__(
        self,
        d_model: int,  # 嵌入维度
        n_head: int,  # 多头注意力的头数
        d_ff: int,  # 前向网络维度
        theta: float,  # RoPE 的 theta
        vocab_size: int,  # 词表大小
        context_length: int,  # 最大长度
        num_layers: int,  # 块数
    ):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by " f"n_head ({n_head})"
            )

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer, got {num_layers}"
            )

        head_dim = d_model // n_head

        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")


        self.rope = RoPE(theta=theta, d_k=head_dim)
        self.rope._maybe_extend_cache(
            needed_len=context_length,
            device=self.rope.inv_freq.device,
        )
        self.layers = nn.ModuleList(
            [Block(d_model, n_head, d_ff) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-6)  # 层归一化
        self.context_length = context_length  # 最大长度
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 语言模型头
        self.gradient_checkpointing = False
        self.reset_parameters()

    def _init_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_std)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def reset_parameters(self) -> None:
        """统一初始化模型，并按残差分支数缩放输出投影。"""
        self.apply(self._init_module)

        # 每个 block 有 attention 和 FFN 两个残差分支。把写回
        # residual stream 的投影缩放 1/sqrt(2L)，避免残差方差随深度累积。
        residual_std = self.initializer_std / math.sqrt(2.0 * len(self.layers))
        for layer in self.layers:
            nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.w2.weight, mean=0.0, std=residual_std)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def create_static_kv_cache(
        self,
        batch_size: int,
        max_cache_length: int | None = None,
    ) -> StaticKVCache:
        if max_cache_length is None:
            max_cache_length = self.context_length
        if max_cache_length > self.context_length:
            raise ValueError(
                f"Static cache length {max_cache_length} exceeds "
                f"model context_length {self.context_length}"
            )
        return StaticKVCache(
            num_layers=len(self.layers),
            batch_size=batch_size,
            max_cache_length=max_cache_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: (
            torch.Tensor | None
        ) = None,  # 长文本对话的时候，历史对话有个位置编码，模型内部不知道
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache=False,
    ) -> tuple[
        torch.Tensor,
        StaticKVCache | None,
    ]:
        B, T = x.shape

        if T == 0:
            raise ValueError("Input sequence cannot be empty")

        if use_cache and past_key_values is None:
            past_key_values = self.create_static_kv_cache(batch_size=B)

        past_length = past_key_values.get_seq_length() if use_cache else 0
        total_length = past_length + T

        if attention_mask is not None:
            if attention_mask.shape != (B, total_length):
                raise ValueError(
                    f"Expected attention_mask shape {(B, total_length)}, "
                    f"got {tuple(attention_mask.shape)}"
                )

            attention_mask = attention_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

        if total_length > self.context_length:
            raise ValueError(
                f"Total sequence length {total_length} exceeds "
                f"context_length {self.context_length}"
            )

        if token_positions is None:
            token_positions = torch.arange(
                past_length,
                total_length,
                device=x.device,
                dtype=torch.long,
            )

            # 普通训练和生成路径完全不需要 .item()
            rope_cache_length = total_length
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)

            if token_positions.ndim not in (1, 2):
                raise ValueError(
                    "token_positions must have shape (T), (1, T), or (B, T), "
                    f"got {tuple(token_positions.shape)}"
                )

            if token_positions.shape[-1] != T:
                raise ValueError(
                    f"Expected token_positions length {T}, "
                    f"got {token_positions.shape[-1]}"
                )

            if token_positions.ndim == 2 and token_positions.shape[0] not in (1, B):
                raise ValueError(
                    f"Expected token_positions batch dimension 1 or {B}, "
                    f"got {token_positions.shape[0]}"
                )

            # 只有外部传入任意 positions 时同步一次
            min_position = int(token_positions.min().item())
            max_position = int(token_positions.max().item())
            if min_position < 0:
                raise ValueError(
                    f"token_positions must be non-negative, got {min_position}"
                )
            if max_position >= self.context_length:
                raise ValueError(
                    f"token position {max_position} exceeds model position range "
                    f"[0, {self.context_length})"
                )
            rope_cache_length = max_position + 1

        x = self.embedding(x)  # (batch, seq_len, d_model) 词嵌入
        position_embeddings = self.rope.get_cos_sin(
            token_positions=token_positions,
            needed_len=rope_cache_length,
            device=x.device,
        )
        rope_dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype
        )
        position_embeddings = tuple(
            table.to(dtype=rope_dtype) for table in position_embeddings
        )

        checkpointing = self.training and getattr(
            self, "gradient_checkpointing", False
        )
        if checkpointing and (past_key_values is not None or use_cache):
            raise ValueError(
                "gradient checkpointing is incompatible with past_key_values/use_cache"
            )

        for i, layer in enumerate(self.layers):
            if checkpointing:

                def checkpointed_layer(hidden_states, current_layer=layer):
                    return current_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                    )

                x = checkpoint(
                    checkpointed_layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    kv_cache=past_key_values if use_cache else None,
                    layer_index=i,
                )  # 注意力

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, past_key_values if use_cache else None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens,
        temperature=1.0,
        top_p=0.9,
        eos_id=None,
        context_length: int | None = None,
    ) -> torch.Tensor:
        """
        参数说明：
            idx: 代表完整的序列
            max_new_tokens: 最大生成长度
            temperature: 温度缩放参数（1.0 = 无效果，<1.0 = 更确定性，>1.0 = 更随机）
            top_p: 核采样阈值（取值范围 0.0 ~ 1.0）
            eos_id: 序列结束标记的ID
            context_length: 模型支持的最大上下文长度
        """

        if idx.ndim != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")

        if idx.size(0) == 0:
            raise ValueError("Batch size cannot be zero")

        if idx.size(1) == 0:
            raise ValueError("Prompt cannot be empty")

        if idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"idx must contain integer token ids, got {idx.dtype}")

        if not isinstance(max_new_tokens, int):
            raise TypeError(
                f"max_new_tokens must be int, got " f"{type(max_new_tokens).__name__}"
            )

        if max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be non-negative, " f"got {max_new_tokens}"
            )

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        if eos_id is not None:
            if not isinstance(eos_id, int):
                raise TypeError(
                    f"eos_id must be int or None, got " f"{type(eos_id).__name__}"
                )

            if not 0 <= eos_id < self.lm_head.out_features:
                raise ValueError(
                    f"eos_id {eos_id} is outside vocabulary range "
                    f"[0, {self.lm_head.out_features})"
                )

        if context_length is None:
            context_length = self.context_length

        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")

        if context_length > self.context_length:
            raise ValueError(
                f"Generation context_length {context_length} exceeds "
                f"model context_length {self.context_length}"
            )
        was_training = self.training
        self.eval()

        try:
            finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
            prompt_length = min(idx.size(1), context_length)
            max_cache_length = min(
                context_length,
                prompt_length + max(0, max_new_tokens - 1),
            )
            past_key_values = (
                self.create_static_kv_cache(
                    batch_size=idx.size(0),
                    max_cache_length=max_cache_length,
                )
                if max_new_tokens > 0
                else None
            )

            for _ in range(max_new_tokens):
                # 增量推理阶段（Decode）：既然已经有了历史缓存，我们只需要把上一步生成的【最后 1 个词】喂给模型
                cached_length = past_key_values.get_seq_length()
                if cached_length > 0:

                    if cached_length >= context_length:
                        break

                    idx_cond = idx[:, -1:]
                else:
                    # 首次计算阶段（Prefill）：喂入完整的 Prompt。若太长则从开头截断，确保不爆窗口
                    idx_cond = (
                        idx
                        if idx.size(1) <= context_length
                        else idx[:, -context_length:]
                    )

                logits, past_key_values = self(
                    idx_cond, past_key_values=past_key_values, use_cache=True
                )
                logits = logits[:, -1, :]  # (B, V)

                # 温度缩放
                logits = logits / max(temperature, 1e-5)

                # Top-p（核）过滤
                if top_p < 1.0:
                    filter_top_p_logits(logits, top_p)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(
                    probs, num_samples=1
                )  # (B, 1). Random sampling based on probability distribution, not greedy search for max. B == 1

                if eos_id is not None:
                    # 已结束的样本后续固定补 EOS，避免继续生成随机内容
                    idx_next = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(idx_next, eos_id),
                        idx_next,
                    )

                idx = torch.cat((idx, idx_next), dim=1)

                if eos_id is not None:
                    # 更新每个样本的结束状态
                    finished |= idx_next.squeeze(-1).eq(eos_id)

                    # 所有样本都结束后才退出整个生成循环
                    if finished.all():
                        break

            return idx  # (full_sentence, generated_new_tokens)
        finally:
            self.train(was_training)
