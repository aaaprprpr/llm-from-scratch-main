import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RoPE(nn.Module):
    """
    这是一种 Lazy（懒加载）/ 自动扩展的 RoPE 实现。属于更主流的工程化解决方案。
    采用这种方式，Attention 模块 / Transformer Block 无需再传入最大序列长度（max_seq_len）参数。
    初始化阶段：无需指定最大序列长度
    前向传播阶段：会根据 token_positions 的最大值自动扩展 cos/sin 缓存
    """

    def __init__(self, theta: float, d_k: int, device=None):
        super().__init__()
        # 注意力头维度必须偶数，两两一对旋转
        assert d_k % 2 == 0, "RoPE requires d_k to be even."
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.device = device
        # 生成一组下标 p：0,1,2,...,d_k/2-1，标记位置
        p = torch.arange(0, d_k // 2, dtype=torch.float64, device=device)
        # 生成固定频率序列，乘上位置之后得到旋转位置向量
        inv_freq = 1.0 / (self.theta ** (2.0 * p / d_k))

        # 关键修复：需在转换为 float32 类型之前注册缓冲区，否则无法在 MPS 设备上正常运行。
        # 注册频率buffer：非训练参数，自动设备对齐，不保存到模型
        self.register_buffer(
            "inv_freq", inv_freq.to(torch.float32), persistent=False
        )  # (d_k/2,)

        # Lazy caches
        self.register_buffer(
            "cos_cached",
            torch.empty(0, d_k // 2, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.empty(0, d_k // 2, dtype=torch.float32, device=device),
            persistent=False,
        )

    @staticmethod
    def _apply_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        x:   (..., seq_len, d_k)
        cos: (..., seq_len, d_k/2)
        sin: (..., seq_len, d_k/2)
        拆积偶进行分块旋转，然后拼接回去
        """
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave back (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    @torch.no_grad()
    def _maybe_extend_cache(self, needed_len: int, device: torch.device):
        """确保缓存至少覆盖 [0, needed_len) positions。"""
        if needed_len < 0:
            raise ValueError(f"needed_len must be non-negative, got {needed_len}")

        cur_len = int(self.cos_cached.size(0))
        cache_ready = (
            needed_len <= cur_len
            and self.cos_cached.device == device
            and self.sin_cached.device == device
            and self.cos_cached.dtype == torch.float32
            and self.sin_cached.dtype == torch.float32
        )
        if cache_ready:
            return

        # 训练时通常一次覆盖完整窗口；增量生成若超过已有容量，则按倍数扩容，
        # 避免每生成一个 token 都重新拼接整份缓存。
        if needed_len > cur_len:
            target_len = max(needed_len, max(16, cur_len * 2))
        else:
            # 这里只是在修复 device/dtype，不应无故扩大已有容量。
            target_len = cur_len
        positions = torch.arange(
            target_len,
            dtype=torch.float32,
            device=device,
        )
        if self.inv_freq.dtype != torch.float32:
            frequency_indices = torch.arange(
                0,
                self.d_k // 2,
                dtype=torch.float64,
                device=device,
            )
            self.inv_freq = (
                1.0
                / (self.theta ** (2.0 * frequency_indices / self.d_k))
            ).to(torch.float32)
        elif self.inv_freq.device != device:
            self.inv_freq = self.inv_freq.to(device=device)
        inv_freq = self.inv_freq
        angles = torch.outer(positions, inv_freq)

        # RoPE 基表固定使用 FP32；AMP 只在应用到 q/k 前转换一次。
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()

    def get_cos_sin(
        self,
        token_positions: torch.Tensor,
        needed_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """一次取出当前 batch 所需的位置表，供所有 Transformer 层共享。"""
        token_positions = token_positions.to(device=device, dtype=torch.long)
        self._maybe_extend_cache(needed_len=needed_len, device=device)

        flat_positions = token_positions.reshape(-1)
        output_shape = (*token_positions.shape, -1)
        cos = self.cos_cached.index_select(0, flat_positions).reshape(output_shape)
        sin = self.sin_cached.index_select(0, flat_positions).reshape(output_shape)
        return cos, sin

    @classmethod
    def apply_rotary(
        cls,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """把预先取出的共享位置表应用到 q 或 k。"""
        cos, sin = position_embeddings
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        return cls._apply_rope(x, cos, sin)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None,
        needed_len: int | None = None,
    ) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (seq_len,) or (..., seq_len)
        """
        assert (
            x.size(-1) == self.d_k
        ), f"Expected last dim d_k={self.d_k}, got {x.size(-1)}"
        seq_len = x.size(-2)

        if token_positions is None:
            token_positions = torch.arange(
                seq_len,
                device=x.device,
                dtype=torch.long,
            )

            if needed_len is None:
                needed_len = seq_len
        else:
            token_positions = token_positions.to(
                device=x.device,
                dtype=torch.long,
            )

        if needed_len is None:
            needed_len = (
                int(token_positions.max().item()) + 1
                if token_positions.numel() > 0
                else 0
            )

        position_embeddings = self.get_cos_sin(
            token_positions=token_positions,
            needed_len=needed_len,
            device=x.device,
        )

        # 合理性检查：序列长度维度对齐
        if position_embeddings[0].shape[-2] != seq_len:
            raise ValueError(
                f"token_positions seq dim {position_embeddings[0].shape[-2]} "
                f"!= x seq_len {seq_len}"
            )

        return self.apply_rotary(x, position_embeddings)


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


class CausalSelfAttention_RoPE(nn.Module):
    """
    Causal multi-head self-attention.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if n_head <= 0:
            raise ValueError(f"n_head must be positive, got {n_head}")

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
        past_key_value=None,
        use_cache=False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.size()  # batch, seq_len, d_model

        past_length = 0
        past_k = None
        past_v = None
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.size(-2)

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
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
        while cos.ndim < q.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        q = RoPE._apply_rope(q, cos, sin)
        k = RoPE._apply_rope(k, cos, sin)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_key_value = (k, v) if use_cache else None

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

        return y, present_key_value


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
        past_key_value=None,
        use_cache=False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present_key_value = self.attn(
            self.attn_norm(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present_key_value


class Transformer(nn.Module):
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

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if n_head <= 0:
            raise ValueError(f"n_head must be positive, got {n_head}")

        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by " f"n_head ({n_head})"
            )

        head_dim = d_model // n_head

        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")

        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")

        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

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

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        token_positions: (
            torch.Tensor | None
        ) = None,  # 长文本对话的时候，历史对话有个位置编码，模型内部不知道
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache=False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        B, T = x.shape

        if T == 0:
            raise ValueError("Input sequence cannot be empty")

        past_length = 0

        if past_key_values is not None:
            if len(past_key_values) != len(self.layers):
                raise ValueError(
                    f"Expected {len(self.layers)} cache layers, "
                    f"got {len(past_key_values)}"
                )

            past_length = past_key_values[0][0].size(-2)

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

        checkpointing = self.training and getattr(
            self, "gradient_checkpointing", False
        )
        if checkpointing and (past_key_values is not None or use_cache):
            raise ValueError(
                "gradient checkpointing is incompatible with past_key_values/use_cache"
            )

        next_decoder_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if checkpointing:

                def checkpointed_layer(hidden_states, current_layer=layer):
                    return current_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_value=None,
                        use_cache=False,
                    )[0]

                x = checkpoint(
                    checkpointed_layer,
                    x,
                    use_reentrant=False,
                )
                present_key_value = None
            else:
                x, present_key_value = layer(
                    x,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )  # 注意力
            if use_cache:
                next_decoder_cache.append(present_key_value)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, next_decoder_cache

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
            past_key_values = None
            finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

            for _ in range(max_new_tokens):
                # 增量推理阶段（Decode）：既然已经有了历史缓存，我们只需要把上一步生成的【最后 1 个词】喂给模型
                if past_key_values is not None:
                    cached_length = past_key_values[0][0].size(-2)

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
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # 找到累积概率超过top_p的索引（掩码）
                    # 将掩码右移一位，保留刚好超过top_p的那个标记
                    # 强制保留概率最高的第一个标记，避免因第一个标记概率超过p而移除所有标记
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # 将需要移除的标记的逻辑值设为负无穷
                    for b in range(logits.size(0)):
                        indices_to_remove = sorted_indices[b][
                            sorted_indices_to_remove[b]
                        ]
                        logits[b, indices_to_remove] = -float("Inf")

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
