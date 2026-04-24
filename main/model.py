import torch
from torch import nn
import math
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = num_embeddings    # 词表大小 V
        self.d_model = embedding_dim        # 嵌入维度 C
        # 创建词嵌入矩阵，形状为 (V, C) 
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)# 截断正态分布，截断为 [-3, 3]，初始化
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 根据给定的ids，查找对应的嵌入向量
        return self.weight[token_ids] # pytorch提供的高级索引，逐个元素查表，返回一个张量

class RMSNorm(nn.Module):
    def __init__(self, 
            d_model: int, 
            eps: float = 1e-5, 
            device=None, 
            dtype=None
        ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps # 极小值ε
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        # gain 是一个可训练参数，初始化为全 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        # rms shape: (B, T, 1) 
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms # normalize
        x = x.to(in_dtype)
        return self.gain * x  # 把 self.gain shape (C,) 广播到 (1, 1, C) 再和 x 逐元素相乘

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype))  # row-major memory ordering (必须这么搞)

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

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
        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)  # (d_k/2,)

        # Lazy caches
        self.register_buffer("cos_cached", torch.empty(0, d_k // 2, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, d_k // 2, dtype=torch.float32, device=device), persistent=False)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
        """确保缓存至少覆盖 [0, needed_len) positions。
        """
        cur_len = int(self.cos_cached.size(0))
        if needed_len <= cur_len:
            return

        # 待扩展的缓存位置区间Range to extend cache [cur_len, needed_len)
        new_positions = torch.arange(cur_len, needed_len, dtype=torch.float32, device=device)  # (ΔL,)
        # angles: (ΔL, d_k/2)
        # inv_freq 可能位于不同设备（例如模型初始化时未指定设备），在此处统一设备
        inv_freq = self.inv_freq.to(device=device)
        # 用 einsum 做外积，生成角度矩阵
        angles = torch.einsum("i,j->ij", new_positions, inv_freq)  # float64 * float32 -> float64

        new_cos = angles.cos().to(dtype=torch.float32)  # (ΔL, d_k/2)
        new_sin = angles.sin().to(dtype=torch.float32)

        # 如果缓存所在设备不匹配（例如模型初始化在 CPU，当前输入在 GPU），将缓存迁移到正确设备
        if self.cos_cached.device != device:
            self.cos_cached = self.cos_cached.to(device=device)
            self.sin_cached = self.sin_cached.to(device=device)

        if cur_len == 0:
            self.cos_cached = new_cos
            self.sin_cached = new_sin
        else:
            self.cos_cached = torch.cat([self.cos_cached, new_cos], dim=0)
            self.sin_cached = torch.cat([self.sin_cached, new_sin], dim=0)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (seq_len,) or (..., seq_len)
        """
        assert x.size(-1) == self.d_k, f"Expected last dim d_k={self.d_k}, got {x.size(-1)}"
        seq_len = x.size(-2)

        # 将 token_positions 转为 long 类型，用于张量索引取值
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)

        max_pos = int(token_positions.max().item()) if token_positions.numel() > 0 else 0 #找要用的最大位置编号
        needed_len = max_pos + 1 # 算缓存至少需要多长
        self._maybe_extend_cache(needed_len=needed_len, device=x.device) # 看看要不要扩展

        # 从缓存中取出对应位置的 cos/sin 值，并重塑回与位置编码相同的批次形状
        cos = self.cos_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        sin = self.sin_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)

        # 合理性检查：序列长度维度对齐
        if cos.shape[-2] != seq_len:
            raise ValueError(f"token_positions seq dim {cos.shape[-2]} != x seq_len {seq_len}")

        # 数据类型对齐
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return self._apply_rope(x, cos, sin)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Params:
     - q: (..., seq_len_q, d_q)
     - k: (..., seq_len_k, d_k)
     - v: (..., seq_len_v, d_v)
     - mask: boolean mask, (seq_len, seq_len)
    
    Returns:
     - tensor with shape (..., seq_len_q, d_v)
    """
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)

    return attn_weights @ v


class CausalSelfAttention_RoPE(nn.Module):
    """
    Causal multi-head self-attention.
    """
    def __init__(self, d_model: int, n_head: int, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head # 每个头的尺寸

        # qkv投影，对所有头，但是写成一次矩阵乘法
        self.qkv_proj = Linear(self.d_model, 3 * self.d_model)
        # output projection
        self.out_proj = Linear(self.d_model, self.d_model)

        # RoPE：注意 d_k = head_dim
        self.rope = RoPE(theta=theta, d_k=self.head_dim)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        B, T, C = x.size()# batch, seq_len, d_model

        qkv = self.qkv_proj(x) # (batch, seq_len, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=-1) # each is (batch, seq_len, d_model)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)

        if token_positions is None:
            token_positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1,T)
        elif token_positions.ndim == 1:
            token_positions = token_positions.unsqueeze(0)  # (1,T)

        # 旋转qk
        q = self.rope(q, token_positions)  # (B,H,T,hd)
        k = self.rope(k, token_positions)

        # 构建下三角矩阵，只保留下三角部分，上三角不让看
        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        attn_output = scaled_dot_product_attention(q, k, v, mask=attn_mask) # (batch, n_head, seq_len, head_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # 多头拼接(batch, seq_len, d_model) <- `concatenation` operation
        y = self.out_proj(attn_output) # (batch, seq_len, d_model)
        
        return y

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    # 实现 SwiGLU 前馈网络，由 SiLU 激活函数与 GLU 门控单元组合构成。
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype) # d_model -> d_ff  gate preact生成“门”
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) # d_model -> d_ff  value生成“内容”
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype) # d_ff -> d_model projection back把过滤后的内容还原回去
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        x1 = self.w1(x)
        x3 = self.w3(x)
        x = silu(x1) * x3 # gated silu. Hardamard product.门控SiLU激活，逐元素相乘（哈达玛积）
        x = self.w2(x)
        return x

class RoPE_llama(nn.Module):
    """Rotary Position Embeddings (RoPE) for queries/keys. This is the Llama-style RoPE (precomputed cache version), which is more traditional.

    Inputs:
      x: (..., seq_len, d_k)
      token_positions: (..., seq_len) or (seq_len,) or (batch, seq_len)
    Outputs:
      Same shape as x: (..., seq_len, d_k)
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        - theta: float Θ value for the RoPE
        - d_k: int dimension of query and key vectors
        - max_seq_len: int Maximum sequence length that will be inputted
        - device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even."
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        self.device = device

        p = torch.arange(0, d_k // 2, dtype=torch.float64, device=device)
        inv_freq = 1.0 / (self.theta ** (2.0 * p / d_k))

        # Key fix: register_buffer before converting to float32, otherwise it cannot run on mps
        inv_freq = inv_freq.to(torch.float32)
        
        # positions: 0..max_seq_len-1
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (L,)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (L, d_k/2)

        # Precompute and cache; persistent=False means not saved in state_dict (not saved to checkpoint files - because it can be recomputed anytime)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)  # (L, d_k/2)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)  # (L, d_k/2)
    
    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        x:   (..., seq_len, d_k)
        cos: (..., seq_len, d_k/2)  (already aligned to batch/seq dims)
        sin: (..., seq_len, d_k/2)

        Treat (x_even, x_odd) as 2D vectors and rotate:
          [x_even'] = x_even*cos - x_odd*sin
          [x_odd' ] = x_even*sin + x_odd*cos
        """
        # Python slicing syntax: start:stop:step
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2). Ellipsis means all preceding dimensions are retained. Take all even positions. 
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k/2) Take all odd positions. 

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # Interleave back to (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """This function mainly performs the computation in _apply_rope, and itself is only responsible for handling the shape and type conversion of inputs and outputs.
        x: (..., seq_len, d_k)
        token_positions: shape can be
          - (seq_len,)
          - (..., seq_len)  (e.g., (batch, seq_len) or more batch dims)
        """
        assert x.size(-1) == self.d_k, f"Expected last dim d_k={self.d_k}, got {x.size(-1)}"
        seq_len = x.size(-2)

        # token_positions converted to long for indexing
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)
        
        # Check that the maximum position does not exceed the precomputed length (torch.numel() returns the total number of elements in an input tensor.)
        max_pos = int(token_positions.max().item()) if token_positions.numel() > 0 else 0 
        if max_pos >= self.max_seq_len:
            raise ValueError(
                f"token_positions has max={max_pos}, but max_seq_len={self.max_seq_len}. "
                "Please increase max_seq_len in RoPE init."
            )
        
        # Fetch cos/sin from cache
        cos = self.cos_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        sin = self.sin_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        
        # Align batch dims of cos/sin with x: target shape should be x.shape[:-1] with last dim d_k/2
        if cos.shape[-2] != seq_len:
            raise ValueError(f"token_positions seq dim {cos.shape[-2]} != x seq_len {seq_len}")
        
        # dtype alignment
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return self._apply_rope(x, cos, sin)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax over specified dimension.
    
    Params:
     - x: input tensor of any shape
     - dim: dimension to compute softmax over. Last dimension by default.
    
    Returns:
     - tensor of the same shape as input, containing softmax values for each element
    """
    x_normalized = x - x.max(dim=dim, keepdim=True).values 
    
    x_exp = x_normalized.exp()
    
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, theta: float = 10000.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)# 可学习权重的归一化
        self.attn = CausalSelfAttention_RoPE(d_model, n_head, theta) # 带旋转位置编码的自注意力，旋转注意力是相对位置，所以每次都要加。最简单那个位置编码是绝对位置，加一次就够了
        self.ffn_norm = RMSNorm(d_model)# 前置归一化设计，啥操作之前都带一个
        self.ffn = SwiGLU(d_model, d_ff)# 激活

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self, 
            d_model: int, # 嵌入维度
            n_head: int, # 多头注意力的头数
            d_ff: int, # 前向网络维度
            theta: float, # RoPE 的 theta
            vocab_size: int, # 词表大小
            context_length: int, # 最大长度
            num_layers: int # 块数
        ):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(d_model, n_head, d_ff, theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)# 层归一化
        self.context_length = context_length # 最大长度
        self.embedding = Embedding(vocab_size, d_model)# 词嵌入
        self.lm_head = Linear(d_model, vocab_size)# 语言模型头

    def forward(self, 
            x: torch.Tensor, 
            token_positions: torch.Tensor | None = None #长文本对话的时候，历史对话有个位置编码，模型内部不知道
        ) -> torch.Tensor:
        B, T = x.shape # (batch, seq_len)
        assert T <= self.context_length, f"无法前向传播长度为 {T}的序列, 上下文长度仅为 {self.context_length}"

        x = self.embedding(x) # (batch, seq_len, d_model) 词嵌入
        for layer in self.layers:
            x = layer(x, token_positions=token_positions) # 注意力
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits





