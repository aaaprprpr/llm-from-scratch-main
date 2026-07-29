import torch
from torch import nn


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
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires d_k to be even, got {d_k}")
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
                1.0 / (self.theta ** (2.0 * frequency_indices / self.d_k))
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
