"""训练 checkpoint 的统一读取入口。

背景：run_train_model.py 把 ``torch.__version__`` 直接写进了 checkpoint 的
``run_config["runtime"]["torch_version"]``。``torch.__version__`` 是
``torch.torch_version.TorchVersion`` 实例（str 的子类），而
``torch.load(weights_only=True)`` 的反序列化器只放行默认白名单里的类，
遇到它就会抛 ``UnpicklingError: Weights only load failed``。

``TorchVersion`` 是纯数据类，放行它是安全的；这里只加这一个白名单项，
``weights_only`` 的其余限制保持不变。不要图省事改成 ``weights_only=False``，
那等于关掉整个反序列化防护。

新写入的 checkpoint 不再包含该类型（save 端已改为存 ``str``），这里的白名单
只是为了还能读旧文件。
"""

from __future__ import annotations

from os import PathLike
from typing import Any, BinaryIO, IO

import torch

# 兼容旧版 PyTorch：没有该 API 时直接跳过，旧版本本来也不会有这个问题。
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


def load_checkpoint(
    src: str | PathLike | BinaryIO | IO[bytes],
    map_location: Any = "cpu",
    mmap: bool = False,
) -> Any:
    """按训练 checkpoint 的既定格式读取，保留 ``weights_only`` 防护。

    ``mmap=True`` 只对文件路径生效，适合一次性读取大 checkpoint 的场景，
    需要调用方保证 ``src`` 是路径。
    """
    return torch.load(
        src,
        map_location=map_location,
        weights_only=True,
        mmap=mmap,
    )
