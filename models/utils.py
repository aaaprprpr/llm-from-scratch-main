import torch
import torch.nn.functional as F


def filter_top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """原地过滤核采样范围外的 logits。"""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(
        F.softmax(sorted_logits, dim=-1),
        dim=-1,
    )

    # 保留第一个累计概率超过 top_p 的 token。
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
        ..., :-1
    ].clone()
    sorted_indices_to_remove[..., 0] = 0

    for batch_index in range(logits.size(0)):
        indices_to_remove = sorted_indices[batch_index][
            sorted_indices_to_remove[batch_index]
        ]
        logits[batch_index, indices_to_remove] = -float("Inf")

    return logits
