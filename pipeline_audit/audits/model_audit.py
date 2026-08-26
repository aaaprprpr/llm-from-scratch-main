from __future__ import annotations

import torch
import torch.nn.functional as F

from pipeline_audit.common import AuditSection
from models.model import Transformer


def _tiny_model() -> Transformer:
    torch.manual_seed(1234)
    return Transformer(
        d_model=32,
        n_head=4,
        d_ff=64,
        theta=10000.0,
        vocab_size=97,
        context_length=64,
        num_layers=2,
        tie_word_embeddings=True,
    ).eval()


def _cache_parity(model: Transformer, tokens: torch.Tensor) -> tuple[float, int]:
    cache = None
    maximum_difference = 0.0
    top1_matches = 0
    with torch.inference_mode():
        for index in range(tokens.size(1)):
            current = tokens[:, index : index + 1]
            cached_logits, cache = model(
                current,
                past_key_values=cache,
                use_cache=True,
            )
            full_logits, _ = model(tokens[:, : index + 1], use_cache=False)
            cached_last = cached_logits[:, -1]
            full_last = full_logits[:, -1]
            maximum_difference = max(
                maximum_difference,
                (cached_last - full_last).abs().max().item(),
            )
            top1_matches += int(
                cached_last.argmax(dim=-1).item()
                == full_last.argmax(dim=-1).item()
            )
    return maximum_difference, top1_matches


def run(_config: dict, _report_dir) -> AuditSection:
    section = AuditSection("model")
    model = _tiny_model()
    tied = model.embedding.weight is model.lm_head.weight
    section.add(
        "tiny_model_weight_tying",
        "pass" if tied else "fail",
        "输入 embedding 与 lm_head 引用同一个 Parameter。",
    )

    torch.manual_seed(5678)
    tokens = torch.randint(0, 97, (2, 20))
    with torch.inference_mode():
        baseline, _ = model(tokens, use_cache=False)

        future_changed = tokens.clone()
        future_changed[:, 11:] = torch.randint(0, 97, future_changed[:, 11:].shape)
        future_logits, _ = model(future_changed, use_cache=False)
        causal_difference = (
            baseline[:, :11] - future_logits[:, :11]
        ).abs().max().item()

        other_row_changed = tokens.clone()
        other_row_changed[1] = torch.randint(0, 97, other_row_changed[1].shape)
        row_logits, _ = model(other_row_changed, use_cache=False)
        row_difference = (baseline[0] - row_logits[0]).abs().max().item()

    causal_ok = causal_difference <= 1e-6
    section.add(
        "causal_no_future_leak",
        "pass" if causal_ok else "fail",
        "修改未来 token 不会改变过去位置 logits。",
        max_absolute_difference=causal_difference,
    )
    row_ok = row_difference <= 1e-6
    section.add(
        "batch_row_isolation",
        "pass" if row_ok else "fail",
        "修改另一条 batch 样本不会影响当前样本。",
        max_absolute_difference=row_difference,
    )

    cache_difference, cache_matches = _cache_parity(model, tokens[:1, :16])
    cache_ok = cache_matches == 16 and cache_difference <= 1e-4
    section.add(
        "kv_cache_full_forward_parity",
        "pass" if cache_ok else "fail",
        "逐 token KV-cache 与完整上下文重算一致。",
        steps=16,
        top1_matches=cache_matches,
        max_absolute_difference=cache_difference,
    )

    train_model = _tiny_model().train()
    x = tokens[:1, :-1]
    y = tokens[:1, 1:]
    logits, _ = train_model(x, use_cache=False)
    loss = F.cross_entropy(logits.reshape(-1, 97), y.reshape(-1))
    loss.backward()
    missing_gradients = [
        name
        for name, parameter in train_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    finite = bool(torch.isfinite(loss)) and all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in train_model.parameters()
    )
    gradient_ok = finite and not missing_gradients
    section.add(
        "forward_backward_finite",
        "pass" if gradient_ok else "fail",
        "next-token loss 能反向传播到全部可训练参数。",
        loss=float(loss.detach()),
        missing_gradients=missing_gradients,
    )

    oracle_targets = torch.tensor([2, 5, 1, 7, 4], dtype=torch.long)
    oracle_logits = torch.full((5, 11), -20.0)
    oracle_logits.scatter_(1, oracle_targets[:, None], 20.0)
    correct_loss = F.cross_entropy(oracle_logits, oracle_targets).item()
    wrong_loss = F.cross_entropy(
        oracle_logits,
        oracle_targets.roll(1),
    ).item()
    alignment_ok = correct_loss < 1e-5 and wrong_loss > 10.0
    section.add(
        "loss_label_alignment_sentinel",
        "pass" if alignment_ok else "fail",
        "正确 next-token 标签与偏移标签在交叉熵中可被明确区分。",
        correct_loss=correct_loss,
        wrong_shift_loss=wrong_loss,
    )
    return section
