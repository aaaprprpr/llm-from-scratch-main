import torch
import torch.nn.functional as F

from sft.tokenize_sft import IGNORE_INDEX


def sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_logprob: bool = False,
) -> torch.Tensor:
    # logits[:, t] 预测 labels[:, t + 1]。labels == -100 的位置不参与 logprob。
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_mask = shift_labels.ne(IGNORE_INDEX)

    safe_labels = shift_labels.masked_fill(~loss_mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    )
    token_logps = token_logps.squeeze(-1) * loss_mask
    sequence_logps = token_logps.sum(dim=-1)

    if average_logprob:
        token_counts = loss_mask.sum(dim=-1).clamp_min(1)
        sequence_logps = sequence_logps / token_counts

    return sequence_logps


def model_sequence_logprob(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    average_logprob: bool = False,
) -> torch.Tensor:
    logits, _ = model(input_ids, attention_mask=attention_mask, use_cache=False)
    return sequence_logprob(
        logits,
        labels,
        average_logprob=average_logprob,
    )


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # DPO 比较的是 policy 相对 reference 更偏好 chosen 还是 rejected。
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = policy_logratios - reference_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (
        policy_rejected_logps - reference_rejected_logps
    ).detach()

    metrics = {
        "loss": losses.mean().detach(),
        "reward_accuracy": chosen_rewards.gt(rejected_rewards).float().mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        "policy_chosen_logps": policy_chosen_logps.mean().detach(),
        "policy_rejected_logps": policy_rejected_logps.mean().detach(),
        "reference_chosen_logps": reference_chosen_logps.mean().detach(),
        "reference_rejected_logps": reference_rejected_logps.mean().detach(),
    }

    return losses.mean(), metrics
