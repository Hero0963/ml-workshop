# src/lm_course/rl.py
"""Post-training with rewards and preferences (lesson 12).

- ``policy_gradient_loss``: the PPO-clip / GRPO objective on per-token log-probabilities.
  With a single update per batch of fresh samples the ratio is 1 and it reduces to REINFORCE
  with a baseline, which is what nanochat's ``chat_rl`` does.
- ``group_advantages``: GRPO's baseline = the mean reward of the other samples of the same
  prompt (optionally divided by their standard deviation).
- ``dpo_loss``: Direct Preference Optimization.
- ``pass_at_k``: the unbiased estimator from the HumanEval paper.
"""

import math

import torch
import torch.nn.functional as F

from lm_course.model import GPT


def completion_logprobs(
    model: GPT,
    prompt: list[int],
    completions: list[list[int]],
    pad_id: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """log p(token | everything before it) for every completion token, and a mask.

    Returns (logprobs, mask), both (num_completions, max_completion_length); the mask is 0 on
    padding. All completions share one prompt, so they are processed as one batch.
    ``temperature`` must match the one the completions were sampled with: the policy being
    trained is the tempered distribution softmax(logits / temperature).
    """
    device = next(model.parameters()).device
    length = max(len(c) for c in completions)
    batch = torch.full(
        (len(completions), len(prompt) + length), pad_id, dtype=torch.long
    )
    mask = torch.zeros(len(completions), length)
    for i, completion in enumerate(completions):
        batch[i, : len(prompt)] = torch.tensor(prompt)
        batch[i, len(prompt) : len(prompt) + len(completion)] = torch.tensor(completion)
        mask[i, : len(completion)] = 1
    batch, mask = batch.to(device), mask.to(device)
    # keep only the positions whose next token is a completion token
    logits = model(batch[:, :-1])[:, len(prompt) - 1 :]
    logprobs = F.log_softmax(logits.float() / temperature, dim=-1)
    targets = batch[:, len(prompt) :]
    return logprobs.gather(-1, targets[..., None]).squeeze(-1) * mask, mask


def group_advantages(
    rewards: torch.Tensor, normalize_std: bool = False
) -> torch.Tensor:
    """A_i = r_i - mean(r) over the samples of one prompt; GRPO also divides by std(r)."""
    advantages = rewards - rewards.mean()
    if normalize_std:
        advantages = advantages / (rewards.std() + 1e-6)
    return advantages


def policy_gradient_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    old_logprobs: torch.Tensor | None = None,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """-mean over completion tokens of min(ratio A, clip(ratio, 1 - eps, 1 + eps) A).

    ratio = pi(token) / pi_old(token). ``old_logprobs=None`` means "the samples come from the
    current policy" (ratio = 1 with gradient d log pi): plain REINFORCE with a baseline.
    Tokens are averaged over the whole batch (DAPO-style), not per sequence first.
    """
    old = logprobs.detach() if old_logprobs is None else old_logprobs
    ratio = torch.exp(logprobs - old)
    adv = advantages[:, None]
    per_token = torch.minimum(
        ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    )
    return -(per_token * mask).sum() / mask.sum().clamp(min=1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """-log sigmoid(beta [(log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))]).

    Inputs are sequence log-probabilities (sums over the completion tokens), shape (B,).
    """
    chosen = policy_chosen - reference_chosen
    rejected = policy_rejected - reference_rejected
    return -F.logsigmoid(beta * (chosen - rejected)).mean()


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k samples is correct), estimated from n samples with c correct:
    1 - C(n - c, k) / C(n, k) (Chen et al. 2021). ``c`` may be a float count (a sum of 0/1
    rewards)."""
    n, c = int(n), int(round(c))
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
