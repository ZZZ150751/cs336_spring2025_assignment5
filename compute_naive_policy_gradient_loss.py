import torch
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """朴素策略梯度损失
    Args:
        raw_rewards_or_advantages: 张量形状(batch_size, 1)：每一轮回复的优势
            the raw rewards or advantages for each rollout response.
        policy_log_probs: 张量形状(batch_size, sequence_length): 每个 token 的对数概率

    Returns:
        张量形状：(batch_size, sequence_length): 每个 token 的策略梯度损失（在后续的训练循环中，该张量将沿 batch 和 sequence 维度进行聚合/求均值）
    """
    loss = -raw_rewards_or_advantages * policy_log_probs
    return loss