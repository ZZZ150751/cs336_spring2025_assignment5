import torch
from compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss
from compute_grpo_clip_loss import compute_grpo_clip_loss
def compute_policy_gradient_loss(
    policy_log_probs,
    loss_type,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    策略梯度的损失包装器（三种）
    Args:
    policy_log_probs:形状为 (batch_size, sequence_length),当前正在训练的策略生成的逐 token 对数概率
    loss_type: 字符串字面量，"no_baseline"、"reinforce_with_baseline" 或 "grpo_clip" 之一
    raw_rewards: 每一轮回复的奖励，如果 loss_type == "no_baseline"，则必填；形状为 (batch_size, 1)
    advantages: 优势，如果 loss_type 为 "reinforce_with_baseline" 或 "grpo_clip"，则必填；形状为 (batch_size, 1)
    old_log_probs: 过去策略生成的逐 token 对数概率，如果使用 "grpo_clip"，则必填；形状为 (batch_size, sequence_length)
    
    Return:
    loss: 形状为 (batch_size, sequence_length) 的逐 token 损失
    metadata: 字典类型，包含底层例程返回的统计信息（记录）
    """
    if loss_type == 'no_baseline':
        loss = compute_naive_policy_gradient_loss(raw_rewards_or_advantages=raw_rewards,
                                                  policy_log_probs=policy_log_probs)
        return loss,{}
    elif loss_type == 'reinforce_with_baseline':
        loss = compute_naive_policy_gradient_loss(raw_rewards_or_advantages=advantages,
                                                  policy_log_probs=policy_log_probs)
        return loss,{}
    elif loss_type == 'grpo_clip':
        loss,metadata = compute_grpo_clip_loss(advantages=advantages,policy_log_probs=policy_log_probs,
                                      old_log_probs=old_log_probs,cliprange=cliprange)
        return loss,metadata
    else:
        print("unexpected loss_type")