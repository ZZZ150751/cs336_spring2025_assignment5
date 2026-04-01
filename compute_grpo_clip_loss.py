import torch
def compute_grpo_clip_loss(
    advantages,
    policy_log_probs,
    old_log_probs,
    cliprange = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 GRPO-Clip 损失
    Args:
        advantages: 张量形状 (batch_size, 1):优势 
        policy_log_probs: 张量形状 (batch_size, sequence_length): 正在训练的策略在每个token的对数概率
        old_log_probs: 张量形状(batch_size, sequence_length): 旧策略下每个token的对数概率
        cliprange: 梯度裁剪系数，一般设置为0.2

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            张量形状(batch_size, sequence_length): GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    #左边
    ratio_n_o = torch.exp(policy_log_probs - old_log_probs)
    loss_1 = ratio_n_o * advantages

    #右边
    ratio_clip = torch.clamp(ratio_n_o, 1.0 - cliprange, 1.0 + cliprange)
    loss_2 = ratio_clip * advantages

    #损失
    grpo_clip = torch.min(loss_1, loss_2)
    loss = -grpo_clip

    #记录
    is_clipped = (loss_1 < loss_2).to(torch.float32)
    metadata = {
        "is_clipped": is_clipped
    }
    return loss,metadata