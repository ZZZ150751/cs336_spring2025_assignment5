import torch
from compute_policy_gradient_loss import compute_policy_gradient_loss
from masked_mean import masked_mean
def grpo_microbatch_train_step(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    loss_type,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    GRPO微批次训练步骤
    Args:
        policy_log_probs: 形状为 (batch_size, sequence_length)，当前正在训练的策略生成的逐 token 对数概率
        response_mask: 形状为 (batch_size, sequence_length)，回复掩码（遮盖提示词和填充部分）
        gradient_accumulation_steps: 每次优化器更新所包含的微批次数量
        loss_type: 必须是 "no_baseline"、"reinforce_with_baseline" 或 "grpo_clip" 之一
        raw_rewards:每一轮回复的奖励，当 loss_type == "no_baseline" 时需要提供；形状为 (batch_size, 1)
        advantages: 优势，如果 loss_type 为 "reinforce_with_baseline" 或 "grpo_clip"，则必填；形状为 (batch_size, 1)
        old_log_probs: 过去策略生成的逐 token 对数概率，如果使用 "grpo_clip"，则必填；形状为 (batch_size, sequence_length)
        cliprange: GRPO-Clip 所需的裁剪参数
    Returns:
        loss: 标量张量（scalar tensor）。经过梯度累积调整后的微批次损失。
        metadata: 字典类型，包含从底层 loss 函数调用中获得的元数据
    """
    #用损失包装器计算pre_token损失
    pre_token_loss,metadata = compute_policy_gradient_loss(policy_log_probs=policy_log_probs,loss_type=loss_type,raw_rewards=raw_rewards,
                                                  advantages=advantages,old_log_probs=old_log_probs,cliprange=cliprange)
    
    #使用掩码平均聚合为每个示例的标量损失(对文本序列掩码平均)
    scalar_loss = masked_mean(tensor=pre_token_loss, mask=response_mask, dim=1)

    #对batch_size维度求平均
    scalar_loss = scalar_loss.mean()

    #对梯度累计进行调整
    loss = scalar_loss / gradient_accumulation_steps

    loss.backward()

    metadata["scalar_loss"] = scalar_loss.detach()

    return loss, metadata

    
    raise NotImplementedError