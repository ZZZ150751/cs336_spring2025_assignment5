import torch
from masked_normalize import masked_normalize
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,#每个token的对数概率(batch_size, sequence_length)
    response_mask: torch.Tensor,#掩码：1——回复，0——提示或者填充
    gradient_accumulation_steps: int,#每个优化器步骤中包含的微批次数量（用于在这些梯度累积步骤中对梯度取平均）
    normalize_constant: float = 1.0,#用于除以求和结果的常数（默认为1.0）
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    输入：1.每个token的对数概率
    2.掩码：1——回复，0——提示或者填充
    3.每个优化器步骤中包含的微批次数量（用于在这些梯度累积步骤中对梯度取平均）
    4.用于除以求和结果的常数
    """
    #要注意在掩码归一化时，要在sequence_length维度进行
    unscaled_loss = masked_normalize(tensor= -policy_log_probs, mask= response_mask, normalize_constant=normalize_constant,dim=1)
    #再对batch_size取平均
    unscaled_loss = unscaled_loss.mean()
    loss = unscaled_loss / gradient_accumulation_steps
    loss.backward()
    #记录
    metadata = {
        "unscaled_loss": unscaled_loss.detach(),
    }
    return loss, metadata