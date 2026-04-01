import torch
def masked_mean(tensor: torch.Tensor,
                 mask: torch.Tensor, 
                 dim: int | None = None) -> torch.Tensor:
    """
    计算掩码均值
    Args:
        tensor:需要进行掩码归一化的张量
        mask:和张量相同形状的掩码
        dim: 求和维度（如果为None则对所有求和）
    Returns:
        掩码均值之后的张量
    """
    mask_tensor = tensor * mask
    sum_mask_tensor = torch.sum(mask_tensor,dim=dim)
    sum_mask = torch.sum(mask,dim=dim)
    mean_mask = sum_mask_tensor / sum_mask
    return mean_mask
    raise NotImplementedError