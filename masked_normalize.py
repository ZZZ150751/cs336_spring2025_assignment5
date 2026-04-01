import torch

def masked_normalize(
    tensor: torch.Tensor,#需要进行掩码归一化的张量
    mask: torch.Tensor,#和张量相同形状的掩码
    normalize_constant: float,#除的归一化常数
    dim: int | None = None,#求和维度（如果为None则对所有求和）
    ) -> torch.Tensor:
    mask_tensor = tensor * mask
    sum_mask_tensor = torch.sum(mask_tensor,dim=dim)
    sum_mask_tensor = sum_mask_tensor / normalize_constant
    return sum_mask_tensor