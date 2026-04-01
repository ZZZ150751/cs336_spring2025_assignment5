import torch

def compute_entropy(logits#张量形状：（batch_size，sequence_length，vocab_size）
                        ) -> torch.Tensor:#返回张量形状（batch_size，sequence_length）
    #获取log p(x) 
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    #获取p(x)
    p = torch.softmax(logits, dim=-1)

    #计算熵
    H_p = -log_p * p
    H_p = torch.sum(H_p, dim=-1)
    return H_p