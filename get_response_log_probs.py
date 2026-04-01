import torch
from compute_entropy import compute_entropy
from transformers import PreTrainedModel

def get_response_log_probs(
    model:PreTrainedModel,
    input_ids: torch.Tensor,#形状（batch_size，sequence_length）分词后的提示和输出x
    labels: torch.Tensor,#形状（batch_size，sequence_length）分词后的标签,也就是y
    return_token_entropy: bool = False,#是否返回下一个token的预测（即字典中的token_entropy）
) -> dict[str,torch.Tensor]:
    """
    返回一个字典：
    “log_probs” 形状（batch_size，sequence_length）：
    表示在给定提示词（prompt）的条件下，模型生成回复（response）的条件对数概率（conditional log-probs）
    （mask操作会在train loop操作中完成）

    “token_entropy” 形状（batch_size，sequence_length）：
                （仅当 return_token_entropy = True 时存在）
    表示预测下一个标记（next token）时的熵值
    """

    logits = model(input_ids).logits#形状为 (batch_size, sequence_length, vocab_size)
    log_prob_f = torch.nn.functional.log_softmax(logits, dim = -1)
    labels = labels.unsqueeze(-1)
    log_probs = torch.gather(log_prob_f, dim=-1,index=labels)
    log_probs = log_probs.squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits=logits)
        res = {
            "log_probs":log_probs,
            "token_entropy":token_entropy
        }
    else:
        res = {
            "log_probs":log_probs        
            }
    return res