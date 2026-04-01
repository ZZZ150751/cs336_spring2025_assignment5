import torch
import torch.nn.functional as F

#文本拼接：Alpaca 模板来格式化字符串，并确保在回答末尾添加序列结束标记
def format_alpaca(prompt: str, response: str, tokenizer):
    #提示词
    prompt_file_path = "cs336_alignment/prompts/alpaca_sft.prompt"
    with open(prompt_file_path, "r", encoding="utf-8") as f1:
        prompt_template = f1.read().strip()

    text_token_seq = prompt_template.format(instruction=prompt, response=response)
    
    tokens = tokenizer.encode(text_token_seq, add_special_tokens=True)
    if tokenizer.eos_token_id is not None:
        tokens.append(tokenizer.eos_token_id)
    
    return torch.tensor(tokens)

#计算对数概率和
def get_log_prob(model, input_ids):
    #增加 batch 维度并移动到模型设备
    input_ids = input_ids.unsqueeze(0).to(model.device)
    logits = model(input_ids).logits #获取 logits
    
    #计算所有位置的 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    #错位对齐：位置 i 的 logit 预测位置 i+1 的 token
    shift_log_probs = log_probs[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    #提取实际 token 的对数概率并求和
    return torch.gather(shift_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1).sum()


def compute_per_instance_dpo_loss(
    lm,      # 正在优化的语言模型 (πθ)
    lm_ref,  # 参考语言模型 (πref)
    tokenizer,  # 两个模型共用的分词器
    beta: float,      # DPO 的 beta 超参数 (控制 KL 散度惩罚)
    prompt: str,          # 针对该样本的提示词串 (x)
    response_chosen: str,     # 偏好的回答字符串 (yw)
    response_rejected: str,       # 拒绝的回答字符串 (yl)
) -> torch.Tensor:
    """
    给定两个语言模型（`lm` 和参考模型 `lm_ref`）、它们的分词器、DPO 的 beta 超参数、
    一段提示词以及针对该提示词的一对回答，计算该样本的 DPO 损失值。

    参数说明：
    lm: torch.nn.Module
        正在接受训练的语言模型。
    lm_ref: torch.nn.Module
        参考语言模型（通常是 SFT 后的冻结模型）。
    tokenizer: PreTrainedTokenizerBase
        两个模型共用的分词器。
    beta: float
        DPO beta 超参数，决定了对参考模型偏离程度的惩罚强度系数。
    prompt: str
        偏好对样本对应的提示词 (Prompt)。
    response_chosen: str
        针对该提示词的“偏好”（较好）回答。
    response_rejected: str
        针对该提示词的“拒绝”（较差）回答。

    返回值：
    torch.Tensor：包含该样本 DPO 损失值的张量。
    """

    token_yw_x = format_alpaca(prompt=prompt,response=response_chosen,tokenizer=tokenizer)
    token_yl_x = format_alpaca(prompt=prompt,response=response_rejected,tokenizer=tokenizer)

    lp_lm_yw_x = get_log_prob(lm,token_yw_x) 
    lp_lm_yl_x = get_log_prob(lm,token_yl_x) 

    with torch.no_grad():
        lp_lm_ref_yw_x = get_log_prob(lm_ref,token_yw_x) 
        lp_lm_ref_yl_x = get_log_prob(lm_ref,token_yl_x)

    #将参考模型的结果移动到训练模型 (lm) 的设备上
    lp_lm_ref_yw_x = lp_lm_ref_yw_x.to(lm.device)
    lp_lm_ref_yl_x = lp_lm_ref_yl_x.to(lm.device)

    chosen_log_ratio = lp_lm_yw_x - lp_lm_ref_yw_x
    rejected_log_ratio = lp_lm_yl_x - lp_lm_ref_yl_x

    loss = -F.logsigmoid(beta * (chosen_log_ratio - rejected_log_ratio))

    return loss

