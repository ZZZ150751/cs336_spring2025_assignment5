import torch
from torch import Tensor
def tokenize_prompt_and_output(
    prompt_strs,#提示词
    output_strs,#输出结果
    tokenizer,#使用的分词器
) -> dict[str, Tensor]:
    """
    返回结果：设 prompt_and_output_lens 为包含分词后提示词和输出字符串长度的列表
    "input_ids":(batch_size, max(prompt_and_output_lens) - 1)分词后的提示词和输出字符串，并切掉最后一个标记
    "labels":(batch_size, max(prompt_and_output_lens) - 1)：平移后的 input_ids，即不包含第一个标记的 input_ids(标签y)
    "response_mask":(batch_size, max(prompt_and_output_lens) - 1)：针对 labels 中响应标记的掩码
    """
    batch_input_ids = []
    batch_response_masks = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_token = tokenizer.encode(prompt, add_special_tokens=False)
        output_token = tokenizer.encode(output, add_special_tokens=False)
    
        all_token = prompt_token + output_token
        mask_token = [0] * len(prompt_token) + [1] * len(output_token)

        batch_input_ids.append(all_token)
        batch_response_masks.append(mask_token)

    #最大长度
    max_len = max(len(p_o) for p_o in batch_input_ids)
    
    #不知道这是干什么的，看意思好像是测试用例不能直接填零，他可能会有一个期望填充值
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        pad_id = tokenizer.eos_token_id
    else:
        pad_id = 0

    #对长度小于最大长度的补齐长度
    batch_input_ids_new = []
    batch_response_masks_new = []
    for ids, mask in zip(batch_input_ids, batch_response_masks):
        d = max_len - len(ids)
        batch_input_ids_new.append(ids + [pad_id] * d)
        batch_response_masks_new.append(mask + [0] * d)
    
    #转换为张量
    full_input_ids = torch.tensor(batch_input_ids_new, dtype=torch.long)
    full_masks = torch.tensor(batch_response_masks_new, dtype=torch.long)
    
    inputs_id = full_input_ids[:,:-1]
    labels = full_input_ids[:, 1:]
    response_mask = full_masks[: , 1:]

    res = {
        "input_ids":inputs_id,
        "labels":labels,
        "response_mask":response_mask
    }
    return res