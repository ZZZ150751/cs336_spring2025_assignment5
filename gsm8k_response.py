import re

def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
   提取预测输出中的最后一个数字作为其预测答案来解析生成内容

    model_output: str
        包含模型针对某个 GSM8K 题目生成的完整回答内容的字符串

    Returns:
        如果模型输出可以被解析为预测结果，则返回包含预测数值答案的字符串；否则返回 None
    """
    #匹配千分位规则的捕获方法
    pattern = r"(\d+(?:,\d{3})*)"

    #re.findall 查找所有匹配项
    match = re.findall(pattern, model_output)
    
    if match:
        #返回最后一个数字
        last_match = match[-1]
        return last_match.replace(",", "")
    return None