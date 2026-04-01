import re

def parse_mmlu_response(
    mmlu_example,
    model_output,
) -> str | None:
    """
   给定一个 MMLU 示例和模型的输出内容，将模型输出解析为一个预测的选项字母（即 'A'、'B'、'C' 或 'D'） 。如果模型输出无法被解析为有效的选项字母，则返回 None

    mmlu_example: dict[str, Any]
        包含单个 MMLU 样本数据的字典 。包含以下键值对：
        - "subject": 字符串，表示问题的学科主题
        - "question": 字符串，表示问题的具体文本内容
        - "options": 字符串列表，按顺序包含四个备选答案 。列表中的第一个选项对应字母 "A"，第二个对应 "B"，依此类推 。
        - "answer": 字符串，表示正确答案的选项字母
    model_output: str
        字符串，表示模型针对该 MMLU 示例生成的原始回答文本

    Returns:
        如果模型输出可以被成功解析，返回 "A"、"B"、"C" 或 "D" 其中之一 。
        如果无法从中提取出有效的预测选项，则返回 None
    """
    #提示词要求
    pattern = r"The correct answer is\s*([A-D])"
    #re.search 查找匹配项,re.IGNORECASE 用于忽略大小写
    match = re.search(pattern, model_output, re.IGNORECASE)
    
    if match:
        #match.group(0)会返回完整的匹配句子，match.group(1) 会返回第一个括号 ()里捕获的内容
        return match.group(1).upper()#.upper() 确保模型输出小写转大写
    else:
        return None