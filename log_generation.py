import numpy as np
import wandb

def log_generations(
    step: int,#迭代步数
    prompts: list[str],#提示词
    responses: list[str],#生成的回复
    ground_truths: list[str],#真实的答案
    rewards_info: list[dict], #包含 total, format, answer reward 的字典列表
    avg_entropies: list[float], #提前算好的平均熵
    tokenizer
):
    """
    输入：迭代步数，提示词，生成的回复，真实的答案，包含 total, format, answer reward 的字典列表，提前算好的平均熵
    """
    lengths = []
    correct_lengths = []
    incorrect_lengths = []
    
    #建立表格
    columns = [
        "Prompt", "Response", "Ground Truth", 
        "Total Reward", "Format Reward", "Answer Reward", 
        "Avg Entropy", "Token Length"
    ]
    log_table = wandb.Table(columns=columns)
    
    for prompt, response, gt, rew_info, entropy in zip(
        prompts, responses, ground_truths, rewards_info, avg_entropies
    ):
        #统计长度
        token_length = len(tokenizer.encode(response))
        lengths.append(token_length)
        
        answer_reward = rew_info.get("answer_reward", 0.0)
        
        if answer_reward > 0:
            correct_lengths.append(token_length)
        else:
            incorrect_lengths.append(token_length)
            
        # 填入表格
        log_table.add_data(
            prompt, response, gt, 
            rew_info.get("reward", 0.0), 
            rew_info.get("format_reward", 0.0), 
            answer_reward, 
            entropy, token_length
        )
        
    #计算均值
    avg_length = np.mean(lengths) if lengths else 0.0
    avg_correct_length = np.mean(correct_lengths) if correct_lengths else 0.0
    avg_incorrect_length = np.mean(incorrect_lengths) if incorrect_lengths else 0.0
    
    # 输出到 wandb
    wandb.log({
        "eval/generations_table": log_table,
        "eval/avg_length": avg_length,
        "eval/avg_correct_length": avg_correct_length,
        "eval/avg_incorrect_length": avg_incorrect_length,
        "eval_step": step
    })