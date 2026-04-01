import torch
def compute_group_normalized_rewards(
    reward_fn,#字典，记录每一轮的奖励（"reward","format_reward","answer_reward"）
    rollout_responses: list[str],#策略模型生成的回复（提示词个数 * 组数）
    repeated_ground_truths: list[str],#每一个回复的正确答案（）
    group_size: int,#组数
    advantage_eps: float,#归一化中防止除0的常数
    normalize_by_std: bool,#若为True，则除组内标准差
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算组归一化奖励
    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948
    Args:
        reward_fn: 字典，记录每一轮的奖励（"reward","format_reward","answer_reward"）
        rollout_responses: 策略模型生成的回复（提示词个数 * 组数）
        repeated_ground_truths:每一个回复的正确答案，和rollout_responses长度相同
        group_size: 组数
        advantage_eps: 归一化中防止除0的常数
        normalize_by_std: 若为True，则除组内标准差
    Returns:
        torch.Tensor of shape (rollout_batch_size,): 组归一化奖励
            torch.Tensor of shape (rollout_batch_size,): 原始（未归一化）奖励
            dict[str, float]: 选择记录的其他统计信息（例如奖励的均值、标准差、最大/最小值等）
    """
    #先得到奖励（输入生成结果和真实结果）
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    for responses, ground_truths in zip(rollout_responses, repeated_ground_truths):
        rewards_dict = reward_fn(responses, ground_truths)
        raw_rewards_list.append(rewards_dict.get("reward", 0.0))
        format_rewards_list.append(rewards_dict.get("format_reward", 0.0))
        answer_rewards_list.append(rewards_dict.get("answer_reward", 0.0))
    
    #将奖励的列表都转换为张量(因为优势是通过总奖励计算，所以不用将格式和答案奖励换成张量形式)
    #同时满足题目要求形状：（rollout_batch_size, ）
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    #转换形状（视角）：(num_prompts, group_size)
    rewards_matrix = raw_rewards.view(-1, group_size)

    #得到组内均值
    group_mean = rewards_matrix.mean(dim=1, keepdim=True)

    advantages = rewards_matrix - group_mean

    if normalize_by_std:
        #计算组内标准差
        group_std = rewards_matrix.std(dim=1, keepdim=True)
        advantages = advantages / (group_std + advantage_eps)
    
    #改变为作业要求形状
    advantages = advantages.view(-1)

    #记录
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "format_reward_mean": sum(format_rewards_list) / len(format_rewards_list),
        "answer_reward_mean": sum(answer_rewards_list) / len(answer_rewards_list),
    }

    return advantages, raw_rewards, metadata

    raise NotImplementedError