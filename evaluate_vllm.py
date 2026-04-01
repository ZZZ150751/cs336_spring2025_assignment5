import json
from typing import List, Callable
from vllm import LLM, SamplingParams
from datasets import load_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn#验证函数


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],#验证函数
    prompts: List[str],#提示词
    answer_p: List[str],#标准答案
    eval_sampling_params: SamplingParams,#超参数设置
    output_path: str #保存路径
):
    """
    在给定的提示词列表上评估语言模型，
    计算评估指标，并将结果序列化保存到磁盘。
    """
    print(f"开始为 {len(prompts)} 个问题生成回答：")
    #根据提示生成文本。输出结果为RequestOutput对象列表（包含提示、生成文本及其他信息的文件）
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results_to_save = []
    responses = []#计算 Entropy
    rewards_info = []#log_generations

    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_all_reward = 0.0

    for i,output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text#得到第一个候选
        res = answer_p[i]
        responses.append(generated_text)
        #验证
        reward_dict = reward_fn(response=generated_text, ground_truth=res)
        rewards_info.append(reward_dict)

        total_format_reward += reward_dict.get("format_reward", 0.0)
        total_answer_reward += reward_dict.get("answer_reward", 0.0)
        total_all_reward += reward_dict.get("reward", 0.0)
        #存储示例，结果，生成文本，评估分数
        results_to_save.append({
            "prompt": prompt,
            "answer_p": res,
            "generated_text": generated_text,
            "rewards": reward_dict
        })

    print(f"评估完成")
    avg_format_reward = total_format_reward / len(prompts)
    avg_answer_reward = total_answer_reward / len(prompts)
    avg_total_reward = total_all_reward / len(prompts)

    print(f"Format Reward: {avg_format_reward:.4f}")
    print(f"Answer Reward: {avg_answer_reward:.4f}")
    print(f"Total Reward: {avg_total_reward:.4f}")


    with open(output_path, "w", encoding="utf-8") as f:
        for item in results_to_save:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"结果已保存至 {output_path}")

    return responses, rewards_info, avg_format_reward, avg_answer_reward, avg_total_reward



def extract_answer(gsm8k_answer: str) -> str:
    return gsm8k_answer.split("#### ")[-1].strip()


if __name__ == "__main__":
    test_data = []
    #本地测试集
    with open("/root/autodl-tmp/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    #提示词
    prompt_file_path = "cs336_alignment/prompts/r1_zero.prompt"

    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompts = []
    answer_p = []
    for item in test_data: #.select(range(100))
        prompts.append(prompt_template.replace("{question}", item["question"]))
        answer_p.append(extract_answer(item["answer"]))

    #超参数
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    llm = LLM(model="/root/autodl-tmp/Qwen2.5-Math-1.5B", dtype="bfloat16")
    
    #评估
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn, 
        prompts=prompts,
        answer_p=answer_p,
        eval_sampling_params=sampling_params,
        output_path="gsm8k_zero_shot_results.jsonl"
    )