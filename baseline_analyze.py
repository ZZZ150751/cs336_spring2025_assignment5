import json
import random
import argparse

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Analyze model evaluation results.")
    parser.add_argument("--file", type=str, default="gsm8k_zero_shot_results.jsonl", help="Path to the JSONL file containing the results.")
    args = parser.parse_args()

    #初始化统计计数器
    cat1_count = 0  # format reward 1.0, answer reward 1.0
    cat2_count = 0  # format reward 1.0, answer reward 0.0
    cat3_count = 0  # format reward 0.0, answer reward 0.0
    cat4_count = 0  # format reward 0.0, answer reward 1.0 
    
    #存储format_reward == 0.0 且 answer_reward == 0.0 的数据
    two_zero_examples = []
    #存储format_reward == 1.0 且 answer_reward == 0.0 的数据
    zero_reward_examples = []

    # 读取文件
    with open(args.file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            rewards = data.get("rewards", {})
            format_reward = rewards.get("format_reward", 0.0)
            answer_reward = rewards.get("answer_reward", 0.0)

            #统计各类别数目
            if format_reward == 1.0 and answer_reward == 1.0:
                cat1_count += 1
            elif format_reward == 1.0 and answer_reward == 0.0:
                cat2_count += 1
                zero_reward_examples.append(data)
            elif format_reward == 0.0 and answer_reward == 0.0:
                cat3_count += 1
                two_zero_examples.append(data)
            elif format_reward == 0.0 and answer_reward == 1.0:
                cat4_count += 1

    total_records = cat1_count + cat2_count + cat3_count + cat4_count
    print("="*50)
    print("评估结果统计")
    print("="*50)
    print(f"总数据量: {total_records}")
    print(f"(1) Format 1.0 & Answer 1.0 (双对): {cat1_count}")
    print(f"(2) Format 1.0 & Answer 0.0 (格式对，答案错): {cat2_count}")
    print(f"(3) Format 0.0 & Answer 0.0 (双错): {cat3_count}")
    print(f"(4) Format 0.0 & Answer 1.0 (格式错，答案对): {cat4_count}")
    print("="*50)

    #随机抽取 10 个双错的样例
    print("\n随机抽取 10 个 Format=0.0 且 Answer=0.0 的样例:\n")
    
    #如果不足10个，就抽取实际的全部数量
    sample_size_1 = min(10, len(two_zero_examples))
    
    if sample_size_1 > 0:
        sampled_examples = random.sample(two_zero_examples, sample_size_1)
        
        for i, example in enumerate(sampled_examples, 1):
            print(f"--- 样例 {i} ---")
            print(f"【Prompt】\n{example.get('prompt', '').strip()}\n")
            print(f"【正确的 Answer】\n{example.get('answer_p', '').strip()}\n")
            print(f"【模型 Generated Text】\n{example.get('generated_text', '').strip()}\n")
            print("-" * 50 + "\n")
    else:
        print("没有找到符合条件 (format_reward=0.0 且 answer_reward=0.0) 的数据。")

#############################################################################################

    #随机抽取 10 个格式对答案错的样例
    print("\n随机抽取 10 个 Format=1.0 且 Answer=0.0 的样例:\n")
    
    #如果不足10个，就抽取实际的全部数量
    sample_size_2 = min(10, len(zero_reward_examples))

    if sample_size_2 > 0:
        sampled_examples = random.sample(zero_reward_examples, sample_size_2)
        
        for i, example in enumerate(sampled_examples, 1):
            print(f"--- 样例 {i} ---")
            print(f"【Prompt】\n{example.get('prompt', '').strip()}\n")
            print(f"【正确的 Answer】\n{example.get('answer_p', '').strip()}\n")
            print(f"【模型 Generated Text】\n{example.get('generated_text', '').strip()}\n")
            print("-" * 50 + "\n")
    else:
        print("没有找到符合条件 (format_reward=1.0 且 answer_reward=0.0) 的数据。")

if __name__ == "__main__":
    main()