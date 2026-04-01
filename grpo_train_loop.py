import json
import torch
import wandb
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from datasets import load_dataset

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tokenize_prompt_and_output import tokenize_prompt_and_output
from get_response_log_probs import get_response_log_probs
from compute_group_normalized_rewards import compute_group_normalized_rewards
from grpo_microbatch_train_step import grpo_microbatch_train_step
from evaluate_vllm import evaluate_vllm
from log_generation import log_generations

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    启动推理进程，这里我们使用 vLLM 将模型加载到独立于策略模型的另一张 GPU 上。
    """
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    将 PyTorch 策略模型的权重同步到 vLLM 实例中。
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

#GRPO训练循环
def grpo_train_loop(
    model,
    tokenizer,
    n_grpo_steps,
    epochs_per_rollout_batch,
    rollout_batch_size,
    train_batch_size,
    dataloader,
    reward_fn, 
    group_size,
    advantage_eps,
    normalize_by_std,
    cliprange,
    loss_type,
    gradient_accumulation_steps,
    optimizer,
    eval_steps,
    vllm_engine,
    eval_prompts: list,  
    eval_gts: list,       
    sampling_params
):
    """
    Args:
    model:模型
    tokenizer:分词器
    n_grpo_steps:总迭代步数
    epochs_per_rollout_batch:对一小批数据学习的次数
    rollout_batch_size:group_size * n_prompts_per_rollout_batch（一个批次的数据量）
    train_batch_size:实际用来更新批次的数据量
    dataloader:训练集数据加载器
    reward_fn:奖励函数（输入回答和正确答案，返回奖励字典）
    group_size:组的大小
    advantage_eps:归一化中防止除零的常数
    normalize_by_std:判断是否除组内标准差的布尔值
    cliprange:裁剪参数
    loss_type:损失类型
    gradient_accumulation_steps:梯度累计步数（几步更新一次）
    optimizer:优化器
    eval_steps:模型评估间隔
    vllm_engine:vllm引擎
    eval_prompts:传入验证集的 prompts
    eval_gts:#传入验证集的 ground truths
    sampling_params：超参数
    """

    device = model.device
    global_step = 0

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    dataloader_iter = iter(dataloader)

    #开始迭代（n_grpo_steps为总迭代次数）
    for step in range(n_grpo_steps):
        model.train()
        #从数据集中随机抽取batch个问题
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        #复制问题和正确答案
        repeated_prompts = [p for p in batch["prompt"] for _ in range(group_size)]
        repeated_ground_truths = [p for p in batch["ground_truth"] for _ in range(group_size)]

        #模型生成的回答
        load_policy_into_vllm_instance(model, vllm_engine)
        outputs = vllm_engine.generate(repeated_prompts, sampling_params)
        rollout_responses = [out.outputs[0].text for out in outputs]#得到第一个候选

        #得到组优势，原始奖励，以及原始奖励的记录值
        advantages, raw_rewards, metadata1 = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std
            )

        #将 advantages,raw_rewards调整形状为 (batch_size, 1)
        advantages = advantages.to(device).unsqueeze(1)
        raw_rewards = raw_rewards.to(device).unsqueeze(1)

        #分词，生成inputs_id，labels和response_mask（与SFT监督微调不同，这里用模型自己的回答）
        inputs = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = inputs["input_ids"].to(device)
        labels = inputs["labels"].to(device)#标签
        response_mask = inputs["response_mask"].to(device)#掩码

        #微批次的大小（train_batch_size：训练批次，模型更新阶段，真正用于计算梯度并更新权重的样本子集大小）
        #这里将其分为了gradient_accumulation_steps个微批次（每次传入GPU真实的样本数，即更新参数时，参考的样本数）
        #gradient_accumulation_steps：为了完成一次权重更新，需要积累的梯度次数
        micro_train_batch_size = train_batch_size // gradient_accumulation_steps

        #group_size * n_prompts_per_rollout_batch（总批次数）
        rollout_batch_size = len(repeated_prompts)

            
        #得到旧的策略产生的对数概率(使用微批次)
        old_log_probs_list = []
        with torch.no_grad():
            for start_idx in range(0, rollout_batch_size, micro_train_batch_size):
                end_idx = start_idx + micro_train_batch_size
                mb_input_ids = input_ids[start_idx:end_idx]
                mb_labels = labels[start_idx:end_idx]
                
                mb_probs_dict = get_response_log_probs(
                    model=model, 
                    input_ids=mb_input_ids, 
                    labels=mb_labels, 
                    return_token_entropy=False
                )
                old_log_probs_list.append(mb_probs_dict["log_probs"].detach())
                
        # 拼接回完整的形状 (batch_size, seq_len)
        old_log_probs = torch.cat(old_log_probs_list, dim=0)
        
        #固定旧的策略产生概率，开始微批次更新策略模型
        for train_step in range(epochs_per_rollout_batch):#（epochs_per_rollout_batch：在一批数据上学习的次数）

            optimizer.zero_grad()

            #每个微批次的起点
            micro_starts = list(range(0, rollout_batch_size, micro_train_batch_size))

            for idx, start_idx in enumerate(micro_starts):
                end_idx = start_idx + micro_train_batch_size

                mb_input_ids = input_ids[start_idx:end_idx]
                mb_labels = labels[start_idx:end_idx]
                mb_response_mask = response_mask[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]
                mb_raw_rewards = raw_rewards[start_idx:end_idx]
                mb_old_log_probs = old_log_probs[start_idx:end_idx]

                #获取当前训练的 log probs 和熵(预测下一个token)
                probs_dict = get_response_log_probs(
                    model=model, 
                    input_ids=mb_input_ids, 
                    labels=mb_labels, 
                    return_token_entropy=True
                )
                policy_log_probs = probs_dict["log_probs"]

                #计算损失，并记录
                loss, metadata2 = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs, 
                    cliprange=cliprange
                )

                if (idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    #记录训练指标到 wandb
                    wandb.log({
                        "train/loss": metadata2.get("microbatch_loss", loss.item() * gradient_accumulation_steps),
                        "train/entropy": probs_dict.get("token_entropy", torch.tensor(0.0)).mean().item(),
                        "train/mean_reward": raw_rewards.mean().item(), # 记录整个 batch 的平均奖励
                        "train_step": global_step
                    })

                    #评估
                    if global_step > 0 and global_step % eval_steps == 0:
                        print(f"\n--- Running Evaluation at Step {global_step} ---")
                        
                        torch.cuda.empty_cache()

                        # 同步权重到 vLLM
                        load_policy_into_vllm_instance(model, vllm_engine)
                        
                        #评估 (这里定义了 responses)
                        output_path = f"sft_eval_results_step_{global_step}.jsonl"
                        responses, rewards_info, avg_format_reward, avg_answer_reward, avg_total_reward = evaluate_vllm(
                            vllm_model=vllm_engine,
                            reward_fn=reward_fn,
                            prompts=eval_prompts,
                            answer_p=eval_gts,
                            eval_sampling_params=sampling_params,
                            output_path=output_path
                        )

                        # 计算 Entropy （只算前 10 个 eval 样本）
                        avg_entropies = []
                        for p, r in zip(eval_prompts[:10], responses[:10]):
                            eval_inputs = tokenize_prompt_and_output([p], [r], tokenizer)
                            eval_input_ids = eval_inputs["input_ids"]
                            if eval_input_ids.dim() == 1:
                                eval_input_ids = eval_input_ids.unsqueeze(0)
                            eval_input_ids = eval_input_ids.to(device)
                        
                            eval_labels = eval_inputs["labels"]
                            if eval_labels.dim() == 1:
                                eval_labels = eval_labels.unsqueeze(0)
                            eval_labels = eval_labels.to(device)
                            
                            with torch.no_grad():
                                probs_dict = get_response_log_probs(
                                    model=model, 
                                    input_ids=eval_input_ids, 
                                    labels=eval_labels, 
                                    return_token_entropy=True
                                )
                                entropy = probs_dict.get("token_entropy", torch.tensor(0.0)).mean().float().item()
                                avg_entropies.append(entropy)
                        
                        avg_entropy_all = sum(avg_entropies) / len(avg_entropies) if avg_entropies else 0.0
                        
                        # 记录数据到 wandb
                        wandb.log({
                            "eval/format_reward": avg_format_reward,
                            "eval/answer_reward": avg_answer_reward,
                            "eval/entropy": avg_entropy_all,
                            "eval_step": global_step
                        })
                        print(f"--- Evaluation Completed at Step {global_step} ---\n")
                        
                    model.train() 

def extract_answer(gsm8k_answer: str) -> str:
    return gsm8k_answer.split("#### ")[-1].strip()

if __name__ == "__main__":
    #按照作业说明提供的集群路径（我这里是本地路径）
    MODEL_PATH = "/root/autodl-tmp/Qwen2.5-Math-1.5B"
    
    # 超参数与实验设置
    n_grpo_steps: int = 20 # GRPO 训练的总步数
    learning_rate: float = 1e-5 #学习率
    advantage_eps: float = 1e-6 #优势归一化中防止除以零的常数
    rollout_batch_size: int = 256 # 每次用策略模型生成采样的总回复数（group_size * n_prompts_per_rollout_batch）
    group_size: int = 8 #组大小
    sampling_temperature: float = 1.0 #采样的温度参数
    sampling_min_tokens: int = 4 #最小生成长度，就像专家迭代(EI)中一样，禁止生成空字符串
    sampling_max_tokens: int = 1024 #最大生成长度
    epochs_per_rollout_batch: int = 4#4 #每个采样批次更新几轮（学几遍）（1代表同策略 On-policy）
    train_batch_size: int = 64#64 #训练批次大小（用来更新数据的批次）（由于是同策略，等于 rollout_batch_size）
    gradient_accumulation_steps: int = 32#32 # 梯度累积步数（代表 microbatch size 为 2：每次在GPU实际训练的批次数）
    gpu_memory_utilization: float = 0.85 # vLLM 的显存占用率
    loss_type = "reinforce_with_baseline"#"grpo_clip" # 当前选用的损失函数类型
    use_std_normalization: bool = True #是否在组内优势计算时除以标准差
    cliprange = 0.2 #GRPO 裁剪参数
    eval_steps = 10 #每隔多少步验证一次

    # 计算衍生参数
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size

    #加载 HuggingFace 模型 (用于训练) 和分词器
    print("Loading Policy Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    policy_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    ).to("cuda:0")
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(
    policy_model.parameters(),
    lr=learning_rate,
    weight_decay=0.0,
    betas=(0.9, 0.95),)


    #初始化 vLLM 引擎 (用于评估)
    print("Initializing vLLM Engine...")
    vllm_engine = init_vllm(
        model_id=MODEL_PATH,
        device="cuda:1", 
        seed=42,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    #R1-zero 的 prompt 模板路径
    prompt_file_path = "cs336_alignment/prompts/r1_zero.prompt"
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    #数据
    print("Loading local GSM8K dataset...")
    #读取本地训练集
    train_data = []
    with open("/root/autodl-tmp/gsm8k_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))
                
    #读取本地测试集（用作验证集）
    test_data = []
    with open("/root/autodl-tmp/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    #准备训练集
    print("Preparing train data...")
    grpo_dataset = []
    
    for item in train_data:
        #构造 Prompt（利用已经读取的 prompt_template）
        prompt = prompt_template.replace("{question}", item["question"])
        
        #构造 SFT 需要的目标 Response (按照作业的 <think> 和 <answer> 格式)
        raw_answer = item["answer"]
        if "####" in raw_answer:
            reasoning, final_answer = raw_answer.split("####")
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = raw_answer.strip()
            final_answer = ""
            
        response = f"\n{reasoning}\n</think> <answer> {final_answer} </answer>"
        
        #将单条数据作为字典存入列表
        grpo_dataset.append({
            "prompt": prompt,
            "response": response,
            "ground_truth": extract_answer(item["answer"])
        })
        
    # 直接将列表传给 DataLoader，使用默认批处理即可
    dataloader = DataLoader(
        dataset=grpo_dataset, 
        batch_size=n_prompts_per_rollout_batch, 
        shuffle=True, 
        drop_last=True 
    )

   
    #准备验证集
    print("Preparing validation data...")
    eval_prompts = []
    eval_gts = []
    
    for item in test_data:
        # 使用 replace 替换提示词模板，并用 extract_answer 提取准确答案
        eval_prompts.append(prompt_template.replace("{question}", item["question"]))
        eval_gts.append(extract_answer(item["answer"]))

    #超参数
    sampling_params = SamplingParams(
        temperature=sampling_temperature, 
        top_p=1.0, 
        max_tokens=sampling_max_tokens,  
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=1 
    )

    wandb.init(project="cs336-assignment5", name="grpo-gsm8k-run1")

    print(f"Starting GRPO Training...")
    grpo_train_loop(
        model=policy_model,
        tokenizer=tokenizer,
        n_grpo_steps=n_grpo_steps,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        rollout_batch_size = rollout_batch_size,
        train_batch_size = train_batch_size,
        dataloader=dataloader,
        reward_fn=r1_zero_reward_fn,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=use_std_normalization,
        cliprange=cliprange,
        loss_type=loss_type,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer=optimizer,
        eval_steps=eval_steps,
        vllm_engine=vllm_engine,
        eval_prompts=eval_prompts,
        eval_gts=eval_gts,
        sampling_params=sampling_params
    )