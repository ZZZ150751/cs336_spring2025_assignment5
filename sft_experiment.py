import json
import torch
import wandb
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from datasets import load_dataset

from tokenize_prompt_and_output import tokenize_prompt_and_output
from get_response_log_probs import get_response_log_probs
from sft_microbatch_train_step import sft_microbatch_train_step
from log_generation import log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from transformers import get_cosine_schedule_with_warmup
from evaluate_vllm import evaluate_vllm



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


#SFT 训练循环
def train_sft(
    model,
    tokenizer,
    num_epochs: int,
    dataloader, #数据加载器
    gradient_accumulation_steps: int,
    optimizer,
    scheduler, 
    eval_steps: int,
    vllm_engine,          #传入 vLLM 引擎
    eval_prompts: list,   #传入验证集的 prompts
    eval_gts: list,       #传入验证集的 ground truths
    reward_fn,             #需要的 reward 函数
    sampling_params
):
    """
    输入：模型，分词器，训练次数，数据加载器，梯度累计步数，优化器，学习率调度器，评估步数间隔，
    vLLM 引擎，验证集的 prompts，验证集的 ground truths，reward 函数，超参数
    """
    #SFT 训练主循环
    device = model.device
    global_step = 0

    # 设置 wandb 的评估指标
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(dataloader): #batch（字典）中包含 prompt 和 response 字符串
        
            #分词，生成inputs_id，labels和response_mask
            inputs = tokenize_prompt_and_output(batch["prompt"], batch["response"], tokenizer)

            input_ids = inputs["input_ids"].to(device)
            labels = inputs["labels"].to(device)
            response_mask = inputs["response_mask"].to(device)

            #获取 log probs 和熵(预测下一个token)
            probs_dict = get_response_log_probs(
                model=model, 
                input_ids=input_ids, 
                labels=labels, 
                return_token_entropy=True
            )
        
            #计算 Loss 并反向传播 (内部用了 masked_normalize)
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=probs_dict["log_probs"], 
                response_mask=response_mask, 
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            #记录训练指标到 wandb
            wandb.log({
                "train/loss": metadata.get("unscaled_loss", loss.item() * gradient_accumulation_steps),
                "train/entropy": probs_dict.get("token_entropy").mean().item() if "token_entropy" in probs_dict else 0,
                "train_step": global_step
            })

            if (i + 1) % gradient_accumulation_steps == 0:
                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                #每隔 gradient_accumulation_steps 个 批 次 更 新 一 次 权 重
                optimizer.step()
                scheduler.step()
                #每隔 gradient_accumulation_steps 个 批 次 清 零 一 次 梯 度
                optimizer.zero_grad()
                global_step += 1
                wandb.log({"train/lr": scheduler.get_last_lr()[0], "train_step": global_step})
        
        
            #评估
            if global_step > 0 and global_step % eval_steps == 0 and (i + 1) % gradient_accumulation_steps == 0:
                print(f"\n--- Running Evaluation at Step {global_step} ---")
                
                torch.cuda.empty_cache()

                # 同步权重到 vLLM
                load_policy_into_vllm_instance(model, vllm_engine)
                
                #评估
                output_path = f"sft_eval_results_step_{global_step}.jsonl"
                responses, rewards_info, avg_format_reward, avg_answer_reward, avg_total_reward = evaluate_vllm(
                    vllm_model=vllm_engine,
                    reward_fn=reward_fn,
                    prompts=eval_prompts,
                    answer_p=eval_gts,
                    eval_sampling_params=sampling_params,
                    output_path=output_path
                )

                #计算 Entropy 
                avg_entropies = []
                for p, r in zip(eval_prompts, responses):
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

                #记录数据到 wandb
                wandb.log({
                    "eval/format_reward": avg_format_reward,
                    "eval/answer_reward": avg_answer_reward,
                    "eval/entropy": avg_entropy_all,
                    "eval_step": global_step
                })

                #logging
                log_generations(
                    step=global_step,
                    prompts=eval_prompts,
                    responses=responses,
                    ground_truths=eval_gts,
                    rewards_info=rewards_info,
                    avg_entropies=avg_entropies,
                    tokenizer=tokenizer
                )
                
                model.train()

def extract_answer(gsm8k_answer: str) -> str:
    return gsm8k_answer.split("#### ")[-1].strip()

if __name__ == "__main__":
    #按照作业说明提供的集群路径（我这里是本地路径）
    MODEL_PATH = "/root/autodl-tmp/Qwen2.5-Math-1.5B"
    
    # 超参数与实验设置
    DATASET_SIZE = None # 可选项: 128, 256, 512, 1024, 或设为 None 表示使用全部数据 
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 2
    EVAL_STEPS = 32

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
    lr=LEARNING_RATE, 
    weight_decay=0.05)

    #初始化 vLLM 引擎 (用于评估)
    print("Initializing vLLM Engine...")
    vllm_engine = init_vllm(
        model_id=MODEL_PATH,
        device="cuda:1", 
        seed=42
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
    full_sft_dataset = []
    
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
        full_sft_dataset.append({
            "prompt": prompt,
            "response": response
        })
    
    # 根据 DATASET_SIZE 截取数据（如果设置了限制的话）
    if DATASET_SIZE is not None:
        sft_dataset = full_sft_dataset[:DATASET_SIZE]
    else:
        sft_dataset = full_sft_dataset
        
    # 直接将列表传给 DataLoader，使用默认批处理即可
    dataloader = DataLoader(
        dataset=sft_dataset, 
        batch_size=BATCH_SIZE, 
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

    # 计算总步数
    num_update_steps_per_epoch = len(dataloader) // GRAD_ACCUM_STEPS
    total_training_steps = num_update_steps_per_epoch * NUM_EPOCHS
    
    # 设置预热步数，通常设为总步数的 5% - 10%
    num_warmup_steps = int(0.1 * total_training_steps)

    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")

    # 创建余弦退火调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )

    #超参数
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    wandb.init(project="cs336-assignment5", name=f"sft-{DATASET_SIZE}")

    print(f"Starting Training on {len(sft_dataset)} examples...")
    train_sft(
        model=policy_model, 
        tokenizer=tokenizer,
        num_epochs=NUM_EPOCHS,
        dataloader=dataloader, 
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_steps=EVAL_STEPS, 
        vllm_engine=vllm_engine,
        eval_prompts=eval_prompts,
        eval_gts=eval_gts,
        reward_fn=r1_zero_reward_fn,
        sampling_params=sampling_params
    )