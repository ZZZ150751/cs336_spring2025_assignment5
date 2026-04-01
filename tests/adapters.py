from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from tokenize_prompt_and_output import tokenize_prompt_and_output
def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    return tokenize_prompt_and_output(prompt_strs,output_strs,tokenizer)
    raise NotImplementedError

from compute_group_normalized_rewards import compute_group_normalized_rewards
def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    return compute_group_normalized_rewards(reward_fn=reward_fn, rollout_responses= rollout_responses, repeated_ground_truths=repeated_ground_truths,
                                            group_size=group_size, advantage_eps=advantage_eps, normalize_by_std=normalize_by_std)
    raise NotImplementedError

from compute_entropy import compute_entropy
def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    return compute_entropy(logits=logits)
    raise NotImplementedError

from get_response_log_probs import get_response_log_probs
def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    return get_response_log_probs(model=model, input_ids=input_ids, labels=labels, return_token_entropy=return_token_entropy)
    raise NotImplementedError

from compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss
def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return compute_naive_policy_gradient_loss(raw_rewards_or_advantages=raw_rewards_or_advantages, policy_log_probs=policy_log_probs)
    raise NotImplementedError

from compute_grpo_clip_loss import compute_grpo_clip_loss
def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return compute_grpo_clip_loss(policy_log_probs= policy_log_probs, old_log_probs=old_log_probs,
                                  cliprange=cliprange, advantages=advantages)
    raise NotImplementedError

from compute_policy_gradient_loss import compute_policy_gradient_loss
def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return compute_policy_gradient_loss(policy_log_probs=policy_log_probs,loss_type=loss_type,raw_rewards=raw_rewards,
                                  advantages=advantages,old_log_probs=old_log_probs,cliprange=cliprange)
    raise NotImplementedError

from masked_mean import masked_mean
def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    return masked_mean(tensor=tensor, mask=mask, dim=dim)
    raise NotImplementedError

from sft_microbatch_train_step import sft_microbatch_train_step
def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return sft_microbatch_train_step(policy_log_probs=policy_log_probs, response_mask=response_mask, gradient_accumulation_steps=gradient_accumulation_steps
                                     , normalize_constant=normalize_constant)
    raise NotImplementedError

from grpo_microbatch_train_step import grpo_microbatch_train_step
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return grpo_microbatch_train_step(policy_log_probs=policy_log_probs, response_mask=response_mask, gradient_accumulation_steps=gradient_accumulation_steps,
                                      loss_type=loss_type, raw_rewards=raw_rewards,advantages=advantages,old_log_probs=old_log_probs,cliprange=cliprange)
    raise NotImplementedError

from masked_normalize import masked_normalize
def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    return masked_normalize(tensor=tensor, mask=mask, dim=dim, normalize_constant=normalize_constant)
    raise NotImplementedError


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""

from PackedDataset import PackedDataset
def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    给定一个分词器（tokenizer）和指向包含指令微调（instruction-tuning）示例的数据集路径，
    构建一个用于语言建模的 PyTorch Dataset 。数据集中所有的序列都具有恒定的长度（seq_length）
    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            用于对文本进行分词（tokenizing）和编码（encoding）的 Transformers 分词器
        dataset_path: str
            包含指令微调示例的文件路径(（提示词，回答）对的集合)
        seq_length: int
            每个示例（example）中应包含的 Token 数量
        shuffle: bool
            如果为 True，在将文档打包成示例之前，先对文档顺序进行打乱（shuffle)

    Returns:
        Dataset
        一个用于语言建模的 PyTorch Dataset 对象
        该数据集中的每个示例都是一个包含 "input_ids" 和 "labels" 键的字典
        （两者都是形状为 (seq_length, ) 的张量）
        "input_ids" 包含语言建模输入的 Token ID，"labels" 包含语言建模标签的 Token ID
    """
    return PackedDataset(tokenizer=tokenizer,dataset_path=dataset_path,seq_length=seq_length,
                         shuffle=shuffle)
    raise NotImplementedError

from iterate_batches import iterate_batches
def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return iterate_batches(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    

from parse_mmlu_response import parse_mmlu_response
def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    return parse_mmlu_response(mmlu_example=mmlu_example, model_output=model_output)
    raise NotImplementedError

from gsm8k_response import parse_gsm8k_response
def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    return parse_gsm8k_response(model_output=model_output)
    raise NotImplementedError

from compute_per_instance_dpo_loss import compute_per_instance_dpo_loss
def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    return compute_per_instance_dpo_loss(lm=lm,
                                         lm_ref=lm_ref,
                                         beta=beta,
                                         prompt=prompt,
                                         response_chosen=response_chosen,
                                         response_rejected=response_rejected,
                                         tokenizer=tokenizer)
    raise NotImplementedError
