import os
import gzip
import json
import torch
import random
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class PackedDataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        dataset_path: str | os.PathLike, 
        seq_length: int, 
        shuffle: bool
    ):
        """
        构建数据集 
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        #提示词
        prompt_file_path = "cs336_alignment/prompts/alpaca_sft.prompt"
        with open(prompt_file_path, "r", encoding="utf-8") as f1:
            prompt_template = f1.read().strip()

        #数据读取与格式化
        documents = []
        with open(dataset_path, 'rt', encoding='utf-8') as f2:
            for line in f2:
                data = json.loads(line)
                # 转换成字符串（Alpaca 的模板）
                text_token_seq = prompt_template.format(
                    instruction=data['prompt'],
                    response=data['response']
                )
                documents.append(text_token_seq)

        #shuffle判断是否对文档顺序进行打乱
        if shuffle:
            random.shuffle(documents)

        #分词并拼接成单一 Token 序列，并转换为张量
        all_token_ids = []
        for doc in documents:
            tokens = tokenizer.encode(doc, add_special_tokens=True)
            all_token_ids.extend(tokens)

            if tokenizer.eos_token_id is not None:
                all_token_ids.append(tokenizer.eos_token_id)
            
        self.all_tokens_tensor = torch.tensor(all_token_ids, dtype=torch.long)

        #计算可以分成的固定块数，丢弃最后不足 seq_length 的部分
        self.num_blocks = (len(self.all_tokens_tensor) - 1) // self.seq_length

    def __len__(self):
        """
        返回一个整数，表示该数据集中的序列数量
        """
        return self.num_blocks

    def __getitem__(self, i):
        """
        返回数据集的第 i 个元素。i 必须小于由 __len__() 返回的数据集长度
        """
        if i >= self.num_blocks or i < 0:
            raise IndexError("Index out of bounds")

        #计算切片的起始和结束索引
        start = i * self.seq_length
        end = start + self.seq_length
            
        input_ids_chunk = self.all_tokens_tensor[start:end]
        labels_chunk = self.all_tokens_tensor[start + 1 : end + 1]
            
        #返回字典，包含 input_ids 和 labels
        return {
            "input_ids": input_ids_chunk.clone(),
            "labels": labels_chunk.clone()
        }