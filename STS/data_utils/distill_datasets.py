import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer

class STSDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        student_tokenizer: AutoTokenizer,
        teacher_tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = args.max_length

        self.dataset = self._load_and_process_data()

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.csv")

        if os.path.exists(path):
            df = pd.read_csv(path)
            required_columns = ['sentence1', 'sentence2', 'score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"CSV file {path} must contain columns: {required_columns}. Missing: {missing_columns}")
            
            log_rank("Processing dataset for STS task...")  
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                # Encode sentence pair for student model
                encoded_pair = self.student_tokenizer.encode_plus(
                    row['sentence1'],
                    row['sentence2'],
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors=None
                )
                
                tokenized_data = {
                    "student_input_ids": encoded_pair["input_ids"],
                    "student_attention_mask": encoded_pair["attention_mask"],
                    "student_token_type_ids": encoded_pair.get("token_type_ids", [0] * len(encoded_pair["input_ids"])),
                    "score": float(row['score'])
                }
        
                if self.teacher_tokenizer:
                    # Encode sentence pair for teacher model
                    teacher_encoded_pair = self.teacher_tokenizer.encode_plus(
                        row['sentence1'],
                        row['sentence2'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors=None
                    )
                    
                    tokenized_data.update({
                        "teacher_input_ids": teacher_encoded_pair["input_ids"],
                        "teacher_attention_mask": teacher_encoded_pair["attention_mask"],
                        "teacher_token_type_ids": teacher_encoded_pair.get("token_type_ids", [0] * len(teacher_encoded_pair["input_ids"]))
                    })

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def collate(self, samples):
        bs = len(samples)
        
        # Initialize model data dict
        model_data = {
            "input_ids": torch.stack([torch.tensor(sample["student_input_ids"], dtype=torch.long) for sample in samples]),
            "attention_mask": torch.stack([torch.tensor(sample["student_attention_mask"], dtype=torch.float) for sample in samples]),
            "token_type_ids": torch.stack([torch.tensor(sample["student_token_type_ids"], dtype=torch.long) for sample in samples]),
        }
        
        # Initialize output data with similarity scores
        output_data = {
            "scores": torch.tensor([sample["score"] for sample in samples], dtype=torch.float).view(-1, 1)
        }

        # Add teacher inputs if available
        if self.teacher_tokenizer:
            model_data.update({
                "teacher_input_ids": torch.stack([torch.tensor(sample["teacher_input_ids"], dtype=torch.long) for sample in samples]),
                "teacher_attention_mask": torch.stack([torch.tensor(sample["teacher_attention_mask"], dtype=torch.float) for sample in samples]),
                "teacher_token_type_ids": torch.stack([torch.tensor(sample["teacher_token_type_ids"], dtype=torch.long) for sample in samples]),
            })
        
        return model_data, output_data

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)