import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from SentencePair.utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer

class DistillDataset(Dataset):
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
            required_cols = ['premise', 'hypothesis']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file {path} must contain 'premise' and 'hypothesis' columns")
            label_col = 'label' if 'label' in df.columns else 'labels'
            
            log_rank("Processing dataset for BERT-style sentence pair classification...")  
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                # Tokenize premise and hypothesis as a single sequence for student (BERT-style)
                student_encoding = self.student_tokenizer(
                    row['premise'], 
                    row['hypothesis'],
                    add_special_tokens=True,  # Adds [CLS] and [SEP]
                    max_length=self.max_length,
                    truncation=True,
                    padding=False  # Padding will be handled in collate
                )
                
                tokenized_data = {
                    "student_input_ids": student_encoding['input_ids'],
                    "student_attention_mask": student_encoding['attention_mask'],
                    "label": int(row[label_col])
                }
        
                # Tokenize for teacher if provided (also BERT-style)
                if self.teacher_tokenizer:
                    teacher_encoding = self.teacher_tokenizer(
                        row['premise'],
                        row['hypothesis'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False
                    )
                    tokenized_data.update({
                        "teacher_input_ids": teacher_encoding['input_ids'],
                        "teacher_attention_mask": teacher_encoding['attention_mask'],
                    })

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_sentence_pair(
        self, i, samp, model_data, no_model_data
    ):
        # Process student input (combined premise and hypothesis)
        input_ids = np.array(samp["student_input_ids"])
        seq_len = len(input_ids)
        model_data["input_ids"][i][:seq_len] = torch.tensor(input_ids, dtype=torch.long)
        model_data["attention_mask"][i][:seq_len] = torch.tensor(samp["student_attention_mask"], dtype=torch.long)

        # Process label
        no_model_data["labels"][i] = torch.tensor(samp["label"], dtype=torch.long)

        # Process teacher data if available
        if "teacher_input_ids" in samp:
            t_input_ids = np.array(samp["teacher_input_ids"])
            t_seq_len = len(t_input_ids)
            model_data["teacher_input_ids"][i][:t_seq_len] = torch.tensor(t_input_ids, dtype=torch.long)
            model_data["teacher_attention_mask"][i][:t_seq_len] = torch.tensor(samp["teacher_attention_mask"], dtype=torch.long)

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        student_pad_token_id = self.student_tokenizer.pad_token_id or 0
        
        # Initialize model_data for student (BERT-style single sequence)
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * student_pad_token_id,
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
        }
        
        output_data = {
            "labels": torch.zeros(bs, dtype=torch.long)
        }

        # Add teacher data if tokenizer is provided
        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id or 0
            model_data.update({
                "teacher_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
            })

        # Process each sample
        for i, samp in enumerate(samples):
            self._process_sentence_pair(i, samp, model_data, output_data)
        
        return model_data, output_data
