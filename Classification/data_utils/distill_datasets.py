import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from Classification.utils import log_rank
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
            # Read CSV file
            df = pd.read_csv(path)
            label_col = 'label' if 'label' in df.columns else 'labels'
            
            log_rank("Processing dataset for classification...")  
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                # Student tokenizer processing
                student_input_ids = self.student_tokenizer.encode(
                    row['text'], 
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                
                tokenized_data = {
                    "student_input_ids": student_input_ids,
                    "label": int(row[label_col]) 
                }
        
                # Teacher tokenizer processing (if provided)
                if self.teacher_tokenizer:
                    teacher_input_ids = self.teacher_tokenizer.encode(
                        row['text'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    tokenized_data["teacher_input_ids"] = teacher_input_ids

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_classification(
        self, i, samp, model_data, no_model_data
    ):
        # Student data
        input_ids = np.array(samp["student_input_ids"])
        input_len = len(input_ids)
        
        model_data["student_input_ids"][i][:input_len] = torch.tensor(input_ids, dtype=torch.long)
        model_data["student_attention_mask"][i][:input_len] = 1.0
        no_model_data["label"][i] = torch.tensor(samp["label"], dtype=torch.long)

        # Teacher data (if exists)
        if "teacher_input_ids" in samp:
            t_input_ids = np.array(samp["teacher_input_ids"])
            t_input_len = len(t_input_ids)
            model_data["teacher_input_ids"][i][:t_input_len] = torch.tensor(t_input_ids, dtype=torch.long)
            model_data["teacher_attention_mask"][i][:t_input_len] = 1.0

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        model_data = {
            "student_input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.student_tokenizer.pad_token_id,
            "student_attention_mask": torch.zeros(bs, max_length),
        }
        
        no_model_data = {
            "label": torch.zeros(bs, dtype=torch.long)
        }

        # Add teacher data structures if teacher tokenizer exists
        if self.teacher_tokenizer:
            model_data.update({
                "teacher_input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.teacher_tokenizer.pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, max_length),
            })

        for i, samp in enumerate(samples):
            self._process_classification(i, samp, model_data, no_model_data)
        
        return model_data, no_model_data
