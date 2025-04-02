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

        # Ensure pad_token_id is set for student_tokenizer
        if self.student_tokenizer.pad_token_id is None:
            log_rank(f"No pad_token_id found in student_tokenizer for {split}. Setting to eos_token.")
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
            self.student_tokenizer.pad_token_id = self.student_tokenizer.eos_token_id

        # Ensure pad_token_id is set for teacher_tokenizer if provided
        if self.teacher_tokenizer and self.teacher_tokenizer.pad_token_id is None:
            log_rank(f"No pad_token_id found in teacher_tokenizer for {split}. Setting to eos_token.")
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
            self.teacher_tokenizer.pad_token_id = self.teacher_tokenizer.eos_token_id

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
            if 'text' not in df.columns:
                raise ValueError(f"CSV file {path} must contain a 'text' column")
            label_col = 'label' if 'label' in df.columns else 'labels'
            
            log_rank("Processing dataset for classification with list labels...")  
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                student_input_ids = self.student_tokenizer.encode(
                    row['text'], 
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                
                # Handle labels: convert integer to list or keep as list
                raw_label = row[label_col]
                if isinstance(raw_label, (int, np.integer)):
                    label = [int(raw_label)]  # Convert single integer to list
                elif isinstance(raw_label, str):  # If label is a string (e.g., "[1, 2, 3]"), parse it
                    label = eval(raw_label) if raw_label.startswith('[') else [int(raw_label)]
                elif isinstance(raw_label, list):
                    label = raw_label  # Already a list
                else:
                    raise ValueError(f"Unsupported label format: {raw_label}")

                tokenized_data = {
                    "student_input_ids": student_input_ids,
                    "label": label  # Store as list
                }
        
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
        input_ids = np.array(samp["student_input_ids"])
        input_len = len(input_ids)
        
        model_data["input_ids"][i][:input_len] = torch.tensor(input_ids, dtype=torch.long)
        model_data["attention_mask"][i][:input_len] = 1.0
        
        # Convert label list to tensor
        label_tensor = torch.tensor(samp["label"], dtype=torch.long)
        no_model_data["labels"][i, :len(samp["label"])] = label_tensor

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
        
        # Determine maximum number of labels per sample (for padding)
        max_label_len = max(len(samp["label"]) for samp in samples)

        student_pad_token_id = self.student_tokenizer.pad_token_id
        if student_pad_token_id is None:
            student_pad_token_id = 0
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * student_pad_token_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        # Labels are now a 2D tensor: (batch_size, max_label_len)
        no_model_data = {
            "labels": torch.full((bs, max_label_len), -1, dtype=torch.long)  # Use -1 as padding
        }

        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id
            if teacher_pad_token_id is None:
                teacher_pad_token_id = 0
            model_data.update({
                "teacher_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, max_length),
            })

        for i, samp in enumerate(samples):
            self._process_classification(i, samp, model_data, no_model_data)
        
        return model_data, no_model_data
