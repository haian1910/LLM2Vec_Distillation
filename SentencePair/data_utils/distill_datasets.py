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
            # Expect MultiNLI-like columns: 'premise', 'hypothesis', and 'label'
            required_cols = ['premise', 'hypothesis']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file {path} must contain 'premise' and 'hypothesis' columns")
            label_col = 'label' if 'label' in df.columns else 'labels'
            
            log_rank("Processing dataset for sentence pair classification...")  
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                # Tokenize premise and hypothesis separately for student
                student_premise_ids = self.student_tokenizer.encode(
                    row['premise'], 
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                student_hypo_ids = self.student_tokenizer.encode(
                    row['hypothesis'], 
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                
                tokenized_data = {
                    "student_premise_input_ids": student_premise_ids,
                    "student_hypo_input_ids": student_hypo_ids,
                    "label": int(row[label_col])
                }
        
                # Tokenize for teacher if provided
                if self.teacher_tokenizer:
                    teacher_premise_ids = self.teacher_tokenizer.encode(
                        row['premise'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    teacher_hypo_ids = self.teacher_tokenizer.encode(
                        row['hypothesis'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    tokenized_data.update({
                        "teacher_premise_input_ids": teacher_premise_ids,
                        "teacher_hypo_input_ids": teacher_hypo_ids
                    })

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_sentence_pair(
        self, i, samp, model_data, no_model_data
    ):
        # Process student premise
        premise_ids = np.array(samp["student_premise_input_ids"])
        premise_len = len(premise_ids)
        model_data["student_premise_input_ids"][i][:premise_len] = torch.tensor(premise_ids, dtype=torch.long)
        model_data["student_premise_attention_mask"][i][:premise_len] = 1.0

        # Process student hypothesis
        hypo_ids = np.array(samp["student_hypo_input_ids"])
        hypo_len = len(hypo_ids)
        model_data["student_hypo_input_ids"][i][:hypo_len] = torch.tensor(hypo_ids, dtype=torch.long)
        model_data["student_hypo_attention_mask"][i][:hypo_len] = 1.0

        # Process label
        no_model_data["labels"][i] = torch.tensor(samp["label"], dtype=torch.long)

        # Process teacher data if available
        if "teacher_premise_input_ids" in samp:
            t_premise_ids = np.array(samp["teacher_premise_input_ids"])
            t_premise_len = len(t_premise_ids)
            model_data["teacher_premise_input_ids"][i][:t_premise_len] = torch.tensor(t_premise_ids, dtype=torch.long)
            model_data["teacher_premise_attention_mask"][i][:t_premise_len] = 1.0

            t_hypo_ids = np.array(samp["teacher_hypo_input_ids"])
            t_hypo_len = len(t_hypo_ids)
            model_data["teacher_hypo_input_ids"][i][:t_hypo_len] = torch.tensor(t_hypo_ids, dtype=torch.long)
            model_data["teacher_hypo_attention_mask"][i][:t_hypo_len] = 1.0

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        student_pad_token_id = self.student_tokenizer.pad_token_id
        if student_pad_token_id is None:
            student_pad_token_id = 0
        
        # Initialize model_data for student premise and hypothesis
        model_data = {
            "student_premise_input_ids": torch.ones(bs, max_length, dtype=torch.long) * student_pad_token_id,
            "student_premise_attention_mask": torch.zeros(bs, max_length),
            "student_hypo_input_ids": torch.ones(bs, max_length, dtype=torch.long) * student_pad_token_id,
            "student_hypo_attention_mask": torch.zeros(bs, max_length),
        }
        
        output_data = {
            "labels": torch.zeros(bs, dtype=torch.long)
        }

        # Add teacher data if tokenizer is provided
        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id
            if teacher_pad_token_id is None:
                teacher_pad_token_id = 0
            model_data.update({
                "teacher_premise_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_premise_attention_mask": torch.zeros(bs, max_length),
                "teacher_hypo_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_hypo_attention_mask": torch.zeros(bs, max_length),
            })

        # Process each sample
        for i, samp in enumerate(samples):
            self._process_sentence_pair(i, samp, model_data, output_data)
        
        return model_data, output_data
