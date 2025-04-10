import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from QuestionAnswer.utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer
import ast  # For parsing string lists

class DistillDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        student_tokenizer: AutoTokenizer,
        teacher_tokenizer: Optional[AutoTokenizer] = None,
        num_choices: int = 8,  # QASC has 8 choices (A-H)
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = args.max_length
        self.num_choices = args.num_labels

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
            required_columns = ['question', 'choices', 'answerKey']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file {path} must contain 'question', 'choices', and 'answerKey' columns")
            
            log_rank("Processing QASC-like dataset for multiple-choice question answering...")
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                # Parse choices (stringified list of dicts)
                choices = ast.literal_eval(row['choices'])
                if len(choices) != self.num_choices:
                    raise ValueError(f"Expected {self.num_choices} choices, but got {len(choices)}")
                
                # Extract choice texts and labels
                choice_texts = [choice['text'] for choice in choices]
                choice_labels = [choice['label'] for choice in choices]
                
                # Map answerKey (e.g., "B") to its index
                answer_key = row['answerKey']
                try:
                    answer_idx = choice_labels.index(answer_key)
                except ValueError:
                    raise ValueError(f"Answer key '{answer_key}' not found in labels {choice_labels}")

                # Prepare inputs: combine question with each choice
                question = row['question']
                inputs = [f"{question} {choice_text}" for choice_text in choice_texts]
                
                # Tokenize for student model
                student_encoding = self.student_tokenizer(
                    inputs,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None
                )
                
                tokenized_data = {
                    "student_input_ids": student_encoding['input_ids'],
                    "student_attention_mask": student_encoding['attention_mask'],
                    "label": answer_idx  # Index of correct choice (0-7)
                }
        
                # Tokenize for teacher model if provided
                if self.teacher_tokenizer:
                    teacher_encoding = self.teacher_tokenizer(
                        inputs,
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None
                    )
                    tokenized_data["teacher_input_ids"] = teacher_encoding['input_ids']
                    tokenized_data["teacher_attention_mask"] = teacher_encoding['attention_mask']

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_multiple_choice(
        self, i, samp, model_data, no_model_data
    ):
        # Student inputs
        for j in range(self.num_choices):
            input_ids = np.array(samp["student_input_ids"][j])
            input_len = min(len(input_ids), self.max_length)
            
            model_data["input_ids"][i, j, :input_len] = torch.tensor(input_ids[:input_len], dtype=torch.long)
            model_data["attention_mask"][i, j, :input_len] = torch.tensor(samp["student_attention_mask"][j][:input_len], dtype=torch.long)

        no_model_data["labels"][i] = torch.tensor(samp["label"], dtype=torch.long)

        # Teacher inputs if available
        if "teacher_input_ids" in samp:
            for j in range(self.num_choices):
                t_input_ids = np.array(samp["teacher_input_ids"][j])
                t_input_len = min(len(t_input_ids), self.max_length)
                model_data["teacher_input_ids"][i, j, :t_input_len] = torch.tensor(t_input_ids[:t_input_len], dtype=torch.long)
                model_data["teacher_attention_mask"][i, j, :t_input_len] = torch.tensor(samp["teacher_attention_mask"][j][:t_input_len], dtype=torch.long)

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length
        num_choices = self.num_choices

        student_pad_token_id = self.student_tokenizer.pad_token_id or 0
        
        # Shape: (batch_size, num_choices, max_length)
        model_data = {
            "input_ids": torch.ones(bs, num_choices, max_length, dtype=torch.long) * student_pad_token_id,
            "attention_mask": torch.zeros(bs, num_choices, max_length, dtype=torch.long),
        }
        
        output_data = {
            "labels": torch.zeros(bs, dtype=torch.long)  # Correct choice index (0-7)
        }

        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id or 0
            model_data.update({
                "teacher_input_ids": torch.ones(bs, num_choices, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, num_choices, max_length, dtype=torch.long),
            })

        for i, samp in enumerate(samples):
            self._process_multiple_choice(i, samp, model_data, output_data)
        
        return model_data, output_data
