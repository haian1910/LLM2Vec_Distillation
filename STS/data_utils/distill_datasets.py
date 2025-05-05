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
            required_cols = ['sentence1', 'sentence2']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file {path} must contain 'sentence1' and 'sentence2' columns")
            
            # For STS, the score column could be named differently across datasets
            score_col = None
            possible_score_cols = ['score', 'similarity', 'label', 'labels']
            for col in possible_score_cols:
                if col in df.columns:
                    score_col = col
                    break
            
            if not score_col:
                raise ValueError(f"CSV file {path} must contain a score column (one of {possible_score_cols})")
            
            log_rank(f"Processing STS dataset with scores from column '{score_col}'...")
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                # Tokenize sentence pair as a single sequence for student (BERT-style)
                student_encoding = self.student_tokenizer(
                    row['sentence1'], 
                    row['sentence2'],
                    add_special_tokens=True,  # Adds [CLS] and [SEP]
                    max_length=self.max_length,
                    truncation=True,
                    padding=False  # Padding will be handled in collate
                )
                
                # Convert score to float (STS scores are typically between 0-5)
                score = float(row[score_col])
                
                # Normalize score if needed (e.g., if scores are on different scales)
                if hasattr(self.args, 'normalize_scores') and self.args.normalize_scores:
                    if hasattr(self.args, 'score_range') and self.args.score_range:
                        min_score, max_score = self.args.score_range
                        score = (score - min_score) / (max_score - min_score) * 5.0  # Scale to 0-5
                
                tokenized_data = {
                    "student_input_ids": student_encoding['input_ids'],
                    "student_attention_mask": student_encoding['attention_mask'],
                    "score": score
                }
                
                # Add token_type_ids if the model uses them (BERT does, but not all models)
                if 'token_type_ids' in student_encoding:
                    tokenized_data["student_token_type_ids"] = student_encoding['token_type_ids']
        
                # Tokenize for teacher if provided (also BERT-style)
                if self.teacher_tokenizer:
                    teacher_encoding = self.teacher_tokenizer(
                        row['sentence1'],
                        row['sentence2'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False
                    )
                    tokenized_data.update({
                        "teacher_input_ids": teacher_encoding['input_ids'],
                        "teacher_attention_mask": teacher_encoding['attention_mask'],
                    })
                    
                    if 'token_type_ids' in teacher_encoding:
                        tokenized_data["teacher_token_type_ids"] = teacher_encoding['token_type_ids']

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_sentence_pair(self, i, samp, model_data, output_data):
        # Process student input (combined sentence1 and sentence2)
        input_ids = np.array(samp["student_input_ids"])
        seq_len = len(input_ids)
        model_data["input_ids"][i][:seq_len] = torch.tensor(input_ids, dtype=torch.long)
        model_data["attention_mask"][i][:seq_len] = torch.tensor(samp["student_attention_mask"], dtype=torch.long)
        
        # Add token_type_ids if available
        if "student_token_type_ids" in samp:
            model_data["token_type_ids"][i][:seq_len] = torch.tensor(samp["student_token_type_ids"], dtype=torch.long)

        # Process score - using float for regression
        output_data["labels"][i] = torch.tensor(samp["score"], dtype=torch.float)

        # Process teacher data if available
        if "teacher_input_ids" in samp:
            t_input_ids = np.array(samp["teacher_input_ids"])
            t_seq_len = len(t_input_ids)
            model_data["teacher_input_ids"][i][:t_seq_len] = torch.tensor(t_input_ids, dtype=torch.long)
            model_data["teacher_attention_mask"][i][:t_seq_len] = torch.tensor(samp["teacher_attention_mask"], dtype=torch.long)
            
            if "teacher_token_type_ids" in samp:
                model_data["teacher_token_type_ids"][i][:t_seq_len] = torch.tensor(samp["teacher_token_type_ids"], dtype=torch.long)

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
        
        # Check if we need token_type_ids
        if "student_token_type_ids" in samples[0]:
            model_data["token_type_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
        
        # For STS, use float labels for regression
        output_data = {
            "labels": torch.zeros(bs, dtype=torch.float)
        }

        # Add teacher data if tokenizer is provided
        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id or 0
            model_data.update({
                "teacher_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
            })
            
            # Check if teacher model needs token_type_ids
            if "teacher_token_type_ids" in samples[0]:
                model_data["teacher_token_type_ids"] = torch.zeros(bs, max_length, dtype=torch.long)

        # Process each sample
        for i, samp in enumerate(samples):
            self._process_sentence_pair(i, samp, model_data, output_data)
        
        return model_data, output_data