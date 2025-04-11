import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from utils import log_rank
from typing import Dict, Optional, List
from transformers import AutoTokenizer
import ast
import json
import math

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
    
    def _parse_choices(self, choices_str):
        """Safely parse the choices string without using eval()"""
        try:
            # First try to parse as a dictionary using ast.literal_eval
            choices_dict = ast.literal_eval(choices_str)
            # If it contains numpy arrays, need to handle specially
            if isinstance(choices_dict, dict):
                result = {}
                for key, value in choices_dict.items():
                    if isinstance(value, str) and 'array(' in value:
                        # Extract array content
                        content = value.split('array(')[1].split(']')[0] + ']'
                        content = content.replace("'", '"')  # Convert single quotes to double quotes
                        try:
                            parsed_content = json.loads(content)
                            result[key] = parsed_content
                        except json.JSONDecodeError:
                            # Fall back to simple string splitting for array content
                            content = content.strip('[]()')
                            items = [item.strip(" '\"") for item in content.split(',')]
                            result[key] = items
                    else:
                        result[key] = value
                return result
            return choices_dict
        except (ValueError, SyntaxError):
            # If it fails, try simple parsing
            if 'text' in choices_str and 'label' in choices_str:
                # Try to extract text and label parts
                text_part = choices_str.split("'text': array(")[1].split("], dtype=object)")[0] + "]"
                label_part = choices_str.split("'label': array(")[1].split("], dtype=object)")[0] + "]"
                
                # Clean up and parse
                text_items = [item.strip(" '\"") for item in text_part.strip('[]').split(',')]
                label_items = [item.strip(" '\"") for item in label_part.strip('[]').split(',')]
                
                return {
                    'text': text_items,
                    'label': label_items
                }
            
            # If all else fails, return the original string
            return choices_str
    
    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.csv")

        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Verify that this is a multiple choice dataset
            if not all(col in df.columns for col in ['question', 'choices', 'answerKey']):
                raise ValueError(f"CSV file {path} must contain 'question', 'choices', and 'answerKey' columns")
                
            log_rank(f"Processing dataset for multiple choice task: {self.split}...")
            
            for idx, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                question = row['question']
                
                # Handle choices based on type
                if isinstance(row['choices'], str):
                    choices = self._parse_choices(row['choices'])
                else:
                    choices = row['choices']
                
                # Extract choice texts
                if isinstance(choices, dict) and 'text' in choices:
                    choice_texts = choices['text']
                    # Handle if it's a numpy array
                    if hasattr(choice_texts, 'tolist'):
                        choice_texts = choice_texts.tolist()
                else:
                    # Assume choices is directly a list of texts
                    choice_texts = choices
                
                answer_key = row['answerKey']
                
                # Skip rows with NaN answer keys
                if isinstance(answer_key, float) and math.isnan(answer_key):
                    log_rank(f"Warning: Skipping row {idx} due to NaN answer key")
                    continue
                
                # Find the index of the correct answer
                label_idx = 0  # Default to first answer if we can't determine
                try:
                    if isinstance(choices, dict) and 'label' in choices:
                        answer_labels = choices['label']
                        # Handle if it's a numpy array
                        if hasattr(answer_labels, 'tolist'):
                            answer_labels = answer_labels.tolist()
                        
                        # Try to find the answer key in labels
                        try:
                            label_idx = answer_labels.index(answer_key)
                        except (ValueError, TypeError):
                            # If the answer key is not found or it's not a valid type
                            if isinstance(answer_key, str) and len(answer_key) == 1:
                                # Try to use alphabetical index
                                label_idx = ord(answer_key.upper()) - ord('A')
                                if label_idx < 0 or label_idx >= len(choice_texts):
                                    label_idx = 0  # Fallback to first choice
                            else:
                                # Try to use as direct index if it's a number
                                try:
                                    if isinstance(answer_key, (int, float)) and not math.isnan(answer_key):
                                        label_idx = int(answer_key)
                                        if label_idx < 0 or label_idx >= len(choice_texts):
                                            label_idx = 0
                                except (ValueError, TypeError):
                                    label_idx = 0
                    else:
                        # Default to alphabetical index for A, B, C, D style answer keys
                        if isinstance(answer_key, str) and len(answer_key) == 1:
                            label_idx = ord(answer_key.upper()) - ord('A')
                            if label_idx < 0 or label_idx >= len(choice_texts):
                                label_idx = 0
                        else:
                            # Try to use as direct index
                            try:
                                if isinstance(answer_key, (int, float)) and not math.isnan(answer_key):
                                    label_idx = int(answer_key)
                                    if label_idx < 0 or label_idx >= len(choice_texts):
                                        label_idx = 0
                            except (ValueError, TypeError):
                                label_idx = 0
                
                except Exception as e:
                    log_rank(f"Warning: Error processing answer key for row {idx}: {e}")
                    label_idx = 0  # Default to first choice
                
                # Ensure label_idx is within bounds
                if label_idx < 0 or label_idx >= len(choice_texts):
                    label_idx = 0
                
                student_encoded_choices = []
                teacher_encoded_choices = [] if self.teacher_tokenizer else None
                
                for choice_text in choice_texts:
                    # Format as: question + choice
                    formatted_text = f"{question} {choice_text}"
                    
                    student_input_ids = self.student_tokenizer.encode(
                        formatted_text,
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    student_encoded_choices.append(student_input_ids)
                    
                    if self.teacher_tokenizer:
                        teacher_input_ids = self.teacher_tokenizer.encode(
                            formatted_text,
                            add_special_tokens=True,
                            max_length=self.max_length,
                            truncation=True
                        )
                        teacher_encoded_choices.append(teacher_input_ids)
                
                sample = {
                    "student_input_ids": student_encoded_choices,
                    "label": label_idx,
                    "num_choices": len(choice_texts)
                }
                
                if teacher_encoded_choices:
                    sample["teacher_input_ids"] = teacher_encoded_choices
                    
                dataset.append(sample)
            
            log_rank(f"Successfully processed {len(dataset)} samples for {self.split}")
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
    
    def _process_multiple_choice(self, i, samp, model_data, no_model_data):
        num_choices = samp["num_choices"]
        
        for choice_idx in range(num_choices):
            input_ids = np.array(samp["student_input_ids"][choice_idx])
            input_len = len(input_ids)
            
            model_data["input_ids"][i, choice_idx, :input_len] = torch.tensor(input_ids, dtype=torch.long)
            model_data["attention_mask"][i, choice_idx, :input_len] = 1.0
            
            if "teacher_input_ids" in samp:
                t_input_ids = np.array(samp["teacher_input_ids"][choice_idx])
                t_input_len = len(t_input_ids)
                model_data["teacher_input_ids"][i, choice_idx, :t_input_len] = torch.tensor(t_input_ids, dtype=torch.long)
                model_data["teacher_attention_mask"][i, choice_idx, :t_input_len] = 1.0
                
        no_model_data["labels"][i] = torch.tensor(samp["label"], dtype=torch.long)

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length
        
        # Get maximum number of choices
        max_choices = max(sample.get("num_choices", 4) for sample in samples)
        
        student_pad_token_id = self.student_tokenizer.pad_token_id
        if student_pad_token_id is None:
            student_pad_token_id = 0
        
        # For multiple choice, we need a 3D tensor [batch_size, num_choices, seq_length]
        model_data = {
            "input_ids": torch.ones(bs, max_choices, max_length, dtype=torch.long) * student_pad_token_id,
            "attention_mask": torch.zeros(bs, max_choices, max_length),
        }
        
        output_data = {
            "labels": torch.zeros(bs, dtype=torch.long)
        }
        
        if any("teacher_input_ids" in sample for sample in samples):
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id if self.teacher_tokenizer else 0
            if teacher_pad_token_id is None:
                teacher_pad_token_id = 0
            model_data.update({
                "teacher_input_ids": torch.ones(bs, max_choices, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, max_choices, max_length),
            })
            
        for i, samp in enumerate(samples):
            self._process_multiple_choice(i, samp, model_data, output_data)
        
        return model_data, output_data
