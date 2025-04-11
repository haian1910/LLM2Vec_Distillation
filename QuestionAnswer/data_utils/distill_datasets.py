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
        self.max_length = args.max_length or 256
        self.num_choices = args.num_labels  # e.g., 8 choices

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
            log_rank(f"Columns in {path}: {df.columns.tolist()}")
            if not all(col in df.columns for col in ['question', 'choices', 'answerKey']):
                raise ValueError(f"CSV file {path} must contain 'question', 'choices', and 'answerKey' columns")

            log_rank("Processing dataset for multiple-choice question answering...")

            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                question = row['question']
                # Parse choices (assuming choices is a string like "array([{'text': 'choice1', 'label': 'A'}, ...])")
                try:
                    # Simplified parsing; adjust based on actual 'choices' format
                    choices_str = row['choices']
                    text_start = choices_str.find("'text':") + 7
                    text_end = choices_str.find("'label':")
                    text_content = choices_str[text_start:text_end].strip().strip("array([").strip("],").strip()
                    choices = [s.strip().strip("'\"") for s in text_content.split(",") if s.strip()]
                    if len(choices) != self.num_choices:
                        log_rank(f"Warning: Expected {self.num_choices} choices, got {len(choices)} for question: {question}")
                        continue
                except Exception as e:
                    log_rank(f"Error parsing choices for question {question}: {e}")
                    continue

                # Map answerKey to index (e.g., 'A' -> 0, 'B' -> 1, etc.)
                answer_key = row['answerKey']
                label = ord(answer_key) - ord('A') if answer_key and answer_key.isalpha() else -1
                if label < 0 or label >= self.num_choices:
                    log_rank(f"Invalid answerKey {answer_key} for question: {question}")
                    continue

                # Tokenize each question-choice pair for student
                student_input_ids_list = []
                for choice in choices:
                    input_text = f"{question} {choice}"
                    student_input_ids = self.student_tokenizer.encode(
                        input_text,
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    student_input_ids_list.append(student_input_ids)

                tokenized_data = {
                    "student_input_ids": student_input_ids_list,  # List of num_choices token sequences
                    "label": int(label)
                }

                # Tokenize for teacher if provided
                if self.teacher_tokenizer:
                    teacher_input_ids_list = []
                    for choice in choices:
                        input_text = f"{question} {choice}"
                        teacher_input_ids = self.teacher_tokenizer.encode(
                            input_text,
                            add_special_tokens=True,
                            max_length=self.max_length,
                            truncation=True
                        )
                        teacher_input_ids_list.append(teacher_input_ids)
                    tokenized_data["teacher_input_ids"] = teacher_input_ids_list

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")

    def _process_multiple_choice(
        self, i, samp, model_data, no_model_data
    ):
        # Process student inputs
        for j, input_ids in enumerate(samp["student_input_ids"]):
            input_len = min(len(input_ids), self.max_length)
            model_data["input_ids"][i, j, :input_len] = torch.tensor(input_ids[:input_len], dtype=torch.long)
            model_data["attention_mask"][i, j, :input_len] = 1.0

        no_model_data["labels"][i] = torch.tensor(samp["label"], dtype=torch.long)

        # Process teacher inputs if available
        if "teacher_input_ids" in samp:
            for j, t_input_ids in enumerate(samp["teacher_input_ids"]):
                t_input_len = min(len(t_input_ids), self.max_length)
                model_data["teacher_input_ids"][i, j, :t_input_len] = torch.tensor(t_input_ids[:t_input_len], dtype=torch.long)
                model_data["teacher_attention_mask"][i, j, :t_input_len] = 1.0

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

        model_data = {
            "input_ids": torch.ones(bs, num_choices, max_length, dtype=torch.long) * student_pad_token_id,
            "attention_mask": torch.zeros(bs, num_choices, max_length),
        }

        output_data = {
            "labels": torch.zeros(bs, dtype=torch.long)
        }

        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id or 0
            model_data.update({
                "teacher_input_ids": torch.ones(bs, num_choices, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, num_choices, max_length),
            })

        for i, samp in enumerate(samples):
            self._process_multiple_choice(i, samp, model_data, output_data)

        return model_data, output_data
