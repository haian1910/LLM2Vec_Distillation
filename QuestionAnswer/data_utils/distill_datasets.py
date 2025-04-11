import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from QuestionAnswer.utils import log_rank
from transformers import BertTokenizer
from datasets import Dataset

class DistillDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        student_tokenizer=None,
        teacher_tokenizer=None,
        num_choices: int = 8,
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = args.max_length or 256
        self.num_choices = num_choices

        self.dataset = self._load_and_process_data()

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        path = os.path.join(self.args.data_dir, f"{self.split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file named {path}")

        # Load CSV into a Hugging Face Dataset
        df = pd.read_csv(path)
        log_rank(f"Columns in {path}: {df.columns.tolist()}")
        if not all(col in df.columns for col in ['question', 'choices', 'answerKey']):
            raise ValueError(f"CSV file {path} must contain 'question', 'choices', and 'answerKey' columns")
        
        dataset = Dataset.from_pandas(df)

        # Map answerKey to index
        def map_answer_to_index(example):
            answer_key = example['answerKey']
            index = ord(answer_key) - ord('A') if answer_key else -1  # A->0, B->1, ..., H->7, or -1 if empty
            return {'label': index}

        dataset = dataset.map(map_answer_to_index)

        # Preprocess and tokenize
        def preprocess_function(examples):
            questions = examples['question']
            choices_str = examples['choices']

            # Extract 'text' list from choices string
            choices = []
            for choice_str in choices_str:
                # Simple string parsing: extract content after 'text': and before 'label'
                text_start = choice_str.find("'text':") + 7
                text_end = choice_str.find("'label':")
                text_content = choice_str[text_start:text_end].strip().strip("array([").strip("],").strip()
                choice_texts = [s.strip().strip("'\"") for s in text_content.split(",") if s.strip()]
                choices.append(choice_texts)

            # Repeat question for each of the 8 choices
            questions = [[q] * self.num_choices for q in questions]
            questions_flat = [item for sublist in questions for item in sublist]
            choices_flat = [item for sublist in choices for item in sublist]

            # Tokenize
            inputs = self.student_tokenizer(
                questions_flat,
                choices_flat,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Reshape to batch_size x num_choices
            input_ids = [inputs['input_ids'][i:i+8].tolist() for i in range(0, len(inputs['input_ids']), 8)]
            attention_mask = [inputs['attention_mask'][i:i+8].tolist() for i in range(0, len(inputs['attention_mask']), 8)]
            token_type_ids = [inputs['token_type_ids'][i:i+8].tolist() for i in range(0, len(inputs['token_type_ids']), 8)]

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': examples['label']
            }

        dataset = dataset.map(preprocess_function, batched=True)
        
        # Convert to list of dicts for PyTorch Dataset
        return dataset

    def collate(self, samples):
        # Convert list of dicts to tensors
        input_ids = torch.tensor([sample['input_ids'] for sample in samples], dtype=torch.long)
        attention_mask = torch.tensor([sample['attention_mask'] for sample in samples], dtype=torch.long)
        token_type_ids = torch.tensor([sample['token_type_ids'] for sample in samples], dtype=torch.long)
        labels = torch.tensor([sample['labels'] for sample in samples], dtype=torch.long)

        model_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        output_data = {"labels": labels}

        if self.teacher_tokenizer:
            # Add teacher tokenization if needed (simplified here)
            teacher_inputs = self.teacher_tokenizer(
                [f"{sample['question']} {choice}" for sample in samples for choice in sample['choices']['text']],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            teacher_input_ids = [teacher_inputs['input_ids'][i:i+8].tolist() for i in range(0, len(teacher_inputs['input_ids']), 8)]
            teacher_attention_mask = [teacher_inputs['attention_mask'][i:i+8].tolist() for i in range(0, len(teacher_inputs['attention_mask']), 8)]
            model_data["teacher_input_ids"] = torch.tensor(teacher_input_ids, dtype=torch.long)
            model_data["teacher_attention_mask"] = torch.tensor(teacher_attention_mask, dtype=torch.long)

        return model_data, output_data
