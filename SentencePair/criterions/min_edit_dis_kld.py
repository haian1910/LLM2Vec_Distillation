import logging
import torch
import torch.nn.functional as F
import numpy as np
import transformers
import editdistance
from typing import Dict, List
from .cross_entropy_loss import CrossEntropyLoss



class MinEditDisForwardKLD(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        # Simplified initialization for classification task
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        # Store token mapping references
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller=distiller 

        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers

        # Bản đồ token đặc biệt
        TOKENIZER_TO_SPECIAL_TOKEN = {
            type(tokenizer_teacher): "<s>",  # Token đặc biệt của teacher
            type(tokenizer_student): "[CLS]"   # Token đặc biệt của student
        }

        # Get student outputs
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        
        # Compute standard cross-entropy loss for classification
        log = {}
        
        # Compute cross-entropy loss with ground-truth labels
        loss_ce = self.compute_cross_entropy_loss(
            outputs.logits, output_data["labels"]
        )[0]
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
            
        # Get aligned teacher logits
        teacher_logits = self.get_aligned_teacher_logits(
            logits, 
            teacher_outputs.logits, 
            input_data,
            distiller
        )
        
        # Compute KL divergence loss
        kd_loss = self.compute_forward_kl_divergence(logits, teacher_logits)
        print("min_ed_loss:", kd_loss)
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * kd_loss
        
        # Calculate accuracy 
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"]
        )
        log["accuracy"] = accuracy
        
        # Create or update logging output to match the expected return format
        
        return loss, logging_output

    def get_aligned_teacher_logits(
        self, logits, teacher_logits, input_data, distiller,
    ):
        # For classification, we just need to align the CLS token logits
        # or the pooled representation logits
        
        stu_tokenizer = distiller.student_tokenizer
        tea_tokenizer = distiller.teacher_tokenizers
        
        # Initialize token mappings if available in distiller
        if hasattr(distiller, 'tea2stu_id_mapping'):
            tea2stu_id_mapping = distiller.tea2stu_id_mapping
        else:
            tea2stu_id_mapping = None
            
        if hasattr(distiller, 'stu2tea_id_mapping_tea'):
            stu2tea_id_mapping_tea = distiller.stu2tea_id_mapping_tea
        else:
            stu2tea_id_mapping_tea = None
            
        if hasattr(distiller, 'stu2tea_id_mapping_stu'):
            stu2tea_id_mapping_stu = distiller.stu2tea_id_mapping_stu
        else:
            stu2tea_id_mapping_stu = None
        
        # Get input token sequences
        bsz = input_data["input_ids"].shape[0]
        aligned_tea_logits = []
        
        for i in range(bsz):
            # Get student tokens
            stu_input_ids = input_data["input_ids"][i]
            
            # Get teacher tokens
            tea_input_ids = input_data["teacher_input_ids"][i]
            
            # Extract logits - for classification tasks, we just need the final/CLS token
            # For BERT-like models, typically the first position [CLS] token is used
            # For decoder-only models, typically the last token is used
            # This may need adjustment based on your specific model architecture
            
            # For simplicity, we'll align all token representations and then use the appropriate ones
            aligned_logits = self.transform_step_logits_fast(
                stu_tokenizer,
                tea_tokenizer,
                stu_input_ids,
                logits[i],
                tea_input_ids,
                teacher_logits[i],
                blending_to_base_mapping=tea2stu_id_mapping,
                base_to_blending_mapping_blending_ids=stu2tea_id_mapping_tea,
                base_to_blending_mapping_base_ids=stu2tea_id_mapping_stu
            )
            
            aligned_tea_logits.append(aligned_logits)
        
        aligned_tea_logits = torch.stack(aligned_tea_logits, 0)
        return aligned_tea_logits

    def transform_step_logits_fast(
        self,
        base_model_tokenizer,
        blending_model_tokenizer,
        base_model_input_ids,
        base_model_logits,
        blending_model_input_ids,
        blending_model_logits,
        blending_to_base_mapping=None,
        base_to_blending_mapping_blending_ids=None,
        base_to_blending_mapping_base_ids=None,
    ):
        """Faster implementation to align logits for classification"""
        # Convert tensor to list for tokenizer conversion if needed
        if isinstance(base_model_input_ids, torch.Tensor):
            base_model_input_ids = base_model_input_ids.cpu().tolist()
        if isinstance(blending_model_input_ids, torch.Tensor):
            blending_model_input_ids = blending_model_input_ids.cpu().tolist()
            
        # Filter out padding tokens
        base_model_input_ids = [id for id in base_model_input_ids if id != base_model_tokenizer.pad_token_id]
        blending_model_input_ids = [id for id in blending_model_input_ids if id != blending_model_tokenizer.pad_token_id]
        
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        
        # Get special tokens for tokenizers
        base_model_special_token = "[CLS]"
        
        blending_model_special_token = "<s>"
        

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        # Use dynamic time warping to align tokens
        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        ) 
        
        # For classification tasks, we primarily care about the CLS token or pooled representation
        device = base_model_logits.device
        
        # For one-to-one mapping, align their logits
        unalign_mask = [1 if len(a) == 1 else 0 for a in base_to_blending]
        unalign_mask = torch.tensor(unalign_mask).to(device)

        base_to_blending = [a[0] if len(a) == 1 else 0 for a in base_to_blending]
        base_to_blending = torch.LongTensor(base_to_blending).to(device)
        
        # Get aligned logits
        # For classification models like BERT, we typically use the first token ([CLS]) logits
        # For transformer models, we often use the last token logits
        if isinstance(base_model_tokenizer, (transformers.BertTokenizer, transformers.BertTokenizerFast)):
            # For BERT-like models, the classification representation is at the [CLS] token (index 0)
            aligned_blending_logits = blending_model_logits
        else:
            # For autoregressive models like LLama, use the last token's representation
            aligned_blending_logits = blending_model_logits
        
        return aligned_blending_logits

    def compute_forward_kl_divergence(self, logits, teacher_logits):
        """
        Compute KL divergence loss for classification
        For classification, we primarily care about the final output logits
        """
        # Apply temperature scaling
        temp_scaled_logits = logits / self.kd_temp
        temp_scaled_teacher_logits = teacher_logits / self.kd_temp
        
        # KL divergence
        kd_loss = F.kl_div(
            F.log_softmax(temp_scaled_logits, dim=-1),
            F.softmax(temp_scaled_teacher_logits, dim=-1),
            reduction="batchmean"
        ) * (self.kd_temp ** 2)
        
        return kd_loss

    def dtw(self, series_1, series_2, norm_func=np.linalg.norm):
        """
        Use dynamic time warping to align two tokenizers
        Modified from: https://github.com/talcs/simpledtw/blob/master/simpledtw.py
        """
        # Create cost matrix
        matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        
        # Fill cost matrix
        for i, vec1 in enumerate(series_1):
            for j, vec2 in enumerate(series_2):
                cost = norm_func(vec1, vec2)
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                )
        
        # Backtrack to find optimal alignment
        matrix = matrix[1:, 1:]
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        matches = []
        mappings_series_1 = [list() for v in range(matrix.shape[0])]
        mappings_series_2 = [list() for v in range(matrix.shape[1])]
        
        while i > 0 or j > 0:
            matches.append((i, j))
            mappings_series_1[i].append(j)
            mappings_series_2[j].append(i)
            option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
            option_up = matrix[i - 1, j] if i > 0 else np.inf
            option_left = matrix[i, j - 1] if j > 0 else np.inf
            move = np.argmin([option_diag, option_up, option_left])
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        
        matches.append((0, 0))
        mappings_series_1[0].append(0)
        mappings_series_2[0].append(0)
        matches.reverse()
        
        for mp in mappings_series_1:
            mp.reverse()
        for mp in mappings_series_2:
            mp.reverse()

        return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
