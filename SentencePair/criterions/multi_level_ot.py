import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoTokenizer
import re
from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence


def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

def normalize(value):
    """Safely normalize tensor values to prevent NaN issues"""
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Replace any existing NaN or Inf values
    value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Calculate mean and std with more numerical stability
    means = value.mean(dim=-1, keepdim=True)
    
    # Calculate std with epsilon for numerical stability
    var = torch.var(value, dim=-1, keepdim=True, unbiased=False) + epsilon
    stds = torch.sqrt(var)
    
    # Normalize with a minimum standard deviation threshold
    z_score_normalized = (value - means) / (stds + epsilon)
    
    # Clip values to prevent extreme outliers
    z_score_normalized = torch.clamp(z_score_normalized, min=-10.0, max=10.0)
    
    # Final check for NaN values
    return torch.nan_to_num(z_score_normalized, nan=0.0, posinf=10.0, neginf=-10.0)

def KL_wo(y_s, y_t,T=1):
    p_s = F.log_softmax(y_s/T, dim=-1)
    p_t = F.softmax(y_t/T, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = 2   
        
    def sinkhorn_normalized(self, x, n_iters=20):
        """Apply Sinkhorn normalization with safety checks for NaN values"""
        # Replace any NaN or Inf values
        x = torch.nan_to_num(x, nan=1e-8, posinf=1e6, neginf=-1e6)
        
        # Add small epsilon to all elements to avoid zeros
        x = x + 1e-8
        
        # Sinkhorn iterations with safety checks
        for _ in range(n_iters):
            # Row normalization
            row_sums = torch.sum(x, dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-8)  # Prevent division by zero
            x = x / row_sums
            
            # Check for NaN and fix
            x = torch.nan_to_num(x, nan=1e-8)
            
            # Column normalization
            col_sums = torch.sum(x, dim=0, keepdim=True)
            col_sums = torch.clamp(col_sums, min=1e-8)  # Prevent division by zero
            x = x / col_sums
            
            # Check for NaN and fix
            x = torch.nan_to_num(x, nan=1e-8)
            
        return x

    def manual_cdist(self, x, y, p=1):
        """Manual implementation of cdist that supports BFloat16 tensors"""
        n = x.size(0)
        m = y.size(0)
        
        # Convert inputs to float32 for computation
        x_float = x.to(torch.float32)
        y_float = y.to(torch.float32)
        
        # Compute pairwise distances
        result = torch.zeros(n, m, device=x.device, dtype=torch.float32)
        for i in range(n):
            for j in range(m):
                # L1 distance (p=1)
                result[i, j] = torch.sum(torch.abs(x_float[i] - y_float[j]))
        
        # Convert back to original dtype
        return result.to(x.dtype)

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=10):
        """Compute Sinkhorn loss with safety checks, handling BFloat16"""
        try:
            # Convert to float32 for processing if needed
            orig_dtype = x.dtype
            if orig_dtype == torch.bfloat16:
                x = x.to(torch.float32)
                y = y.to(torch.float32)
                
            # Check inputs for NaN values
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Input x to sinkhorn_loss contains NaN or Inf values")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
                
            if torch.isnan(y).any() or torch.isinf(y).any():
                print("Input y to sinkhorn_loss contains NaN or Inf values")
                y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Compute distance matrix - use manual implementation for BFloat16
            if orig_dtype == torch.bfloat16:
                Wxy = self.manual_cdist(x, y, p=1)
            else:
                # Try to use torch.cdist
                try:
                    Wxy = torch.cdist(x, y, p=1)
                except RuntimeError as e:
                    # Fallback to manual implementation if cdist fails
                    print(f"Falling back to manual cdist implementation: {e}")
                    Wxy = self.manual_cdist(x, y, p=1)
            
            # Check distance matrix for extreme values
            if torch.isnan(Wxy).any() or torch.isinf(Wxy).any():
                print("Distance matrix contains NaN or Inf values")
                Wxy = torch.nan_to_num(Wxy, nan=0.0, posinf=1e6, neginf=-1e6)
                
            # Compute kernel with numerical stability
            # Clip distances to prevent exp overflow
            Wxy_clipped = torch.clamp(Wxy, max=10.0)
            K = torch.exp(-Wxy_clipped / max(epsilon, 1e-5))
            
            # Check kernel for NaN values
            if torch.isnan(K).any() or torch.isinf(K).any():
                print("Kernel matrix contains NaN or Inf values")
                K = torch.nan_to_num(K, nan=1e-8, posinf=1.0, neginf=0.0)
            
            # Apply Sinkhorn normalization
            P = self.sinkhorn_normalized(K, n_iters)
            
            # Compute loss with safety check
            loss = torch.sum(P * Wxy)
            
            # Final safety check
            if torch.isnan(loss) or torch.isinf(loss):
                print("Final Sinkhorn loss is NaN or Inf")
                return torch.tensor(0.0, device=x.device)
                
            # Convert back to original dtype
            return loss.to(orig_dtype)
            
        except Exception as e:
            print(f"Error in sinkhorn_loss: {e}")
            return torch.tensor(0.0, device=x.device)
            
    def forward(self, y_s, y_t):
        """Forward pass with safety handling"""
        try:
            # Check inputs
            if y_s.size(0) == 0 or y_t.size(0) == 0:
                return torch.tensor(0.0, device=y_s.device)
                
            softmax = nn.Softmax(dim=-1)
            
            # Apply softmax with temperature scaling
            p_s = softmax(y_s/self.T)
            p_t = softmax(y_t/self.T)
            
            # Check for NaN values after softmax
            p_s = torch.nan_to_num(p_s, nan=1e-8)
            p_t = torch.nan_to_num(p_t, nan=1e-8)
            
            # Print information about data type
            print(f"Sinkhorn processing tensors of dtype: {p_s.dtype}")
            
            emd_loss = 0
            valid_samples = 0
            
            for i in range(p_s.shape[0]):
                try:
                    sample_loss = self.sinkhorn_loss(x=p_s[i], y=p_t[i])
                    
                    # Skip if loss is NaN or Inf
                    if torch.isnan(sample_loss) or torch.isinf(sample_loss):
                        continue
                        
                    emd_loss += 0.001 * sample_loss
                    valid_samples += 1
                except Exception as e:
                    print(f"Error processing sample {i} in Sinkhorn forward: {e}")
                    
            # Return mean loss if there are valid samples, otherwise return zero
            if valid_samples > 0:
                return emd_loss / valid_samples
            else:
                return torch.tensor(0.0, device=y_s.device)
                
        except Exception as e:
            print(f"Error in Sinkhorn forward pass: {e}")
            return torch.tensor(0.0, device=y_s.device)

def greedy_algorithm_adjust_s(t, s):
    batch_size, T, k = t.shape
    _, n, _ = s.shape
    
    # Initialize the adjusted source tensor
    s_adjusted = torch.zeros_like(t)
    
    for b in range(batch_size):
        # Initialize set of available source indices for each batch
        available_indices = list(range(n))
        
        for i in range(T):
            C_min = float('inf')
            j_star = -1
            
            for j in available_indices:
                # Compute cost as the sum of absolute differences for each batch
                C = torch.sum(torch.abs(t[b,:,i] - s[b,:,j]))
                
                if C < C_min:
                    C_min = C
                    j_star = j
            
            # Assign the best matching source vector to the adjusted tensor
            s_adjusted[b,:,i] = s[b,:,j_star]
            
            # Remove the selected index from available indices
            available_indices.remove(j_star)

    return s_adjusted
    

class MULTI_LEVEL_OT(VariousDivergence):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
    
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
        self.distiller = distiller
        
        # Student forward pass
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        
        # Compute cross-entropy loss with ground-truth labels
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["labels"]
        )[0]

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute distillation loss
        kd_loss, log = self.compute_multi_level_ot_distillation_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("multi_level_ot_loss:", kd_loss)
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"]
        )
        log["accuracy"] = accuracy

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss , logging_output
    
    
    def __get_start_and_size_answers(self, answer_tensors):
        """
        Extract answer positions and sizes from tensors, even for 2D logits
        with no clear ignore index.
        """
        answers_index = []
        answers_size = []
        
        # Debug the shape of the input
        print(f"Answer tensors shape: {answer_tensors.shape if hasattr(answer_tensors, 'shape') else 'No shape'}")
        
        # Handle case where answer_tensors is a single tensor, not a list
        if isinstance(answer_tensors, torch.Tensor) and answer_tensors.dim() <= 2:
            # For logits that are [batch_size, vocab_size] or [batch_size, seq_len]
            # Just assume each example has a valid answer covering the entire sequence
            batch_size = answer_tensors.size(0)
            for i in range(batch_size):
                answers_index.append(0)  # Start at position 0
                if answer_tensors.dim() == 2:
                    answers_size.append(answer_tensors.size(1))  # Use entire sequence length
                else:
                    answers_size.append(1)  # Default to size 1 for unexpected dimensions
            return answers_index, answers_size
            
        # Standard processing for list of tensors or properly structured tensors
        for answer in answer_tensors:
            ignore_index = -100
            
            # Check if the tensor is 0-dimensional (scalar)
            if answer.dim() == 0:
                # Handle scalar tensor by assuming it's a valid answer of size 1
                answers_index.append(0)
                answers_size.append(1)
                continue
                
            # For tensors that don't have ignore tokens (like logits), treat the whole tensor as valid
            if not torch.any(answer.eq(ignore_index)):
                answers_index.append(0)  # Start at beginning
                answers_size.append(answer.numel())  # Count all elements
                continue
                
            # Standard processing for tensors with ignore tokens
            is_value = answer.eq(ignore_index)
            answers_size.append(answer.numel() - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() + 1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
                
        # Ensure we have at least some valid answers for processing
        if all(size == 0 for size in answers_size):
            print("Warning: No valid answers found in any tensor. Setting default values.")
            for i in range(len(answers_size)):
                answers_index[i] = 0
                answers_size[i] = 1  # Assume at least one token is valid
                
        return answers_index, answers_size

    def compute_multi_level_ot_distillation_loss(self, outputs, teacher_outputs, output_data, distiller, log):
        student = outputs.logits
        teacher = teacher_outputs.logits
        target = output_data["labels"]

        # Print tensor shapes and dtypes for debugging
        print(f"Student logits shape: {student.shape}, dtype: {student.dtype}, min: {student.min().item()}, max: {student.max().item()}")
        print(f"Teacher logits shape: {teacher.shape}, dtype: {teacher.dtype}, min: {teacher.min().item()}, max: {teacher.max().item()}")
        print(f"Target shape: {target.shape if hasattr(target, 'shape') else 'No shape'}")
        
        # Get answer first token and answer size
        student_answer_index, student_answer_size = self.__get_start_and_size_answers(target)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(target)
        
        print(f"Student answer sizes: {student_answer_size}")
        print(f"Teacher answer sizes: {teacher_answer_size}")
        
        # Ensure sizes are reasonable
        valid_samples = sum(1 for size in student_answer_size if size > 0)
        if valid_samples == 0:
            # Force valid samples for all batch entries
            student_answer_size = [1] * len(student_answer_index)
            teacher_answer_size = [1] * len(teacher_answer_index)
            print("Forcing valid answers for all samples")
        
        # Handle 2D case specifically - reshape to 3D if needed
        if student.dim() == 2:
            # If [batch_size, vocab_size] - reshape to [batch_size, 1, vocab_size]
            student = student.unsqueeze(1)
            print(f"Reshaped student to: {student.shape}")
        
        if teacher.dim() == 2:
            # If [batch_size, vocab_size] - reshape to [batch_size, 1, vocab_size]
            teacher = teacher.unsqueeze(1)
            print(f"Reshaped teacher to: {teacher.shape}")
            
        # Ensure 3D tensors for processing
        if student.dim() < 3:
            student = student.unsqueeze(0)  # Add batch dimension if missing
        if teacher.dim() < 3:
            teacher = teacher.unsqueeze(0)  # Add batch dimension if missing
        
        # Replace NaN values with zeros before normalization
        student = torch.nan_to_num(student, nan=0.0, posinf=1e6, neginf=-1e6)
        teacher = torch.nan_to_num(teacher, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize the tensors
        try:
            student = normalize(student)
            teacher = normalize(teacher)
        except Exception as e:
            print(f"Error during normalization: {e}")
            student = torch.nan_to_num(student, nan=0.0)
            teacher = torch.nan_to_num(teacher, nan=0.0)
        
        # For 2D logits case, we need special handling
        if student.size(1) == 1 or teacher.size(1) == 1:
            # For single token per batch, don't need complex processing
            # Just apply softmax directly
            batch_size = student.size(0)
            student_processed = []
            teacher_processed = []
            
            for i in range(batch_size):
                # Handle student softmax
                student_logits = student[i].squeeze(0)  # Remove seq_len dimension if it's 1
                student_soft = F.softmax(student_logits, dim=-1)
                student_processed.append(student_soft.unsqueeze(0))  # Add seq_len dimension back
                
                # Handle teacher softmax
                teacher_logits = teacher[i].squeeze(0)  # Remove seq_len dimension if it's 1
                teacher_soft = F.softmax(teacher_logits, dim=-1)
                teacher_processed.append(teacher_soft.unsqueeze(0))  # Add seq_len dimension back
            
            # Stack processed tensors
            student = torch.stack(student_processed, dim=0)
            teacher = torch.stack(teacher_processed, dim=0)
            
            print(f"After processing for single token case - Student: {student.shape}, Teacher: {teacher.shape}")
        else:
            # Standard processing for multi-token case
            student_processed = []
            teacher_processed = []
            
            for i in range(student.size(0)):
                shift = student_answer_index[i]
                size = student_answer_size[i]
                end_shift = min(shift + size, student.size(1))
                
                # Process student
                student_slice = student[i, shift:end_shift, :]
                student_soft = F.softmax(student_slice, dim=-1)
                padding = torch.zeros_like(student[i, :(student.size(1)-size), :])
                student_processed.append(torch.cat((student_soft, padding), dim=0))
                
                # Process teacher
                shift_t = teacher_answer_index[i]
                size_t = teacher_answer_size[i]
                end_shift_t = min(shift_t + size_t, teacher.size(1))
                
                teacher_slice = teacher[i, shift_t:end_shift_t, :]
                teacher_soft = F.softmax(teacher_slice, dim=-1)
                padding_t = torch.zeros_like(teacher[i, :(teacher.size(1)-size_t), :])
                teacher_processed.append(torch.cat((teacher_soft, padding_t), dim=0))
            
            # Stack processed tensors
            student = torch.stack(student_processed, dim=0)
            teacher = torch.stack(teacher_processed, dim=0)
        
        # Cut to max answer length
        max_length = max(max(student_answer_size), max(teacher_answer_size))
        max_length = min(max_length, student.size(1), teacher.size(1))
        
        student = student[:, :max_length, :]
        teacher = teacher[:, :max_length, :]
        
        # Sort values in descending order
        student = student.sort(dim=-1, descending=True).values
        teacher = teacher.sort(dim=-1, descending=True).values
        
        # Apply improved sort
        teacher = improved_sort(teacher)
        student = improved_sort(student)
        
        # Take top K values
        k = min(50, teacher.size(2), student.size(2))
        teacher = teacher[:,:,:k]
        student = student[:,:,:k]
        
        # Ensure equal vocab size for both tensors
        diff_size = student.size(2) - teacher.size(2)
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)
        
        # Calculate L1 loss
        distillation_loss = torch.zeros(student.size(0), device=student.device)
        for i in range(student.size(0)):
            size = min(student_answer_size[i], teacher_answer_size[i])
            size = max(1, size)  # Ensure at least 1 token is used
            size = min(size, student.size(1))  # Ensure size doesn't exceed tensor dimension
            
            # Calculate L1 loss between distributions
            distillation_loss[i] = torch.abs(student[i, :size] - teacher[i, :size]).sum(-1).mean(-1)
        
        # Calculate KL divergence (optional based on stability)
        try:
            kl_loss = KL_wo(teacher, student) * 0.1
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.0, device=student.device)
        except Exception as e:
            print(f"Error computing KL divergence: {e}")
            kl_loss = torch.tensor(0.0, device=student.device)
        
        # Calculate Sinkhorn loss
        try:
            sinkorn_loss_fn = Sinkhorn_seq()
            # Convert to float32 if necessary before passing to Sinkhorn
            if student.dtype == torch.bfloat16 or teacher.dtype == torch.bfloat16:
                print("Converting BFloat16 tensors to Float32 for Sinkhorn loss calculation")
            sinkhorn_loss = sinkorn_loss_fn(teacher, student) * 0.1
            if torch.isnan(sinkhorn_loss) or torch.isinf(sinkhorn_loss):
                sinkhorn_loss = torch.tensor(0.0, device=student.device)
        except Exception as e:
            print(f"Error computing Sinkhorn loss: {e}")
            sinkhorn_loss = torch.tensor(0.0, device=student.device)
        
        # Combine all losses
        mean_distillation_loss = distillation_loss.mean()
        multi_level_ot_loss = mean_distillation_loss + kl_loss + sinkhorn_loss
        
        # Safety check for final loss
        if torch.isnan(multi_level_ot_loss) or torch.isinf(multi_level_ot_loss):
            print("Final loss is NaN or Inf, returning zero")
            multi_level_ot_loss = torch.tensor(0.0, device=student.device)
        
        print(f"L_HAD: {mean_distillation_loss.item()}, L_SL: {kl_loss.item()}, L_SD: {sinkhorn_loss.item()}")
        print(f"Final multi_level_ot_loss: {multi_level_ot_loss.item()}")
        
        log["multi_level_ot_loss"] = multi_level_ot_loss
        return multi_level_ot_loss, log
