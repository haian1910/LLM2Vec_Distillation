import torch
from .cross_entropy_loss import CrossEntropyLoss
import torch.nn as nn
import math
import editdistance
from transformers import AutoTokenizer, AutoConfig, AutoModel
import re

class OT_PRO(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.sinkhorn_alpha = 0.1
        self.stopThr = 1e-9
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'attention'
        self.importance_scaling = 0.5
    
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
        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers

        # Bản đồ token đặc biệt
        TOKENIZER_TO_SPECIAL_TOKEN = {
            type(tokenizer_teacher): "<s>",  # Token đặc biệt của teacher
            type(tokenizer_student): "[CLS]"   # Token đặc biệt của student
        }
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
                output_hidden_states=True,
                output_attentions=True
            )
        
        # Compute distillation loss using optimal transport
        kd_loss, log = self.compute_ot_loss(
            input_data=input_data,
            outputs=outputs, 
            teacher_outputs=teacher_outputs, 
            attention_mask_student=input_data["attention_mask"],
            attention_mask_teacher=input_data["teacher_attention_mask"],
            log=log,
            distiller=distiller
        )
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"]
        )
        log["accuracy"] = accuracy

        # # Update logging output
        # logging_output = self.record_logging_output(
        #     logging_output, batch_denom, log
        # )
        return loss, logging_output
    
    def pairwise_euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2)
    
    def pairwise_cosine_distance(self, a, b, eps=1e-8):
        """
        Computes pairwise cosine distance with numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=a.dtype))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=b.dtype))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        
        sim_mt = 1 - sim_mt
        return sim_mt

    def pairwise_attention_distance(self, x, y, eps=1e-8):
        d = x.shape[1]
        sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
        attention_weights = torch.softmax(sim_mt, dim=1)
        dist_mt = 1.0 - attention_weights
        return dist_mt
    
    def compute_token_importance(self, attention_weights, tokens):
        device = attention_weights.device
        
        # Check if attention_weights is 3D (with multiple heads) or 2D (single attention matrix)
        if len(attention_weights.shape) == 3:
            # Average attention across heads: [seq_len, seq_len]
            avg_attention = attention_weights.mean(dim=0)
        else:
            # Already a 2D attention matrix
            avg_attention = attention_weights
        
        # Ensure dimensions match
        seq_len = min(avg_attention.shape[0], len(tokens))
        
        # Truncate attention matrix if needed
        avg_attention = avg_attention[:seq_len, :seq_len]
        
        # Sum attention that each token receives: [seq_len]
        token_importance = avg_attention.sum(dim=0)
        
        # Normalize importance scores (add small epsilon to avoid division by zero)
        norm_importance = torch.softmax(token_importance, dim=0)
        
        return norm_importance
    
    def find_best_mapping(self, x, base_tokens, blending_special, base_special, best_one=True):
        tmp_x = x.replace(blending_special, base_special)
        if tmp_x in base_tokens:
            return tmp_x, tmp_x
        else:
            if best_one:
                best = None
                best_dist = None
                for y in base_tokens:
                    d = editdistance.eval(tmp_x, y)
                    if best is None or d < best_dist:
                        best = y
                        best_dist = d
                return tmp_x, best
            else:
                token_and_distance = [(y, editdistance.eval(tmp_x, y)) for y in base_tokens]
                min_distance = min(d for _, d in token_and_distance)
                shortest_distance_tokens = [y for y, d in token_and_distance if d == min_distance]
                return tmp_x, shortest_distance_tokens

    def align_tokens(self, teacher_tokens, student_tokens, teacher_special="<s>", student_special="[CLS]"):
        # Create mapping dictionary
        teacher_to_student = {}
        
        # Handle empty token lists
        if not teacher_tokens or not student_tokens:
            return teacher_to_student
        
        # Process special tokens mapping
        if teacher_special in teacher_tokens and student_special in student_tokens:
            teacher_to_student[teacher_special] = student_special
        
        # Create a set of student tokens for faster lookup
        student_token_set = set(student_tokens)
        
        for t in teacher_tokens:
            # Try direct replacement first
            tmp_t = t.replace(teacher_special, student_special)
            if tmp_t in student_token_set:
                teacher_to_student[t] = tmp_t
                continue
            
            # If direct replacement doesn't work, find closest match
            best_s = None
            best_dist = float('inf')
            
            for s in student_tokens:
                # Skip special tokens in this loop
                if s == student_special:
                    continue
                    
                # Calculate edit distance
                d = editdistance.eval(tmp_t, s)
                if d < best_dist:
                    best_s = s
                    best_dist = d
            
            # Only add mapping if we found a reasonable match
            if best_s is not None:
                teacher_to_student[t] = best_s
        
        return teacher_to_student
    
    def project_importance(self, teacher_importance, teacher_tokens, student_tokens, mapping):
        device = teacher_importance.device
        student_importance = torch.zeros(len(student_tokens), device=device)
        
        # Get valid teacher tokens based on attention mask
        valid_teacher_tokens = teacher_tokens[:teacher_importance.shape[0]]
        
        # Map valid tokens to importance scores
        teacher_token_to_importance = {token: score.item() for token, score in zip(valid_teacher_tokens, teacher_importance)}
        
        # Keep track of mapped student indices
        mapped_student_indices = set()
        
        # Project importance scores
        for t_idx, t in enumerate(valid_teacher_tokens):
            if t in mapping:
                s = mapping[t]
                # Find all occurrences of this student token
                s_indices = [i for i, token in enumerate(student_tokens) if token == s]
                for s_idx in s_indices:
                    if s_idx < len(student_importance):  # Ensure index is valid
                        student_importance[s_idx] = teacher_importance[t_idx]
                        mapped_student_indices.add(s_idx)
        
        # Find minimum importance score from teacher for unmapped tokens
        min_importance = teacher_importance.min().item() if len(teacher_importance) > 0 else 0.0
        
        # Assign minimum importance to unmapped student tokens
        for s_idx in range(len(student_tokens)):
            if s_idx not in mapped_student_indices and s_idx < len(student_importance):
                student_importance[s_idx] = min_importance
        
        # Re-normalize student importance (add small epsilon to avoid division by zero)
        student_importance = torch.softmax(student_importance, dim=0)
        
        return student_importance
    
    def compute_ot_loss(
        self, input_data, outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log, distiller, logits=False
    ):
        # Get the last hidden state from both models
        student_features = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        teacher_features = teacher_outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Extract and determine target dtype (float32 for highest precision)
        target_dtype = torch.float32  # Use a consistent dtype for all tensors
        
        tokenizer_teacher = distiller.teacher_tokenizers
        tokenizer_student = distiller.student_tokenizer
        batch_size = teacher_features.size(0)
        total_loss = 0
        
        # Check if projector exists
        if not hasattr(distiller, 'projectors') or 't2s' not in distiller.projectors:
            raise AttributeError("Distiller missing 't2s' projector. Make sure projectors are properly initialized.")
            
        projector = distiller.projectors["t2s"]
        teacher_special = "<s>"
        student_special = "[CLS]"
        
        for b in range(batch_size):
            # Get tokens for current batch
            teacher_input_ids = input_data["teacher_input_ids"][b]
            student_input_ids = input_data["input_ids"][b]
            
            # Truncate teacher input_ids to remove padding
            # Fix for the TypeError: Convert tensor to integer using int() before slicing
            valid_teacher_len = int(attention_mask_teacher[b].sum().item())
            valid_teacher_input_ids = teacher_input_ids[:valid_teacher_len]
            
            # Truncate student input_ids to remove padding
            # Fix for potential similar error in student input_ids
            valid_student_len = int(attention_mask_student[b].sum().item())
            valid_student_input_ids = student_input_ids[:valid_student_len]
            
            # Convert to tokens
            teacher_tokens = tokenizer_teacher.convert_ids_to_tokens(valid_teacher_input_ids)
            student_tokens = tokenizer_student.convert_ids_to_tokens(valid_student_input_ids)
            
            # Get sequences for current batch
            teacher_seq = teacher_features[b]  # Shape: (seq_len, hidden_dim)
            student_seq = student_features[b]  # Shape: (seq_len, hidden_dim)

            # Get masks for current batch
            teacher_mask = attention_mask_teacher[b]  # (seq_len)
            student_mask = attention_mask_student[b]  # (seq_len)
            
            # Prune sequences based on the mask
            valid_teacher_seq = teacher_seq[teacher_mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            valid_student_seq = student_seq[student_mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            
            # Project each row of teacher_seq to student space
            projected_teacher_seq = projector(valid_teacher_seq)  # Now project after pruning
            
            # Ensure both tensors are in the same dtype
            dtype = valid_student_seq.dtype
            projected_teacher_seq = projected_teacher_seq.to(dtype)
            
            # Process attention weights
            if hasattr(teacher_outputs, 'attentions') and teacher_outputs.attentions is not None:
                teacher_attention = teacher_outputs.attentions[-1][b]
                
                # Ensure teacher_attention has the right shape for current batch
                valid_teacher_attention = teacher_attention[:, :valid_teacher_len, :valid_teacher_len]
                
                # Compute token importance from teacher attention
                teacher_importance = self.compute_token_importance(valid_teacher_attention, teacher_tokens[:valid_teacher_len])
            else:
                # Fallback if attentions not available
                teacher_importance = torch.ones(len(teacher_tokens), device=teacher_seq.device)
                teacher_importance = torch.softmax(teacher_importance, dim=0)
            
            # Create token mapping between teacher and student
            token_mapping = self.align_tokens(teacher_tokens, student_tokens, 
                                          teacher_special, student_special)
            
            # Project importance from teacher to student
            student_importance = self.project_importance(teacher_importance, 
                                                      teacher_tokens, 
                                                      student_tokens, 
                                                      token_mapping)
            
            # Ensure importance vectors are reshaped properly for matrix multiplication
            tea_mass = teacher_importance.view(-1, 1)  # Column vector
            stu_mass = student_importance.view(-1, 1)  # Column vector
            
            # Ensure mass vectors match sequence lengths
            tea_mass = tea_mass[:valid_teacher_seq.size(0)]
            stu_mass = stu_mass[:valid_student_seq.size(0)]
            
            # Convert all tensors to the target dtype (float32) for computation safety
            valid_student_seq = valid_student_seq.to(torch.float32)
            projected_teacher_seq = projected_teacher_seq.to(torch.float32)
            
            # Compute cost matrix based on specified distance metric
            if self.ot_dist_type == 'euclidean':
                cost_matrix = self.pairwise_euclidean_distance(valid_student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'cosine':
                cost_matrix = self.pairwise_cosine_distance(valid_student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'attention':
                cost_matrix = self.pairwise_attention_distance(valid_student_seq, projected_teacher_seq)
            else:
                raise ValueError(f"Unknown distance type: {self.ot_dist_type}")
            
            # Ensure cost matrix is in float32 for numerical stability
            cost_matrix = cost_matrix.to(torch.float32)
            
            # Check dimensions
            if tea_mass.size(0) != cost_matrix.size(1) or stu_mass.size(0) != cost_matrix.size(0):
                # Reshape tea_mass and stu_mass to match cost_matrix
                tea_mass = torch.ones(cost_matrix.size(1), 1, device=cost_matrix.device, dtype=torch.float32) / cost_matrix.size(1)
                stu_mass = torch.ones(cost_matrix.size(0), 1, device=cost_matrix.device, dtype=torch.float32) / cost_matrix.size(0)
            else:
                # Convert existing mass vectors to float32
                tea_mass = tea_mass.to(torch.float32)
                stu_mass = stu_mass.to(torch.float32)
            
            # Compute OT plan and loss
            ot_loss, transport_plan = self.sinkhorn(cost_matrix, stu_mass, tea_mass)
            total_loss += ot_loss
        
        avg_loss = total_loss / batch_size
        log["ot_loss"] = avg_loss.item()
        
        return avg_loss, log
    
    def sinkhorn(self, cost_matrix, a, b, num_iters=None):
        if num_iters is None:
            num_iters = self.OT_max_iter
        
        m, n = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype
        
        # Handle edge cases where one of the dimensions is 0
        if m == 0 or n == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), torch.zeros((m, n), device=device, dtype=dtype)
        
        # Ensure a and b have the right shape and dtype
        if a.dim() == 1:
            a = a.view(-1, 1)
        if b.dim() == 1:
            b = b.view(-1, 1)
            
        # Convert all tensors to the same dtype (use cost_matrix's dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        
        # Ensure a and b have the correct length
        if a.shape[0] != m:
            a = torch.ones(m, 1, device=device, dtype=dtype) / m
        if b.shape[0] != n:
            b = torch.ones(n, 1, device=device, dtype=dtype) / n
        
        # Ensure the mass sums to 1
        if torch.sum(a) < self.epsilon or torch.sum(b) < self.epsilon:
            a = torch.ones(m, 1, device=device, dtype=dtype) / m
            b = torch.ones(n, 1, device=device, dtype=dtype) / n
        else:
            a = a / torch.sum(a)
            b = b / torch.sum(b)
        
        # Initialize K matrix (Gibbs kernel)
        K = torch.exp(-cost_matrix / self.sinkhorn_alpha)
        
        # Initialize dual variables with same dtype as cost_matrix
        u = torch.ones(m, 1, device=device, dtype=dtype)
        v = torch.ones(n, 1, device=device, dtype=dtype)
        
        # Sinkhorn iterations
        for _ in range(num_iters):
            u_prev = u.clone()
            
            # Update v = b / (K.T @ u)
            KTu = torch.matmul(K.t(), u)
            v = b / (KTu + self.epsilon)
            
            # Update u = a / (K @ v)
            Kv = torch.matmul(K, v)
            u = a / (Kv + self.epsilon)
            
            # Check convergence
            err = torch.norm(u - u_prev, p=float('inf'))
            if err < self.stopThr:
                break
        
        # Compute transport plan
        P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())
        
        # Compute OT loss
        ot_loss = torch.sum(P * cost_matrix)
        
        return ot_loss, P
