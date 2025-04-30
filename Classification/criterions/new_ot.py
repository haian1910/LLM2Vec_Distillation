import torch
from .cross_entropy_loss import CrossEntropyLoss
import torch.nn as nn
import math
import editdistance
from transformers import AutoTokenizer, AutoConfig, AutoModel
import re

class NEW_OT(CrossEntropyLoss):
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
                output_hidden_states=True
            )
        
        # Compute distillation loss using optimal transport
        kd_loss, log = self.compute_ot_loss(
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
        sim_mt = torch.mm(a_norm.transpose(0, 1), b_norm)
        
        sim_mt = 1 - sim_mt
        return sim_mt

    def pairwise_attention_distance(self, x, y, eps=1e-8):
        d = x.shape[1]
        sim_mt = torch.mm(x.transpose(0, 1), y) / math.sqrt(d)
        attention_weights = torch.softmax(sim_mt, dim=1)
        dist_mt = 1.0 - attention_weights
        return dist_mt
    def compute_token_importance(self, attention_weights, tokens):
        """
        Compute importance scores for tokens based on attention weights
        attention_weights: [num_heads, seq_len, seq_len] tensor
        """
        # Average attention across heads: [seq_len, seq_len]
        avg_attention = attention_weights.mean(dim=0)
        
        # Sum attention that each token receives: [seq_len]
        token_importance = avg_attention.sum(dim=0)
        
        # Normalize importance scores
        norm_importance = torch.softmax(token_importance, dim=0)
        
        return norm_importance
    
    def find_best_mapping(x, base_tokens, blending_special, base_special, best_one=True):
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

    # Hàm ánh xạ token song hướng giữa teacher và student
    def align_tokens(self, teacher_tokens, student_tokens, teacher_special="<s>", student_special="[CLS]"):
        """
        Create token mapping between teacher and student tokenizers
        """
        # Create mapping dictionary
        teacher_to_student = {}
        for t in teacher_tokens:
            tmp_t = t.replace(teacher_special, student_special)
            best_s = None
            best_dist = float('inf')

            for s in student_tokens:
                d = editdistance.eval(tmp_t, s)
                if d < best_dist:
                    best_s = s
                    best_dist = d
                    
            teacher_to_student[t] = best_s

        return teacher_to_student
    
    def project_importance(self, teacher_importance, teacher_tokens, student_tokens, mapping):
        """
        Project token importance from teacher to student based on token mapping
        """
        device = teacher_importance.device
        student_importance = torch.zeros(len(student_tokens), device=device)
        
        # Keep track of mapped student tokens
        mapped_student_indices = set()
        
        # Project importance scores
        for t_idx, t in enumerate(teacher_tokens):
            if t in mapping:
                s = mapping[t]
                s_idx = student_tokens.index(s)
                student_importance[s_idx] = teacher_importance[t_idx]
                mapped_student_indices.add(s_idx)
        
        # Find minimum importance score from teacher
        min_importance = teacher_importance.min().item()
        
        # Assign minimum importance to unmapped student tokens
        for s_idx in range(len(student_tokens)):
            if s_idx not in mapped_student_indices:
                student_importance[s_idx] = min_importance
        
        # Re-normalize student importance
        student_importance = torch.softmax(student_importance, dim=0)
        
        return student_importance
    def compute_ot_loss(
        self, input_data, outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log, distiller, logits=False
    ):
        # Get the last hidden state from both models
        student_features = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        teacher_features = teacher_outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        
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
            teacher_tokens = tokenizer_teacher.convert_ids_to_tokens(input_data["teacher_input_ids"][b])
            student_tokens = tokenizer_student.convert_ids_to_tokens(input_data["input_ids"][b])
            # Get sequences for current batch
            teacher_seq = teacher_features[b]  # Shape: (seq_len, hidden_dim)
            student_seq = student_features[b]  # Shape: (seq_len, hidden_dim)

            # Get masks for current batch
            teacher_mask = attention_mask_teacher[b]  # (seq_len)
            student_mask = attention_mask_student[b]  # (seq_len)
            
            # Prune sequences based on the mask
            teacher_seq = teacher_seq[teacher_mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            student_seq = student_seq[student_mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            
            # Project each row of teacher_seq to student space
            projected_teacher_seq = projector(teacher_seq)  # Now project after pruning
            
            # Ensure both tensors are in the same dtype
            dtype = student_seq.dtype
            projected_teacher_seq = projected_teacher_seq.to(dtype)
            teacher_attention = teacher_outputs.attentions[-1]
            
            # Compute token importance from teacher attention
            teacher_importance = self.compute_token_importance(teacher_attention, teacher_tokens)
            
            # Create token mapping between teacher and student
            token_mapping = self.align_tokens(teacher_tokens, student_tokens, 
                                            teacher_special, student_special)
            
            # Project importance from teacher to student
            student_importance = self.project_importance(teacher_importance, 
                                                        teacher_tokens, 
                                                        student_tokens, 
                                                        token_mapping)
            
            tea_mass = teacher_importance.unsqueeze(1)
            stu_mass = student_importance.unsqueeze(1)
            # Compute cost matrix based on specified distance metric
            if self.ot_dist_type == 'euclidean':
                cost_matrix = self.pairwise_euclidean_distance(student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'cosine':
                cost_matrix = self.pairwise_cosine_distance(student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'attention':
                cost_matrix = self.pairwise_attention_distance(student_seq, projected_teacher_seq)
            else:
                raise ValueError(f"Unknown distance type: {self.ot_dist_type}")
            
            # Ensure cost matrix is in the right dtype
            cost_matrix = cost_matrix.to(dtype)
            
            # Compute OT plan and loss
            ot_loss, transport_plan = self.sinkhorn(cost_matrix, stu_mass, tea_mass)
            total_loss += ot_loss
        
        avg_loss = total_loss / batch_size
        log["ot_loss"] = avg_loss.item()
        
        return avg_loss, log
    
    def sinkhorn(self, cost_matrix, stu_mass, tea_mass, num_iters=None):
        """
        Sinkhorn algorithm for computing optimal transport
        
        Args:
            cost_matrix: Cost matrix of shape (m, n)
            num_iters: Number of iterations (uses self.OT_max_iter if None)
            
        Returns:
            ot_loss: Optimal transport loss
            transport_plan: Transport plan matrix of shape (m, n)
        """
        if num_iters is None:
            num_iters = self.OT_max_iter
        
        m, n = cost_matrix.shape
        dtype = cost_matrix.dtype
        device = cost_matrix.device
        
        # Initialize uniform marginals - ensure correct dtype
        a = stu_mass
        b = tea_mass
        
        # Initialize transport plan - ensure correct dtype
        K = torch.exp(-cost_matrix / self.sinkhorn_alpha)
        
        # Initialize u with correct dtype
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        # Sinkhorn iterations
        for _ in range(num_iters):
            u_prev = u.clone()
            
            # Use a more stable implementation without walrus operator
            # (original line that caused error):
            # u = a / (torch.matmul(K, v := b / (torch.matmul(K.t(), u) + self.epsilon)) + self.epsilon)
            
            # First compute v
            v = b / (torch.matmul(K.t(), u) + self.epsilon)
            # Then compute u
            u = a / (torch.matmul(K, v) + self.epsilon)
            
            # Check convergence
            err = torch.norm(u - u_prev, p=float('inf'))
            if err < self.stopThr:
                break
        
        # Compute transport plan
        # Create diagonal matrices manually to ensure correct dtype
        diag_u = torch.diag(u)
        diag_v = torch.diag(v)
        transport_plan = torch.matmul(torch.matmul(diag_u, K), diag_v)
        
        # Compute OT loss
        ot_loss = torch.sum(transport_plan * cost_matrix)
        
        return ot_loss, transport_plan
