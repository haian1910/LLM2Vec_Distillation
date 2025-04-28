import torch
import torch.nn as nn
import math
import editdistance
from .cross_entropy_loss import CrossEntropyLoss

class OT(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.sinkhorn_alpha = args.get('sinkhorn_alpha', 0.1)
        self.stopThr = args.get('stopThr', 1e-9)
        self.OT_max_iter = args.get('OT_max_iter', 100)
        self.epsilon = args.get('epsilon', 1e-9)
        self.ot_dist_type = args.get('ot_dist_type', 'attention')
        self.importance_scaling = args.get('importance_scaling', 0.5)
        
    def pairwise_euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2)  # Computes pairwise Euclidean distance
        
    def pairwise_cosine_distance(self, a, b, eps=1e-8):
        """
        Compute cosine distance (1 - cosine similarity)
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
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
    
    def solve_optimal_transport(self, M, a, b):
        """
        Solve optimal transport problem using Sinkhorn algorithm
        M: cost matrix
        a: source distribution
        b: target distribution
        """
        # Initialize variables
        u = torch.ones_like(a) / a.size(0)
        
        # K matrix (Gibbs kernel)
        K = torch.exp(-M * self.sinkhorn_alpha)
        
        # Sinkhorn iterations
        err = 1.0
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u_new = torch.div(a, torch.matmul(K, v) + self.epsilon)
            
            # Check convergence every 50 iterations
            if cpt % 50 == 1:
                err = torch.norm(u_new - u, p=float('inf'))
            
            u = u_new
            cpt += 1
        
        # Transport plan
        transp = u * (K * v.t())
        
        return transp
    
    def compute_ot_loss(self, teacher_hidden_states, student_hidden_states, teacher_attention, 
                        teacher_tokens, student_tokens, teacher_special="<s>", student_special="[CLS]"):
        """
        Align teacher and student hidden states using optimal transport
        """
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
        
        # Calculate cost matrix based on the specified distance type
        if self.ot_dist_type == 'euclidean':
            M = self.pairwise_euclidean_distance(student_hidden_states, teacher_hidden_states)
        elif self.ot_dist_type == 'cosine':
            M = self.pairwise_cosine_distance(student_hidden_states, teacher_hidden_states)
        else:  # attention distance
            M = self.pairwise_attention_distance(student_hidden_states, teacher_hidden_states)
        
        # Prepare mass distributions
        teacher_mass = teacher_importance.unsqueeze(1)
        student_mass = student_importance.unsqueeze(1)
        
        # Solve optimal transport
        transport_plan = self.solve_optimal_transport(M, student_mass, teacher_mass)
        
        # Calculate OT loss
        ot_loss = torch.sum(transport_plan * M)
        
        return ot_loss
    
    def compute_ot_batch_loss(self, distiller, input_data, k=1):
        """
        Process a batch of inputs and compute OT loss
        """
        teacher_model = distiller.teacher_model
        student_model = distiller.student_model
        
        device = teacher_model.device
        batch_size = input_data["input_ids"].shape[0]
        
        total_loss = 0.0
        
        # Get tokenizers
        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers
        
        # Special tokens mapping
        teacher_special = "<s>"
        student_special = "[CLS]"
        
        for i in range(batch_size):
            # Run teacher model with teacher inputs
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_data["teacher_input_ids"][i].unsqueeze(0),
                    attention_mask=input_data["teacher_attention_mask"][i].unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=True
                )
            
            # Run student model with student inputs
            student_outputs = student_model(
                input_data["input_ids"][i].unsqueeze(0),
                attention_mask=input_data["attention_mask"][i].unsqueeze(0),
                output_hidden_states=True,
                output_attentions=True
            )
            
            # Get tokens
            teacher_tokens = tokenizer_teacher.convert_ids_to_tokens(input_data["teacher_input_ids"][i])
            student_tokens = tokenizer_student.convert_ids_to_tokens(input_data["input_ids"][i])
            
            # Get hidden states from last k layers
            teacher_hidden_states = teacher_outputs.hidden_states[-k:]
            student_hidden_states = student_outputs.hidden_states[-k:]
            
            # Get attention from last layer
            teacher_attention = teacher_outputs.attentions[-1]
            
            # Process each layer pair
            for t_layer, s_layer in zip(teacher_hidden_states, student_hidden_states):
                # Apply projector if available
                if hasattr(distiller, 'projectors') and 't2s' in distiller.projectors:
                    t_hidden = distiller.projectors["t2s"](t_layer[0])  # [seq_len, hidden_dim]
                else:
                    t_hidden = t_layer[0]  # [seq_len, hidden_dim]
                
                s_hidden = s_layer[0]  # [seq_len, hidden_dim]
                
                # Compute OT loss
                layer_loss = self.compute_ot_loss(
                    t_hidden, 
                    s_hidden,
                    teacher_attention[0],
                    teacher_tokens,
                    student_tokens,
                    teacher_special,
                    student_special
                )
                
                total_loss += layer_loss
                
        return total_loss
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        """
        Forward method integrating OT loss with cross-entropy loss
        """
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model

        # Calculate OT loss
        ot_loss = self.compute_ot_batch_loss(distiller, input_data, k=4)
        
        # Run student model for CE loss calculation
        outputs = model(
            input_ids=input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )

        logits = outputs.logits

        # Calculate cross-entropy loss
        loss_ce = self.compute_cross_entropy_loss(
            logits,
            output_data["labels"],
        )[0]
        
        # Combine losses according to kd_rate
        log = {}
        print("loss_ce:", loss_ce)
        print("ot_loss:", ot_loss)
        
        # Combined loss (CE + OT loss)
        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * ot_loss
        log["loss"] = loss
        log["ce_loss"] = loss_ce
        log["ot_loss"] = ot_loss

        # Calculate accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"], 
        )
        log["accuracy"] = accuracy

        # Update logging
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        
        return loss, logging_output
