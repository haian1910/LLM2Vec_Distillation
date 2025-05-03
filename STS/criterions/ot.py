import torch
from .cross_entropy_loss import CrossEntropyLoss
import torch.nn as nn
import math

class OT(CrossEntropyLoss):
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
    
    def compute_ot_loss(
        self, outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log, distiller, logits=False
    ):
        # Get the last hidden state from both models
        student_features = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        teacher_features = teacher_outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        
        batch_size = teacher_features.size(0)
        total_loss = 0
        
        # Check if projector exists
        if not hasattr(distiller, 'projectors') or 't2s' not in distiller.projectors:
            raise AttributeError("Distiller missing 't2s' projector. Make sure projectors are properly initialized.")
            
        projector = distiller.projectors["t2s"]
        
        for b in range(batch_size):
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
            ot_loss, transport_plan = self.sinkhorn(cost_matrix)
            total_loss += ot_loss
        
        avg_loss = total_loss / batch_size
        log["ot_loss"] = avg_loss.item()
        
        return avg_loss, log
    
    def sinkhorn(self, cost_matrix, num_iters=None):
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
        a = torch.ones(m, device=device, dtype=dtype) / m
        b = torch.ones(n, device=device, dtype=dtype) / n
        
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
