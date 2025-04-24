import torch
from .various_divergence import VariousDivergence

class DualSpaceKDWithCMA(VariousDivergence):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized

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
        
        # Forward pass through student model
        outputs = model(
            input_ids=input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        
        logits = outputs.logits  # Shape: [batch_size, num_choices]
        log = {}
        
        # Cross-entropy loss with ground-truth labels
        loss = self.compute_cross_entropy_loss(logits, output_data["labels"])[0]
        
        # Forward pass through teacher model (with no gradient tracking)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_ids=input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute dual-space KD loss with CMA for multiple choice task
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(logits, output_data["labels"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output
    
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # Target for multiple choice task: shape [batch_size]
        target = output_data["labels"]
        batch_size = target.size(0)
        num_choices = outputs.logits.size(1)
        
        # Extract hidden states for both student and teacher
        # For multiple choice tasks, we need to handle the [batch_size, num_choices, ...] shape
        # We'll use the CLS token representation for each choice
        
        # Get the CLS token embedding for all choices
        # Shape of hidden_states: [batch_size * num_choices, seq_len, hidden_dim]
        student_hidden_states = outputs.hidden_states[-1]  
        teacher_hidden_states = teacher_outputs.hidden_states[-1]
        
        # Get the CLS token representation (first token) for each choice
        # Shape: [batch_size * num_choices, hidden_dim]
        student_cls_hiddens = student_hidden_states[:, 0, :]
        teacher_cls_hiddens = teacher_hidden_states[:, 0, :]
        
        # Reshape to [batch_size, num_choices, hidden_dim]
        student_cls_hiddens = student_cls_hiddens.view(batch_size, num_choices, -1)
        teacher_cls_hiddens = teacher_cls_hiddens.view(batch_size, num_choices, -1)
        
        # Get the hidden state for the correct answer for each example in the batch
        # Gather based on target indices - shape: [batch_size, hidden_dim]
        student_correct_hiddens = student_cls_hiddens[torch.arange(batch_size), target]
        teacher_correct_hiddens = teacher_cls_hiddens[torch.arange(batch_size), target]
        
        # Get embeddings for input tokens (for CMA context)
        if hasattr(distiller.student_model, "get_input_embeddings"):
            stu_embed_tokens = distiller.student_model.get_input_embeddings()
        else:
            raise NotImplementedError("Unsupported student model architecture for embedding extraction")

        if hasattr(distiller.teacher_model, "get_input_embeddings"):
            tea_embed_tokens = distiller.teacher_model.get_input_embeddings()
        else:
            raise NotImplementedError("Unsupported teacher model architecture for embedding extraction")
        
        # Get embeddings for CLS tokens of correct choices
        # First get the input_ids for correct choices - shape: [batch_size]
        correct_student_input_ids = input_data["input_ids"][torch.arange(batch_size), target, 0]
        correct_teacher_input_ids = input_data["teacher_input_ids"][torch.arange(batch_size), target, 0]
        
        # Then get embeddings - shape: [batch_size, hidden_dim]
        stu_input_embeds = stu_embed_tokens(correct_student_input_ids).detach()
        tea_input_embeds = tea_embed_tokens(correct_teacher_input_ids).detach()
        
        # Normalize teacher embeddings
        norm_tea_input_embeds = tea_input_embeds / (tea_input_embeds.std() + 1e-6)
        norm_teacher_hiddens = teacher_correct_hiddens / (teacher_correct_hiddens.std() + 1e-6)
        
        # CMA projections
        stu_q_hiddens = distiller.projectors["query"](stu_input_embeds).float()
        tea_k_hiddens = norm_tea_input_embeds.float()
        
        stu_v_hiddens = distiller.projectors["s2t"](student_correct_hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens).float()
        
        # Alignment computation
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / (student_correct_hiddens.shape[-1] ** 0.5)  # Scale by sqrt of hidden size
        
        # Teacher-to-Student (t2s) projection
        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(student_correct_hiddens)
        
        # Project back to logits space for each choice
        # First expand t2s_hiddens to match each choice
        t2s_hiddens_expanded = t2s_hiddens.unsqueeze(1).expand(-1, num_choices, -1)
        t2s_hiddens_flat = t2s_hiddens_expanded.reshape(batch_size * num_choices, -1)
        
        # Use MultipleChoiceModel's classifier to get logits
        if hasattr(distiller.student_model, "classifier"):
            t2s_all_logits = distiller.student_model.classifier(t2s_hiddens_flat)
            t2s_logits = t2s_all_logits.view(batch_size, num_choices)
        else:
            raise AttributeError("Student model has no 'classifier' attribute")
        
        # Compute t2s losses
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_kd_loss = self.dist_func(outputs.logits, t2s_logits.detach(), target, reduction="mean")
        
        # Student-to-Teacher (s2t) projection
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(teacher_correct_hiddens)
        
        # Project back to logits space for each choice
        s2t_hiddens_expanded = s2t_hiddens.unsqueeze(1).expand(-1, num_choices, -1)
        s2t_hiddens_flat = s2t_hiddens_expanded.reshape(batch_size * num_choices, -1)
        
        # Use teacher's classifier to get logits
        if hasattr(distiller.teacher_model, "classifier"):
            s2t_all_logits = distiller.teacher_model.classifier(s2t_hiddens_flat)
            s2t_logits = s2t_all_logits.view(batch_size, num_choices)
        else:
            raise AttributeError("Teacher model has no 'classifier' attribute")
        
        # Compute s2t loss
        s2t_kd_loss = self.compute_forward_kl_divergence(s2t_logits, teacher_outputs.logits, target, reduction="mean")
        
        # Combine KD losses
        kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
        
        # Compute accuracies
        t2s_acc = (t2s_logits.argmax(-1) == target).float().mean()
        s2t_acc = (s2t_logits.argmax(-1) == target).float().mean()
        
        # Logging
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_acc"] = t2s_acc
        log["s2t_acc"] = s2t_acc
        log["kd_loss"] = kd_loss
        
        return kd_loss, log
