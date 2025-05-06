import torch
import torch.nn as nn
from .sts_loss import STSLoss

class DualSpaceKDWithCMA(STSLoss):
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
        
        # Get model's dtype
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        
        predictions = outputs.scores
        log = {}
        loss_mse = nn.MSELoss()
        
        # STS task uses MSE loss with regression scores
        # Convert labels to model's dtype
        labels = output_data["labels"].to(dtype)
        loss_ce = loss_mse(predictions, labels)
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute dual-space KD loss with CMA
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        print("dskd_cma_loss:", kd_loss)
        
        # Combine losses - ensure same dtype
        kd_loss = kd_loss.to(dtype)
        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * kd_loss
        log["loss"] = loss

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output
    
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # Get model dtype and device for consistent type conversion
        student_dtype = next(distiller.student_model.parameters()).dtype
        teacher_dtype = next(distiller.teacher_model.parameters()).dtype
        device = next(distiller.student_model.parameters()).device
        
        # For STS tasks, target is regression scores: shape [batch_size]
        target = output_data["labels"].to(student_dtype)
        
        # For BERT-like models: use [CLS] token (index 0); adjust if needed for other architectures
        hiddens = outputs.hidden_states[-1][:, 0, :]
        teacher_hiddens = teacher_outputs.hidden_states[-1][:, 0, :]

        # Embedding extraction for student and teacher
        if hasattr(distiller.student_model, "get_input_embeddings"):
            stu_embed_tokens = distiller.student_model.get_input_embeddings()  # Works for BERT, LLaMA, etc.
        elif hasattr(distiller.student_model, "bert") and hasattr(distiller.student_model.bert, "embeddings"):
            stu_embed_tokens = distiller.student_model.bert.embeddings.word_embeddings  # BERT-specific
        elif hasattr(distiller.student_model, "model") and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens  # LLaMA-like
        elif hasattr(distiller.student_model, "transformer") and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte  # GPT-like
        else:
            raise NotImplementedError("Unsupported student model architecture for embedding extraction")

        # Embedding extraction for teacher (LLaMA or similar)
        teacher_model = distiller.teacher_model
        if hasattr(teacher_model, "get_input_embeddings"):
            tea_embed_tokens = teacher_model.get_input_embeddings()  # Universal method, should work for LLaMA
        elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.embed_tokens  # LLaMA-specific
        elif hasattr(teacher_model, "bert") and hasattr(teacher_model.bert, "embeddings"):
            tea_embed_tokens = teacher_model.bert.embeddings.word_embeddings  # BERT-specific
        else:
            raise NotImplementedError("Unsupported teacher model architecture for embedding extraction")

        # Use input_ids as context for CMA
        stu_input_embeds = stu_embed_tokens(input_data["input_ids"][:, 0]).detach()
        tea_input_embeds = tea_embed_tokens(input_data["teacher_input_ids"][:, 0]).detach()

        # Normalize teacher embeddings
        norm_tea_input_embeds = tea_input_embeds / (tea_input_embeds.std() + 1e-6)
        norm_teacher_hiddens = teacher_hiddens / (teacher_hiddens.std() + 1e-6)

        # CMA projections - keep in original dtype until the end
        stu_q_hiddens = distiller.projectors["query"](stu_input_embeds)
        tea_k_hiddens = norm_tea_input_embeds

        stu_v_hiddens = distiller.projectors["s2t"](hiddens)
        tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens)

        # Alignment computation in student's dtype
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / ((hiddens.shape[-1] ** 0.5) + 1e-6)  # Scale by sqrt of hidden size

        # Teacher-to-Student (t2s) projection
        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)

        # Get scores for STS task
        # For STS task, we need to use the regression head to get scores
        if hasattr(distiller.student_model, "score"):
            t2s_scores = distiller.student_model.score(t2s_hiddens)
        elif hasattr(distiller.student_model, "regression"):
            t2s_scores = distiller.student_model.regression(t2s_hiddens)
        elif hasattr(distiller.student_model, "regressor"):
            t2s_scores = distiller.student_model.regressor(t2s_hiddens)
        else:
            # Try to use the same method that produced the original outputs.scores
            t2s_scores = getattr(distiller.student_model, "score", lambda x: x)(t2s_hiddens)

        # Compute t2s losses for regression (STS)
        t2s_mse_loss = nn.MSELoss()(t2s_scores, target)
        t2s_kd_loss = nn.MSELoss()(outputs.scores, t2s_scores.detach())

        # Student-to-Teacher (s2t) projection
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)

        # Get scores for teacher model
        if hasattr(distiller.teacher_model, "score"):
            s2t_scores = distiller.teacher_model.score(s2t_hiddens)
        elif hasattr(distiller.teacher_model, "regression"):
            s2t_scores = distiller.teacher_model.regression(s2t_hiddens)
        elif hasattr(distiller.teacher_model, "regressor"):
            s2t_scores = distiller.teacher_model.regressor(s2t_hiddens)
        else:
            # Try to access teacher scores the same way they appear in teacher_outputs
            s2t_scores = getattr(distiller.teacher_model, "score", lambda x: x)(s2t_hiddens)

        # Use MSE for STS tasks instead of KL divergence
        s2t_kd_loss = nn.MSELoss()(s2t_scores, teacher_outputs.scores)

        # Combine KD losses - ensure same dtype for all losses
        t2s_mse_loss = t2s_mse_loss.to(student_dtype)
        t2s_kd_loss = t2s_kd_loss.to(student_dtype)  
        s2t_kd_loss = s2t_kd_loss.to(student_dtype)
        
        kd_loss = t2s_mse_loss + t2s_kd_loss + s2t_kd_loss

        # Compute RMSE (root mean squared error) for logging
        with torch.no_grad():
            t2s_rmse = torch.sqrt(nn.MSELoss(reduction='mean')(t2s_scores, target) + 1e-6)
            s2t_rmse = torch.sqrt(nn.MSELoss(reduction='mean')(s2t_scores, teacher_outputs.scores) + 1e-6)

        # Logging
        log["t2s_mse_loss"] = t2s_mse_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_rmse"] = t2s_rmse
        log["s2t_rmse"] = s2t_rmse
        log["kd_loss"] = kd_loss

        return kd_loss, log

    def compute_accuracy(self, scores, labels):
        # For STS (regression), use RMSE or other regression metrics instead of accuracy
        # Ensure both tensors are the same dtype
        dtype = scores.dtype
        labels = labels.to(dtype)
        rmse = torch.sqrt(nn.MSELoss(reduction='mean')(scores, labels) + 1e-6)
        return rmse