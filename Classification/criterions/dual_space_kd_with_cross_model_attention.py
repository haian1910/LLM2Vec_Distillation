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
        
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        
        # Cross-entropy loss with ground-truth labels
        loss = self.compute_cross_entropy_loss(outputs.logits, output_data["labels"])[0]
        
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
        # Target for classification: shape [batch_size]
        target = output_data["labels"]
        
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

        # Use input_ids as context for CMA (no padding_id needed for classification)
        stu_input_embeds = stu_embed_tokens(input_data["input_ids"][:, 0]).detach()  # [CLS] token embedding
        tea_input_embeds = tea_embed_tokens(input_data["teacher_input_ids"][:, -1]).detach()  # [CLS] token embedding

        # Normalize teacher embeddings
        norm_tea_input_embeds = tea_input_embeds / tea_input_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        # CMA projections
        stu_q_hiddens = distiller.projectors["query"](stu_input_embeds).float()
        tea_k_hiddens = norm_tea_input_embeds.float()

        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens).float()

        # Alignment computation
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / (hiddens.shape[-1] ** 0.5)  # Scale by sqrt of hidden size

        # Teacher-to-Student (t2s) projection
        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)

        # Use appropriate classification head for student
        if hasattr(distiller.student_model, "classifier"):
            t2s_logits = distiller.student_model.classifier(t2s_hiddens)
        elif hasattr(distiller.student_model, "score"):
            t2s_logits = distiller.student_model.score(t2s_hiddens)
        else:
            raise AttributeError("Student model has neither 'classifier' nor 'score' attribute")

        # Compute t2s losses
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_kd_loss = self.dist_func(outputs.logits, t2s_logits.detach(), target, reduction="mean")

        # Student-to-Teacher (s2t) projection
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)

        # Use appropriate classification head for teacher
        if hasattr(distiller.teacher_model, "classifier"):
            s2t_logits = distiller.teacher_model.classifier(s2t_hiddens)
        elif hasattr(distiller.teacher_model, "score"):
            s2t_logits = distiller.teacher_model.score(s2t_hiddens)
        else:
            raise AttributeError("Teacher model has neither 'classifier' nor 'score' attribute")

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
