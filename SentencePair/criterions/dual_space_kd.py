import torch
from .various_divergence import VariousDivergence

class DualSpaceKD(VariousDivergence):
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
        
        # Compute dual-space KD loss
        kd_loss, log = self.compute_dual_space_kd_loss(outputs, teacher_outputs, output_data, distiller, log)
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(logits, output_data["labels"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss , logging_output

    def compute_dual_space_kd_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        # Target for classification: shape [batch_size]
        target = output_data["labels"]

        # For BERT: use [CLS] token (index 0); for LLaMA: use last token
        hiddens = outputs.hidden_states[-1][:, 0, :]

        teacher_hiddens = teacher_outputs.hidden_states[-1][:, -1, :]


        t2s_hiddens = distiller.projectors["t2s"](teacher_hiddens)
        # Use appropriate classification head for student
        if hasattr(distiller.student_model, "classifier"):
            t2s_logits = distiller.student_model.classifier(t2s_hiddens)
        elif hasattr(distiller.student_model, "score"):
            t2s_logits = distiller.student_model.score(t2s_hiddens)
        else:
            raise AttributeError("Student model has neither 'classifier' nor 'score' attribute")
        
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]  # Scalar
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="mean"
        )  # Mean over batch

        # Teacher space: Student-to-Teacher projection
        s2t_hiddens = distiller.projectors["s2t"](hiddens)
        # Use appropriate classification head for teacher
        if hasattr(distiller.teacher_model, "classifier"):
            s2t_logits = distiller.teacher_model.classifier(s2t_hiddens)
        elif hasattr(distiller.teacher_model, "score"):
            s2t_logits = distiller.teacher_model.score(s2t_hiddens)
        else:
            raise AttributeError("Teacher model has neither 'classifier' nor 'score' attribute")
        
        s2t_kd_loss = self.compute_forward_kl_divergence(
            s2t_logits, teacher_outputs.logits, target, reduction="mean"
        )  # Mean over batch

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
