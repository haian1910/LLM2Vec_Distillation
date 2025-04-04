import torch
from .various_divergence import VariousDivergence


class DualSpaceKD(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

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
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["labels"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True)
        
        kd_loss, log = self.compute_dual_space_kd_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["labels"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output

    def compute_dual_space_kd_loss(
    self, outputs, teacher_outputs, output_data, distiller, log
):
        # Target cho classification: shape (batch_size,)
        target = output_data["labels"]  # Không cần pad_mask
        teacher_target = output_data["labels"]

        # Lấy hidden state của token [CLS] thay vì toàn bộ chuỗi
        hiddens = outputs.hidden_states[-1]# (batch_size, hidden_size)
        teacher_hiddens = teacher_outputs.hidden_states[-1] # (batch_size, hidden_size)

        # Student space
        t2s_hiddens = distiller.projectors["t2s"](teacher_hiddens)  # (batch_size, hidden_size)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # (batch_size, num_classes)
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]  # Scalar

        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="sum"
        )  # Tính tổng loss trên batch

        # Teacher space
        s2t_hiddens = distiller.projectors["s2t"](hiddens)  # (batch_size, hidden_size)
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)  # (batch_size, num_classes)
        s2t_kd_loss = self.compute_forward_kl_divergence(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="sum"
        )  # Tính tổng loss trên batch

        # Tổng hợp KD loss
        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

        # Tính accuracy trên toàn bộ batch
        t2s_acc = (t2s_logits.argmax(-1) == target).float().sum()  # Tổng số dự đoán đúng
        s2t_acc = (s2t_logits.argmax(-1) == teacher_target).float().sum()  # Tổng số dự đoán đúng

        # Lưu log
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_acc"] = t2s_acc
        log["s2t_acc"] = s2t_acc
        log["kd_loss"] = kd_loss

        return kd_loss, log
    
    
