import torch
import torch.nn as nn
import torch.distributed as dist

class CrossEntropyLoss(nn.Module):
    def __init__(self, args) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = args.label_smoothing
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute cross-entropy loss and accuracy for text classification.
        - Expects logits (batch_size, num_classes), target (batch_size,).
        - batch_denom is typically the batch size.
        """
        self.distiller = distiller
        model = distiller.student_model
        target = output_data["labels"]

        logits = model(input_ids=input_data['input_ids'],attention_mask=input_data['attention_mask']).logits

        # Compute loss and accuracy
        loss, nll_loss = self.compute_cross_entropy_loss(logits, target)
        correct = self.compute_accuracy(logits, target)
        
        # Update logging output, return to main distillation
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "nll_loss": nll_loss,
                "correct": correct
            }
        )
        return loss, logging_output

    def compute_cross_entropy_loss(self, logits, target):
        # Tính log softmax trên chiều lớp
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        
        # Tính negative log likelihood loss
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        
        if self.label_smoothing > 0:
            # Tính mất mát mịn (smooth loss)
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss, nll_loss

    def compute_accuracy(self, logits, target):
        # Lấy chỉ số lớp có xác suất cao nhất
        pred = logits.argmax(dim=-1)
        
        # Tính số lượng mẫu dự đoán đúng
        correct = pred.eq(target).sum().float()
        accu = correct / target.size(0)
        return accu

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and accuracy for logging, handling distributed training.
        content = {
                "loss": loss,
                "nll_loss": nll_loss,
                "correct": correct
            }
        """
        
        for k, v in content.items():
            if k == "correct":
                # Sum the correct counts across processes
                record_v = v.clone()
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item()
            else:
                # Normalize loss by batch_denom and average across processes
                record_v = v / batch_denom
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item() / dist.get_world_size()
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        return logging_output
