import torch
import torch.nn as nn
import torch.distributed as dist

class STSLoss(nn.Module):
    def __init__(self, args) -> None:
        super(STSLoss, self).__init__()
        self.loss_type = getattr(args, "sts_loss_type", "mse")  # Default to MSE if not specified
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute loss for STS (Semantic Textual Similarity) tasks.
        - Expects model output (batch_size, 1), target (batch_size, 1).
        - batch_denom is typically the batch size.
        """
        self.distiller = distiller
        model = distiller.student_model
        target = output_data["scores"]  # Change from "labels" to "scores" for STS

        # Forward pass through the model
        model_output = model(
            input_ids=input_data['input_ids'],
            attention_mask=input_data['attention_mask'],
            token_type_ids=input_data['token_type_ids']
        )
        
        # For regression, we expect a single output value
        # Check if the model outputs a dictionary with a specific key or just the prediction
        if hasattr(model_output, "logits"):
            predictions = model_output.logits
        else:
            predictions = model_output
            
        # Ensure predictions are the right shape
        if predictions.shape[-1] != 1:
            # If the model outputs multiple values, use mean pooling or a linear layer
            if hasattr(self.distiller, "regression_head"):
                predictions = self.distiller.regression_head(predictions)
            else:
                # Create a simple regression head on first use
                self.distiller.regression_head = nn.Linear(predictions.size(-1), 1).to(predictions.device)
                self.distiller.regression_head.weight.data.normal_(mean=0.0, std=0.02)
                self.distiller.regression_head.bias.data.zero_()
                predictions = self.distiller.regression_head(predictions)

        # Compute loss based on specified type
        loss = self.compute_sts_loss(predictions, target)
        
        # Compute correlation for evaluation
        pearson, spearman = self.compute_correlations(predictions, target)
        
        # Update logging output, return to main distillation
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "pearson": pearson,
                "spearman": spearman,
                "predictions": predictions.detach().mean(),
                "target": target.detach().mean()
            }
        )
        return loss, logging_output

    def compute_sts_loss(self, predictions, target):
        if self.loss_type == "mse":
            return nn.MSELoss()(predictions, target)
        elif self.loss_type == "mae":
            return nn.L1Loss()(predictions, target)
        elif self.loss_type == "huber":
            return nn.SmoothL1Loss()(predictions, target)
        else:
            # Default to MSE
            return nn.MSELoss()(predictions, target)

    def compute_correlations(self, predictions, target):
        """
        Compute Pearson and Spearman correlations between predictions and targets.
        Returns naive local estimates (proper correlation will be computed at evaluation).
        """
        # Detach from computation graph and flatten
        pred_flat = predictions.detach().view(-1).cpu()
        target_flat = target.view(-1).cpu()
        
        # Calculate mean and standard deviation for Pearson correlation
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        
        # Simple Pearson correlation estimate (for logging only)
        try:
            # Calculate Pearson correlation coefficient
            numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
            denominator = torch.sqrt(((pred_flat - pred_mean) ** 2).sum() * ((target_flat - target_mean) ** 2).sum())
            pearson = numerator / denominator if denominator != 0 else torch.tensor(0.0)
            
            # Simple placeholder for Spearman (proper calculation requires rank conversion)
            # In practice, this would be calculated during evaluation, not training
            spearman = pearson  # Placeholder
        except:
            # Handle potential numerical issues
            pearson = torch.tensor(0.0)
            spearman = torch.tensor(0.0)
            
        return pearson, spearman

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and correlations for logging, handling distributed training.
        """
        for k, v in content.items():
            if k in ["pearson", "spearman"]:
                # For correlations, we don't need to divide by batch_denom
                # but we still average across processes
                if isinstance(v, torch.Tensor):
                    record_v = v.clone().to(torch.float32)
                    dist.all_reduce(record_v, dist.ReduceOp.SUM)
                    record_v = record_v.item() / dist.get_world_size()
                else:
                    record_v = v / dist.get_world_size()
            elif k in ["predictions", "target"]:
                # Just record mean values for monitoring
                if isinstance(v, torch.Tensor):
                    record_v = v.item()
                else:
                    record_v = v
            else:
                # Normalize loss by batch_denom and average across processes
                if isinstance(v, torch.Tensor):
                    record_v = v / batch_denom
                    dist.all_reduce(record_v, dist.ReduceOp.SUM)
                    record_v = record_v.item() / dist.get_world_size()
                else:
                    record_v = v / batch_denom / dist.get_world_size()
                    
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
                
        return logging_output