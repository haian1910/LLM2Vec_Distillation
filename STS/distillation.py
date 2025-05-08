import time
import os

from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import shutil
import json
from tqdm import tqdm
import math
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModel,
)
from transformers.integrations import HfDeepSpeedConfig
from STS.arguments import get_args
from STS.distiller import Distiller
from STS.data_utils.distill_datasets import STSDataset
from STS.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from STS.criterions import build_criterion

torch.set_num_threads(4) # limit the number of threads used by torch for cpu

def prepare_dataset(args, distiller):
    data = {}
    if args.do_train:
        data["train"] = STSDataset(
            args, "train", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = STSDataset(
            args, "dev", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test.csv")):
            data["test"] = STSDataset(
                args, "test", distiller.student_tokenizer,
                distiller.teacher_tokenizers
            )
            log_rank("Num of test data: {}".format(len(data["test"])))

    elif args.do_eval:
        data["test"] = STSDataset(
            args, "test", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of test data: {}".format(len(data["test"])))
    else:
        raise ValueError("Do train and do eval must set one")
        
    return data

def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device):
    log_rank("Start Fine-tuning")
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_loader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    step = 0
    model_list = []
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [], 
        "pearson": [],
        "spearman": [],
        "kd_loss": [],
        "micro_step_time": [],
        "step_time": []
    }
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        print("Training mode?", model.student_model.training)  # True

        epoch_start_time = time.time()
        step = 0
        total_samples = 0
        total_time = 0.0

        data_iter = train_loader
        if dist.get_rank() == 0:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for batch in data_iter:
            st_time = time.time()
            input_batch, output_batch = batch
            dataset["train"].move_to_device([input_batch, output_batch], device)

            loss, logging_output = model(
                criterion,
                {"input_batch": input_batch, "output_batch": output_batch},
                logging_output,
                loss_denom=1, #deepspeed support sync gradient, no need to calculate loss_denom
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue

            
            model.backward(loss)
            model.step()
            torch.cuda.synchronize()  # correctly compute time

            elapsed_time = time.time() - st_time
            num_samples = input_batch["input_ids"].size(0)
            total_samples += num_samples
            total_time += elapsed_time
            step += 1

            logging_output["global_step"] += 1
            logging_output["micro_step_time"].append(elapsed_time)
            logging_output["step_time"].append(elapsed_time)

            if dist.get_rank() == 0:
                data_iter.set_postfix(loss=loss.item())


        if args.save_dir and (epoch + 1) % args.save_interval == 0: #save_interval = 1 then save each epoch
            #eval_interval = 1 then evaluate each epoch
            log_rank("Evaluating before saving model...")
            eval_loss, eval_pearson, eval_spearman = evaluate(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
            if "test" in dataset: #evaluate for test, no affect
                _, _, _ = evaluate(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
            ckpt_name = f"epoch{epoch + 1}_step{logging_output['global_step']}_loss{eval_loss:.4f}_pearson{eval_pearson:.4f}"
            save_dir_path = os.path.join(args.save_dir, ckpt_name)
            
            if dist.get_rank() == 0:
                os.makedirs(save_dir_path, exist_ok=True)
                if not args.only_save_projector:
                    log_rank("Saving tokenizer...")
                    tokenizer.save_pretrained(save_dir_path)
                    log_rank("Saving model...")
                    model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    log_rank("Saving config")
                    model.module.student_model.config.save_pretrained(save_dir_path)
                
                if hasattr(model.module, "projectors"):
                    log_rank("Saving projector...")
                    torch.save(
                        model.module.projectors.state_dict(), 
                        os.path.join(save_dir_path, "projector.pt")
                    )
                
                # Use Pearson correlation as the primary metric for STS tasks
                model_list.append({"path": save_dir_path, "score": eval_pearson})
                model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)  # Higher is better
                
                if len(model_list) > args.keep_best_n_checkpoints:
                    removed_model = model_list.pop(-1)  # Remove worst model
                    shutil.rmtree(removed_model["path"])

                log_rank(f"Model has been saved to {save_dir_path}")
            dist.barrier()
            
    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad()
def evaluate(args, tokenizer, student_model, dataset, split, device):
    if dist.get_rank() != 0:
        return None, None, None, None        
    
    # Use regular DataLoader without DistributedSampler
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    eval_info = {
        "loss": 0.0,
        "sample_num": 0
    }

    all_preds = []
    all_targets = []
    total_loss = 0
    
    for input_batch, output_batch in tqdm(dataloader, desc="Processing batches"):
        dataset.move_to_device([input_batch, output_batch], device)
        targets = output_batch["labels"]
        
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        predictions = outputs.scores 
        # Compute MSE loss
        loss = F.mse_loss(predictions, targets)
        
        all_preds.append(predictions)
        all_targets.append(targets)
        sample_num = targets.size(0)
        total_loss += loss.item() * sample_num  # Scale loss by batch size for proper averaging

        eval_info["sample_num"] += sample_num
        
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # No need for gathering across processes

    # Convert to float32 before converting to numpy (BFloat16 is not supported by numpy)
    all_preds = all_preds.to(torch.float32)
    all_targets = all_targets.to(torch.float32)

    # Convert to numpy for correlation metrics
    all_preds_np = all_preds.cpu().numpy().flatten()
    all_targets_np = all_targets.cpu().numpy().flatten()

    # Calculate Pearson and Spearman correlations
    pearson_correlation, _ = pearsonr(all_preds_np, all_targets_np)
    spearman_correlation, _ = spearmanr(all_preds_np, all_targets_np)
    
    # Update evaluation info
    eval_info["loss"] = float(total_loss / eval_info["sample_num"])
    eval_info["pearson"] = round(float(pearson_correlation), 6)
    eval_info["spearman"] = round(float(spearman_correlation), 6)
    eval_info["mse"] = round(float(((all_preds_np - all_targets_np) ** 2).mean()), 6)

    if hasattr(args, 'local_rank') and args.local_rank == 0 or not hasattr(args, 'local_rank'):
        print(f"Evaluated: {split} | {eval_info}")

    student_model.train()

    return eval_info["loss"], eval_info["pearson"], eval_info["spearman"]
def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    # save arguments
    if dist.get_rank() == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    print('user ds_config', ds_config)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * dp_world_size

    log_rank("Initializing a distiller for knowledge distillation...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size))
        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

        log_rank("Total_iters = {}".format(args.total_iters))
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    optimizer_grouped_parameters = get_optimizer(args, distiller.student_model)
    optimizer_grouped_parameters = distiller.add_optimizer_param_group(optimizer_grouped_parameters)

    lr_scheduler = get_learning_rate_scheduler(args, optimizer_grouped_parameters)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer_grouped_parameters,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
       
    if args.do_eval:
        evaluate(args, distiller.student_tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
        
    
if __name__ == "__main__":
    main()
