import time
import os

from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support

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
from Classification.arguments import get_args
from Classification.distiller import Distiller
from Classification.data_utils.distill_datasets import DistillDataset
from Classification.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from Classification.criterions import build_criterion
# from rouge_metric import compute_metrics

torch.set_num_threads(4) # giới hạn số lượng thread torch sử dụng cho cpu

def prepare_dataset(args, distiller):
    data = {}
    if args.do_train:
        data["train"] = DistillDataset(
            args, "train", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = DistillDataset(
            args, "dev", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test.csv")):
            data["test"] = DistillDataset(
                args, "test", distiller.student_tokenizer,
                distiller.teacher_tokenizers
            )
            log_rank("Num of test data: {}".format(len(data["test"])))

    elif args.do_eval:
        data["test"] = DistillDataset(
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
        "nll_loss": [],
        "kd_loss": [],
        "accuracy": [],
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
            torch.cuda.synchronize()  # correctlyc compute time

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
            eval_loss, eval_accu, eval_precision, eval_recall = evaluate(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
            if "test" in dataset: #evaluate for test, no affect
                _, _, _, _ = evaluate(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
            ckpt_name = "epoch{}_step{}_loss{:.4f}".format(epoch + 1, logging_output["global_step"], eval_loss)
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
                
                model_list.append({"path": save_dir_path, "score": eval_loss}) #store model list in term of eval_loss
                model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)
                
                if len(model_list) > args.keep_best_n_checkpoints:
                    removed_model = model_list.pop(0)
                    shutil.rmtree(removed_model["path"])

                log_rank(f"Model has been saved to {save_dir_path}")
            dist.barrier()
            
    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad
def evaluate(args, tokenizer, student_model, dataset, split, device):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None

    if dist.get_rank() == 0:
        print(f"Evaluating on {split} set with {dp_world_size} GPU(s)")
        
    sampler = DistributedSampler(
        dataset,
        shuffle=False,
        drop_last=False,
        rank=dp_rank,
        num_replicas=dp_world_size
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    eval_info = {
        "loss": 0.0,
        "sample_num": 0,
        "correct_samples": 0
    }

    all_preds = []
    all_labels = []
    local_loss = 0
    for input_batch, output_batch in tqdm(dataloader, desc="Processing batches"):
        
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]       
        outputs = student_model(
            input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            position_ids=input_batch.get("position_ids", None),
            labels = labels
        )
        logits = outputs.logits
        loss = outputs.loss

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        all_preds.append(preds)
        all_labels.append(labels)
        sample_num = labels.size(0)
        local_loss += loss

        eval_info["sample_num"] += sample_num
        eval_info["correct_samples"] += correct

        
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)


    all_preds_gathered = [torch.zeros_like(all_preds) for _ in range(dp_world_size)]
    all_labels_gathered = [torch.zeros_like(all_labels) for _ in range(dp_world_size)]
    
    dist.all_gather(all_preds_gathered, all_preds, group=dp_group)
    dist.all_gather(all_labels_gathered, all_labels, group=dp_group)
   
    all_dp_loss = [torch.zeros_like(local_loss) for _ in range(dp_world_size)]
    dist.all_gather(all_dp_loss, local_loss, group=dp_group)

    all_dp_loss = sum(all_dp_loss).item()

    all_preds = torch.cat(all_preds_gathered, dim=0)
    all_labels = torch.cat(all_labels_gathered, dim=0)

    all_preds_np = all_preds.cpu().numpy()
    all_labels_np = all_labels.cpu().numpy()

    precision = precision_score(all_labels_np, all_preds_np, average='macro')
    recall = recall_score(all_labels_np, all_preds_np, average='macro')

    eval_info["precision"] = round(precision, 6)
    eval_info["recall"] = round(recall, 6)

    eval_info["loss"] = all_dp_loss / len(all_preds)
    eval_info["accuracy"] = (all_preds==all_labels).sum().item() / len(all_preds)
    eval_info["sample_num"] = len(all_preds)
    eval_info["correct_samples"] = (all_preds==all_labels).sum().item()

    for key in eval_info:
        if isinstance(eval_info[key], float):
            eval_info[key] = round(eval_info[key], 6)
    
    if dist.get_rank() == 0:
        print(f"Evaluated: {split} | {eval_info}")

    student_model.train()

    return eval_info["loss"], eval_info["accuracy"], eval_info.get("precision", 0.0), eval_info.get("recall", 0.0)

def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    # save arguments
    if dist.get_rank() == 0:
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
