import numpy as np
import os
import time
import torch.distributed as dist
import random
import logging
import torch
import torch.nn as nn
from datetime import timedelta
import deepspeed
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_constant_schedule_with_warmup, 
    get_polynomial_decay_schedule_with_warmup,
    AutoModelForSequenceClassification,
    AutoModel
)


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s]  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Logging
def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def log_rank(content, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        logging.info(content)


# Distributed
def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t


# Initialize
def set_random_seed(seed, mp=False):
    """Set random seed for reproducability."""
    seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def init_distributed(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    log_rank(f"Using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    log_rank(f"Using world size: {args.world_size}")
    
    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()

    if args.local_rank is not None:
        device = args.local_rank
    
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=30))


def initialize(args):
    # init bmt
    if args.deepspeed:
        init_distributed_ds(args)
    else:
        init_distributed(args)

    if args.model_parallel:
        raise NotImplementedError

    set_random_seed(args.seed, args.model_parallel)
    # init save folder
    if args.save_dir != None:
        os.makedirs(args.save_dir, exist_ok=True)


# Load and save model
'''
        Sửa cả đoạn này cho đúng BERT của mình
'''
def get_model(args, device):
    config = AutoConfig.from_pretrained(args.model_path)
    
    st_time = time.time()
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if hasattr(config, "n_embed"):
            hidden_size = config.n_embed
        else:
            hidden_size = config.hidden_size
        if args.peft is not None: #for LLM2Vec
            if args.peft == "lora":
                dtype = torch.float32 if args.fp32 else torch.float16
                try:
                    model = AutoModel.from_pretrained(
                        args.model_path,
                        config=config,
                        device_map={"": device},
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    )
                except:
                    model = AutoModel.from_pretrained(
                        args.model_path,
                        config=config,
                        device_map={"": device},
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                    )
                    model = model.half()

                # Apply a new LoRA adapter for fine-tuning
                if args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        inference_mode=(not args.do_train),
                        r=args.peft_lora_r,
                        lora_alpha=args.peft_lora_alpha,
                        lora_dropout=args.peft_lora_dropout,
                    )
                    model = get_peft_model(model, peft_config)

                # Initialize LLM2Vec
                l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

                # Classification Head (can be LoRA-enhanced)
                classification_head = nn.Linear(hidden_size, args.num_labels)

                # Create a wrapper for the classification model
                class TeacherModelForClassification(nn.Module):
                    def __init__(self, l2v, classification_head):
                        super().__init__()
                        self.l2v = l2v
                        self.classification_head = classification_head

                    def forward(self, input_ids, attention_mask=None, **kwargs):
                        # Encode text into vector representation using LLM2Vec
                        encoded = self.l2v.encode(input_ids=input_ids, attention_mask=attention_mask)
                        # Apply classification head
                        logits = self.classification_head(encoded)
                        return logits

                # Initialize the teacher model
                model = TeacherModelForClassification(l2v, classification_head)
            else:
                raise NotImplementedError
        else: #for BERT
            dtype = torch.float32 if args.fp32 else torch.float16
            config.num_labels = args.num_labels
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_path, 
                    config=config, 
                    device_map={"": device}, 
                    torch_dtype=dtype,
                    trust_remote_code=True,)
            except:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_path, 
                    config=config, 
                    device_map={"": device}, 
                    torch_dtype=torch.float32,
                    trust_remote_code=True,)
                model = model.half()
            if dist.get_rank() == 0:
                log_rank(' > number of parameters: {:n}'.format(
                    sum([p.nelement() for p in model.parameters()])))
                
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    ed_time = time.time()
    
    log_rank(f"Model load time: {ed_time - st_time}s")
    
    return model


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(
            args.teacher_model_path,
            trust_remote_code=True
        )
    
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
        if hasattr(config, "n_embed"):
            teacher_hidden_size = config.n_embed
        else:
            teacher_hidden_size = config.hidden_size
    
        # Load the base model
        try:
            model = AutoModel.from_pretrained(
                args.teacher_model_path,
                config=config,
                device_map={"": device},
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except:
            model = AutoModel.from_pretrained(
                args.teacher_model_path,
                config=config,
                device_map={"": device},
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model = model.half()
            
        
        l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        # Thêm lớp phân loại
        classification_head = torch.nn.Linear(teacher_hidden_size, args.num_labels)
        
        # Tạo lớp wrapper cho mô hình phân loại
        class TeacherModelForClassification(torch.nn.Module):
            def __init__(self, l2v, classification_head):
                super().__init__()
                self.l2v = l2v
                self.classification_head = classification_head
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                # Mã hóa văn bản thành vector với LLM2Vec
                encoded = self.l2v.encode(input_ids=input_ids, attention_mask=attention_mask)
                # Áp dụng lớp phân loại
                logits = self.classification_head(encoded)
                return logits
        
        # Khởi tạo mô hình teacher
        teacher_model = TeacherModelForClassification(l2v, classification_head)

    teacher_model.eval()
    
    return teacher_model

                
def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


def get_optimizer_params_peft(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    log_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.num_epochs

    print(f'get_learning_rate_scheduler total_iters: {args.total_iters} warmup_iters {args.warmup_iters}')

    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters
        )
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min
        )
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5
        )
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler
