import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,  
    AutoModelForSequenceClassification,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from llm2vec import LLM2Vec
from utils import log_rank


class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student_model, self.student_tokenizer = self.load_student_model()
        
        if self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizers = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizers = None, {}


    @staticmethod
    def add_distiller_args(parser):
        group = parser.add_argument_group("distiller", "distiller configurations")
        group.add_argument("--projector-config-path", type=str, default=None,
                           help='path to projector_config.json')
        group.add_argument("--projector-path", type=str, default=None,
                           help='path to pretrained projector')
        group.add_argument("--projector-lr", type=float, default=0.001,
                           help='learning rate only for projection')
        group.add_argument("--pretrained-projector", type=str, default=None,
                           help='pretrained projector name')
        group.add_argument("--pretrained-projector-lr", type=float, default=0.001,
                           help='learning rate only for pretrained projector')
        group.add_argument("--vocab-alignment-path", type=str, default=None,
                           help='path for the vocab alignment file')
        group.add_argument("--teacher-to-student-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, teacher-to-student)')
        group.add_argument("--teacher-to-student-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, teacher-to-student)')
        group.add_argument("--student-to-teacher-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, student-to-teacher)')
        group.add_argument("--student-to-teacher-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, student-to-teacher)')
        return parser
    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return tokenizer
    
    def load_student_model(self):
        log_rank("Loading student model...")
        config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)
        config.is_model_parallel = False

        # lấy tokenizer
        tokenizer = self.load_tokenizer(self.args.model_path)
        
        if hasattr(config, "n_embed"):
            self.hidden_size = config.n_embed
        else:
            self.hidden_size = config.hidden_size
        
        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError("Invalid model_dtype for f`{self.args.model_dtype}`")

        if self.args.peft is not None: #for LLM2Vec
            if self.args.peft == "lora":
                base_model = AutoModel.from_pretrained(
                    self.args.model_path,
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )


                # Apply a new LoRA adapter for fine-tuning
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=(not self.args.do_train),
                    r=self.args.peft_lora_r,
                    lora_alpha=self.args.peft_lora_alpha,
                    lora_dropout=self.args.peft_lora_dropout,
                )
                model = get_peft_model(base_model, peft_config)

                # Initialize LLM2Vec
                l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

                # Classification Head (can be LoRA-enhanced)
                classification_head = nn.Linear(self.hidden_size, self.args.num_labels)

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

                model.print_trainable_parameters()
            else:
                raise NotImplementedError
        else: #for BERT
            config.num_labels = self.args.num_labels
            model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_path, 
                config=config, 
                device_map=None, 
                torch_dtype=self.dtype,
                trust_remote_code=True,)
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))


        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(
            self.args.teacher_model_path,
            num_labels=self.num_labels,
            trust_remote_code=True
        )
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.args.teacher_model_path)

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        # Load the base model
        model = AutoModel.from_pretrained(
            self.args.teacher_model_path,
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        
        l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        # Thêm lớp phân loại
        classification_head = torch.nn.Linear(self.teacher_hidden_size, self.args.num_labels)
        
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
        
        # Vô hiệu hóa requires_grad cho tất cả tham số
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Trả về mô hình và tokenizer
        return teacher_model, tokenizer
    
    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors"):
            if self.args.projector_lr:
                pretrained_proj = self.args.pretrained_projector.split(",") if self.args.pretrained_projector is not None else []
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b not in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.projector_lr
                })
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.pretrained_projector_lr
                })
            else:
                optimizer.add_param_group({
                    "params": [p for b in self.projectors for p in self.projectors[b].parameters()],
                })
        return optimizer

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch["output_batch"]
        loss, logging_output = criterion(
            self,
            input_data, 
            output_data,
            logging_output,
            loss_denom,
        )
        return loss, logging_output