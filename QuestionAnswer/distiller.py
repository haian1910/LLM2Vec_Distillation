import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,  
    AutoModelForMultipleChoice,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from utils import log_rank
from huggingface_hub import login

#login(token="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")
login(token="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")


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
        if self.teacher_model and args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")

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
        return tokenizer
        
    def set_and_load_existing_projectors(self):
        self.projectors = nn.ModuleDict()
        projector_config = json.load(open(self.args.projector_config_path))
        name_dict = {
            "s": self.hidden_size, 
            "t": self.teacher_hidden_size,
            "relu": nn.ReLU()
        }
        # auto-parse projector config strings to construct nn.Module
        for projector_name in projector_config:
            # for d in projector_config[loc]:
            if projector_config[projector_name]["enabled"]:
                self.projectors[projector_name] = nn.Sequential()

                structure = projector_config[projector_name]["structure"].split("-")
                for i in range(len(structure)):
                    if structure[i] not in ["relu"]:
                        coef = 1 if not len(structure[i][:-1]) else int(structure[i][:-1])
                        base_size = name_dict[structure[i][-1]]
                        structure[i] = coef * base_size

                for i in range(len(structure) - 1):
                    if isinstance(structure[i], int) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(structure[i], structure[i+1])
                        )
                    elif isinstance(structure[i], int) and isinstance(structure[i+1], str):
                        self.projectors[projector_name].append(
                            name_dict[structure[i+1]]
                        )
                        last_size = structure[i]
                    elif isinstance(structure[i], str) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(last_size, structure[i+1])
                        )
                    else:
                        raise NotImplementedError(f"Invalid structure for '{structure}'")
                        
        # load existing projectors if already have
        self.load_existing_projectors()

    def load_existing_projectors(self):
        if self.args.projector_path is not None:
            projector_path = os.path.join(self.args.projector_path, "projector.pt")
        else:
            projector_path = os.path.join(self.args.model_path, "projector.pt")

        if os.path.exists(projector_path):
            projector_params = torch.load(projector_path, map_location=f"cuda:{self.device}")
            log_rank("Existing projector params: {}".format(list(projector_params.keys())))
            for key in self.projectors:
                try:
                    state_dict = {
                        n.split('.', 1)[1]: projector_params[n] for n in projector_params if n.startswith(key)
                    }
                    self.projectors[key].load_state_dict(state_dict)
                    log_rank("Load projector '{}' from current path.".format(key))
                except:
                    log_rank("Not compatible for projector '{}'".format(key))
                    continue
    
    def load_student_model(self):
        log_rank("Loading student model...")
    
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
                config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True)
                config.is_model_parallel = False
        
                # lấy tokenizer
                tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp")
                
                if hasattr(config, "n_embed"):
                    self.hidden_size = config.n_embed
                else:
                    self.hidden_size = config.hidden_size
        
                config.num_labels = self.args.num_labels
                model = AutoModelForCausalLM.from_pretrained(
                    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                
                model.config.pad_token_id = 2
                    
                model = PeftModel.from_pretrained(
                    model,
                    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
                )
                model = model.merge_and_unload()  # This can take several minutes on cpu

                model = PeftModel.from_pretrained(
                    model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"
                )

                # Apply new LoRA adapter for fine-tuning
                
                '''if self.args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,  # Use SEQ_CLS instead of FEATURE_EXTRACTION
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                    )'''
                if self.args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,  # SEQ_CLS là hợp lý nếu đang làm classification
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"
                        ]
                    )
                    model = get_peft_model(model, peft_config)

            else:
                raise NotImplementedError
        else: #for BERT
            config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
            config.is_model_parallel = False
    
            # lấy tokenizer
            tokenizer = self.load_tokenizer("bert-base-uncased")
            
            if hasattr(config, "n_embed"):
                self.hidden_size = config.n_embed
            else:
                self.hidden_size = config.hidden_size
            config.num_labels = self.args.num_labels
            model = AutoModel.from_pretrained(
                "bert-base-uncased", 
                config=config, 
                device_map=None, 
                torch_dtype=self.dtype,
                trust_remote_code=True,)
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        model = CustomModelForMultipleChoice(model, config)
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            trust_remote_code=True
        )
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp")

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        config.num_labels = self.args.num_labels
        model = AutoModelForMultipleChoice.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        model.config.pad_token_id = 2
        teacher_model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        )    
        
        teacher_model = teacher_model.merge_and_unload()  # This can take several minutes on cpu

        # Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
        teacher_model = PeftModel.from_pretrained(
            teacher_model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"
        )
        teacher_model = teacher_model.merge_and_unload()  # This can take several minutes on cpu
        teacher_model = PeftModel.from_pretrained(
            teacher_model,
            self.args.teacher_model_path,
        )
      
        for param in teacher_model.parameters():
            param.requires_grad = False
        
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
class CustomModelForMultipleChoice(nn.Module):
    def __init__(self, base_model, config):
        super(CustomModelForMultipleChoice, self).__init__()
        self.base_model = base_model
        self.config = config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, output_hidden_states=True, output_attentions=True, **kwargs):
        # Flatten input for processing
        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)
        
        # Reshape inputs to match the base model's expectations
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        # Get outputs from the base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        
        # Extract the pooled output and other model outputs
        if hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
            hidden_states = outputs.hidden_states if output_hidden_states else None
            attentions = outputs.attentions if output_attentions else None
        else:
            # For models without pooler or with tuple outputs
            if isinstance(outputs, tuple):
                last_hidden_state = outputs[0]
                hidden_states = outputs[1] if len(outputs) > 1 and output_hidden_states else None
                attentions = outputs[2] if len(outputs) > 2 and output_attentions else None
                pooled_output = last_hidden_state[:, 0]  # Use [CLS] token
            else:
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
                pooled_output = hidden_states[-1]  # Use [CLS] token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Reshape logits back to [batch_size, num_choices]
        reshaped_logits = logits.view(batch_size, num_choices)
        
        
        # Return a comprehensive dictionary with all outputs
        return {
            'logits': reshaped_logits,
            'hidden_states': hidden_states,
            'attentions': attentions,
        }
    
    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)
