import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
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
from utils import log_rank
from huggingface_hub import login
import time
import requests

token = os.getenv("HF_TOKEN")
login(token=token)


class MeanPoolingClassifier(nn.Module):
    """Wrapper class that applies mean pooling over model outputs and classifies."""
    def __init__(self, base_model, hidden_size, num_labels, model_type="bert"):
        super(MeanPoolingClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_labels, 
                              dtype=next(base_model.parameters()).dtype)
        self.model_type = model_type.lower()
        self.config = base_model.config
    
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, output_attentions=False, output_hidden_states=False, **kwargs):
        # Remove token_type_ids for LLaMA models
        if self.model_type in ["llama", "mistral", "sheared-llama"]:
            if "token_type_ids" in kwargs:
                del kwargs["token_type_ids"]
            
            # Get the model outputs
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs
            )
        else:
            # For BERT and other models that accept token_type_ids
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs
            )
        
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Ensure mean_embeddings has the same dtype as classifier weights
        mean_embeddings = mean_embeddings.to(self.classifier.weight.dtype)
        logits = self.classifier(mean_embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        class ModelOutput:
            def __init__(self, loss, logits, hidden_states, last_hidden_state, attentions=None):
                self.loss = loss
                self.logits = logits
                self.hidden_states = hidden_states
                self.last_hidden_state = last_hidden_state
                self.attentions = attentions
        
        return ModelOutput(
            loss, 
            logits, 
            outputs.hidden_states if hasattr(outputs, 'hidden_states') else None, 
            token_embeddings,
            outputs.attentions if hasattr(outputs, 'attentions') else None
        )
    
    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        """Save the model to the specified directory."""
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the classifier separately
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)
        
        # Save wrapper config
        config_dict = {
            "hidden_size": self.classifier.in_features,
            "num_labels": self.classifier.out_features,
            "model_type": self.model_type
        }
        with open(os.path.join(save_directory, "mean_pooling_config.json"), "w") as f:
            json.dump(config_dict, f)
        
        # Save the base model
        return self.base_model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from pretrained."""
        # First load the base model
        base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Load classifier config
        config_path = os.path.join(pretrained_model_name_or_path, "mean_pooling_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            hidden_size = config_dict.get("hidden_size")
            num_labels = config_dict.get("num_labels")
            model_type = config_dict.get("model_type", "bert")
        else:
            # Use defaults or passed kwargs
            hidden_size = kwargs.pop("hidden_size", base_model.config.hidden_size if hasattr(base_model.config, "hidden_size") else base_model.config.n_embed)
            num_labels = kwargs.pop("num_labels", 2)  # Default to binary classification
            model_type = kwargs.pop("model_type", "bert")
        
        # Create the wrapper
        model = cls(base_model, hidden_size, num_labels, model_type=model_type)
        
        # Load classifier weights if they exist
        classifier_path = os.path.join(pretrained_model_name_or_path, "classifier.pt")
        if os.path.exists(classifier_path):
            classifier_state_dict = torch.load(classifier_path, map_location="cpu")
            model.classifier.load_state_dict(classifier_state_dict)
        
        return model
        

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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log_rank(f"Loading tokenizer from {path} (attempt {attempt+1}/{max_retries})")
                tokenizer = AutoTokenizer.from_pretrained(
                    path, 
                    trust_remote_code=True,
                    use_fast=True,  # Using fast tokenizer when available
                    local_files_only=False  # Force downloading if not available locally
                )
                return tokenizer
            except Exception as e:
                log_rank(f"Error loading tokenizer: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    log_rank(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    log_rank("Failed to load tokenizer after multiple attempts")
                    raise
        
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
    
    def _load_model_with_retries(self, model_name, model_cls, config, max_retries=3):
        """Load model with retry mechanism"""
        for attempt in range(max_retries):
            try:
                log_rank(f"Loading {model_cls.__name__} from {model_name} (attempt {attempt+1}/{max_retries})")
                model = model_cls.from_pretrained(
                    model_name,
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                log_rank(f"Successfully loaded {model_cls.__name__} from {model_name}")
                return model
            except Exception as e:
                log_rank(f"Error loading model: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    log_rank(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    log_rank("Failed to load model after multiple attempts")
                    raise
    
    def _get_model_type(self, model_name):
        """Determine model type based on model name."""
        model_name = model_name.lower()
        if "llama" in model_name:
            return "llama"
        elif "mistral" in model_name:
            return "mistral"
        elif "bert" in model_name:
            return "bert"
        else:
            # Default to a generic type that doesn't use token_type_ids
            log_rank(f"Unknown model type for {model_name}, defaulting to generic model type")
            return "generic"
    
    def load_student_model(self):
        log_rank("Loading student model...")
    
        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError(f"Invalid model_dtype for `{self.args.model_dtype}`")

        if self.args.peft is not None: #for LLM2Vec
            if self.args.peft == "lora":
                model_name = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
                model_type = self._get_model_type(model_name)
                
                try:
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    config.is_model_parallel = False
                    config.output_attentions = True  # Add this parameter to output attentions
            
                    # Get tokenizer
                    tokenizer = self.load_tokenizer(model_name)
                    
                    if hasattr(config, "n_embed"):
                        self.hidden_size = config.n_embed
                    else:
                        self.hidden_size = config.hidden_size
            
                    # Load base model with retries
                    base_model = self._load_model_with_retries(
                        model_name, AutoModel, config
                    )

                    base_model.config.pad_token_id = tokenizer.pad_token_id or 2
                        
                    base_model = PeftModel.from_pretrained(
                        base_model,
                        model_name,
                    )
                    base_model = base_model.merge_and_unload()  # This can take several minutes on cpu

                    base_model = PeftModel.from_pretrained(
                        base_model, f"{model_name}-unsup-simcse"
                    )

                    # Create wrapper classification model with mean pooling
                    model = MeanPoolingClassifier(
                        base_model, 
                        self.hidden_size, 
                        self.args.num_labels,
                        model_type=model_type
                    )

                    # Apply new LoRA adapter for fine-tuning
                    if self.args.do_train:
                        peft_config = LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,  # Changed to FEATURE_EXTRACTION for base model
                            inference_mode=(not self.args.do_train),
                            r=self.args.peft_lora_r,
                            lora_alpha=self.args.peft_lora_alpha,
                            lora_dropout=self.args.peft_lora_dropout,
                            target_modules=[
                                "q_proj", "k_proj", "v_proj", "o_proj", 
                                "gate_proj", "up_proj", "down_proj"
                            ]
                        )
                        # Only apply LoRA to the base model part
                        model.base_model = get_peft_model(model.base_model, peft_config)
                        model.print_trainable_parameters = lambda: log_rank(
                            f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}/{sum(p.numel() for p in model.parameters())} = {sum(p.numel() for p in model.parameters() if p.requires_grad)/sum(p.numel() for p in model.parameters()):.2%}'
                        )
                        model.print_trainable_parameters()
                        
                except Exception as e:
                    log_rank(f"Error initializing LLM2Vec model: {e}")
                    raise
            else:
                raise NotImplementedError(f"PEFT method {self.args.peft} not implemented")
        else: #for BERT
            model_name = "bert-base-uncased"
            model_type = self._get_model_type(model_name)
            
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config.is_model_parallel = False
                config.output_attentions = True  # Add this parameter to output attentions
        
                # Get tokenizer
                tokenizer = self.load_tokenizer(model_name)
                
                if hasattr(config, "n_embed"):
                    self.hidden_size = config.n_embed
                else:
                    self.hidden_size = config.hidden_size
                    
                # Load base model instead of classification model
                base_model = self._load_model_with_retries(
                    model_name, AutoModel, config
                )
                
                # Create wrapper classification model with mean pooling
                model = MeanPoolingClassifier(
                    base_model, 
                    self.hidden_size, 
                    self.args.num_labels,
                    model_type=model_type
                )
                
                log_rank(' > number of parameters: {:,}'.format(
                    sum([p.nelement() for p in model.parameters()])
                ))
            except Exception as e:
                log_rank(f"Error initializing BERT model: {e}")
                raise

        if self.args.gradient_checkpointing and hasattr(model.base_model, "gradient_checkpointing_enable"):
            model.base_model.gradient_checkpointing_enable()
            log_rank("Gradient checkpointing enabled")

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        
        try:
            teacher_model_path = self.args.teacher_model_path
            log_rank(f"Loading teacher model from path: {teacher_model_path}")
            
            # Check if this is a saved MeanPoolingClassifier model
            mean_pooling_config_path = os.path.join(teacher_model_path, "mean_pooling_config.json")
            config_path = os.path.join(teacher_model_path, "config.json")
            
            if os.path.exists(mean_pooling_config_path) and os.path.exists(config_path):
                log_rank("Found mean_pooling_config.json and config.json - loading directly")
                
                # Read mean pooling config directly from file
                with open(mean_pooling_config_path, "r") as f:
                    mean_pooling_config = json.load(f)
                
                # Load the config directly from file without using from_dict
                config = AutoConfig.from_pretrained(config_path)
                config.output_attentions = True  # Add this parameter to output attentions
                
                # Get the model type and hidden size from configs
                model_type = mean_pooling_config.get("model_type", "llama")
                self.teacher_hidden_size = mean_pooling_config.get("hidden_size", 
                                          config.hidden_size if hasattr(config, "hidden_size") 
                                          else config.n_embed)
                num_labels = mean_pooling_config.get("num_labels", self.args.num_labels)
                
                # Load tokenizer from the same path
                try:
                    tokenizer = self.load_tokenizer(teacher_model_path)
                except Exception as e:
                    log_rank(f"Error loading tokenizer from checkpoint: {e}")
                    # Fall back to original model's tokenizer
                    tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp")
                
                # First, load the base model with the config
                log_rank("Loading base model from checkpoint")
                model_class = AutoModel
                base_model = model_class.from_config(config)
                
                # Load the base model's state dict
                model_bin_path = os.path.join(teacher_model_path, "pytorch_model.bin")
                if os.path.exists(model_bin_path):
                    state_dict = torch.load(model_bin_path, map_location="cpu")
                    base_model.load_state_dict(state_dict, strict=False)
                    log_rank("Loaded model weights from pytorch_model.bin")
                else:
                    # Try loading from sharded files if they exist
                    try:
                        from transformers.modeling_utils import load_sharded_checkpoint
                        log_rank("No pytorch_model.bin found, trying sharded files")
                        load_sharded_checkpoint(base_model, teacher_model_path)
                        log_rank("Loaded model weights from sharded files")
                    except Exception as e:
                        log_rank(f"Error loading model weights: {e}")
                        log_rank("Will try to create model from scratch")
                        
                        # Fall back to loading from original path
                        model_name = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
                        base_model = self._load_model_with_retries(model_name, model_class, config)
                
                # Create the classifier wrapper
                teacher_model = MeanPoolingClassifier(
                    base_model, 
                    self.teacher_hidden_size, 
                    num_labels,
                    model_type=model_type
                )
                
                # Load classifier weights if they exist
                classifier_path = os.path.join(teacher_model_path, "classifier.pt")
                if os.path.exists(classifier_path):
                    classifier_state_dict = torch.load(classifier_path, map_location="cpu")
                    teacher_model.classifier.load_state_dict(classifier_state_dict)
                    log_rank("Loaded classifier weights")
                
            else:
                log_rank("No mean_pooling_config.json found - loading with PEFT adapters")
                model_name = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
                model_type = self._get_model_type(model_name)
                
                # Load config from base model, not from teacher_model_path
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config.is_model_parallel = False
                config.output_attentions = True  # Add this parameter to output attentions
                
                tokenizer = self.load_tokenizer(model_name)
                
                if hasattr(config, "n_embed"):
                    self.teacher_hidden_size = config.n_embed
                else:
                    self.teacher_hidden_size = config.hidden_size
                    
                # Load base model
                base_model = self._load_model_with_retries(
                    model_name, AutoModel, config
                )
                
                base_model.config.pad_token_id = tokenizer.pad_token_id or 2
                
                teacher_base_model = PeftModel.from_pretrained(
                    base_model,
                    model_name,
                )    
                
                teacher_base_model = teacher_base_model.merge_and_unload()
                
                # Loading unsupervised SimCSE model
                teacher_base_model = PeftModel.from_pretrained(
                    teacher_base_model, f"{model_name}-unsup-simcse"
                )
                teacher_base_model = teacher_base_model.merge_and_unload()
                
                teacher_base_model = PeftModel.from_pretrained(
                    teacher_base_model,
                    teacher_model_path,
                )
                
                # Create wrapper classification model with mean pooling
                teacher_model = MeanPoolingClassifier(
                    teacher_base_model, 
                    self.teacher_hidden_size, 
                    self.args.num_labels,
                    model_type=model_type
                )
            
            # Freeze all parameters
            for param in teacher_model.parameters():
                param.requires_grad = False
                
            log_rank("Teacher model loaded successfully")
            
        except Exception as e:
            log_rank(f"Error loading teacher model: {e}")
            raise
        
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