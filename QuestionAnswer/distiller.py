import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from QuestionAnswer.utils import log_rank
from huggingface_hub import login

import os

#login(token="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")
token = os.getenv("HF_TOKEN")
login(token=token)

class MultipleChoiceModel(nn.Module):
    """Wrapper for multiple choice tasks using a base model"""
    def __init__(self, base_model, num_choices=8):
        super(MultipleChoiceModel, self).__init__()
        self.base_model = base_model
        self.num_choices = num_choices
        self.config = base_model.config  # Expose config for save_pretrained
        
        # Get the hidden size from the base model
        self.hidden_size = base_model.config.hidden_size

        # Create a classifier for multiple choice
        self.classifier = nn.Linear(self.hidden_size, 1)

        # Check model type to determine which arguments it accepts
        # self.uses_token_type_ids = "llama" not in base_model.__class__.__name__.lower() and "mistral" not in base_model.__class__.__name__.lower()
        self.uses_token_type_ids = hasattr(base_model.config, "type_vocab_size") and base_model.config.type_vocab_size > 0
    def device(self):
        return next(self.parameters()).device
    def get_input_embeddings(self):
        """Return the input embeddings from the base model"""
        if hasattr(self.base_model, "get_input_embeddings"):
            return self.base_model.get_input_embeddings()
        elif hasattr(self.base_model, "bert") and hasattr(self.base_model.bert, "embeddings"):
            return self.base_model.bert.embeddings.word_embeddings  # BERT-specific
        elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            return self.base_model.model.embed_tokens  # LLaMA-like
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            return self.base_model.transformer.wte  # GPT-like
        else:
            raise NotImplementedError("Unsupported model architecture for embedding extraction")
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, token_type_ids=None, labels=None, **kwargs):
        batch_size = input_ids.size(0) if input_ids is not None else attention_mask.size(0)
        num_choices = input_ids.size(1) if input_ids is not None else attention_mask.size(1)

        # Reshape input tensors to combine batch and choice dimensions
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        # Filter kwargs to only include parameters accepted by the base model
        filtered_kwargs = {}
        for key, value in kwargs.items():
            # Skip labels as they'll be handled separately
            if key == 'labels':
                continue  # Don't pass labels to base model
            if key == 'token_type_ids' and not self.uses_token_type_ids:
                continue  # Don't pass token_type_ids if model doesn't use them

            # Reshape tensors if they have batch and choice dimensions
            if isinstance(value, torch.Tensor) and value.dim() > 2:
                filtered_kwargs[key] = value.view(-1, value.size(-1))
            else:
                filtered_kwargs[key] = value

        # Only pass token_type_ids if the model supports it
        if self.uses_token_type_ids and token_type_ids is not None:
            flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            filtered_kwargs["token_type_ids"] = flat_token_type_ids

        # Make sure we get hidden states and attentions
        filtered_kwargs["output_hidden_states"] = True
        filtered_kwargs["output_attentions"] = True

        # Get outputs from the base model with filtered kwargs
        outputs = self.base_model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            position_ids=flat_position_ids,
            **filtered_kwargs
        )

        # Get the appropriate representation based on model architecture
        # if hasattr(outputs, "pooler_output"):
        #     pooled_output = outputs.pooler_output
        # else:
        #     # Use the [CLS] token representation (first token)
        #     pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = outputs.last_hidden_state[:, 0]

        # Apply the classifier to get logits for each choice
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(batch_size, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # Create a comprehensive output structure matching HuggingFace's transformers output format
        class MultipleChoiceModelOutput:
            def __init__(self, loss, logits, hidden_states, attentions, last_hidden_state=None):
                self.loss = loss
                self.logits = logits
                self.hidden_states = hidden_states
                self.attentions = attentions
                self.last_hidden_state = last_hidden_state

        # Return complete output with original hidden states and attentions
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            last_hidden_state=outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None
        )

    # Add HuggingFace compatibility methods
    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        """Save the model to the specified directory."""
        # Save the classifier separately
        os.makedirs(save_directory, exist_ok=True)
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)

        # Save wrapper config
        config_dict = {
            "num_choices": self.num_choices,
            "uses_token_type_ids": self.uses_token_type_ids
        }
        with open(os.path.join(save_directory, "multiple_choice_config.json"), "w") as f:
            json.dump(config_dict, f)

        # Save the base model
        return self.base_model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from pretrained."""
        # First load the base model
        base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Load classifier config
        config_path = os.path.join(pretrained_model_name_or_path, "multiple_choice_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            num_choices = config_dict.get("num_choices", 8)
        else:
            num_choices = kwargs.pop("num_choices", 8)

        # Create the wrapper
        model = cls(base_model, num_choices=num_choices)

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
        # Add multiple choice specific args
        group.add_argument("--num-choices", type=int, default=8,
                           help='number of choices for multiple choice task')
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

        if self.args.peft is not None: # for LLM2Vec
            if self.args.peft == "lora":
                # Load the base model configuration
                config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True)
                config.is_model_parallel = False

                # Get tokenizer
                tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")

                if hasattr(config, "n_embed"):
                    self.hidden_size = config.n_embed
                else:
                    self.hidden_size = config.hidden_size

                # Load the base model using AutoModel instead of AutoModelForSequenceClassification
                base_model = AutoModel.from_pretrained(
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )

                if hasattr(base_model.config, "pad_token_id"):
                    base_model.config.pad_token_id = 2

                # Apply PEFT
                base_model = PeftModel.from_pretrained(
                    base_model,
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                )
                base_model = base_model.merge_and_unload()  # This can take several minutes on cpu

                base_model = PeftModel.from_pretrained(
                    base_model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
                )

                # Wrap the base model with our multiple choice model
                model = MultipleChoiceModel(base_model, num_choices=self.args.num_choices)

                # Apply new LoRA adapter for fine-tuning
                if self.args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,  # Use FEATURE_EXTRACTION for AutoModel
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"
                        ]
                    )
                    # Apply PEFT to the base model
                    base_model = get_peft_model(base_model, peft_config)
                    base_model.print_trainable_parameters()

                    # Update our wrapped model to use the PEFT-adapted base model
                    model.base_model = base_model
            else:
                raise NotImplementedError(f"PEFT method {self.args.peft} not implemented for multiple choice tasks")
        else: # for BERT
            config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
            config.is_model_parallel = False

            # Get tokenizer
            tokenizer = self.load_tokenizer("bert-base-uncased")

            if hasattr(config, "n_embed"):
                self.hidden_size = config.n_embed
            else:
                self.hidden_size = config.hidden_size

            # Load base model using AutoModel
            base_model = AutoModel.from_pretrained(
                "bert-base-uncased", 
                config=config, 
                device_map=None, 
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

            # Wrap with multiple choice model
            model = MultipleChoiceModel(base_model, num_choices=self.args.num_choices)

            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        if self.args.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model.base_model, "gradient_checkpointing_enable"):
                model.base_model.gradient_checkpointing_enable()

        return model, tokenizer

    def load_teacher_model(self):
        log_rank("Loading teacher model from checkpoint...")

        if not os.path.exists(self.args.teacher_model_path):
            raise ValueError(f"Teacher model path does not exist: {self.args.teacher_model_path}")

        model_files = os.listdir(self.args.teacher_model_path)
        log_rank(f"Found files in teacher model directory: {model_files}")

        # normal loading
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            trust_remote_code=True
        )
        config.is_model_parallel = False
        tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        base_model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        if hasattr(base_model.config, "pad_token_id"):
            base_model.config.pad_token_id = 2

        teacher_base_model = PeftModel.from_pretrained(
            base_model,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        )    

        teacher_base_model = teacher_base_model.merge_and_unload()

        teacher_base_model = PeftModel.from_pretrained(
            teacher_base_model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
        )
        teacher_base_model = teacher_base_model.merge_and_unload()

        teacher_base_model = PeftModel.from_pretrained(
            teacher_base_model,
            self.args.teacher_model_path,
        )

        teacher_model = MultipleChoiceModel(teacher_base_model, num_choices=self.args.num_choices)
        # Load classifier if available
        if os.path.exists("classifier.pt"):
            log_rank("Loading classifier weights")
            classifier_state_dict = torch.load("classifier.pt", map_location="cpu")
            teacher_model.classifier.load_state_dict(classifier_state_dict)
        else:
            log_rank("No classifier.pt found, using initialized classifier")

        # Freeze the teacher model parameters
        for param in teacher_model.parameters():
            param.requires_grad = False

        log_rank("Teacher model loaded successfully")
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
