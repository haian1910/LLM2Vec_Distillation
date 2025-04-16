import torch
from .various_divergence import VariousDivergence
import editdistance
import torch.nn as nn
import re

class DSKD_CMA_ATT_MINED(VariousDivergence):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized

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

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )

        logits = outputs.logits
        log = {}
        
        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers

        # Bản đồ token đặc biệt
        TOKENIZER_TO_SPECIAL_TOKEN = {
            type(tokenizer_teacher): "<s>",  # Token đặc biệt của teacher
            type(tokenizer_student): "[CLS]"   # Token đặc biệt của student
        }
        # Hàm tìm ánh xạ token tốt nhất bằng MinED
        def find_best_mapping(x, base_tokens, blending_special, base_special, best_one=True):
            tmp_x = x.replace(blending_special, base_special)
            if tmp_x in base_tokens:
                return tmp_x, tmp_x
            else:
                if best_one:
                    best = None
                    best_dist = None
                    for y in base_tokens:
                        d = editdistance.eval(tmp_x, y)
                        if best is None or d < best_dist:
                            best = y
                            best_dist = d
                    return tmp_x, best
                else:
                    token_and_distance = [(y, editdistance.eval(tmp_x, y)) for y in base_tokens]
                    min_distance = min(d for _, d in token_and_distance)
                    shortest_distance_tokens = [y for y, d in token_and_distance if d == min_distance]
                    return tmp_x, shortest_distance_tokens

        # Hàm ánh xạ token song hướng giữa teacher và student
        def align_text_tokens(text):
            teacher_tokens = set(tokenizer_teacher.tokenize(text))
            student_tokens = set(tokenizer_student.tokenize(text))
            teacher_special = TOKENIZER_TO_SPECIAL_TOKEN[type(tokenizer_teacher)]
            student_special = TOKENIZER_TO_SPECIAL_TOKEN[type(tokenizer_student)]
            teacher_to_student = {}
            for t in teacher_tokens:
                _, s = find_best_mapping(t, student_tokens, teacher_special, student_special, best_one=True)
                teacher_to_student[t] = s
            student_to_teacher = {}
            for s in student_tokens:
                _, t = find_best_mapping(s, teacher_tokens, student_special, teacher_special, best_one=True)
                student_to_teacher[s] = t
            reciprocal_mapping = {}
            for t, s in teacher_to_student.items():
                if s in student_to_teacher and student_to_teacher[s] == t:
                    reciprocal_mapping[t] = s
            return reciprocal_mapping

        # Hàm lấy chỉ số (indices) từ ánh xạ
        def get_indices_from_mapping(text, reciprocal_mapping):
            input_ids_teacher = tokenizer_teacher.encode(text, return_tensors='pt')[0]
            input_ids_student = tokenizer_student.encode(text, return_tensors='pt')[0]
            
            # Tạo tập hợp các token_id duy nhất từ reciprocal_mapping
            teacher_token_ids = {tokenizer_teacher.convert_tokens_to_ids(t) for t in reciprocal_mapping.keys()}
            student_token_ids = {tokenizer_student.convert_tokens_to_ids(s) for s in reciprocal_mapping.values()}
            
            # Chọn chỉ số đầu tiên cho mỗi token_id trong teacher
            teacher_indices = []
            seen_teacher = set()  # Theo dõi các token_id đã xử lý
            for idx, token_id in enumerate(input_ids_teacher):
                tid = token_id.item()
                if tid in teacher_token_ids and tid not in seen_teacher:
                    teacher_indices.append(idx)
                    seen_teacher.add(tid)
            # Chọn chỉ số đầu tiên cho mỗi token_id trong student
            student_indices = []
            seen_student = set()  # Theo dõi các token_id đã xử lý
            for idx, token_id in enumerate(input_ids_student):
                tid = token_id.item()
                if tid in student_token_ids and tid not in seen_student:
                    student_indices.append(idx)
                    seen_student.add(tid)
            
            return teacher_indices, student_indices
        def preprocess_text(text):

            text = text.lower()
        
            text = re.sub(r'[^\w\s]', '', text)

            return text

        # Hàm tính att_loss cho toàn bộ batch
        def compute_att_loss(teacher_model, student_model, input_data, k):
            att_loss_total = 0.0
            loss_mse = nn.MSELoss()
            device = teacher_model.device

            # Lấy tokenizer từ distiller (giả sử đã được định nghĩa trong class)
            tokenizer_student = distiller.student_tokenizer
            tokenizer_teacher = distiller.teacher_tokenizers

            # Lấy batch_size từ input_ids
            batch_size = input_data["input_ids"].shape[0]

            # Hàm decode input_ids thành văn bản
            def decode_input_ids(tokenizer, input_ids):
                return tokenizer.decode(input_ids, skip_special_tokens=True)

            # Duyệt qua từng sample trong batch
            for i in range(batch_size):
                # Decode input_ids để lấy văn bản (giả sử teacher và student dùng cùng input)
                text = decode_input_ids(tokenizer_student, input_data["input_ids"][i])
                print(f"Processing text: {text}")

                # Tiền xử lý văn bản
                text = preprocess_text(text)

                # Tokenize văn bản cho teacher và student
                input_ids_teacher = tokenizer_teacher.encode(text, return_tensors='pt').to(device)
                input_ids_student = tokenizer_student.encode(text, return_tensors='pt').to(device)
                attention_mask_teacher = tokenizer_teacher(text, return_tensors='pt')['attention_mask'].to(device)
                attention_mask_student = tokenizer_student(text, return_tensors='pt')['attention_mask'].to(device)

                # Lấy reciprocal_mapping và indices
                reciprocal_mapping = align_text_tokens(text)
                teacher_indices, student_indices = get_indices_from_mapping(text, reciprocal_mapping)

                # Chạy mô hình với output_attentions=True
                teacher_outputs = teacher_model(input_ids_teacher, attention_mask=attention_mask_teacher, output_attentions=True)
                student_outputs = student_model(input_ids_student, attention_mask=attention_mask_student, output_attentions=True)

                # Lấy attention weights từ outputs
                teacher_atts = teacher_outputs.attentions
                student_atts = student_outputs.attentions

                # Tính layers_per_block để ánh xạ layer của teacher sang student
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                layers_per_block = teacher_layer_num // student_layer_num

                # Chọn các layer của teacher tương ứng
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)]

                # Lấy k layer cuối
                teacher_last_k_layers = new_teacher_atts[-k:]
                student_last_k_layers = student_atts[-k:]

                # Lặp qua từng layer trong k layer cuối
                for teacher_att, student_att in zip(teacher_last_k_layers, student_last_k_layers):
                    # Lấy ma trận attention cho n token
                    teacher_att_for_n_token = teacher_att[0, :, teacher_indices, :][:, :, teacher_indices].mean(dim=0)  # (num_heads, n, n)
                    student_att_for_n_token = student_att[0, :, student_indices, :][:, :, student_indices].mean(dim=0)   # (num_heads, n, n)
                    # Xử lý giá trị nhỏ
                    teacher_att_for_n_token = torch.where(
                        teacher_att_for_n_token <= -1e2,
                        torch.zeros_like(teacher_att_for_n_token).to(device),
                        teacher_att_for_n_token
                    )
                    student_att_for_n_token = torch.where(
                        student_att_for_n_token <= -1e2,
                        torch.zeros_like(student_att_for_n_token).to(device),
                        student_att_for_n_token
                    )
                    # Tính MSE và cộng vào att_loss_total
                    att_loss_total += loss_mse(student_att_for_n_token, teacher_att_for_n_token)

            return att_loss_total

        att_loss_total = compute_att_loss(teacher_model, model,input_data, 9) # define lại batches 
        # Cross-entropy loss with ground-truth labels
        loss_ce = self.compute_cross_entropy_loss(outputs.logits, output_data["labels"])[0]
        
        # Compute dual-space KD loss with CMA
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        
        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * (kd_loss + att_loss_total) # Hàm loss cuối cùng
        log["loss"] = loss

        accuracy = self.compute_accuracy(
            logits, output_data["labels"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss , logging_output
    
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # Target for classification: shape [batch_size]
        target = output_data["labels"]
        
        # For BERT-like models: use [CLS] token (index 0); adjust if needed for other architectures
        hiddens = outputs.hidden_states[-1][:, 0, :]
        teacher_hiddens = teacher_outputs.hidden_states[-1][:, 0, :]

        # Embedding extraction for student and teacher
        if hasattr(distiller.student_model, "get_input_embeddings"):
            stu_embed_tokens = distiller.student_model.get_input_embeddings()  # Works for BERT, LLaMA, etc.
        elif hasattr(distiller.student_model, "bert") and hasattr(distiller.student_model.bert, "embeddings"):
            stu_embed_tokens = distiller.student_model.bert.embeddings.word_embeddings  # BERT-specific
        elif hasattr(distiller.student_model, "model") and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens  # LLaMA-like
        elif hasattr(distiller.student_model, "transformer") and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte  # GPT-like
        else:
            raise NotImplementedError("Unsupported student model architecture for embedding extraction")

        # Embedding extraction for teacher (LLaMA or similar)
        teacher_model = distiller.teacher_model
        if hasattr(teacher_model, "get_input_embeddings"):
            tea_embed_tokens = teacher_model.get_input_embeddings()  # Universal method, should work for LLaMA
        elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.embed_tokens  # LLaMA-specific
        elif hasattr(teacher_model, "bert") and hasattr(teacher_model.bert, "embeddings"):
            tea_embed_tokens = teacher_model.bert.embeddings.word_embeddings  # BERT-specific
        else:
            raise NotImplementedError("Unsupported teacher model architecture for embedding extraction")

        # Use input_ids as context for CMA (no padding_id needed for classification)
        stu_input_embeds = stu_embed_tokens(input_data["input_ids"][:, 0]).detach()  # [CLS] token embedding
        tea_input_embeds = tea_embed_tokens(input_data["teacher_input_ids"][:, 0]).detach()  # [CLS] token embedding

        # Normalize teacher embeddings
        norm_tea_input_embeds = tea_input_embeds / tea_input_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        # CMA projections
        stu_q_hiddens = distiller.projectors["query"](stu_input_embeds).float()
        tea_k_hiddens = norm_tea_input_embeds.float()

        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens).float()

        # Alignment computation
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / (hiddens.shape[-1] ** 0.5)  # Scale by sqrt of hidden size

        # Teacher-to-Student (t2s) projection
        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)

        # Use appropriate classification head for student
        if hasattr(distiller.student_model, "classifier"):
            t2s_logits = distiller.student_model.classifier(t2s_hiddens)
        elif hasattr(distiller.student_model, "score"):
            t2s_logits = distiller.student_model.score(t2s_hiddens)
        else:
            raise AttributeError("Student model has neither 'classifier' nor 'score' attribute")

        # Compute t2s losses
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_kd_loss = self.dist_func(outputs.logits, t2s_logits.detach(), target, reduction="mean")

        # Student-to-Teacher (s2t) projection
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)

        # Use appropriate classification head for teacher
        if hasattr(distiller.teacher_model, "classifier"):
            s2t_logits = distiller.teacher_model.classifier(s2t_hiddens)
        elif hasattr(distiller.teacher_model, "score"):
            s2t_logits = distiller.teacher_model.score(s2t_hiddens)
        else:
            raise AttributeError("Teacher model has neither 'classifier' nor 'score' attribute")

        # Compute s2t loss
        s2t_kd_loss = self.compute_forward_kl_divergence(s2t_logits, teacher_outputs.logits, target, reduction="mean")

        # Combine KD losses
        kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss

        # Compute accuracies
        t2s_acc = (t2s_logits.argmax(-1) == target).float().mean()
        s2t_acc = (s2t_logits.argmax(-1) == target).float().mean()

        # Logging
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_acc"] = t2s_acc
        log["s2t_acc"] = s2t_acc
        log["kd_loss"] = kd_loss

        return kd_loss, log
