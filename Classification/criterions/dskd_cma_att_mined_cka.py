import torch
from .various_divergence import VariousDivergence
import editdistance
import torch.nn as nn
import re

class DSKD_CMA_ATT_MINED_CKA(VariousDivergence):
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
        logits = outputs.logits
        log = {}
        
        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers

        # Bản đồ token đặc biệt
        TOKENIZER_TO_SPECIAL_TOKEN = {
            type(tokenizer_teacher): "<s>",  # Token đặc biệt của teacher
            type(tokenizer_student): "[CLS]"   # Token đặc biệt của student
        }
        
        def preprocess_text(text):
            text = text.lower()
        
            text = re.sub(r'[^\w\s]', '', text)

            # Remove numbers if specified
            text = re.sub(r'\d+', '', text)

            # Custom list of English stopwords (a common subset)
            stop_words = [
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'would', 'could', 'should', 'ought', 'i\'m', 'you\'re', 'he\'s',
                'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve',
                'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll',
                'she\'ll', 'we\'ll', 'they\'ll', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t',
                'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t',
                'shan\'t', 'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s',
                'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s',
                'why\'s', 'how\'s', '.'
            ]


            words = [word for word in text.split() if word not in stop_words]
            text = ' '.join(words)

            return text

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
            # Giả sử tokenizer_teacher và tokenizer_student đã được khởi tạo
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

        # Hàm lấy chỉ số (indices) từ ánh xạ reciprocal_mapping
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
        
        # Hàm trích xuất top k tokens dựa trên attention của lớp cuối cùng
        def extract_top_k_tokens(text, k):
            # Tiền xử lý văn bản: loại stopwords và dấu câu
            text = preprocess_text(text)

            # Load model và tokenizer
            # phải lấy output từ teacher model để rank
            
            tokenizer = tokenizer_teacher

            # Tokenize văn bản
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Lấy output và attention weights
            with torch.no_grad():
                outputs = teacher_model(**inputs,
                output_hidden_states=True)

            # Lấy attention từ lớp cuối cùng: [num_heads, seq_len, seq_len]
            last_layer_attention = outputs.attentions[-1].squeeze(0)  # loại bỏ batch dimension

            # Trung bình hoá attention trên các head: kết quả [seq_len, seq_len]
            avg_attention = last_layer_attention.mean(dim=0)

            # Tính tổng attention mà mỗi token nhận được
            token_importance = avg_attention.sum(dim=0).cpu().numpy()

            # Lấy danh sách các token gốc
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

            # Ghép token với importance
            token_importance_pairs = list(zip(tokens, token_importance))

            # Sắp xếp giảm dần theo importance và lấy top k
            top_k_tokens = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)[:k]

            return top_k_tokens

        # Hàm kết hợp reciprocal mapping và lọc ra top k token dựa trên attention
        def get_top_k_reciprocal_mapping(text):
            # Lấy ánh xạ song phương giữa teacher và student
            text = text.lower()
        
            text = re.sub(r'[^\w\s]', '', text)
            reciprocal_mapping = align_text_tokens(text)
            n = len(reciprocal_mapping)
            
            top_k = extract_top_k_tokens(text, n//3)
            top_k_tokens_set = {token for token, _ in top_k}
            # Lọc reciprocal mapping chỉ giữ các token teacher có trong top k
            reciprocal_mapping_top_k = {t: s for t, s in reciprocal_mapping.items() if t in top_k_tokens_set}
            return reciprocal_mapping_top_k
        
        class CKALoss(nn.Module):
            """
            Loss with knowledge distillation.
            """
            def __init__(self, eps ):
                super().__init__()
                self.eps = eps
            def forward(self, SH, TH): 
                dT = TH.size(-1)
                dS = SH.size(-1)
                SH = SH.view(-1,dS).to(SH.device,torch.float64)
                TH = TH.view(-1,dT).to(SH.device,torch.float64)
                
                slen = SH.size(0)
                        # Dropout on Hidden State Matching
                SH = SH - SH.mean(0, keepdim=True)
                TH = TH - TH.mean(0, keepdim=True)
                        
                num = torch.norm(SH.t().matmul(TH),'fro')
                den1 = torch.norm(SH.t().matmul(SH),'fro') + self.eps
                den2 = torch.norm(TH.t().matmul(TH),'fro') + self.eps
                
                return 1 - num/torch.sqrt(den1*den2)
            
        def compute_att_loss_2(teacher_model, student_model, input_data, k):
            att_loss_total = 0.0
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

                input_ids_teacher = tokenizer_teacher.encode(text, return_tensors='pt').to(device)
                input_ids_student = tokenizer_student.encode(text, return_tensors='pt').to(device)
                attention_mask_teacher = tokenizer_teacher(text, return_tensors='pt')['attention_mask'].to(device)
                attention_mask_student = tokenizer_student(text, return_tensors='pt')['attention_mask'].to(device)

                # Lấy reciprocal_mapping top k và các chỉ số tương ứng
                reciprocal_mapping_top_k = get_top_k_reciprocal_mapping(text)
                teacher_indices, student_indices = get_indices_from_mapping(text, reciprocal_mapping_top_k)
                print("Teacher indices (top-k):", teacher_indices)
                print("Student indices (top-k):", student_indices)
                print('Lấy xong mapping')

                # Chạy mô hình với output_attentions=True
                teacher_outputs = teacher_model(input_ids_teacher, attention_mask=attention_mask_teacher, output_attentions=True)
                student_outputs = student_model(input_ids_student, attention_mask=attention_mask_student, output_attentions=True)
                print('Đã chạy mô hình')

                # Lấy attention weights từ outputs
                teacher_atts = teacher_outputs.attentions
                student_atts = student_outputs.attentions

                # Tính layers_per_block để ánh xạ layer của teacher sang student
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                layers_per_block = teacher_layer_num // student_layer_num

                # Chọn các layer của teacher tương ứng
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)]

                # Lấy k layer cuối (k tương ứng với số layer sử dụng để tính loss)
                teacher_last_k_layers = new_teacher_atts[-k:]
                student_last_k_layers = student_atts[-k:]
                print('Bắt đầu vào vòng lặp tính loss')

                # Lặp qua từng layer trong k layer cuối
                for teacher_att, student_att in zip(teacher_last_k_layers, student_last_k_layers):
                    print(f"Processing layer với {teacher_att.shape[0]} head(s)")
                    # Lấy ma trận attention cho k token đối với tất cả các token:
                    # - Với teacher: shape (k, t) với t là số token toàn bộ của text theo tokenizer_teacher
                    # - Với student: shape (k, s) với s là số token toàn bộ của text theo tokenizer_student

                    teacher_att_for_k_token = teacher_att[0, :, teacher_indices, :].mean(dim=0)  # (k, t)
                    student_att_for_k_token = student_att[0, :, student_indices, :].mean(dim=0)   # (k, s)
                    print('Lấy xong ma trận attention cho k token')

                    # Xử lý các giá trị attention nhỏ
                    teacher_att_for_k_token = torch.where(
                        teacher_att_for_k_token <= -1e2,
                        torch.zeros_like(teacher_att_for_k_token).to(device),
                        teacher_att_for_k_token
                    )
                    student_att_for_k_token = torch.where(
                        student_att_for_k_token <= -1e2,
                        torch.zeros_like(student_att_for_k_token).to(device),
                        student_att_for_k_token
                    )
                    print("Teacher attention shape (k x t):", teacher_att_for_k_token.shape)
                    print("Student attention shape (k x s):", student_att_for_k_token.shape)

                    # Khởi tạo CKALoss
                    cka_loss_fn = CKALoss(eps=1e-6).to(device)

                    # Tính CKALoss giữa 2 ma trận
                    cka_loss = cka_loss_fn(student_att_for_k_token, teacher_att_for_k_token)

                    print("CKA Loss:", cka_loss.item())
                    att_loss_total  += cka_loss   

            return att_loss_total
    
        att_loss_total = compute_att_loss_2(teacher_model, model, input_data, 3) # define lại batches

        # Cross-entropy loss with ground-truth labels
        loss_ce = self.compute_cross_entropy_loss(outputs.logits, output_data["labels"])[0]
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute dual-space KD loss with CMA
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * (kd_loss + batch_denom * att_loss_total) # Hàm loss cuối cùng
        log["loss"] = loss

        accuracy = self.compute_accuracy(
            logits, output_data["labels"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output
    
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