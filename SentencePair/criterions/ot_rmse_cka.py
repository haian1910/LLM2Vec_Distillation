import editdistance
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
import torch.nn as nn
import re
from .cross_entropy_loss import CrossEntropyLoss
import math

class OT_RMSE_CKA(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.sinkhorn_alpha = 0.1
        self.stopThr = 1e-9
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'attention'
        self.importance_scaling = 0.5
    
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
        
        # Student forward pass
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        
        # Compute cross-entropy loss with ground-truth labels
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["labels"]
        )[0]

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers

        # Bản đồ token đặc biệt
        TOKENIZER_TO_SPECIAL_TOKEN = {
            type(tokenizer_teacher): "<s>",  # Token đặc biệt của teacher
            type(tokenizer_student): "[CLS]"   # Token đặc biệt của student
        }
        
        def preprocess_text(text):

            # Remove numbers if specified
            text = re.sub(r'\d+', '', text)

            # Custom list of English stopwords (a common subset)
            stop_words = [
                'a', 'an', 'the', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through',
                'during', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'other', 'such',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now',
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'would', 'could', 'should', 'ought', 'i\'m', 'you\'re', 'he\'s',
                'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve',
                'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll',
                'she\'ll', 'we\'ll', 'they\'ll', 'let\'s', 'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s',
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
            inputs = {key: value.to(teacher_model.device) for key, value in inputs.items()}

            # Lấy output và attention weights
            with torch.no_grad():
                outputs = teacher_model(**inputs,
                output_hidden_states=True,
                output_attentions=True)

            # Lấy attention từ lớp cuối cùng: [num_heads, seq_len, seq_len]
            last_layer_attention = outputs.attentions[-1].squeeze(0)  # loại bỏ batch dimension

            # Trung bình hoá attention trên các head: kết quả [seq_len, seq_len]
            avg_attention = last_layer_attention.mean(dim=0)

            # Tính tổng attention mà mỗi token nhận được
            token_importance = avg_attention.sum(dim=0).to(torch.float32).cpu().numpy()

            # Lấy danh sách các token gốc
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            # Ghép token với importance
            token_importance_pairs = list(zip(tokens, token_importance))

            # Sắp xếp giảm dần theo importance và lấy top k
            top_k_tokens = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)[:k]

            return top_k_tokens

        # Hàm kết hợp reciprocal mapping và lọc ra top k token dựa trên attention
        def get_top_k_reciprocal_mapping(text):
            # Lấy ánh xạ song phương giữa teacher và student
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
        def compute_att_loss_1(teacher_model, student_model, input_data, k):
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
                # print(f"Processing text: {text}")

                # Tiền xử lý văn bản
                text = text.lower()
        
                text = re.sub(r'[^\w\s]', '', text)

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
                text = text.lower()
        
                text = re.sub(r'[^\w\s]', '', text)

                input_ids_teacher = tokenizer_teacher.encode(text, return_tensors='pt').to(device)
                input_ids_student = tokenizer_student.encode(text, return_tensors='pt').to(device)
                attention_mask_teacher = tokenizer_teacher(text, return_tensors='pt')['attention_mask'].to(device)
                attention_mask_student = tokenizer_student(text, return_tensors='pt')['attention_mask'].to(device)

                # Lấy reciprocal_mapping top k và các chỉ số tương ứng
                reciprocal_mapping_top_k = get_top_k_reciprocal_mapping(text)
                teacher_indices, student_indices = get_indices_from_mapping(text, reciprocal_mapping_top_k)
                # print("Teacher indices (top-k):", teacher_indices)
                # print("Student indices (top-k):", student_indices)
                # print('Lấy xong mapping')

                # Chạy mô hình với output_attentions=True
                teacher_outputs = teacher_model(input_ids_teacher, attention_mask=attention_mask_teacher, output_attentions=True)
                student_outputs = student_model(input_ids_student, attention_mask=attention_mask_student, output_attentions=True)
                # print('Đã chạy mô hình')

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
                # print('Bắt đầu vào vòng lặp tính loss')

                # Lặp qua từng layer trong k layer cuối
                for teacher_att, student_att in zip(teacher_last_k_layers, student_last_k_layers):
                    # print(f"Processing layer với {teacher_att.shape[0]} head(s)")
                    # Lấy ma trận attention cho k token đối với tất cả các token:
                    # - Với teacher: shape (k, t) với t là số token toàn bộ của text theo tokenizer_teacher
                    # - Với student: shape (k, s) với s là số token toàn bộ của text theo tokenizer_student

                    teacher_att_for_k_token = teacher_att[0, :, teacher_indices, :].mean(dim=0)  # (k, t)
                    student_att_for_k_token = student_att[0, :, student_indices, :].mean(dim=0)   # (k, s)

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
                    # print("Teacher attention shape (k x t):", teacher_att_for_k_token.shape)
                    # print("Student attention shape (k x s):", student_att_for_k_token.shape)

                    # Khởi tạo CKALoss
                    cka_loss_fn = CKALoss(eps=1e-8).to(device)

                    # Tính CKALoss giữa 2 ma trận
                    cka_loss = cka_loss_fn(student_att_for_k_token, teacher_att_for_k_token)

                    
                    att_loss_total  += cka_loss   

            return att_loss_total
        
        att_loss_total_1 = compute_att_loss_1(teacher_model, model,input_data, 3) # define lại batches 
        att_loss_total_2 = compute_att_loss_2(teacher_model, model, input_data, 3) 
        print("rmse_loss:", att_loss_total_1)
        print("cka_loss:", att_loss_total_2)
        
        # Compute distillation loss using optimal transport
        kd_loss, log = self.compute_ot_loss(
            outputs=outputs, 
            teacher_outputs=teacher_outputs, 
            attention_mask_student=input_data["attention_mask"],
            attention_mask_teacher=input_data["teacher_attention_mask"],
            log=log,
            distiller=distiller
        )
        
        print("ot_loss:", kd_loss) 

        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate *(att_loss_total_1 + 0.1*att_loss_total_2 + kd_loss) 
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"]
        )
        log["accuracy"] = accuracy
        return loss, logging_output
    
    def pairwise_euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2)
    
    def pairwise_cosine_distance(self, a, b, eps=1e-8):
        """
        Computes pairwise cosine distance with numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=a.dtype))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=b.dtype))
        sim_mt = torch.mm(a_norm.transpose(0, 1), b_norm)
        
        sim_mt = 1 - sim_mt
        return sim_mt

    def pairwise_attention_distance(self, x, y, eps=1e-8):
        d = x.shape[1]
        sim_mt = torch.mm(x.transpose(0, 1), y) / math.sqrt(d)
        attention_weights = torch.softmax(sim_mt, dim=1)
        dist_mt = 1.0 - attention_weights
        return dist_mt
    
    def compute_ot_loss(
        self, outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log, distiller, logits=False
    ):
        # Get the last hidden state from both models
        student_features = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        teacher_features = teacher_outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
        
        batch_size = teacher_features.size(0)
        total_loss = 0
        
        # Check if projector exists
        if not hasattr(distiller, 'projectors') or 't2s' not in distiller.projectors:
            raise AttributeError("Distiller missing 't2s' projector. Make sure projectors are properly initialized.")
            
        projector = distiller.projectors["t2s"]
        
        for b in range(batch_size):
            # Get sequences for current batch
            teacher_seq = teacher_features[b]  # Shape: (seq_len, hidden_dim)
            student_seq = student_features[b]  # Shape: (seq_len, hidden_dim)

            # Get masks for current batch
            teacher_mask = attention_mask_teacher[b]  # (seq_len)
            student_mask = attention_mask_student[b]  # (seq_len)
            
            # Prune sequences based on the mask
            teacher_seq = teacher_seq[teacher_mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            student_seq = student_seq[student_mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            
            # Project each row of teacher_seq to student space
            projected_teacher_seq = projector(teacher_seq)  # Now project after pruning
            
            # Ensure both tensors are in the same dtype
            dtype = student_seq.dtype
            projected_teacher_seq = projected_teacher_seq.to(dtype)
            
            # Compute cost matrix based on specified distance metric
            if self.ot_dist_type == 'euclidean':
                cost_matrix = self.pairwise_euclidean_distance(student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'cosine':
                cost_matrix = self.pairwise_cosine_distance(student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'attention':
                cost_matrix = self.pairwise_attention_distance(student_seq, projected_teacher_seq)
            else:
                raise ValueError(f"Unknown distance type: {self.ot_dist_type}")
            
            # Ensure cost matrix is in the right dtype
            cost_matrix = cost_matrix.to(dtype)
            
            # Compute OT plan and loss
            ot_loss, transport_plan = self.sinkhorn(cost_matrix)
            total_loss += ot_loss
        
        avg_loss = total_loss / batch_size
        log["ot_loss"] = avg_loss.item()
        
        return avg_loss, log
    
    def sinkhorn(self, cost_matrix, num_iters=None):
        """
        Sinkhorn algorithm for computing optimal transport
        
        Args:
            cost_matrix: Cost matrix of shape (m, n)
            num_iters: Number of iterations (uses self.OT_max_iter if None)
            
        Returns:
            ot_loss: Optimal transport loss
            transport_plan: Transport plan matrix of shape (m, n)
        """
        if num_iters is None:
            num_iters = self.OT_max_iter
        
        m, n = cost_matrix.shape
        dtype = cost_matrix.dtype
        device = cost_matrix.device
        
        # Initialize uniform marginals - ensure correct dtype
        a = torch.ones(m, device=device, dtype=dtype) / m
        b = torch.ones(n, device=device, dtype=dtype) / n
        
        # Initialize transport plan - ensure correct dtype
        K = torch.exp(-cost_matrix / self.sinkhorn_alpha)
        
        # Initialize u with correct dtype
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        # Sinkhorn iterations
        for _ in range(num_iters):
            u_prev = u.clone()
            
            # Use a more stable implementation without walrus operator
            # (original line that caused error):
            # u = a / (torch.matmul(K, v := b / (torch.matmul(K.t(), u) + self.epsilon)) + self.epsilon)
            
            # First compute v
            v = b / (torch.matmul(K.t(), u) + self.epsilon)
            # Then compute u
            u = a / (torch.matmul(K, v) + self.epsilon)
            
            # Check convergence
            err = torch.norm(u - u_prev, p=float('inf'))
            if err < self.stopThr:
                break
        
        # Compute transport plan
        # Create diagonal matrices manually to ensure correct dtype
        diag_u = torch.diag(u)
        diag_v = torch.diag(v)
        transport_plan = torch.matmul(torch.matmul(diag_u, K), diag_v)
        
        # Compute OT loss
        ot_loss = torch.sum(transport_plan * cost_matrix)
        
        return ot_loss, transport_plan