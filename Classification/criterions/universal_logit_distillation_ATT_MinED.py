from huggingface_hub import login
import editdistance
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
import torch.nn as nn
import re
from .cross_entropy_loss import CrossEntropyLoss


class UniversalLogitDistillation_ATT_MinED(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.k = args.k # số layer mình distill

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        self.distiller = distiller
        model = distiller.student_model # BERT
        teacher_model = distiller.teacher_model # LLM2VEC

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_ids=input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None),
                output_hidden_states=True
            )


        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizer

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
            teacher_token_ids = {tokenizer_teacher.convert_tokens_to_ids(t): tokenizer_student.convert_tokens_to_ids(s)
                                for t, s in reciprocal_mapping.items()}
            teacher_indices = [idx for idx, token_id in enumerate(input_ids_teacher) if token_id.item() in teacher_token_ids]
            student_indices = [idx for idx, token_id in enumerate(input_ids_student) if token_id.item() in teacher_token_ids.values()]
            return teacher_indices, student_indices

        def preprocess_text(text, remove_stopwords=True, remove_punctuation=True,
                    lowercase=True, remove_numbers=True):

            text = text.lower()

            # Remove punctuation if specified

            # Loại bỏ mọi ký tự không phải chữ cái, số hoặc khoảng trắng
        
            text = re.sub(r'[^\w\s]', '', text)

            '''# Remove numbers if specified
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
            text = ' '.join(words)'''

            return text

        # Hàm tính att_loss cho toàn bộ batch
        def compute_att_loss(teacher_model, student_model, batches, k):
            att_loss_total = 0.0
            loss_mse = nn.MSELoss()
            device = teacher_model.l2v.model.device

            # Duyệt qua tất cả các batch
            for batch_texts in batches:
                print(f"Processing batch with {len(batch_texts)} texts")
                batch_att_loss = 0.0
                # Duyệt qua tất cả các text trong batch hiện tại
                for text in batch_texts: 
                    text = preprocess_text(text, remove_stopwords=True, remove_punctuation=True,
                    lowercase=True, remove_numbers=True)
                    
                    print(f"Processing text: {text}")
                    # Tokenize văn bản cho teacher và student
                    input_ids_teacher = tokenizer_teacher.encode(text, return_tensors='pt').to(device)
                    input_ids_student = tokenizer_student.encode(text, return_tensors='pt').to(device)
                    attention_mask_teacher = tokenizer_teacher(text, return_tensors='pt')['attention_mask'].to(device)
                    attention_mask_student = tokenizer_student(text, return_tensors='pt')['attention_mask'].to(device)

                    # Lấy reciprocal_mapping và indices
                    reciprocal_mapping = align_text_tokens(text)
                    teacher_indices, student_indices = get_indices_from_mapping(text, reciprocal_mapping)

                    # Chạy mô hình với output_attentions=True
                    teacher_outputs = teacher_model.l2v.model(input_ids_teacher, attention_mask=attention_mask_teacher, output_attentions=True)
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
                        print(teacher_att_for_n_token.shape)
                        print(student_att_for_n_token.shape)
                        # Tính MSE và cộng vào batch_att_loss
                        batch_att_loss += loss_mse(student_att_for_n_token, teacher_att_for_n_token)

                # Cộng batch_att_loss vào att_loss_total
                att_loss_total += batch_att_loss

            return att_loss_total
        

        att_loss_total = compute_att_loss(teacher_model, model, batches, 3) # define lại batches 
        outputs = model(
            input_ids=input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True
        )

        logits = outputs.logits

        loss_ce = self.compute_cross_entropy_loss(
            logits,
            output_data["label"],
            log=log
        )[0]

        kd_loss, log = self.compute_universal_logit_distillation_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )

        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * (kd_loss + batch_denom * att_loss_total) # Hàm loss cuối cùng
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output
    
    def compute_universal_logit_distillation_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        student_target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        # align the start of the student&teacher sequences
        for i in range(student_target.shape[0]):
            stu_start_idx = student_target[i].ne(self.padding_id).nonzero()[0][0]
            tea_start_idx = teacher_target[i].ne(self.padding_id).nonzero()[0][0]
            student_target[i] = torch.cat([
                student_target[i][stu_start_idx:], 
                student_target[i][:stu_start_idx]], dim=0
            )
            student_logits[i] = torch.cat([
                student_logits[i][stu_start_idx:, :],
                student_logits[i][:stu_start_idx, :]], dim=0
            )
            teacher_target[i] = torch.cat([
                teacher_target[i][tea_start_idx:], 
                teacher_target[i][:tea_start_idx]], dim=0
            )
            teacher_logits[i] = torch.cat([
                teacher_logits[i][tea_start_idx:, :],
                teacher_logits[i][:tea_start_idx, :]], dim=0
            )
        
        student_probs = torch.softmax(student_logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        sorted_student_probs = student_probs.sort(-1, descending=True).values
        sorted_teacher_probs = teacher_probs.sort(-1, descending=True).values

        vocab_size_gap = sorted_student_probs.shape[-1] - sorted_teacher_probs.shape[-1]
        bsz, slen = sorted_student_probs.shape[0], sorted_student_probs.shape[1]
        if vocab_size_gap > 0:
            sorted_teacher_probs = torch.cat([
                sorted_teacher_probs, 
                torch.zeros(bsz, slen, vocab_size_gap).to(teacher_probs)], 
                dim=-1
            )
        elif vocab_size_gap < 0:
            sorted_student_probs = torch.cat([
                sorted_student_probs, 
                torch.zeros(bsz, slen, -vocab_size_gap).to(student_probs)], 
                dim=-1
            )
        
        uld_loss = (sorted_student_probs - sorted_teacher_probs).abs().sum(-1)
        pad_mask = student_target.ne(self.padding_id) & teacher_target.ne(self.padding_id)
        uld_loss = (uld_loss * pad_mask).sum()
        log["uld_loss"] = uld_loss
        return uld_loss, log
    


