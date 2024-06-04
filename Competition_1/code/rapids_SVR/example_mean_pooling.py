import torch
from transformers import BertTokenizer, BertModel

# Khởi tạo tokenizer và model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Văn bản đầu vào
text = "This is an example sentence for mean pooling."

# Mã hóa văn bản
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Chạy mô hình để lấy output
model_output = model(**inputs)


# Hàm mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Sử dụng mean pooling để lấy embedding
attention_mask = inputs['attention_mask']
mean_pooled_output = mean_pooling(model_output, attention_mask)

print(mean_pooled_output)