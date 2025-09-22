# MAP - Charting Student Math Misunderstandings
    Mục tiêu Training
    LLM sẽ học pattern: dựa vào Question + Answer + Explanation → predict Misconception:
        Input: Question + Answer + Correct + Explanation
        Output: Predict misconception category
# Create env
+ python3.12 -m venv .venv
+ source .venv/bin/activate
+ pip install -r requirements.txt

# Change function
+ compute_map3
+ format_input
+ get_top_k_ensemble

# Gemma2-9b-it model

```
model_name = "/path/to/gemma2-9b-it-cv945"
Cấu trúc thư mục:
gemma2-9b-it-cv945/
├── tokenizer.json          ← Tokenizer files
├── tokenizer_config.json   ← Tokenizer config  
├── vocab.txt               ← Vocabulary
├── special_tokens_map.json ← Special tokens
├── adapter_config.json     ← PEFT adapter config
├── adapter_model.bin       ← PEFT adapter weights
└── README.md
```
##  1. Tokenizer Files (cho AutoTokenizer), Tạo tokenizer object để convert text ↔ token_ids
+ `tokenizer.json`: Rules để tách text thành tokens
+ `vocab.txt`: Từ điển các tokens
+ `tokenizer_config.json`: Cấu hình tokenizer

## 2.  PEFT Adapter Files (cho PeftModel)
+ `adapter_config.json`: Cấu hình LoRA (rank, alpha, target_modules...)
+ `adapter_model.bin`: Trained adapter weights (A, B matrices)

+ `AutoTokenizer.from_pretrained()` → Đọc tokenizer files
+ `PeftModel.from_pretrained()` → Đọc adapter files

# PEFT(LoRA) = Parameter-Efficient Fine-Tuning 
+ PEFT - kỹ thuật fine-tune chỉ 1-2% parameters thay vì toàn bộ model.
+ Khi training với PEFT, `trainer.save_model()` chỉ save adapter weights, không save base model.
+ `PeftModel.from_pretrained("./ver_1")` đã tự động load trained adapter weights
+ `load_state_dict()` chỉ dành cho full model, không phải PEFT adapters

