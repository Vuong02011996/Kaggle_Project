import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, Math, Latex
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import PeftModel
from sklearn.metrics import average_precision_score


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
VER=1
EPOCHS = 2
DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)

path_model = "/home/server2/Desktop/My_Github/Kaggle/Kaggle_Project/Competition_3/Models/"
path_data = '/home/server2/Desktop/My_Github/Kaggle/01_Charting Student Math Misunderstandings/map-charting-student-math-misunderstandings/train.csv'

#model_name = "google/gemma-2-9b-it"
model_name = path_model +  "gemma2-9b-it-cv945"
# Load model gemma 2 for tokenizer data
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}

# Load data training
def load_training_data(path_data):
    le = LabelEncoder()
    train = pd.read_csv(path_data)
    """ Coloumns: ['row_id', 'QuestionId', 'QuestionText', 'MC_Answer', 'StudentExplanation', 'Category', 'Misconception']"""
    train.Misconception = train.Misconception.fillna('NA') # fill label "NA" for miss data in Misconception columns
    train['target'] = train.Category+":"+train.Misconception
    train['label'] = le.fit_transform(train['target']) # Encode nhãn thành số (0, 1, 2, ...)
    target_classes = le.classes_ # np.unique(train["label"].to_numpy()) = 0-> 64 labels
    n_classes = len(target_classes)
    print(f"Train shape: {train.shape} with {n_classes} target classes")
    train.head()
    return train, n_classes


def find_correct_answer_for_each_question(train):
    """
    Xác định đáp án đúng "chính thức" cho mỗi câu hỏi
    Đáp án được nhiều học sinh chọn nhất (và đều đúng) → đáp án chuẩn
    """
    idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True' # Tìm các câu trả lời đúng (Category bắt đầu bằng 'True')
    correct = train.loc[idx].copy() # Lọc DataFrame chỉ lấy những row mà idx=True, Tạo một bản sao độc lập (không ảnh hưởng đến DataFrame gốc)
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count') # Thêm cột 'c' Cho biết có bao nhiều học sinh chọn đáp án này và trả lời đúng
    correct = correct.sort_values('c',ascending=False) # Sắp xếp theo cột 'c' giảm dần"
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)
    return train, correct

# GET ANSWER CHOICES
def sort_data_follow_rank(train):
    """
    Chuẩn bị data để hiển thị câu hỏi với các lựa chọn A,B,C,D theo thứ tự từ phổ biến nhất đến ít phổ biến nhất
    "DataFrame 'tmp' chứa tất cả đáp án được sắp xếp theo:"
    1. QuestionId (tăng dần)
    2. rank (tăng dần - từ phổ biến nhất đến ít phổ biến nhất)
        rank=0: Đáp án phổ biến nhất (được chọn nhiều nhất)
        rank=1: Đáp án phổ biến thứ 2
        ...
    """
    tmp = train.groupby(['QuestionId', 'MC_Answer']).size().reset_index(name='count')
    tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
    tmp = tmp.drop('count', axis=1)
    tmp = tmp.sort_values(['QuestionId', 'rank'])
    return tmp

# DISPLAY QUESTION AND ANSWER CHOICES
def display_question_and_answer_choices(tmp):
    Q = tmp.QuestionId.unique() # 15 Question: [31772, 31774, 31777, 31778, 32829, 32833, 32835, 33471, 33472, 33474, 76870, 89443, 91695, 104665, 109465]
    for q in Q:
        question = train.loc[train.QuestionId == q].iloc[0].QuestionText
        choices = tmp.loc[tmp.QuestionId == q].MC_Answer.values
        labels = "ABCD"
        choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
        display(Latex(f"QuestionId {q}: {question}"))
        display(Latex(f"MC Answers: {choice_str}"))

def format_input(row):
    """
    Tạo prompt theo template cố định cho LLM
    Tạo template cố định 4 dòng
    Mục tiêu Training
    LLM sẽ học pattern: dựa vào Question + Answer + Explanation → predict Misconception:
        Input: Question + Answer + Correct + Explanation
        Output: Predict misconception category

    Nếu is_correct = 1 (True) → x = 'Yes' (đáp án đúng)
    Nếu is_correct = 0 (False) → x = 'No' (đáp án sai)
    """
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n" # Cung cấp context câu hỏi
        f"Answer: {row['MC_Answer']}\n" # Đáp án học sinh chọn
        f"Correct? {x}\n" # Thông tin quan trọng - đáp án đúng hay sai
        f"Student Explanation: {row['StudentExplanation']}" #  Lời giải thích của học sinh (chứa misconception)
    )

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)


if __name__ == '__main__':
    train, n_classes = load_training_data(path_data)
    train, correct = find_correct_answer_for_each_question(train)
    tmp = sort_data_follow_rank(train)
    # display_question_and_answer_choices(tmp)

    train['text'] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train.text.values[0])

    # Tính độ dài token
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]

    plt.hist(lengths, bins=50)
    plt.title("Token Length Distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    L = (np.array(lengths) > MAX_LEN).sum()
    print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
    np.sort(lengths)

    # Split into train and validation sets
    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

    # Convert to Hugging Face Dataset
    COLS = ['text', 'label']
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Set format for PyTorch
    columns = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format(type='torch', columns=columns)
    val_ds.set_format(type='torch', columns=columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        path_model + "gemma2-9b-it-bf16",
        num_labels=n_classes,
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float16,   # 👈 use float16 instead of bfloat16, FP16 (recommended for Kaggle T4)
        # torch_dtype=torch.float32,   # 👈 force full precision
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(model, model_name)  # <- also pass here)

    training_args = TrainingArguments(
        output_dir=f"./{DIR}",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",  # no for no saving
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        bf16=False,  # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU
        # bf16=True,  # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU
        fp16=False,  # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_map3,
    )

    # 🚀 Start training
    trainer.train()

    # ✅ Save the final trained model & tokenizer
    os.makedirs(f"./{DIR}/final_model", exist_ok=True)
    trainer.save_model(f"./{DIR}/final_model")
    tokenizer.save_pretrained(f"./{DIR}/final_model")