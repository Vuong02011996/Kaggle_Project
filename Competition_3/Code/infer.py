import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import PeftModel

from train import format_input, tokenize, find_correct_answer_for_each_question, load_training_data, path_data, \
    model_name, path_model

saved_model_path = "./ver_1"  # Thư mục đã save từ training


def load_model_after_train():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        path_model + "gemma2-9b-it-bf16",
        num_labels=n_classes,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Load trained PEFT adapter (ĐÃ ĐỦ!)
    # model = PeftModel.from_pretrained(model, "./ver_1")
    model = PeftModel.from_pretrained(model, model_name)

    # Tạo trainer cho inference
    inference_args = TrainingArguments(
        output_dir="./inference",
        per_device_eval_batch_size=16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=inference_args,
        tokenizer=tokenizer
    )
    return trainer


if __name__ == '__main__':
    test = pd.read_csv('/home/server2/Desktop/My_Github/Kaggle/01_Charting Student Math Misunderstandings/map-charting-student-math-misunderstandings/test.csv')
    print( test.shape )
    test.head()

    train, n_classes = load_training_data(path_data)
    train, correct = find_correct_answer_for_each_question(train)
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)

    test['text'] = test.apply(format_input,axis=1)

    test.head()

    ds_test = Dataset.from_pandas(test[['text']])
    ds_test = ds_test.map(tokenize, batched=True)


    trainer = load_model_after_train()
    predictions = trainer.predict(ds_test)
    # Chuyển logits thành xác suất (sum = 1.0)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

    # Get top 3 predicted class indices
    top3 = np.argsort(-probs, axis=1)[:, :]   # shape: [num_samples, 3]

    # Decode numeric class indices to original string labels
    flat_top3 = top3.flatten()
    le = LabelEncoder()
    decoded_labels = le.inverse_transform(flat_top3)
    top3_labels = decoded_labels.reshape(top3.shape)

    # Join 3 labels per row with space
    joined_preds = ["|".join(row) for row in top3_labels]

    # Save submission
    sub = pd.DataFrame({
        "row_id": test.row_id.values,
        "Category:Misconception": joined_preds
    })
    sub.to_csv("submission_gemma.csv", index=False)
    sub.head()