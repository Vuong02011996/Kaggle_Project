import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from tokenizers import AddedToken
import random, os
import numpy as np
import torch
import pickle


os.environ["CUDA_VISIBLE_DEVICES"]="0"

USE_REGRESSION = True

VER = 1

LOAD_FROM = None

COMPUTE_CV = True

warnings.simplefilter('ignore')
print("Pass import library")


class PATHS:
    train_path = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/train.csv'
    # train_path = '/home/oryza/Desktop/KK/Competition_1/data/persuade_2.0_human_scores_demo_id_github.csv'
    test_path = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/test.csv'
    sub_path = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/sample_submission.csv'
    model_pretrain_path = "/home/oryza/Desktop/KK/Competition_1/models/deberta-v3-small"
    dir_save_model = f'/home/oryza/Desktop/KK/Competition_1/models/debearta_v3_small_retrain_v{VER}/'


class CFG:
    n_splits = 5
    seed = 42
    # max_length = 1024
    max_length = 1024
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 1
    weight_decay = 0.01
    # warmup_ratio = 0.0
    warmup_ratio = 0.1
    num_labels = 6


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Tokenize(object):
    def __init__(self, train, valid, tokenizer):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    def get_dataset(self, df):
        ds = Dataset.from_dict({
                'essay_id': [e for e in df['essay_id']],
                'full_text': [ft for ft in df['full_text']],
                'label': [s for s in df['label']],
            })
        return ds

    def tokenize_function(self, example):
        tokenized_inputs = self.tokenizer(
            example['full_text'], truncation=True, max_length=CFG.max_length
        )
        return tokenized_inputs

    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)
        tokenized_train = train_ds.map(
            self.tokenize_function, batched=True
        )
        tokenized_valid = valid_ds.map(
            self.tokenize_function, batched=True
        )

        return tokenized_train, tokenized_valid, self.tokenizer


def compute_metrics_for_regression(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results


def compute_metrics_for_classification(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results


def k_fold_train_valid_data():
    data = pd.read_csv(PATHS.train_path)
    data = data.head(500)
    print(data.head())
    data['label'] = data['score'].apply(lambda x: x-1)

    if USE_REGRESSION:
      data["label"] = data["label"].astype('float32')
    else:
      data["label"] = data["label"].astype('int32')

    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    for i, (_, val_index) in enumerate(skf.split(data, data["score"])):
        data.loc[val_index, "fold"] = i

    print(data.head())
    return data


training_args = TrainingArguments(
    #
    output_dir=PATHS.dir_save_model + f'output_v{VER}',
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    evaluation_strategy='epoch',
    metric_for_best_model='qwk',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to='none',
    warmup_ratio=CFG.warmup_ratio,
    lr_scheduler_type='linear', # "cosine" or "linear" or "constant"
    optim='adamw_torch',
    logging_first_step=True,
)


def train(data):
    all_predictions = []
    if COMPUTE_CV:
        tokenizer = AutoTokenizer.from_pretrained(PATHS.model_pretrain_path)
        tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])

        for fold in range(len(data['fold'].unique())):
            train = data[data['fold'] != fold]
            valid = data[data['fold'] == fold].copy()

            tokenize = Tokenize(train, valid, tokenizer)
            tokenized_train, tokenized_valid, _ = tokenize()
            config = AutoConfig.from_pretrained(PATHS.model_pretrain_path)
            if USE_REGRESSION:
                config.attention_probs_dropout_prob = 0.0
                config.hidden_dropout_prob = 0.0
                config.num_labels = 1
            else:
                config.num_labels = CFG.num_labels

            if LOAD_FROM:
                model = AutoModelForSequenceClassification.from_pretrained(
                    LOAD_FROM + f'deberta-v3-small_AES2_fold_{fold}_v{VER}')
            else:
                model = AutoModelForSequenceClassification.from_pretrained(PATHS.model_pretrain_path, config=config)
                # cập nhật số lượng token embeddings.
                model.resize_token_embeddings(len(tokenizer))

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            if USE_REGRESSION:
                compute_metrics = compute_metrics_for_regression
            else:
                compute_metrics = compute_metrics_for_classification

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_valid,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            if LOAD_FROM is None:
                trainer.train()

            y_true = valid['score'].values
            test = trainer.predict(tokenized_valid)
            predictions0 = trainer.predict(tokenized_valid).predictions
            # tokenized_valid (3462, 6)
            # predictions0 (3462, 0)
            if LOAD_FROM is None:
                trainer.save_model(PATHS.dir_save_model + f'deberta-v3-small_AES2_fold_{fold}_v{VER}')
                tokenizer.save_pretrained(PATHS.dir_save_model + f'deberta-v3-small_AES2_fold_{fold}_v{VER}')
            if USE_REGRESSION:
                valid['pred'] = predictions0 + 1
            else:
                COLS = [f'p{x}' for x in range(CFG.num_labels)]
                valid[COLS] = predictions0
            valid.to_csv(PATHS.dir_save_model + f'valid_df_fold_{fold}_v{VER}.csv', index=False)

            all_predictions.append(predictions0)

            # Get raw logits instead of probabilities
            # logits = trainer.predict(tokenized_valid).predictions
            # all_predictions.append(logits)

    return np.concatenate(all_predictions, axis=0)


def load_file_pkl():
    with open("/home/oryza/Desktop/KK/Competition_1/code/deberta/predictions.pkl", 'rb') as file:
        predictions = pickle.load(file)
        predictions1 = predictions


if __name__ == '__main__':
    data = k_fold_train_valid_data()
    predictions = train(data)
    # Save predictions to a .pkl file
    with open('predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    print(f"Predictions shape: {predictions.shape}")
    print(predictions[0])
    # """
    # Predictions shape: (17307,)
    # 2.703125
    # """
    # load_file_pkl()