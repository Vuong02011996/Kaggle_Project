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


# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# True USES REGRESSION, False USES CLASSIFICATION
USE_REGRESSION = False

# VERSION NUMBER FOR NAMING OF SAVED MODELS
VER=4

# IF "LOAD_FROM" IS None, THEN WE TRAIN NEW MODELS
# LOAD_FROM = "/kaggle/input/deberta-v3-small_1-finetuned-v1/"
LOAD_FROM = None

# WHEN TRAINING NEW MODELS SET COMPUTE_CV = True
# WHEN LOADING MODELS, WE CAN CHOOSE True or False
COMPUTE_CV = True


warnings.simplefilter('ignore')
print("Pass import library")


class PATHS:
    train_path = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/train.csv'
    test_path = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/test.csv'
    sub_path = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/sample_submission.csv'
    model_path = "/home/oryza/Desktop/KK/Competition_1/models/deberta-v3-small"

    dir_save_model = f'/home/oryza/Desktop/KK/Competition_1/models/debearta_v3_small_retrain_v{VER}/'


class CFG:
    n_splits = 5
    seed = 42
    # max_length = 1024
    max_length = 1536
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 15
    weight_decay = 0.01
    # warmup_ratio = 0.0
    warmup_ratio = 0.1
    num_labels = 6


def infer():
    dfs = []
    if COMPUTE_CV:
        for k in range(CFG.n_splits):
            dfs.append(pd.read_csv(PATHS.dir_save_model + f'valid_df_fold_{k}_v{VER}.csv'))
            # os.system(f'rm valid_df_fold_{k}_v{VER}.csv')
        dfs = pd.concat(dfs)
        dfs.to_csv(PATHS.dir_save_model + f'valid_df_v{VER}.csv', index=False)
        print('Valid OOF shape:', dfs.shape)
        print(dfs.head())

    if COMPUTE_CV:
        if USE_REGRESSION:
            m = cohen_kappa_score(dfs.score.values, dfs.pred.values.clip(1, 6).round(0), weights='quadratic')
        else:
            m = cohen_kappa_score(dfs.score.values, dfs.iloc[:, -6:].values.argmax(axis=1) + 1, weights='quadratic')
        print('Overall QWK CV =', m)


if __name__ == '__main__':
    infer()