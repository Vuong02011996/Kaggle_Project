
import warnings
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import random, os


# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# True USES REGRESSION, False USES CLASSIFICATION
USE_REGRESSION = True

# VERSION NUMBER FOR NAMING OF SAVED MODELS
VER=8

# IF "LOAD_FROM" IS None, THEN WE TRAIN NEW MODELS
# LOAD_FROM = "/kaggle/input/deberta-v3-small_1-finetuned-v1/"
LOAD_FROM = None

# WHEN TRAINING NEW MODELS SET COMPUTE_CV = True
# WHEN LOADING MODELS, WE CAN CHOOSE True or False
COMPUTE_CV = True


warnings.simplefilter('ignore')
print("Pass import library")


class PATHS:
    dir_save_model = f'/home/oryza/Desktop/KK/Competition_1/models/debearta_v3_small_retrain_v{VER}/'


class CFG:
    n_splits = 5
    seed = 42
    # max_length = 1024
    max_length = 1536
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