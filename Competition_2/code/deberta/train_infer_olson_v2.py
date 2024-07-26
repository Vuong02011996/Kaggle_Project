import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import regex as re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz, load_npz, hstack
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import lightgbm as lgb

from tqdm import tqdm

import gensim
import itertools
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

from transformers import DebertaV2Tokenizer, DebertaV2Model
import torch

import joblib
import unicodedata
import re
import time
from sklearn.metrics.pairwise import cosine_similarity
# Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
import gensim

from Competition_2.code.deberta.extract_features_from_deberta import get_tx_vectors
from Competition_2.code.tfidf_vectorizer.extract_feature_from_tfidf_vectorizer import get_tfidf_vectors
from Competition_2.code.word2vector.extract_feature_from_word2vector import get_word2vec_vectors, \
    extract_feature_word2vector

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train = pd.read_csv("/home/oryza/Desktop/KK/Competition_2/data/train.csv")
test = pd.read_csv("/home/oryza/Desktop/KK/Competition_2/data/test.csv")

# for training on small part of train data
quick_test = False

# override quick_test if we detect actual test data... (to prevent submission sadness)
# if (len(test)) > 3: quick_test = False

if quick_test: train = train.head(5000)

target_columns = ['winner_model_a', 'winner_model_b', 'winner_tie']

columns_to_vectorize = ["prompt", "response_a", "response_b"]

print(train.head(5))
ver_train = "deberta_large"


def train_lgbm(combined_train_vectors):
    model_filename = f'{ver_train}_lgbm_model.pkl'

    # n_estimators is set too high -> overfitting; too low -> underfitting
    max_estimators = 2000

    # meaning training will stop if the model performance does not improve for 50 consecutive rounds.
    early_stopping_limit = 100

    # Data preparation
    X = combined_train_vectors
    y = train[target_columns].idxmax(axis=1)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # [0 1 2 0 ...]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.05, random_state=42)

    # LightGBM parameters
    params = {
        'n_estimators': max_estimators,  # Maximum number of trees.
        'max_depth': 4,  # Maximum depth of each tree
        'subsample': 0.8,  # Fractions of samples and features used for building each tree.
        'colsample_bytree': 0.8,  # Fractions of samples and features used for building each tree.
        'objective': 'multiclass',  # Specifies the task as multi-class classification.
        'num_class': 3,
        'metric': 'multi_logloss',
        'random_state': 42,
        'learning_rate': 0.03,
        'verbose': -1  # keep logs quiet
    }

    # Create the model
    model = lgb.LGBMClassifier(**params)

    def callback(env):
        if env.iteration % 10 == 0: print ("Iteration:", env.iteration, "\tLog Loss:", env.evaluation_result_list[0][2])

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_limit), callback]
    )

    # Save the model to disk
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    y_pred_proba = model.predict_proba(X_test)

    logloss = log_loss(y_test, y_pred_proba)
    print(f"\nLog Loss: {logloss}")

    y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


def train_lgbm_find_max_depth(combined_train_vectors):
    """
    Use grid search to try a range of max_depth values and
    find the optimal one based on a performance metric such as accuracy, log loss, or F1 score.
    :param combined_train_vectors:
    :return:
    """
    model_filename = f'{ver_train}_lgbm_model.pkl'

    # n_estimators is set too high -> overfitting; too low -> underfitting
    max_estimators = 2000

    # meaning training will stop if the model performance does not improve for 50 consecutive rounds.
    early_stopping_limit = 100

    # Data preparation
    X = combined_train_vectors
    y = train[target_columns].idxmax(axis=1) # [0 1 2 0 ...]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.05, random_state=42)

    # LightGBM parameters
    params = {
        'n_estimators': max_estimators,  # Maximum number of trees.
        # 'max_depth': 4,  # Maximum depth of each tree
        'subsample': 0.8,  # Fractions of samples and features used for building each tree.
        'colsample_bytree': 0.8,  # Fractions of samples and features used for building each tree.
        'objective': 'multiclass',  # Specifies the task as multi-class classification.
        'num_class': 3,
        'metric': 'multi_logloss',
        'random_state': 42,
        'learning_rate': 0.03,
        'verbose': -1  # keep logs quiet
    }

    """--------------------Perform grid search-----------------------------"""
    start_time = time.time()
    # Create the model
    model = lgb.LGBMClassifier(**params)
    # Define the grid of max_depth values to search
    param_grid = {'max_depth': [3, 4, 5, 6, 7, 8]}

    # Perform grid search
    print("Find max depth ....")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_log_loss', verbose=1)
    grid_search.fit(X_train, y_train)

    # Best max_depth value
    best_max_depth = grid_search.best_params_['max_depth']
    print(f"Best max_depth: {best_max_depth}")

    # Train the model with the best max_depth
    params['max_depth'] = best_max_depth
    print("Find max_depth cost: ", time.time() - start_time)

    """-------------------------------------------------------------------"""

    def callback(env):
        if env.iteration % 10 == 0: print("Iteration:", env.iteration, "\tLog Loss:", env.evaluation_result_list[0][2])

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_limit), callback]
    )

    # Save the model to disk
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    y_pred_proba = model.predict_proba(X_test)

    logloss = log_loss(y_test, y_pred_proba)
    print(f"\nLog Loss: {logloss}")

    y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


def train_lgbm_find_subsample(combined_train_vectors):
    """
    Use grid search to try a range of max_depth values and
    find the optimal one based on a performance metric such as accuracy, log loss, or F1 score.
    :param combined_train_vectors:
    :return:
    """
    model_filename = f'{ver_train}_lgbm_model.pkl'

    # n_estimators is set too high -> overfitting; too low -> underfitting
    max_estimators = 2000

    # meaning training will stop if the model performance does not improve for 50 consecutive rounds.
    early_stopping_limit = 100

    # Data preparation
    X = combined_train_vectors
    y = train[target_columns].idxmax(axis=1) # [0 1 2 0 ...]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.05, random_state=42)

    # LightGBM parameters
    params = {
        'n_estimators': max_estimators,  # Maximum number of trees.
        'max_depth': 3,  # Maximum depth of each tree
        # 'subsample': 0.8,  # Fractions of samples and features used for building each tree.
        # 'colsample_bytree': 0.8,  # Fractions of samples and features used for building each tree.
        'objective': 'multiclass',  # Specifies the task as multi-class classification.
        'num_class': 3,
        'metric': 'multi_logloss',
        'random_state': 42,
        'learning_rate': 0.03,
        'verbose': -1  # keep logs quiet
    }

    """--------------------Perform grid search-----------------------------"""
    start_time = time.time()
    # Create the model
    model = lgb.LGBMClassifier(**params)
    # Define the grid of subsample and colsample_bytree values to search
    param_grid = {
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_log_loss', verbose=1)
    grid_search.fit(X_train, y_train)
    # Best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    # Train the model with the best parameters
    params.update(best_params)
    print("Find subsample cost: ", time.time() - start_time)
    """-------------------------------------------------------------------"""

    def callback(env):
        if env.iteration % 10 == 0: print("Iteration:", env.iteration, "\tLog Loss:", env.evaluation_result_list[0][2])

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_limit), callback]
    )

    # Save the model to disk
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    y_pred_proba = model.predict_proba(X_test)

    logloss = log_loss(y_test, y_pred_proba)
    print(f"\nLog Loss: {logloss}")

    y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


def train_lgbm_fold(combined_train_vectors):
    model_filename = f'{ver_train}_lightgbm_model_fold.pkl'
    max_estimators = 2000
    early_stopping_limit = 100
    n_splits = 5

    # Data preparation
    X = combined_train_vectors
    y = train[target_columns].idxmax(axis=1)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # # Split dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.05, random_state=42)

    # LightGBM parameters
    params = {
        'n_estimators': max_estimators,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'random_state': 42,
        'learning_rate': 0.03,
        'verbose': -1  # keep logs quiet
    }

    # Collect predictions and metrics
    oof_preds = np.zeros((X.shape[0], params['num_class']))
    fold_models = []
    fold_logloss = []
    fold_accuracy = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"Training fold {fold + 1}/{n_splits}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # Create the model
        model = lgb.LGBMClassifier(**params)

        def callback(env):
            if env.iteration % 10 == 0: print ("Iteration:", env.iteration, "\tLog Loss:", env.evaluation_result_list[0][2])

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_limit), callback]
        )

        # Predict validation set
        y_val_pred_proba = model.predict_proba(X_val)
        oof_preds[val_idx] = y_val_pred_proba

        # Calculate and print metrics
        logloss = log_loss(y_val, y_val_pred_proba)
        fold_logloss.append(logloss)
        y_val_pred = np.argmax(y_val_pred_proba, axis=1)  # Convert probabilities to class labels
        accuracy = accuracy_score(y_val, y_val_pred)
        fold_accuracy.append(accuracy)

        print(f"Fold {fold + 1} Log Loss: {logloss}")
        print(f"Fold {fold + 1} Accuracy: {accuracy}")

        # Save the model for this fold
        fold_models.append(model)

    # Save the model (last fold model or a model ensemble)
    joblib.dump(fold_models, model_filename)
    print(f"Models saved to {model_filename}")

    # Aggregate fold results
    avg_logloss = np.mean(fold_logloss)
    avg_accuracy = np.mean(fold_accuracy)
    print(f"\nAverage Log Loss: {avg_logloss}")
    print(f"Average Accuracy: {avg_accuracy}")

    return fold_models, avg_logloss, avg_accuracy


if __name__ == '__main__':
    # Define the path to the saved feature file
    feature_file_path = os.path.join(".", 'tx_train_vectors_1440_deberta_large.npz')

    # Check if the feature file exists
    if os.path.exists(feature_file_path):
        # Load the features from the file
        start_time = time.time()
        tx_train_vectors_csr = load_npz(feature_file_path)
        tx_train_vectors = tx_train_vectors_csr.toarray()
        print("Loaded features from file.")
        print("get_tx_vectors cost: ", time.time() - start_time, "seconds")
        print("tx_train_vectors.shape: ", tx_train_vectors.shape)
    else:
        # If the file doesn't exist, extract and save the features
        start_time = time.time()
        tx_train_vectors = get_tx_vectors(train, columns_to_vectorize)
        print("get_tx_vectors cost: ", time.time() - start_time, "seconds")
        print("tx_train_vectors.shape: ", tx_train_vectors.shape)

        tx_train_vectors_csr = csr_matrix(tx_train_vectors)
        save_npz(feature_file_path, tx_train_vectors_csr)
        print("Extracted and saved features to file.")

    "-------------------------------------------------------------------------------------------------------"
    combined_train_vectors = tx_train_vectors

    # word2vector features

    # start_time = time.time()
    # vectors = extract_feature_word2vector(train)
    # vectors.save("word2vec_trained.model")
    # print("extract_feature_word2vector cost: ", time.time() - start_time, "seconds")
    #
    # start_time = time.time()
    # word2vec_train_vectors = get_word2vec_vectors(train, vectors, columns_to_vectorize)
    # print("get_word2vec_vectors cost: ", time.time() - start_time, "seconds")
    # word2vec_train_vectors_csr = csr_matrix(word2vec_train_vectors)
    # save_npz(os.path.join(".", 'word2vec_train_vectors.npz'), word2vec_train_vectors_csr)

    # tfidf features
    # start_time = time.time()
    # tfidf_train_vectors = get_tfidf_vectors(train, columns_to_vectorize)
    # print("get_tfidf_vectors cost: ", time.time() - start_time, "seconds")
    # tfidf_train_vectors_csr = csr_matrix(tfidf_train_vectors)
    # save_npz(os.path.join(".", 'tfidf_train_vectors.npz'), tfidf_train_vectors_csr)

    train_lgbm_fold(combined_train_vectors)


