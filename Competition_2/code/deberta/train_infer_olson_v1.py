import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import regex as re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

import lightgbm as lgb

from tqdm import tqdm

# Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
import gensim
import itertools
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import torch
import time
import joblib


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train = pd.read_csv("/home/oryza/Desktop/KK/Competition_2/data/train.csv")
test = pd.read_csv("/home/oryza/Desktop/KK/Competition_2/data/test.csv")

# for training on small part of train data
quick_test = True

# override quick_test if we detect actual test data... (to prevent submission sadness)
# if (len(test)) > 3: quick_test = False

if quick_test: train = train.head(5000)

target_columns = ['winner_model_a', 'winner_model_b', 'winner_tie']

columns_to_vectorize = ["prompt", "response_a", "response_b"]

print(train.head(5))


def batch_extract_tx_features(texts, tokenizer, model, batch_size=16, max_length=1024):
    """
    extract features from text data using the DeBERTa V2 transformer model
    :param texts:  A list of text strings to process.
    :param tokenizer:  The tokenizer for converting text into token IDs.
    :param model: The DeBERTa V2 model for feature extraction.
    :param batch_size: The number of texts to process in each batch (default is 16).
    :param max_length: The maximum sequence length for the tokenizer (default is 1024).
    :return:
    """
    features = []
    for i in range(0, len(texts), batch_size):
        # print(".", end="")
        # Extracts the current batch of texts
        batch_texts = texts[i:i+batch_size]
        # Tokenizes the batch of texts and moves the tokenized inputs to the GPU.
        # Enables truncation of texts longer than max_length.
        # padding=True: Adds padding to texts shorter than max_length to make all sequences in the batch of equal length.
        """
        # This parameter specifies the format of the tokenized output.
        'pt': Returns the output as PyTorch tensors.
        'tf': Returns the output as TensorFlow tensors.
        'np': Returns the output as NumPy arrays.
        """
        inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, truncation=True, padding=True).to('cuda')

        # inputs is a dictionary containing the following key-value pairs:

        with torch.no_grad():
            # Passes the tokenized inputs through the model.
            """
            inputs is a dictionary containing the tokenized text data.
            **inputs: This is equivalent to calling the model with 
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'], etc.
            attention_mask: A mask to distinguish between padding and non-padding tokens            
            
            outputs = model(input_ids=tensor([[101, 2003, 1037,  ...], [101, 2054,  ...], ...]), 
                            attention_mask=tensor([[1, 1, 1,  ...], [1, 1,  ...], ...]))
            The outputs object typically contains several components depending on the model and its configuration.
             For transformer models like DeBERTa, it often includes:
            + last_hidden_state: The hidden states of the last layer for each token in the input sequence.
            + pooler_output: The pooled output (often the hidden state corresponding to the first token [CLS]), used for classification tasks.
            + Other possible outputs (e.g., hidden states of all layers, attention weights) depending on the model and its configuration.
            """
            outputs = model(**inputs)
        # Move the tensor to CPU before converting to numpy
        """
        0: Selects the first token of each sequence (which is usually the [CLS] token in transformer models like BERT, DeBERTa, etc.).
        The [CLS] token is a special token added at the beginning of every input sequence for transformer-based models.
        It stands for "classification" and is used by the model to aggregate information from the entire sequence.
        last_hidden_state is [batch_size, sequence_length, hidden_size]
        """
        batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.extend(batch_features)
        torch.cuda.empty_cache()  # Clear GPU cache to manage memory
    return np.vstack(features)

"""
Deberta: Extract features for prompt and both responses
Also adding difference between two responses to array (seemed to help)
"""
def get_tx_vectors(df):
    vectors = []
    for column in tqdm(columns_to_vectorize, desc="Vectorizing Columns"):
        print("Vectorizing", column)
        vectors.append(batch_extract_tx_features(df[column].tolist(), tokenizer, tx_model))

    vectors = np.array(vectors)
    # from (3, num_row_of_data, feature_dim) to (num_row_of_data, 3, feature_dim)).
    vectors = np.transpose(vectors, (1, 0, 2))

    avg_dif = vectors[:, 1, :] - vectors[:, 2, :]
    avg_dif = avg_dif.reshape(vectors.shape[0], 1, vectors.shape[2])
    vectors = np.concatenate((vectors, avg_dif), axis=1)
    vectors = np.array(vectors).reshape(len(vectors), -1)
    return vectors


time_start = time.time()
tx_train_vectors = get_tx_vectors(train)
print("get_tx_vectors cost: ", time.time() - time_start)
# get_tx_vectors cost:  6475.497596979141
"-------------------------------------------------------------------------------------------------------"
"""
Word2Vec: Initialize / train on our train data...
Here we train Word2Vec to capture word relationships on our text columns....
"""

print("Training Word2Vec...")
train_data = train[['prompt', 'response_a', 'response_b']]
combined_text = train_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
train_tokens = combined_text.map(simple_preprocess)

#performance vector_size much better at 60 than 150
vectors = Word2Vec(train_tokens, vector_size=60, window=3, seed=1, workers=4)
print("Done.")

"""
Word2Vec: Function to return average, min and max vectors for a text body
+ Word2Vec provides vector values for each word - but we need a single vector that represents the entire text
+ We do this by taking the average vector for all words in the text
+ We also can return the minimum / maximum values across all vector components
"""


def get_w2v_doc_vector(model, tokens, mode="mean"):
    def doc_vector(words):
        vectors_in_doc = [model.wv[w] for w in words if w in model.wv]
        if (len(vectors_in_doc) == 0): return np.zeros(vectors.vector_size)
        if (mode == "mean"): return np.mean(vectors_in_doc, axis=0)
        if (mode == "min"): return np.min(vectors_in_doc, axis=0)
        if (mode == "max"): return np.max(vectors_in_doc, axis=0)

    def replace_nan_with_default(x, default_vector):
        return default_vector if np.isnan(x).any() else x

    X = tokens.map(doc_vector)

    # default vector is average of all
    default_vector = X[False == X.isnull()].mean()

    return np.stack([replace_nan_with_default(vector, default_vector) for vector in X])


"""Word2Vec: Vectorize prompt and both responses¶
+ Word2Vec is used to generate mean, min and max vectors for the prompt and both responses
+ We additionally generate a column with vector that's the difference between the two prompts
+ Generating vectors with the differences between the prompt and responses didn't help score
"""


def get_word2vec_vectors(df):
    word2vec_vectors = []
    for column in tqdm(columns_to_vectorize, desc="Vectorizing Columns"):
        print("Vectorizing", column)
        column_tokens = df[column].map(simple_preprocess)

        word2vec_vectors.append(get_w2v_doc_vector(vectors, column_tokens, mode="mean"))
        word2vec_vectors.append(get_w2v_doc_vector(vectors, column_tokens, mode="min"))
        word2vec_vectors.append(get_w2v_doc_vector(vectors, column_tokens, mode="max"))

    # adjust array config
    word2vec_vectors = np.array(word2vec_vectors)
    word2vec_vectors = np.transpose(word2vec_vectors, (1, 0, 2))

    # generate a vector that is response_a - response_b (means values) / append it to array
    avg_dif = word2vec_vectors[:, 3, :] - word2vec_vectors[:, 6, :]
    avg_dif = avg_dif.reshape(word2vec_vectors.shape[0], 1, word2vec_vectors.shape[2])
    word2vec_vectors = np.concatenate((word2vec_vectors, avg_dif), axis=1)

    # flatten
    word2vec_vectors = np.array(word2vec_vectors).reshape(len(word2vec_vectors), -1)
    return word2vec_vectors


word2vec_train_vectors = get_word2vec_vectors(train)

"""TF-IDF: Fit vectorizer on prompts and both responses"""
vector_fit_text = train[['prompt', 'response_a', 'response_b']].astype(str).apply(lambda x: ' '.join(x), axis=1)

# produces better results than using "word" for analyzer
tfidf_word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 5),  # scores improved considerable after decreasing bottom ngram_range from 3 to 1
    tokenizer=lambda x: re.findall(r'[^\W]+', x),
    token_pattern=None,
    strip_accents='unicode',
    min_df=4,
    max_features=300  # larger number of features doesn't see to help
)

tfidf_char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), max_features=1000, min_df=4)


def batch_process(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

#doing in batches so we can see progress
batch_size = 1000
for batch in tqdm(batch_process(vector_fit_text, batch_size), total=np.ceil(len(vector_fit_text) / batch_size)):
    tfidf_word_vectorizer.fit(batch)
    tfidf_char_vectorizer.fit(batch)


"""TF-IDF: Vectorize text columns - and combine in hstack"""
def get_tfidf_vectors(df):
    vectorized_columns = []
    for column in columns_to_vectorize:
        vectorized_columns.append(tfidf_word_vectorizer.transform(df[column]))
        vectorized_columns.append(tfidf_char_vectorizer.transform(df[column]))
    return hstack(vectorized_columns)

tfidf_train_vectors = get_tfidf_vectors(train)


"""Deberta + Word2Vec + TF-IDF: Assemble vectors"""
tx_train_vectors_csr = csr_matrix(tx_train_vectors)  # Convert tx vectors to a CSR matrix
word2vec_train_vectors_csr = csr_matrix(word2vec_train_vectors)  # Convert Word2Vec vectors to a CSR matrix
combined_train_vectors = hstack([tfidf_train_vectors, tx_train_vectors, word2vec_train_vectors_csr])  # Combine TF-IDF and Word2Vec vectors


"""Train LightGBM on the whole mess...¶"""
# Data preparation
X = combined_train_vectors
y = train[target_columns].idxmax(axis=1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# LightGBM parameters
params = {
    'n_estimators': 150,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'random_state': 42,
    'learning_rate': 0.04,
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
    callbacks=[callback]
)

model_filename = 'Competition_2/code/deberta/lightgbm_model.pkl'
# Save the model to disk to load inference
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

y_pred_proba = model.predict_proba(X_test)

logloss = log_loss(y_test, y_pred_proba)
print(f"\nLog Loss: {logloss}")

y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")