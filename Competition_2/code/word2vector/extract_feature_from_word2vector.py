from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm


def get_w2v_doc_vector(model, tokens, mode="mean"):
    """
     Function to return average, min and max vectors for a text body¶
     + Word2Vec provides vector values for each word - but we need a single vector that represents the entire text
     + We do this by taking the average vector for all words in the text
     + We also can return the minimum / maximum values across all vector components

    :param model:
    :param tokens:
    :param mode:
    :return:
    """
    def doc_vector(words):
        vectors_in_doc = [model.wv[w] for w in words if w in model.wv]
        if (len(vectors_in_doc) == 0): return np.zeros(model.vector_size)
        if (mode == "mean"): return np.mean(vectors_in_doc, axis=0)
        if (mode == "min"): return np.min(vectors_in_doc, axis=0)
        if (mode == "max"): return np.max(vectors_in_doc, axis=0)

    def replace_nan_with_default(x, default_vector):
        return default_vector if np.isnan(x).any() else x

    X = tokens.map(doc_vector)

    # default vector is average of all
    default_vector = X[False == X.isnull()].mean()

    return np.stack([replace_nan_with_default(vector, default_vector) for vector in X])


def get_word2vec_vectors(df, vectors, columns_to_vectorize):
    """
    Vectorize prompt and both responses
    + Word2Vec is used to generate mean, min and max vectors for the prompt and both responses
    + We additionally generate a column with vector that's the difference between the two responses
    +
    :param columns_to_vectorize:
    :param vectors:
    :param df:
    :return:
    """
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


def extract_feature_word2vector(train):
    train_text = train[['prompt', 'response_a', 'response_b']].astype(str).apply(lambda x: ' '.join(x), axis=1)

    vector_fit_text = train_text

    print("Training Word2Vec...")
    # Tokenizes the combined text.
    # Applies the simple_preprocess function to each element of vector_fit_text
    # simple_preprocess is a function from the Gensim library that tokenizes the text,
    # removes punctuation, converts to lowercase, etc.
    train_tokens = vector_fit_text.map(simple_preprocess)

    # performance vector_size much better at 60 than 150
    # Trains a Word2Vec model on the tokenized text.
    """
    + train_tokens: The tokenized text used for training.
    + vector_size=60: The size of the word vectors. Smaller vectors (e.g., 60) can perform better than larger ones (e.g., 150) depending on the task.
    + window=3: parameter specifies the number of words before and after the target word to be considered as context.
        VD: The quick brown fox jumps over the lazy dog
        window size of 3, the context words would be the 3 words before 
        and the 3 words after "fox": "The quick brown" (before) and "jumps over the" (after).
    + workers=4: The number of worker threads to use for training.
    
    """
    """
    Word2Vec:
        + Word2Vec is a neural network-based model developed by Tomas Mikolov and his team at Google in 2013.
        + It is used to generate word embeddings, which are dense vector representations of words. 
    Applications of Word2Vec:
        + Text Classification: Embeddings can be used as features for classifiers to categorize text into predefined classes (e.g., spam detection, sentiment analysis)
        + Words can be grouped into clusters based on their embeddings to discover similar words or topics.
    
    """
    vectors = Word2Vec(train_tokens, vector_size=60, window=3, seed=1, workers=4)
    print("Done")
    return vectors
