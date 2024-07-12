from tqdm import tqdm
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix, hstack
import pickle


def extract_feature_from_tfidf(train):
    vector_fit_text = train[['prompt', 'response_a', 'response_b']].astype(str).apply(lambda x: ' '.join(x), axis=1)

    # produces better results than using "word" for analyzer
    tfidf_word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 5),  # scores improved considerable after decreasing bottom ngram_range from 3 to 1
        # tokenizer=lambda x: re.findall(r'[^\W]+', x),
        # token_pattern=None,
        token_pattern=r'[^\W]+',
        strip_accents='unicode',
        # # Ngưỡng Loại bỏ từ xuất hiện quá ít
        min_df=0.10,
        # Ngưỡng loại bỏ từ xuất hiện quá nhiều lần
        max_df=0.85,
        max_features=300  # larger number of features doesn't see to help
    )

    tfidf_char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), max_features=1000, min_df=4)

    def batch_process(texts, batch_size):
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    # doing in batches so we can see progress
    batch_size = 1000
    for batch in tqdm(batch_process(vector_fit_text, batch_size), total=np.ceil(len(vector_fit_text) / batch_size)):
        tfidf_word_vectorizer.fit(batch)
        tfidf_char_vectorizer.fit(batch)
    return tfidf_word_vectorizer, tfidf_char_vectorizer


def get_tfidf_vectors(df, columns_to_vectorize):
    tfidf_word_vectorizer, tfidf_char_vectorizer = extract_feature_from_tfidf(df)
    vectorized_columns = []
    for column in columns_to_vectorize:
        vectorized_columns.append(tfidf_word_vectorizer.fit_transform(df[column]))
        vectorized_columns.append(tfidf_char_vectorizer.fit_transform(df[column]))

    with open(f'tfidf_word_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_word_vectorizer, f)
    with open(f'tfidf_char_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_char_vectorizer, f)
    return hstack(vectorized_columns)

