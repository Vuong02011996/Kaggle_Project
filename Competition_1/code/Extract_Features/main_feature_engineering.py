import polars as pl
from Competition_1.code.Extract_Features.A_Paragraph_Features import Paragraph_Preprocess, Paragraph_Eng
from Competition_1.code.Extract_Features.B_Sentence_Features import Sentence_Preprocess, Sentence_Eng
from Competition_1.code.Extract_Features.C_Word_Features import Word_Preprocess, Word_Eng
import pandas as pd

from Competition_1.code.Extract_Features.D_CountVectorizer_Features import vectorizer_cnt
from Competition_1.code.Extract_Features.E_Tf_idf_features import vectorizer
import time
import pickle
import os

"""
columns là một danh sách chứa các biểu thức xử lý cột được định nghĩa bằng cách sử dụng Polars.
Ở đây chỉ tạo ra một cột mới là paragraph.

pl.col("full_text"): Chọn cột có tên "full_text" trong DataFrame.
.str.split(by="\n\n"): Chia chuỗi văn bản trong cột "full_text" thành các đoạn văn, mỗi đoạn ngăn cách bởi hai ký tự xuống dòng (\n\n).
.alias("paragraph"): Đổi tên cột kết quả thành "paragraph".
biểu thức này sẽ tạo ra một cột mới tên là "paragraph", chứa các đoạn văn từ cột "full_text"
"""

columns = [
    (
        pl.col("full_text").str.split(by="\n\n").alias("paragraph")
    ),
]
PATH = "/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/"
path_vectorizer_models = "/home/oryza/Desktop/KK/Competition_1/models/vectorizer_models/"
"""
.with_columns(columns): Áp dụng danh sách các cột đã định nghĩa trong columns để xử lý DataFrame, 
cụ thể là thêm cột "paragraph" chứa các đoạn văn được tách ra từ cột "full_text".
"""
train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)

train.head(1)
with pl.Config(fmt_str_lengths=1000):
    print(train.head(5))


def extract_features():
    """Paragraph preprocess"""
    start_time = time.time()

    # Load the DataFrame from the pickle file
    pickle_file = 'train_feats.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            train_feats = pickle.load(file)
        print("DataFrame loaded from train_feats.pkl")
    else:
        tmp = Paragraph_Preprocess(train)
        train_feats = Paragraph_Eng(tmp)
        train_feats['score'] = train['score']
        # Save the DataFrame to a pickle file
        with open('train_feats.pkl', 'wb') as file:
            pickle.dump(train_feats, file)
        print("DataFrame saved to train_feats.pkl")

    feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
    print('Number of Features: ', len(feature_names))
    # train_feats is pandas not polars
    # pd.set_option('display.max_columns', None)
    # print(train_feats.head(3))
    print("Paragraph preprocess cost: ", time.time() - start_time)

    """Sentence preprocess"""
    start_time = time.time()
    tmp = Sentence_Preprocess(train)

    # train_feats = Sentence_Eng(tmp)
    # train_feats['score'] = train['score']
    train_feats = train_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')

    feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
    print('Features Number after Sentence preprocess: ', len(feature_names))
    # train_feats is pandas not polars
    # pd.set_option('display.max_columns', None)
    # print(train_feats.head(3))
    print("Sentence preprocess cost: ", time.time() - start_time)

    start_time = time.time()
    # """ Word preprocess """
    tmp = Word_Preprocess(train)
    # train_feats = Word_Eng(tmp)
    train_feats = train_feats.merge(Word_Eng(tmp), on='essay_id', how='left')

    feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
    print('Features Number after Word preprocess: ', len(feature_names))
    # train_feats is pandas not polars
    # pd.set_option('display.max_columns', None)
    # print(train_feats.head(3))
    print("Word preprocess cost: ", time.time() - start_time)

    """TfidfVectorizer"""
    start_time = time.time()
    train_tfid = vectorizer.fit_transform([i for i in train['full_text']])
    dense_matrix = train_tfid.toarray()
    df = pd.DataFrame(dense_matrix)
    tfid_columns = [f'tfid_{i}' for i in range(len(df.columns))]
    df.columns = tfid_columns
    df['essay_id'] = train_feats['essay_id']
    train_feats = train_feats.merge(df, on='essay_id', how='left')
    feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
    print('Number of Features: ', len(feature_names))
    pd.set_option('display.max_columns', None)
    # print("TfidfVectorizer features: ")
    # print(train_feats.head(3))
    print("TfidfVectorizer cost: ", time.time() - start_time)
    with open(f'vectorizer_1_2.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    """CountVectorizer"""
    start_time = time.time()
    train_tfid = vectorizer_cnt.fit_transform([i for i in train['full_text']])
    dense_matrix = train_tfid.toarray()
    df = pd.DataFrame(dense_matrix)
    tfid_columns = [f'tfid_cnt_{i}' for i in range(len(df.columns))]
    df.columns = tfid_columns
    df['essay_id'] = train_feats['essay_id']
    train_feats = train_feats.merge(df, on='essay_id', how='left')

    feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
    print('Features number after CountVectorizer: ', len(feature_names))
    print("CountVectorizer cost: ", time.time() - start_time)

    # Save the fitted vectorizer
    with open(f'vectorizer_cnt_1_2.pkl', 'wb') as f:
        pickle.dump(vectorizer_cnt, f)

    return train_feats


if __name__ == '__main__':
    extract_features()
