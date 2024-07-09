import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""
TF-IDF là viết tắt của Term Frequency-Inverse Document Frequency.
khai thác văn bản để chuyển đổi văn bản thô thành các đặc trưng số mà các mô hình máy học có thể sử dụng

+ Term Frequency (TF): Tần suất xuất hiện của một từ trong một tài liệu.
+ Inverse Document Frequency (IDF): Mức độ quan trọng của từ đó trong toàn bộ tập hợp tài liệu. 
  Nó giảm giá trị của các từ xuất hiện nhiều trong các tài liệu (như "is", "the", "in"), 
  và tăng giá trị của các từ ít xuất hiện.

Khi nào nên sử dụng TF-IDF Vectorizer?
+ Nó đặc biệt hữu ích trong các bài toán phân loại văn bản, truy vấn thông tin, 
và khai thác văn bản khi bạn muốn nắm bắt được mức độ quan trọng của các từ trong tài liệu.

"""

vectorizer = TfidfVectorizer(
            # tokenizer=lambda x: x,
            # preprocessor=lambda x: x,
            token_pattern=r'\b\w+\b',
            # token_pattern=None,
            strip_accents='unicode',
            analyzer='word',
            # ngram_range=(3,6),
            ngram_range=(1, 2),
            min_df=0.05,
            max_df=0.95,
            # # # Ngưỡng Loại bỏ từ xuất hiện quá ít
            # min_df=0.10,
            # # Ngưỡng loại bỏ từ xuất hiện quá nhiều lần
            # max_df=0.85,
            # sublinear_tf=True,
)


def example_tf_idf_vectorizer():

    # Giả sử bạn có một tập dữ liệu văn bản
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Never jump over the lazy dog quickly",
        "Bright brown foxes leap high"
    ]

    # Tạo một DataFrame từ các tài liệu
    df = pd.DataFrame({"text": documents})

    # Khởi tạo TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Chuyển đổi các tài liệu thành ma trận TF-IDF
    # Sử dụng phương thức fit_transform để học từ dữ liệu và chuyển đổi các tài liệu thành ma trận TF-IDF.
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["text"])

    # Chuyển đổi ma trận TF-IDF thành DataFrame để dễ nhìn
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    pd.set_option('display.max_columns', None)
    print(tfidf_df)
    """
             bright     brown       dog       fox     foxes      high      jump  \
        0  0.000000  0.275379  0.275379  0.362091  0.000000  0.000000  0.000000   
        1  0.000000  0.000000  0.329928  0.000000  0.000000  0.000000  0.433816   
        2  0.467351  0.355432  0.000000  0.000000  0.467351  0.467351  0.000000  
        
              jumps      lazy      leap     never      over     quick   quickly  \
        0  0.362091  0.275379  0.000000  0.000000  0.275379  0.362091  0.000000   
        1  0.000000  0.329928  0.000000  0.433816  0.329928  0.000000  0.433816   
        2  0.000000  0.000000  0.467351  0.000000  0.000000  0.000000  0.000000   
    """


if __name__ == '__main__':
    example_tf_idf_vectorizer()
