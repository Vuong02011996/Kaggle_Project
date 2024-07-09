import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

"""
CountVectorizer là một kỹ thuật chuyển đổi văn bản thành các đặc trưng số bằng cách đếm tần suất xuất hiện của các từ trong văn bản.
Mỗi từ sẽ trở thành một đặc trưng, và giá trị của đặc trưng đó là số lần từ đó xuất hiện trong văn bản.

1. N-gram là gì?
Unigram: Các từ đơn lẻ. Ví dụ: "machine", "learning".
Bigram: Các cặp từ. Ví dụ: "machine learning", "learning model".
Trigram: Các nhóm ba từ. Ví dụ: "machine learning model".
ngram_range=(2,3): Tạo cả bigrams và trigrams. Cả cặp từ và ba từ liền kề sẽ được xem xét. 
Điều này có thể giúp mô hình nắm bắt ngữ cảnh tốt hơn, đặc biệt trong các văn bản mà các cụm từ dài hơn có ý nghĩa quan trọng
+ Thông tin ngữ cảnh: Khi tăng kích thước của n-gram, mô hình sẽ nắm bắt được nhiều ngữ cảnh hơn từ văn bản. 
+ Hiệu suất và tính toán: Việc xem xét các n-gram lớn hơn có thể làm tăng thời gian tính toán và yêu cầu bộ nhớ

2. min_df:
Tham số min_df trong CountVectorizer xác định ngưỡng tần suất tối thiểu của từ/ngữ n-gram để được bao gồm/loại bỏ 
+ min_df là số nguyên: Nếu min_df=5, các từ/ngữ n-gram phải xuất hiện ít nhất 5 lần trong toàn bộ tập dữ liệu để được bao gồm trong từ điển.
+ min_df là tỉ lệ phần trăm: Nếu min_df=0.10, các từ/ngữ n-gram phải xuất hiện trong ít nhất 10% các tài liệu để được bao gồm trong từ điển

Giá trị min_df thấp (ví dụ: 1 hoặc 0.01)
+ Bao gồm nhiều từ/ngữ n-gram hơn: Nhiều từ/ngữ n-gram, kể cả những từ xuất hiện rất ít hoặc chỉ xuất hiện trong một vài tài liệu, sẽ được bao gồm
+ Tăng thời gian và bộ nhớ tính toán: Do kích thước từ điển lớn hơn, yêu cầu bộ nhớ và thời gian tính toán sẽ tăng lên.

Giá trị min_df cao (ví dụ: 5 hoặc 0.10)
+ Loại bỏ các từ/ngữ n-gram ít gặp: Các từ/ngữ n-gram xuất hiện ít hơn ngưỡng sẽ bị loại bỏ, chỉ giữ lại các từ/ngữ n-gram phổ biến.
+ Giảm thời gian và bộ nhớ tính toán: Do kích thước từ điển nhỏ hơn, yêu cầu bộ nhớ và thời gian tính toán sẽ giảm.

Tương tự max_df:
+ Ngưỡng loại bỏ những từ xuất hiện quá nhiều lần trong tài liệu (the, a , an , ....)
+ Từ "machine" xuất hiện trong 9 tài liệu: Bị loại bỏ vì 9/10 = 90% > 85%.

# Có thể không 
tokenizer=lambda x: x: Đây là một hàm tokenize tùy chỉnh. Trong trường hợp này, nó chỉ đơn giản trả về đầu vào là chính nó (lambda x: x). 
Điều này có nghĩa là CountVectorizer sẽ không thực hiện bất kỳ sự tách từ nào, và giả định rằng dữ liệu đầu vào đã được tokenize trước đó.

preprocessor=lambda x: x: Đây là một hàm tiền xử lý tùy chỉnh. Cũng tương tự như tokenizer, nó chỉ trả về đầu vào là chính nó (lambda x: x). 
Điều này có nghĩa là CountVectorizer sẽ không thực hiện bất kỳ sự tiền xử lý nào trên dữ liệu đầu vào.
"""


# def identity_tokenizer(x):
#     return x
#
#
# def identity_preprocessor(x):
#     return x


vectorizer_cnt = CountVectorizer(
            # tokenizer=lambda x: x,
            # preprocessor=lambda x: x,

            # token_pattern mặc định là r'\b\w\w+\b' (một mẫu regex để tách các từ từ văn bản),
            # và không có bất kỳ tiền xử lý nào được áp dụng trước khi tokenize.
            # Với mẫu regex này, CountVectorizer sẽ tách các từ dựa trên các khoảng trắng và dấu câu,
            # token_pattern=None,
            token_pattern=r'\b\w+\b',
            # Tùy chọn loại bỏ dấu câu, dấu chữ từ văn bản.
            # Giá trị 'unicode' chỉ ra rằng mọi ký tự Unicode được xem xét và loại bỏ các dấu.
            strip_accents='unicode',

            # Chỉ định cấp độ phân tích.
            # Giá trị 'word' chỉ ra rằng quá trình phân tích sẽ dựa trên các từ.
            analyzer = 'word',

            # Dải n-gram được sử dụng, từ n-gram kích thước nhỏ nhất đến lớn nhất.
            # ngram_range=(2,3) có nghĩa là CountVectorizer sẽ xem xét cả bigram (2 từ liền kề)
            #   và trigram (3 từ liền kề) từ văn bản.
            ngram_range=(1, 2),
            # # Ngưỡng Loại bỏ từ xuất hiện quá ít
            min_df=0.10,
            # Ngưỡng loại bỏ từ xuất hiện quá nhiều lần
            max_df=0.85,

)


def example_count_vectorizer():
    # Giả sử bạn có một tập dữ liệu văn bản
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Never jump over the lazy dog quickly",
        "Bright brown foxes leap high"
    ]

    # Tạo một DataFrame từ các tài liệu
    df = pd.DataFrame({"text": documents})

    # Khởi tạo CountVectorizer
    count_vectorizer = CountVectorizer()

    # Chuyển đổi các tài liệu thành ma trận số đếm
    count_matrix = count_vectorizer.fit_transform(df["text"])

    # Chuyển đổi ma trận số đếm thành DataFrame để dễ nhìn
    count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

    pd.set_option('display.max_columns', None)
    print(count_df)
    """
               bright  brown  dog  fox  foxes  high  jump  jumps  lazy  leap  never  over  \
        0       0      1    1    1      0     0     0      1     1     0      0     1   
        1       0      0    1    0      0     0     1      0     1     0      1     1   
        2       1      1    0    0      1     1     0      0     0     1      0     0   
        
           quick  quickly  the  
        0      1        0    2  
        1      0        1    1  
        2      0        0    0  
    """


if __name__ == '__main__':
    example_count_vectorizer()