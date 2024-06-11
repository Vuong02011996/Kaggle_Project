import polars as pl
from Competition_1.code.Extract_Features.A_Paragraph_Features import Paragraph_Preprocess, Paragraph_Eng
from Competition_1.code.Extract_Features.B_Sentence_Features import Sentence_Preprocess, Sentence_Eng
from Competition_1.code.Extract_Features.C_Word_Features import Word_Preprocess, Word_Eng

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

"""
.with_columns(columns): Áp dụng danh sách các cột đã định nghĩa trong columns để xử lý DataFrame, 
cụ thể là thêm cột "paragraph" chứa các đoạn văn được tách ra từ cột "full_text".
"""
train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)

train.head(1)
with pl.Config(fmt_str_lengths=1000):
    print(train.head(5))


# Paragraph preprocess
tmp = Paragraph_Preprocess(train)
train_feats = Paragraph_Eng(tmp)
train_feats['score'] = train['score']
feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
print('Number of Features: ', len(feature_names))
with pl.Config(fmt_str_lengths=1000):
    print(train_feats.head(3))

# Sentence preprocess
tmp = Sentence_Preprocess(train)
train_feats = train_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')
feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
print('Features Number: ', len(feature_names))
train_feats.head(3)

# Word preprocess
tmp = Word_Preprocess(train)
train_feats = train_feats.merge(Word_Eng(tmp), on='essay_id', how='left')

feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
print('Features Number: ', len(feature_names))
train_feats.head(3)
