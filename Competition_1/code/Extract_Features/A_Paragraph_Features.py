import polars as pl
from Competition_1.code.Extract_Features.preprocessing import dataPreprocessing, remove_punctuation, count_spelling_errors


def Paragraph_Preprocess(tmp):
    """
    explode columns paragraph and preprocessing create more columns such as:
    + paragraph_no_pinctuation
    + paragraph_error_num
    + paragraph_len
    + paragraph_sentence_cnt
    + paragraph_word_cnt
    :param tmp: dataframe from polars
    :return: new dataframe with new columns
    """
    """
    explode tách các danh sách hoặc mảng trong cột paragraph thành các hàng riêng lẻ.
    Mỗi phần tử trong danh sách sẽ trở thành một hàng mới trong DataFrame.
    """
    tmp = tmp.explode('paragraph')

    """
    with_columns tạo ra các cột mới hoặc thay đổi cột hiện tại. 
    Ở đây, nó áp dụng hàm dataPreprocessing lên từng phần tử trong cột paragraph.
    pl.col là một hàm trong thư viện Polars dùng để tham chiếu tới một cột trong DataFrame.
    pl.col('paragraph') trả về một biểu thức (expression) đại diện cho cột paragraph trong DataFrame.
    
    map_elements là một phương thức của biểu thức cột trong Polars. 
    Nó cho phép bạn áp dụng một hàm tùy ý lên từng phần tử (element) của cột đó.
    """
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(dataPreprocessing))
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(remove_punctuation).alias('paragraph_no_pinctuation'))
    tmp = tmp.with_columns(pl.col('paragraph_no_pinctuation').map_elements(count_spelling_errors).alias("paragraph_error_num"))
    # Tính độ dài đoạn văn
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x)).alias("paragraph_len"))
    # Tính số câu trong đoạn văn.
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x.split('.'))).alias("paragraph_sentence_cnt"),
                           # Tính số từ trong đoạn văn.
                    pl.col('paragraph').map_elements(lambda x: len(x.split(' '))).alias("paragraph_word_cnt"),)
    return tmp


# feature_eng
paragraph_fea = ['paragraph_len',
                 'paragraph_sentence_cnt',
                 'paragraph_word_cnt']
paragraph_fea2 = ['paragraph_error_num'] + paragraph_fea


def Paragraph_Eng(train_tmp):
    """

    :param train_tmp: dataframe from polars
    :return:
    """
    aggs = [
        # Đếm số đoạn văn có độ dài lớn hơn một giá trị nhất định.
        *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_>{i}_cnt") for i in [0, 50,75,100,125,150,175,200,250,300,350,400,500,600,700] ],
        # Đếm số đoạn văn có độ dài nhỏ hơn một giá trị nhất định.
        *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_<{i}_cnt") for i in [25,49]],
        # Tính giá trị lớn nhất, trung bình, nhỏ nhất, tổng, giá trị đầu tiên, giá trị cuối cùng, độ lệch,
        # và các giá trị phần tư thứ nhất và thứ ba của các đặc trưng.
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea2],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea2],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea2],
        *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in paragraph_fea2],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea2],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea2],
        *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in paragraph_fea2],
        *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in paragraph_fea2],
        *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in paragraph_fea2],
        ]
    """
    thực hiện việc nhóm các hàng của DataFrame train_tmp theo cột essay_id, 
    tính toán các phép toán tổng hợp (aggregations) được định nghĩa trong danh sách aggs, 
    và sau đó sắp xếp kết quả theo cột essay_id.
    """
    df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df



