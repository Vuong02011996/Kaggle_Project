import polars as pl
import pandas as pd
from IPython.core.display_functions import display

from Competition_1.code.Extract_Features.preprocessing import dataPreprocessing


def explain_with_columns():
    # Sample data to mimic reading from a CSV
    data = {
        "essay_id": [1, 2, 3],
        "full_text": [
            "This is the first paragraph of essay 1.\n\nThis is the second paragraph of essay 1.",
            "This is the first paragraph of essay 2.",
            "This is the first paragraph of essay 3.\n\nThis is the second paragraph of essay 3.\n\nThis is the third paragraph of essay 3."
        ]
    }

    # Create a DataFrame from the sample data
    df = pl.DataFrame(data)

    # Define the column transformation
    columns = [
        (
            pl.col("full_text").str.split(by="\n\n").alias("paragraph")
        ),
    ]

    # Apply the column transformation
    # This method is used to add or modify columns in the DataFrame.
    df = df.with_columns(columns)

    # show full dataframe in polars
    with pl.Config(fmt_str_lengths=1000):
        print(df)

    """
        shape: (3, 3)
    ┌──────────┬──────────────────────────────────────────┬────────────────────────────────────────────┐
    │ essay_id ┆ full_text                                ┆ paragraph                                  │
    │ ---      ┆ ---                                      ┆ ---                                        │
    │ i64      ┆ str                                      ┆ list[str]                                  │
    ╞══════════╪══════════════════════════════════════════╪════════════════════════════════════════════╡
    │ 1        ┆ This is the first paragraph of essay 1.  ┆ ["This is the first paragraph of essay     │
    │          ┆                                          ┆ 1.", "This is the second paragraph of      │
    │          ┆ This is the second paragraph of essay 1. ┆ essay 1."]                                 │
    │ 2        ┆ This is the first paragraph of essay 2.  ┆ ["This is the first paragraph of essay     │
    │          ┆                                          ┆ 2."]                                       │
    │ 3        ┆ This is the first paragraph of essay 3.  ┆ ["This is the first paragraph of essay     │
    │          ┆                                          ┆ 3.", "This is the second paragraph of      │
    │          ┆ This is the second paragraph of essay 3. ┆ essay 3.", "This is the third paragraph of │
    │          ┆                                          ┆ essay 3."]                                 │
    │          ┆ This is the third paragraph of essay 3.  ┆                                            │
    └──────────┴──────────────────────────────────────────┴────────────────────────────────────────────┘
    """


def explain_explode():
    # Sample data
    data = {
        "essay_id": [1, 2, 3],
        "paragraph": [
            ["First paragraph of essay 1", "Second paragraph of essay 1"],
            ["First paragraph of essay 2"],
            ["First paragraph of essay 3", "Second paragraph of essay 3", "Third paragraph of essay 3"]
        ]
    }

    # Create DataFrame
    df = pl.DataFrame(data)

    # Print original DataFrame
    print("Original DataFrame:")
    with pl.Config(fmt_str_lengths=1000):
        print(df)

    # Explode the 'paragraph' column
    # explode : nổ tung
    df_exploded = df.explode('paragraph')

    # Print exploded DataFrame
    print("\nExploded DataFrame:")
    print(df_exploded)
    """
    shape: (6, 2)
    ┌──────────┬─────────────────────────────┐
    │ essay_id ┆ paragraph                   │
    │ ---      ┆ ---                         │
    │ i64      ┆ str                         │
    ╞══════════╪═════════════════════════════╡
    │ 1        ┆ First paragraph of essay 1  │
    │ 1        ┆ Second paragraph of essay 1 │
    │ 2        ┆ First paragraph of essay 2  │
    │ 3        ┆ First paragraph of essay 3  │
    │ 3        ┆ Second paragraph of essay 3 │
    │ 3        ┆ Third paragraph of essay 3  │
    └──────────┴─────────────────────────────┘
    """


def explain_map_elements():
    # Sample data
    data = {
        "essay_id": [1, 2, 3],
        "paragraph": [
            "I'm happy! This is the first paragraph of essay 1.",
            "This is the second paragraph of essay 2 with HTML <b>tags</b>.",
            "It's the third paragraph of essay 3. Check out http://example.com."
        ]
    }

    # Create a DataFrame
    df = pl.DataFrame(data)

    # Print the original DataFrame
    print("Original DataFrame:")
    with pl.Config(fmt_str_lengths=1000):
        print(df)

    # Add a new column with preprocessed paragraphs
    df = df.with_columns(pl.col('paragraph').map_elements(dataPreprocessing).alias("processed_paragraph"))

    # Print the modified DataFrame
    print("\nDataFrame after adding 'processed_paragraph' column:")
    with pl.Config(fmt_str_lengths=1000):
        print(df)

    """
    shape: (3, 3)
    ┌──────────┬───────────────────────────────────────────┬───────────────────────────────────────────┐
    │ essay_id ┆ paragraph                                 ┆ processed_paragraph                       │
    │ ---      ┆ ---                                       ┆ ---                                       │
    │ i64      ┆ str                                       ┆ str                                       │
    ╞══════════╪═══════════════════════════════════════════╪═══════════════════════════════════════════╡
    │ 1        ┆ I'm happy! This is the first paragraph of ┆ i'm happy! this is the first paragraph of │
    │          ┆ essay 1.                                  ┆ essay .                                   │
    │ 2        ┆ This is the second paragraph of essay 2   ┆ this is the second paragraph of essay     │
    │          ┆ with HTML <b>tags</b>.                    ┆ with html tags.                           │
    │ 3        ┆ It's the third paragraph of essay 3.      ┆ it is the third paragraph of essay .      │
    │          ┆ Check out http://example.com.             ┆ check out http://example.com.             │
    └──────────┴───────────────────────────────────────────┴───────────────────────────────────────────┘
    """


def explain_star():
    numbers = [1, 2, 3]

    # Không dùng unpacking
    print(numbers)  # Output: [1, 2, 3]

    # Dùng unpacking
    print(*numbers)  # Output: 1 2 3


def explain_aggs():
    columns = ['column1', 'column2', 'column3']

    # Không dùng unpacking
    agg_list = [
        [pl.col(col).max().alias(f"{col}_max") for col in columns]
    ]
    # agg_list sẽ là danh sách lồng nhau, với một danh sách các phép tính bên trong:
    # [
    #     [
    #         pl.col('column1').max().alias('column1_max'),
    #         pl.col('column2').max().alias('column2_max'),
    #         pl.col('column3').max().alias('column3_max')
    #     ]
    # ]

    # Dùng unpacking
    agg_list = [
        *[pl.col(col).max().alias(f"{col}_max") for col in columns]
    ]
    # agg_list sẽ là một danh sách phẳng chứa các phép tính:
    # [
    #     pl.col('column1').max().alias('column1_max'),
    #     pl.col('column2').max().alias('column2_max'),
    #     pl.col('column3').max().alias('column3_max')
    # ]


def explain_group_by():
    """
    group_by(['essay_id']): Nhóm các hàng trong DataFrame theo giá trị của cột essay_id.
    Các hàng có cùng giá trị essay_id sẽ được nhóm lại với nhau.
    maintain_order=True: Đảm bảo rằng thứ tự của các nhóm được giữ nguyên như trong DataFrame gốc.
    :return:
    """
    # Tạo DataFrame giả sử
    data = {
        "essay_id": [1, 1, 2, 2, 3, 3],
        "paragraph": ["Para1", "Para2", "Para1", "Para2", "Para1", "Para2"],
        "paragraph_len": [100, 150, 200, 250, 300, 350],
    }

    train_tmp = pl.DataFrame(data)

    # Thực hiện group_by
    grouped_df = train_tmp.group_by(['essay_id'], maintain_order=True)
    with pl.Config(fmt_str_lengths=1000):
        print(grouped_df.df)
        """
        shape: (6, 3)
        ┌──────────┬───────────┬───────────────┐
        │ essay_id ┆ paragraph ┆ paragraph_len │
        │ ---      ┆ ---       ┆ ---           │
        │ i64      ┆ str       ┆ i64           │
        ╞══════════╪═══════════╪═══════════════╡
        │ 1        ┆ Para1     ┆ 100           │
        │ 1        ┆ Para2     ┆ 150           │
        │ 2        ┆ Para1     ┆ 200           │
        │ 2        ┆ Para2     ┆ 250           │
        │ 3        ┆ Para1     ┆ 300           │
        │ 3        ┆ Para2     ┆ 350           │
        └──────────┴───────────┴───────────────┘
        """

    # Hiển thị các nhóm
    for group in grouped_df:
        print(group)


def explain_function_agg():
    """
    Hàm .agg trong thư viện Polars được sử dụng để thực hiện các phép toán tổng hợp (aggregation)
    trên các nhóm dữ liệu được tạo bởi hàm group_by
    áp dụng các phép tổng hợp được xác định trong danh sách aggs lên từng nhóm
    :return:
    """
    # Tạo DataFrame giả sử
    data = {
        "essay_id": [1, 1, 2, 2, 3, 3],
        "paragraph_len": [100, 150, 200, 250, 300, 350],
        "paragraph_error_num": [2, 1, 4, 3, 2, 5]
    }

    train_tmp = pl.DataFrame(data)

    # Định nghĩa các phép tổng hợp
    aggs = [
        pl.col('paragraph_len').max().alias('max_paragraph_len'),
        pl.col('paragraph_len').mean().alias('mean_paragraph_len'),
        pl.col('paragraph_len').sum().alias('sum_paragraph_len'),
        pl.col('paragraph_error_num').sum().alias('sum_paragraph_error_num'),
    ]

    # Thực hiện group_by và aggregate
    result = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs)

    # Chuyển kết quả sang pandas DataFrame để hiển thị
    # result_df = result.to_pandas()
    with pl.Config(fmt_str_lengths=1000, tbl_cols=-1):
        # cfg.set_tbl_cols(5)
        # cfg.set_tbl_cols(result_df.width)
        print(result)
        """
        shape: (3, 5)
        ┌──────────┬───────────────────┬────────────────────┬───────────────────┬─────────────────────────┐
        │ essay_id ┆ max_paragraph_len ┆ mean_paragraph_len ┆ sum_paragraph_len ┆ sum_paragraph_error_num │
        │ ---      ┆ ---               ┆ ---                ┆ ---               ┆ ---                     │
        │ i64      ┆ i64               ┆ f64                ┆ i64               ┆ i64                     │
        ╞══════════╪═══════════════════╪════════════════════╪═══════════════════╪═════════════════════════╡
        │ 1        ┆ 150               ┆ 125.0              ┆ 250               ┆ 3                       │
        │ 2        ┆ 250               ┆ 225.0              ┆ 450               ┆ 7                       │
        │ 3        ┆ 350               ┆ 325.0              ┆ 650               ┆ 7                       │
        └──────────┴───────────────────┴────────────────────┴───────────────────┴─────────────────────────┘
        """


if __name__ == '__main__':
    # explain_with_columns()
    # explain_explode()
    # explain_map_elements()
    # explain_group_by()
    explain_function_agg()