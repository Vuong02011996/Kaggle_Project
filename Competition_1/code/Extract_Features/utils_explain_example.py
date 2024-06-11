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

    # Print the resulting DataFrame
    # pl.Config.set_tbl_rows(df.shape[0])
    # with pd.option_context('display.max_colwidth', None,
    #                        'display.max_columns', None,
    #                        'display.max_rows', None):
    #     # display(df)
    #     display(df)
    # show full dataframe in polars
    with pl.Config(fmt_str_lengths=1000):
        print(df)


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
    df_exploded = df.explode('paragraph')

    # Print exploded DataFrame
    print("\nExploded DataFrame:")
    print(df_exploded)


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


if __name__ == '__main__':
    # explain_with_columns()
    # explain_explode()
    explain_map_elements()