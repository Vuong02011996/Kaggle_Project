import pandas as pd


def basic_command_pandas(train_path):
    data = pd.read_csv(train_path)
    print(data.head())
    # Print all column names
    print(data.columns)

    # rename column
    # data = data.rename(columns={"holistic_essay_score": "score"})
    # print(data.columns)


if __name__ == '__main__':
    train_path = '/home/oryza/Desktop/KK/Competition_1/data/persuade_2.0_human_scores_demo_id_github.csv'
    train_path2 = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/train.csv'
    basic_command_pandas(train_path)
    basic_command_pandas(train_path2)