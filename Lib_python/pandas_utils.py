
def basic_command_pandas(train_path):
    import pandas as pd
    data = pd.read_csv(train_path)
    print(data.head())
    # Print all column names
    print(data.columns)

    # rename column
    # data = data.rename(columns={"holistic_essay_score": "score"})
    # print(data.columns)


def explain_apply():
    import pandas as pd
    from gensim.utils import simple_preprocess
    from gensim.models import Word2Vec

    # Sample data
    data = {
        'prompt': ["What is AI?", "How to learn Python?", "Best practices in ML?"],
        'response_a': ["AI stands for Artificial Intelligence.", "Start with basics.", "Follow guidelines."],
        'response_b': ["AI is a field of study.", "Practice a lot.", "Read documentation."]
    }

    # Create a DataFrame
    train = pd.DataFrame(data)

    # Combine text columns
    # train[['prompt', 'response_a', 'response_b']]: Selects the specified columns from the DataFrame.
    # .astype(str): Converts the values in these columns to strings.
    # .apply(lambda x: ' '.join(x), axis=1): Applies a lambda function to each row (axis=1) that joins the text from
    # the three columns with a space in between.
    train_text = train[['prompt', 'response_a', 'response_b']].astype(str).apply(lambda x: ' '.join(x), axis=1)

    # Assign combined text
    vector_fit_text = train_text

    # Print message
    print("Training Word2Vec...")

    # Tokenize text
    train_tokens = vector_fit_text.map(simple_preprocess)

    # Train Word2Vec model
    vectors = Word2Vec(train_tokens, vector_size=60, window=3, seed=1, workers=4)
    vectors.save("word2vec_trained.model")

    # Print message
    print("Done")


if __name__ == '__main__':
    # train_path = '/home/oryza/Desktop/KK/Competition_1/data/persuade_2.0_human_scores_demo_id_github.csv'
    # train_path2 = '/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/train.csv'
    # basic_command_pandas(train_path)
    # basic_command_pandas(train_path2)
    explain_apply()