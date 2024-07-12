import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def explain_last_hidden_state():
    # last_hidden_state is [batch_size, sequence_length, hidden_size]

    last_hidden_state = torch.tensor([
        [  # First sequence
            [0.1, 0.2, 0.3],  # Hidden state for [CLS]
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5]
        ],
        [  # Second sequence
            [0.2, 0.3, 0.4],  # Hidden state for [CLS]
            [0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6]
        ]
    ])

    cls_hidden_states = last_hidden_state[:, 0, :]
    print("cls_hidden_states: ", cls_hidden_states)
    """
    cls_hidden_states = torch.tensor([
    [0.1, 0.2, 0.3],  # Hidden state for [CLS] of the first sequence
    [0.2, 0.3, 0.4]   # Hidden state for [CLS] of the second sequence
    ])
    """


def explain_vstack_hstack():
    # a batch of size 2 and feature dimension of 3:
    features_batch1 = np.array([[0.1, 0.2, 0.3],
                                [0.4, 0.5, 0.6]])

    features_batch2 = np.array([[0.7, 0.8, 0.9],
                                [1.0, 1.1, 1.2]])

    features_batch3 = np.array([[1.3, 1.4, 1.5],
                                [1.6, 1.7, 1.8]])
    features = [features_batch1, features_batch2, features_batch3]

    combined_features = np.vstack(features)
    # stacked vertically: (6,3)
    print(combined_features)
    """combined_features = np.array([[0.1, 0.2, 0.3],
                                  [0.4, 0.5, 0.6],
                                  [0.7, 0.8, 0.9],
                                  [1.0, 1.1, 1.2],
                                  [1.3, 1.4, 1.5],
                                  [1.6, 1.7, 1.8]])"""

    combined_features = np.hstack(features)  # (6,3)
    print(combined_features)
    # (9, 2)
    """
    [[0.1 0.2 0.3 0.7 0.8 0.9 1.3 1.4 1.5]
     [0.4 0.5 0.6 1.  1.1 1.2 1.6 1.7 1.8]]
    """


def explain_transpose():
    """
    np.transpose: Reorders the dimensions of the array. It does not change the number of elements or their layout in memory but rather changes the way dimensions are accessed.
    torch.view: Reshapes the tensor to a new shape. It changes the layout of elements in memory to match the new shape.
    :return:
    """
    # Changing the order of dimensions for multi-dimensional arrays
    #  from (batch_size, sequence_length, feature_dim) to (sequence_length, batch_size, feature_dim)).
    # (0, 1, 2) => (1, 0, 2)
    vectors = np.array([
        [[1, 2], [3, 4], [5, 6], [7, 8]],  # First "slice"
        [[9, 10], [11, 12], [13, 14], [15, 16]],  # Second "slice"
        [[17, 18], [19, 20], [21, 22], [23, 24]]  # Third "slice"
    ])
    print(vectors.shape)
    # Output: (3, 4, 2)
    transposed_vectors = np.transpose(vectors, (1, 0, 2))
    print(transposed_vectors.shape)
    # Output: (4, 3, 2)
    transposed_vectors = np.array([
        [[1, 2], [9, 10], [17, 18]],  # First "slice" along the new Dimension 0
        [[3, 4], [11, 12], [19, 20]],  # Second "slice" along the new Dimension 0
        [[5, 6], [13, 14], [21, 22]],  # Third "slice" along the new Dimension 0
        [[7, 8], [15, 16], [23, 24]]  # Fourth "slice" along the new Dimension 0
    ])

    # View
    tensor = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    reshaped_tensor = tensor.view(4, 2)
    print(reshaped_tensor)
    # Output:
    # tensor([[1, 2],
    #         [3, 4],
    #         [5, 6],
    #         [7, 8]])


def explain_idxmax():
    # Create a toy dataset with one-hot encoded target columns
    data = {
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1.1, 1.2, 1.3, 1.4],
        'target_A': [1, 0, 0, 1],
        'target_B': [0, 1, 0, 0],
        'target_C': [0, 0, 1, 0]
    }

    train = pd.DataFrame(data)

    # Define the target columns
    target_columns = ['target_A', 'target_B', 'target_C']

    # Use idxmax to find the original categories
    # to find the index of the maximum value along a specific axis
    # axis=1 means that we are looking for the maximum value index along the columns for each row.
    y = train[target_columns].idxmax(axis=1)

    print("Original DataFrame:")
    print(train)
    """
       feature1  feature2  target_A  target_B  target_C
    0       0.1       1.1         1         0         0
    1       0.2       1.2         0         1         0
    2       0.3       1.3         0         0         1
    3       0.4       1.4         1         0         0
    """
    print("\nExtracted Target Labels:")
    print(y)
    """
    Extracted Target Labels:
    0    target_A
    1    target_B
    2    target_C
    3    target_A
    dtype: object
    """


def explain_label_encoding():
    # Create a toy dataset with one-hot encoded target columns
    data = {
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1.1, 1.2, 1.3, 1.4],
        'target_A': [1, 0, 0, 1],
        'target_B': [0, 1, 0, 0],
        'target_C': [0, 0, 1, 0]
    }

    train = pd.DataFrame(data)

    # Define the target columns
    target_columns = ['target_A', 'target_B', 'target_C']

    # Use idxmax to find the original categories
    y = train[target_columns].idxmax(axis=1)

    print("Original Target Labels:")
    print(y)

    # Encode labels
    # Creates an instance of LabelEncoder to convert categorical labels to numerical labels.
    label_encoder = LabelEncoder()
    # Fits the label encoder to the target labels and transforms them into numerical labels.
    y_encoded = label_encoder.fit_transform(y)

    print("\nEncoded Target Labels:")
    print(y_encoded)  # [0 1 2 0]

    # To show the mapping from original labels to encoded labels
    print("\nLabel Mapping:")
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(label_mapping)  # {'target_A': 0, 'target_B': 1, 'target_C': 2}


def explain_re_findall():
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Sample text
    text = "Hello, world! This is a test."

    """
    This tokenizer will split the input text into tokens (words) by finding all sequences of word characters and 
    ignoring any non-word characters
    """
    # Custom tokenizer using lambda function and regex pattern
    tokenizer = lambda x: re.findall(r'[^\W]+', x)

    # Apply the tokenizer to the sample text
    tokens = tokenizer(text)
    # Output the tokens
    print(tokens)
    # ['Hello', 'world', 'This', 'is', 'a', 'test']

    # Using token_pattern instead of tokenizer
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 5),  # Consider n-grams from 1 to 5 words
        token_pattern=r'[^\W]+',  # Equivalent regular expression pattern
        strip_accents='unicode',  # Remove accents and convert characters to Unicode
        min_df=4,  # Ignore terms that appear in fewer than 4 documents
        max_features=300  # Limit the number of features to 300
    )

    # Example usage with sample text data
    sample_texts = [
        "Hello, world! This is a test.",
        "Another test, with more words."
    ]

    # Fit and transform the sample text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(sample_texts)

    # Output the feature names
    print(tfidf_vectorizer.get_feature_names_out())

    # Output the TF-IDF matrix
    print(tfidf_matrix.toarray())


if __name__ == '__main__':
    # explain_last_hidden_state()
    # explain_vstack_hstack()
    # explain_idxmax()
    # explain_label_encoding()
    explain_re_findall()