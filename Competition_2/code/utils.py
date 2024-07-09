import torch
import numpy as np

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


if __name__ == '__main__':
    # explain_last_hidden_state()
    explain_vstack_hstack()