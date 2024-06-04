import torch


def ex_unsqueeze():
    """
    unsqueeze() rất hữu ích khi bạn cần thay đổi hình dạng của tensor để thực hiện các phép toán hoặc
     phù hợp với các yêu cầu của mô hình học máy
     Cụ thể, hàm unsqueeze(dim) sẽ thêm một chiều mới tại vị trí dim.
     tensor.unsqueeze(dim)
     dim là vị trí của chiều mới được thêm vào, và nó có thể là bất kỳ giá trị nào trong phạm vi từ -tensor.dim()-1 đến tensor.dim()
     Khi sử dụng dim là số âm, vị trí sẽ được tính từ cuối dãy.

    :return:
    """
    a = torch.randn((4, 4, 1))
    # a = torch.tensor([1, 2, 3, 4])
    print(a)
    print(a.shape)  # torch.size([4, 4, 1])
    print(a.squeeze(2).shape)  # torch.Size([4, 4]), dimension 2 has been removed
    print(a.squeeze(2))

    print(a.squeeze(0).shape)  # torch.Size([4, 4]), dimension 2 has been removed
    print(a.squeeze(0))

    # print(a.squeeze(-1).shape)  # torch.Size([4, 4]), last dimension has been removed (same effect)
    # print(a.unsqueeze(0).shape)  # torch.Size([1, 4, 4, 1]), one new dimension as first
    # print(a.unsqueeze(1).shape)  # torch.Size([4, 1, 4, 1]), one new dimension as second
    # print(a.unsqueeze(3).shape)  # torch.Size([4, 4, 1, 1]), one new dimension as fourth
    # print(a.unsqueeze(-1).shape)  # torch.Size([4, 4, 1, 1]), one new dimension as last (same effect)

    t = torch.tensor([1, 2, 3, 4])
    print(t.shape)
    # Output: torch.Size([4])
    t_unsqueezed = t.unsqueeze(-1)
    print(t_unsqueezed)
    print(t_unsqueezed.shape)
    # Output:
    # tensor([[1],
    #         [2],
    #         [3],
    #         [4]])
    # torch.Size([4, 1])


def ex_torch_expand():
    """
    a.expand(token_embeddings.size()) trong PyTorch được sử dụng để mở rộng một tensor a đến kích thước
    của tensor token_embeddings mà không thực sự sao chép dữ liệu.
    Điều này có nghĩa là nó sẽ tạo một view mới của a với kích thước chỉ định, nhưng dữ liệu thực tế sẽ
    không được sao chép, giúp tiết kiệm bộ nhớ.

    PyTorch sẽ tạo ra một view mới của tensor mà lặp lại các phần tử của tensor ban đầu để phù hợp với kích thước mới
    mà không thực sự sao chép dữ liệu

    Kích thước ban đầu của tensor a phải có số chiều nhỏ hơn hoặc bằng số chiều của kích thước mở rộng,
    và các chiều kích thước ban đầu phải bằng 1 hoặc phù hợp với các chiều tương ứng trong kích thước mở rộng.

    Phương thức expand() thường được sử dụng trong các tình huống mà bạn cần mở rộng một tensor nhỏ hơn để khớp với
    kích thước của một tensor khác
    :return:
    """
    # Tạo một tensor với kích thước ban đầu
    t = torch.tensor([[1], [2], [3]])
    print(f"Kích thước ban đầu của t: {t.size()}")

    # Giả sử bạn có một tensor token_embeddings
    token_embeddings = torch.tensor([[0.1, 0.2, 0.3],
                                     [0.4, 0.5, 0.6],
                                     [0.7, 0.8, 0.9]])
    print(f"Kích thước của token_embeddings: {token_embeddings.size()}")

    # Mở rộng t để có cùng kích thước như token_embeddings
    t_expanded = t.expand(token_embeddings.size())
    print(f"Kích thước của t_expanded: {t_expanded.size()}")

    # Kiểm tra nội dung của t_expanded
    print(t_expanded)
    """
    Kích thước ban đầu của t: torch.Size([3, 1])
    Kích thước của token_embeddings: torch.Size([3, 3])
    Kích thước của t_expanded: torch.Size([3, 3])
    tensor([[1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]])
    """


if __name__ == '__main__':
    ex_torch_expand()