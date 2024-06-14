# kĩ thuật cross-validation
## KFold:  
+ KFold chia dữ liệu thành K phần (folds) bằng nhau, sử dụng K-1 phần để huấn luyện và 1 phần để kiểm tra, 
+ lặp lại K lần để mỗi phần dữ liệu đều được sử dụng làm tập kiểm tra một lần
## StratifiedKFold:
+ StratifiedKFold tương tự như KFold nhưng bảo đảm tỷ lệ các lớp trong mỗi fold giống với tỷ lệ các lớp trong toàn bộ dữ liệu
## GroupKFold:
+ Kỹ thuật này đặc biệt hữu ích trong các trường hợp mà sự phụ thuộc giữa các mẫu trong cùng một nhóm có thể ảnh hưởng đến quá trình huấn luyện và kiểm tra mô hình