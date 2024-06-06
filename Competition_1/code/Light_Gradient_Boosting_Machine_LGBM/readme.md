# Ensemble learning
+ Bài toán về phân loại (classification) hay hồi quy (regression), chắc ai cũng biết phần quan trọng nhất là lựa chọn model
+ Việc chọn này phụ thuộc nhiều yếu tố: số lượng data, đặc điểm data (số chiều, phân phối), v.v...
+ Từ đó ta sẽ có tương quan giữa data và model (bias-variance tradeoff) aka (mối quan hệ đánh đổi giữa bias và variance).

# Tradeoff bias / variance.
+ **Bias (Thiên lệch)**
  + Nếu mô hình có bias cao, nó sẽ không nắm bắt được các đặc điểm quan trọng của dữ liệu. dẫn đến việc dự đoán kém ngay cả trên dữ liệu huấn luyện
  + Điều này thường xảy ra khi mô hình quá đơn giản (underfitting)
+ **Variance (Phương sai)**
  + Một mô hình có variance cao sẽ khớp rất tốt với dữ liệu huấn luyện nhưng lại kém hiệu quả khi gặp dữ liệu mới (overfitting).
  + Điều này thường xảy ra khi mô hình quá phức tạp, nó nắm bắt cả các nhiễu (noise) trong dữ liệu huấn luyện.
+ Nếu mô hình quá đơn giản (bias cao, variance thấp), nó sẽ không dự đoán tốt do underfitting. 
+ Nếu mô hình quá phức tạp (bias thấp, variance cao), nó sẽ dự đoán kém trên dữ liệu mới do overfitting.
+ _Điểm lý tưởng là nơi mà tổng lỗi (bias + variance) là thấp nhất._
+ Để giảm bias:
  + Tăng độ phức tạp của mô hình (sử dụng mô hình phức tạp hơn như neural networks thay vì linear regression).
  + Thêm nhiều đặc trưng (features) hơn vào mô hình.
+ Để giảm variance:
  + Sử dụng nhiều dữ liệu huấn luyện hơn.
  + Sử dụng kỹ thuật regularization (như L1, L2 regularization).
  + Sử dụng kỹ thuật ensemble (như **bagging**, **boosting**).
# 3 biến thể của phương thức ensemble learning
## Bagging
+ Xây dựng một lượng lớn các model (thường là cùng loại) trên những subsamples khác nhau từ tập training dataset (random sample trong 1 dataset để tạo 1 dataset mới).
+ Những model này sẽ được train độc lập và song song với nhau nhưng đầu ra của chúng sẽ được trung bình cộng để cho ra kết quả cuối cùng.

## Boosting
+ Xây dựng một lượng lớn các model (thường là cùng loại). Mỗi model sau sẽ học cách sửa những errors của model trước (dữ liệu mà model trước dự đoán sai) 
-> tạo thành một chuỗi các model mà model sau sẽ tốt hơn model trước bởi trọng số được update qua mỗi model .
+ trọng số của những dữ liệu dự đoán đúng sẽ không đổi, còn trọng số của những dữ liệu dự đoán sai sẽ được tăng thêm
+ Chúng ta sẽ lấy kết quả của model cuối cùng trong chuỗi model này làm kết quả trả về

## Stacking
+ Xây dựng một số model (thường là khác loại) và một meta model (supervisor model), train những model này độc lập, 
sau đó meta model sẽ học cách kết hợp kết quả dự báo của một số mô hình một cách tốt nhất

+ So sánh:
  + Trong 3 biến thể trên thì Bagging giúp ensemble model giảm variance.
  + Còn Boosting và Stacking tập trung vào việc giảm bias (cũng giảm cả variance).

# Boosting
+ Các model trong Bagging đều là học một cách riêng rẽ, không liên quan hay ảnh hưởng gì đến nhau, điều này trong một số trường hợp có thể dẫn đến kết quả tệ khi các model có thể học cùng ra 1 kết quả.
+ Chúng ta mong đợi các model yếu của thể hỗ trợ lẫn nhau, học được từ nhau để tránh đi vào các sai lầm của model trước đó. 
+ Ý tưởng cơ bản là Boosting sẽ tạo ra một loạt các model yếu, học bổ sung lẫn nhau.
+ Nói cách khác, trong Boosting, các model sau sẽ cố gắng học để hạn chế lỗi lầm của các model trước.

+ 2 loại Boosting :
  + Adaptive Boosting (AdaBoost)
  + Gradient Boosting
+ 2 framework phổ biến nhất của Gradient Boosting là **XGBoost** và **LightGBM**
  + XGBoost (Extreme Gradient Boosting): https://github.com/dmlc/xgboost
    +  ra mắt năm 2014
    +  XGBoost gặp một vấn đề là thời gian training khá lâu, đặc biệt với những bộ dữ liệu lớn.
    + Độ chính xác: Rất cao
    + Sử dụng bộ nhớ: Yêu cầu bộ nhớ cao hơn so với LightGBM
  + LightGBM: https://github.com/microsoft/LightGBM/tree/master
    + tháng 1 năm 2016, Microsoft lần đầu realease phiên bản thử nghiệm LightGBM.
    + Độ chính xác: Cao
    + Nhanh hơn XGBoost đáng kể, đặc biệt là với các tập dữ liệu lớn. LightGBM tận dụng các kỹ thuật tăng tốc để giảm thời gian huấn luyện.
    + Sử dụng bộ nhớ: Hiệu quả hơn trong việc sử dụng bộ nhớ do các kỹ thuật tối ưu như histogram-based method.
  + .Thật ra cả xgboost và lightgbm đều sử dụng histogram-based algorithms,
  + điểm tối ưu của lightgbm so với xgboost là ở 2 thuật toán: **GOSS (Gradient Based One Side Sampling)** và **EFB (Exclusive Feature Bundling)** giúp tăng tốc đáng kể trong quá trình tính toán.
  + LightGBM phát triển tree dựa trên leaf-wise, trong khi hầu hết các boosting tool khác (kể cả xgboost) dựa trên level (depth)-wise
  
# LightGBM 
