+ https://viblo.asia/p/cac-ky-thuat-tuning-mo-hinh-large-language-model-llm-aAY4qeZw4Pw

# Content 
+ [History ](#history-)
+ [Fine-Tuning LLMs](#fine-tuning-llms)
+ [Instruction Finetuning LLMs](#instruction-finetuning-llms)
+ [Catastrophic forgetting](#catastrophic-forgetting)
+ [Parameter Efficient Finetuning (PEFT)](#parameter-efficient-finetuning-peft)

# History 
+ Sự phát triển của Large language Model có thể được chia làm 4 bước ngoặt chính:
  + **Statistical languauge model (SLM)**
  + **Neural network languauge model (NLM)**: xây dựng dựa trên mạng nơ-ron, như là RNN, LSTM, GRU,...
  + **Pre-training language model (PLM)**:: Hầu hết chúng đều dựa trên kiến trúc transformer như BERT, BART, T5...
  + **Large language model (LLM)**: Dựa trên nền tảng của PLM, số lượng tham số mô hình tăng lên. Những mô hình kiểu này thường sử dụng cả kỹ thuật pretrained lẫn fintuning.
# Fine-Tuning LLMs
+ Fine-tuning mô hình là việc adapt mô hình có nhiều tác vụ về mô hình có một số tác vụ nhất định một cách chính xác và hiệu quả (ví dụ: fine-tuning model BERT cho bài toán phân loại văn bản..). 
+ Việc fine-tuning có nhiều lợi ích:
  + Tận dụng được những kiến thức mà mô hình pre-training đã được học rồi, điều này sẽ tiết kiệm được rất nhiều thời gian cũng như là tài nguyên tính toán so với việc training from scratch
  + Việc fine-tuning cho phép mô hình hoạt động tốt hơn trên một số task cụ thể.

## Instruction Finetuning LLMs
+ **Instruction fine-tuning** là một kỹ thuật để adapt LLMs thực hiện một số task cụ thể dựa trên những lời dẫn cụ thể (explicit instruction ).
+ mô hình được tinh chỉnh để thực hiện tốt hơn các nhiệm vụ dựa trên các chỉ dẫn cụ thể từ người dùng
+ Dữ liệu huấn luyện cho Instruction Finetuning thường bao gồm các cặp đầu vào-đầu ra, trong đó đầu vào là các chỉ dẫn hoặc câu hỏi từ người dùng, và đầu ra là các phản hồi mong muốn.
+ Tinh chỉnh mô hình: Sử dụng các bộ dữ liệu đã chuẩn bị, mô hình LLM ban đầu được huấn luyện lại hoặc tinh chỉnh để cải thiện khả năng thực hiện các nhiệm vụ theo chỉ dẫn.
+ các kỹ thuật finetuning khác:
  + **Traditional Finetuning**: Tinh chỉnh mô hình LLM trên một tập dữ liệu cụ thể của nhiệm vụ mà không yêu cầu chỉ dẫn từ người dùng.
  + **Prompt Tuning**: hay vì tinh chỉnh toàn bộ mô hình, chỉ tinh chỉnh một số lượng nhỏ các tham số (prompt) để mô hình thực hiện tốt hơn một nhiệm vụ cụ thể.
  + **Adapter Tuning**:  Sử dụng các mạng nơ-ron nhỏ (adapters) được chèn vào giữa các lớp của mô hình gốc để học thêm thông tin mà không làm thay đổi các tham số gốc của mô hình.

## Catastrophic forgetting
+ Catastrophic forgetting là một thách thức lớn trong việc phát triển các mô hình học máy mạnh mẽ và linh hoạt
+ Hiện tượng này xảy ra khi một mô hình học máy quên đi những kiến thức đã học trước đó sau khi học thêm kiến thức mới.
+ Các giải pháp cho Catastrophic Forgetting:
  + Regularization Techniques: 
    + Elastic Weight Consolidation (EWC): Một kỹ thuật thêm các ràng buộc (regularization) vào quá trình huấn luyện để giữ các trọng số quan trọng đối với các nhiệm vụ đã học trước đó
    + Synaptic Intelligence (SI)
  + Replay-based Methods:
    + Experience Replay: Lưu trữ và sử dụng lại một phần dữ liệu từ các nhiệm vụ đã học trước đó để huấn luyện lại mô hình khi học nhiệm vụ mới.
    + Generative Replay
  + Progressive Neural Networks:
    + Mạng nơ-ron tiến hóa: Thay vì điều chỉnh các trọng số của mô hình hiện tại, kỹ thuật này thêm các mạng con mới vào mô hình cho mỗi nhiệm vụ mới.

## Parameter Efficient Finetuning (PEFT)
+ Là một phương pháp tinh chỉnh các mô hình ngôn ngữ lớn (LLMs) sao cho tối ưu hóa hiệu suất mà không cần phải điều chỉnh tất cả các tham số của mô hình gốc. 
+ Thay vì cập nhật toàn bộ các tham số của mô hình, PEFT chỉ cập nhật một số lượng nhỏ các tham số, giúp giảm thiểu yêu cầu về tài nguyên tính toán và thời gian huấn luyện
+ Với các phương pháp như Adapter Tuning và Prompt Tuning, PEFT cho phép tận dụng tối đa khả năng của các mô hình hiện đại mà vẫn giữ được tính linh hoạt và hiệu quả.
