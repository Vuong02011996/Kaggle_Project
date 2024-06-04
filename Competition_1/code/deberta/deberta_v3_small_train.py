import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from tokenizers import AddedToken
import random, os
import numpy as np
import torch


# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# True USES REGRESSION, False USES CLASSIFICATION
USE_REGRESSION = True

# VERSION NUMBER FOR NAMING OF SAVED MODELS
VER=2

# IF "LOAD_FROM" IS None, THEN WE TRAIN NEW MODELS
# LOAD_FROM = "/kaggle/input/deberta-v3-small_1-finetuned-v1/"
LOAD_FROM = None

# WHEN TRAINING NEW MODELS SET COMPUTE_CV = True
# WHEN LOADING MODELS, WE CAN CHOOSE True or False
COMPUTE_CV = True


warnings.simplefilter('ignore')
print("Pass import library")


class PATHS:
    train_path = '/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/train.csv'
    test_path = '/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/test.csv'
    sub_path = '/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/sample_submission.csv'
    model_path = "/Competition_1/models/deberta-v3-small"

    dir_save_model = '/Competition_1/models/debearta_v3_small_retrain_v2/'

class CFG:
    n_splits = 5
    seed = 42
    # max_length = 1024
    max_length = 1440
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 4
    weight_decay = 0.01
    # warmup_ratio = 0.0
    warmup_ratio = 0.1
    num_labels = 6


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Tokenize(object):
    def __init__(self, train, valid, tokenizer):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    def get_dataset(self, df):
      # Tạo một đối tượng Dataset từ dictionary chửa 3 cột lấy từ dataframe df
        ds = Dataset.from_dict({
                'essay_id': [e for e in df['essay_id']],
                'full_text': [ft for ft in df['full_text']],
                'label': [s for s in df['label']],
            })
        return ds

    def tokenize_function(self, example):
      # self.tokenizer(...) sử dụng phương thức tokenizer để token hóa văn bản trong example['full_text']
      # truncation=True để cắt ngắn văn bản nếu vượt quá độ dài tối đa
      #  max_length=CFG.max_length để chỉ định độ dài tối đa.
        tokenized_inputs = self.tokenizer(
            example['full_text'], truncation=True, max_length=CFG.max_length
        )
        return tokenized_inputs
    # khai báo phương thức __call__, cho phép đối tượng của lớp này có thể được gọi như một hàm
    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)
        # train_ds.map(...) áp dụng phương thức tokenize_function lên train_ds
        # bằng cách sử dụng map, với batched=True để xử lý theo lô.

        # map là một phương thức của lớp Dataset từ thư viện datasets của Hugging Face.
        # map áp dụng một hàm (trong trường hợp này là self.tokenize_function) lên mỗi phần tử của Dataset.
        tokenized_train = train_ds.map(
            self.tokenize_function, batched=True
        )
        tokenized_valid = valid_ds.map(
            self.tokenize_function, batched=True
        )

        return tokenized_train, tokenized_valid, self.tokenizer


def compute_metrics_for_regression(eval_pred):
    # predictions (dự đoán của mô hình) và labels (nhãn thực tế).
    predictions, labels = eval_pred

    # clip(0, 5) đảm bảo rằng tất cả các giá trị trong predictions nằm trong khoảng từ 0 đến 5.
    # Nếu một giá trị nhỏ hơn 0, nó sẽ được thay bằng 0; nếu lớn hơn 5, nó sẽ được thay bằng 5

    # round(0) làm tròn các giá trị trong predictions đến số nguyên gần nhất

    # weights='quadratic' chỉ định rằng Kappa được tính với trọng số bậc hai

    # Chỉ số Kappa của Cohen có trọng số bậc hai (qwk) là một thước đo độ phù hợp giữa các nhãn thực tế và nhãn dự đoán,
    # đặc biệt hữu ích trong các bài toán phân loại có thứ tự hoặc hồi quy phân loại, nơi mà khoảng cách giữa các nhãn có ý nghĩa.

    # Trọng số bậc hai (quadratic weights) giúp nhấn mạnh tầm quan trọng của các sai lệch lớn hơn.
    # Đây là lựa chọn phổ biến cho các bài toán phân loại có thứ tự (ordinal classification).

    qwk = cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results


def compute_metrics_for_classification(eval_pred):
    # predictions là một mảng (hoặc tensor) chứa các xác suất dự đoán cho mỗi lớp của mô hình.

    # argmax(-1) là một hàm NumPy được gọi trên predictions.
    # argmax trả về chỉ số của giá trị lớn nhất dọc theo trục được chỉ định.
    # -1 là trục cuối cùng, thường tương ứng với các xác suất dự đoán cho mỗi lớp.
    # Nghĩa là, hàm argmax(-1) tìm chỉ số của lớp có xác suất cao nhất cho mỗi mẫu dữ liệu.

    # predictions.argmax(-1): một mảng chứa nhãn dự đoán (predicted labels) được suy ra từ các xác suất dự đoán.

    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results


def k_fold_train_valid_data():
    data = pd.read_csv(PATHS.train_path)

    # data['score']: Lấy cột score từ DataFrame data. Đây là một Series chứa các giá trị của cột score.
    # .apply(lambda x: x-1): Sử dụng phương thức apply của Series để áp dụng một hàm lên từng giá trị của cột score.
    # Nếu data['score'] chứa các giá trị [3, 4, 5], thì data['score'].apply(lambda x: x-1) sẽ trả về [2, 3, 4].

    data['label'] = data['score'].apply(lambda x: x-1)

    if USE_REGRESSION:
      data["label"] = data["label"].astype('float32')
    else:
      data["label"] = data["label"].astype('int32')

    # Khởi tạo một đối tượng StratifiedKFold từ thư viện sklearn.model_selection.
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    # Chia dữ liệu và gán nhãn fold

    for i, (_, val_index) in enumerate(skf.split(data, data["score"])):
        data.loc[val_index, "fold"] = i

    # data.head(): Hiển thị 5 hàng đầu tiên của DataFrame data sau khi thêm cột label và fold.
    print(data.head())
    return data


training_args = TrainingArguments(
    #
    output_dir=PATHS.dir_save_model + f'output_v{VER}',
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=CFG.train_epochs,

    # weight_decay: Hệ số giảm trọng lượng, được sử dụng để kiểm soát overfitting bằng cách giảm giá trị của các trọng số
    weight_decay=CFG.weight_decay,

    # Chiến lược đánh giá, có thể là 'epoch' (sau mỗi epoch) hoặc 'steps' (sau mỗi bước huấn luyện).
    evaluation_strategy='epoch',

    # Độ đo được sử dụng để chọn mô hình tốt nhất, trong trường hợp này là 'qwk' (Quadratic Weighted Kappa)
    metric_for_best_model='qwk',

    # Chiến lược lưu trữ mô hình, có thể là 'epoch' (sau mỗi epoch) hoặc 'steps' (sau mỗi bước huấn luyện).
    save_strategy='epoch',

    # Số lượng mô hình được lưu trữ tối đa.
    save_total_limit=1,

    # load_best_model_at_end: Có nên tải mô hình tốt nhất vào cuối quá trình huấn luyện hay không.
    load_best_model_at_end=True,

    report_to='none',
    # Tỷ lệ warm-up, là phần trăm epochs sử dụng cho giai đoạn warm-up trong quá trình huấn luyện.
    warmup_ratio=CFG.warmup_ratio,

    lr_scheduler_type='linear', # "cosine" or "linear" or "constant"
    # Thuật toán tối ưu hóa được sử dụng, trong trường hợp này là 'adamw_torch' (AdamW từ thư viện PyTorch).
    optim='adamw_torch',

    logging_first_step=True,
)


def train(data):
    if COMPUTE_CV:

        # ADD NEW TOKENS for ("\n") new paragraph and (" "*2) double space
        # Tạo tokenizer từ mô hình được chỉ định bởi PATHS.model_path.
        # Thêm các token mới cho ký tự xuống dòng ("\n") và khoảng trắng kép (" ").
        tokenizer = AutoTokenizer.from_pretrained(PATHS.model_path)
        tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])

        for fold in range(len(data['fold'].unique())):

            # GET TRAIN AND VALID DATA
            # Dữ liệu huấn luyện là tất cả các mẫu không thuộc fold hiện tại,
            # và dữ liệu kiểm tra là các mẫu thuộc fold hiện tại.
            train = data[data['fold'] != fold]
            valid = data[data['fold'] == fold].copy()

            # # ADD NEW TOKENS for ("\n") new paragraph and (" "*2) double space
            # # Tạo tokenizer từ mô hình được chỉ định bởi PATHS.model_path.
            # # Thêm các token mới cho ký tự xuống dòng ("\n") và khoảng trắng kép (" ").
            # tokenizer = AutoTokenizer.from_pretrained(PATHS.model_path)
            # tokenizer.add_tokens([AddedToken("\n", normalized=False)])
            # tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])

            #  Khởi tạo đối tượng Tokenize để token hóa dữ liệu huấn luyện và kiểm tra.
            # Sau đó, gọi phương thức để thực hiện việc token hóa.
            tokenize = Tokenize(train, valid, tokenizer)
            # vì Tokenize có __call__ nên đối tượng có thể gọi như một hàm
            tokenized_train, tokenized_valid, _ = tokenize()

            # REMOVE DROPOUT FROM REGRESSION
            # AutoConfig: Đây là một lớp từ thư viện transformers của Hugging Face.
            # AutoConfig là một lớp tiện lợi giúp bạn tự động tải cấu hình phù hợp dựa trên tên hoặc đường dẫn của mô hình.

            # from_pretrained: Đây là một phương thức lớp (classmethod) của AutoConfig.
            # Nó tải cấu hình từ một mô hình đã được huấn luyện trước đó
            config = AutoConfig.from_pretrained(PATHS.model_path)
            if USE_REGRESSION:
                config.attention_probs_dropout_prob = 0.0
                config.hidden_dropout_prob = 0.0
                config.num_labels = 1
            else:
                config.num_labels = CFG.num_labels

            if LOAD_FROM:
                # Nếu load model đã train trước đó  không cần truyền config, không cần train
                model = AutoModelForSequenceClassification.from_pretrained(
                    LOAD_FROM + f'deberta-v3-small_AES2_fold_{fold}_v{VER}')
            else:
                model = AutoModelForSequenceClassification.from_pretrained(PATHS.model_path, config=config)
                # cập nhật số lượng token embeddings.
                model.resize_token_embeddings(len(tokenizer))

            # TRAIN WITH TRAINER
            # Tạo đối tượng DataCollatorWithPadding để điều chỉnh độ dài của các chuỗi đầu vào trong batch.
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            if USE_REGRESSION:
                compute_metrics = compute_metrics_for_regression
            else:
                compute_metrics = compute_metrics_for_classification

            # Trong trường hợp này, mỗi fold sẽ huấn luyện 4 lần.
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_valid,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            if LOAD_FROM is None:
                trainer.train()

            # PLOT CONFUSION MATRIX
            y_true = valid['score'].values
            predictions0 = trainer.predict(tokenized_valid).predictions
            if USE_REGRESSION:
                predictions = predictions0.round(0) + 1
            else:
                predictions = predictions0.argmax(axis=1) + 1
            cm = confusion_matrix(y_true, predictions, labels=[x for x in range(1, 7)])
            draw_cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                                             display_labels=[x for x in range(1, 7)])
            draw_cm.plot()
            plt.show()

            # SAVE FOLD MODEL AND TOKENIZER
            if LOAD_FROM is None:
                trainer.save_model(PATHS.dir_save_model + f'deberta-v3-small_AES2_fold_{fold}_v{VER}')
                tokenizer.save_pretrained(PATHS.dir_save_model + f'deberta-v3-small_AES2_fold_{fold}_v{VER}')

            # Lưu dự đoán ngoài mẫu (Out-Of-Fold, OOF).
            # Điều chỉnh dự đoán tùy thuộc vào việc sử dụng hồi quy hay phân loại và lưu kết quả vào tệp CSV

            # SAVE OOF PREDICTIONS
            if USE_REGRESSION:
                valid['pred'] = predictions0 + 1
            else:
                COLS = [f'p{x}' for x in range(CFG.num_labels)]
                valid[COLS] = predictions0
            valid.to_csv(f'valid_df_fold_{fold}_v{VER}.csv', index=False)


if __name__ == '__main__':
    data = k_fold_train_valid_data()
    train(data)