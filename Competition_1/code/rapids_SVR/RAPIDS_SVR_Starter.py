import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np, gc
import pandas as pd

from transformers import AutoModel, AutoTokenizer
import torch, torch.nn.functional as F
from tqdm import tqdm

train = pd.read_csv(
    "/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/train.csv")
print("Train shape", train.shape)
print(train.head())
print()

test = pd.read_csv("/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/test.csv")
print("Test shape", test.shape)
print(test.head())


def Stratified_15_K_Fold():
    from sklearn.model_selection import StratifiedKFold

    FOLDS = 15
    train["fold"] = -1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(train, train["score"])):
        train.loc[val_index, "fold"] = fold
    print('Train samples per fold:')
    train.fold.value_counts().sort_index()


"""Generate Embeddings"""


def mean_pooling(model_output, attention_mask):
    """

    hàm mean_pooling để tính toán giá trị trung bình của các embedding từ mô hình ngôn ngữ (ví dụ như BERT)
    dựa trên mặt nạ attention mask.
    :param model_output: là đầu ra của mô hình ngôn ngữ, thường chứa last_hidden_state, đó là các embedding của các token.
    :param attention_mask: là mặt nạ (mask) dùng để xác định các vị trí trong chuỗi đầu vào cần được chú ý (attention)
    và không phải là các padding token.
    :return:
    """
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    """
    + model_output.last_hidden_state là tensor chứa các embedding của các token từ đầu ra cuối cùng của mô hình.
    + .detach() tách tensor này khỏi cây tính toán của PyTorch để không theo dõi gradient nữa
     (điều này có ích khi bạn chỉ muốn tính toán mà không cần backpropagation)
    + .cpu() chuyển tensor sang CPU (thường cần thiết nếu tensor hiện đang trên GPU). 
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    """
    + attention_mask.unsqueeze(-1) thêm một chiều mới vào cuối của tensor attention_mask.
    + .expand(token_embeddings.size()) mở rộng kích thước của attention_mask để nó có cùng kích thước với token_embeddings.
    + .float() chuyển đổi tensor sang kiểu số thực.
    """

    output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    """
    + token_embeddings * input_mask_expanded nhân từng phần tử của token_embeddings với input_mask_expanded. 
    Điều này giúp bỏ qua các giá trị embedding tại các vị trí padding token (vì các vị trí đó có giá trị attention mask là 0)
    
    + torch.sum(token_embeddings * input_mask_expanded, 1) tính tổng các giá trị đã nhân theo chiều thứ nhất (chiều của các token trong chuỗi)\
    + input_mask_expanded.sum(1) tính tổng các giá trị của mặt nạ attention_mask theo chiều thứ nhất.
    + torch.clamp(input_mask_expanded.sum(1), min=1e-9) đảm bảo rằng tổng các giá trị mặt nạ không bao giờ bằng 0
     (tránh chia cho 0) bằng cách đặt giá trị nhỏ nhất là 1e-9.
     
     Kết quả của hàm mean_pooling là một tensor chứa các giá trị trung bình của các embedding, 
     tương ứng với các chuỗi đầu vào sau khi đã loại bỏ ảnh hưởng của các padding token
    """
    return output


class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "full_text"]
        tokens = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max,
            return_tensors="pt")
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        return tokens


"""Extract Embeddings"""


def get_embeddings(model_name='', max_length=1024, batch_size=32, compute_train=True, compute_test=True):
    global train, test

    DEVICE = "cuda:1"  # EXTRACT EMBEDDINGS WITH GPU #2
    path = "/kaggle/input/download-huggingface-models/"
    disk_name = path + model_name.replace("/", "_")
    model = AutoModel.from_pretrained(disk_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(disk_name, trust_remote_code=True)

    ds_tr = EmbedDataset(train, tokenizer, max_length)
    embed_dataloader_tr = torch.utils.data.DataLoader(ds_tr,
                                                      batch_size=batch_size,
                                                      shuffle=False)
    ds_te = EmbedDataset(test, tokenizer, max_length)
    embed_dataloader_te = torch.utils.data.DataLoader(ds_te,
                                                      batch_size=batch_size,
                                                      shuffle=False)

    model = model.to(DEVICE)
    model.eval()

    # COMPUTE TRAIN EMBEDDINGS
    all_train_text_feats = []
    if compute_train:
        for batch in tqdm(embed_dataloader_tr, total=len(embed_dataloader_tr)):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
            # Normalize the embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = sentence_embeddings.squeeze(0).detach().cpu().numpy()
            all_train_text_feats.extend(sentence_embeddings)
    all_train_text_feats = np.array(all_train_text_feats)

    # COMPUTE TEST EMBEDDINGS
    all_test_text_feats = []
    if compute_test:
        for batch in embed_dataloader_te:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
            # Normalize the embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = sentence_embeddings.squeeze(0).detach().cpu().numpy()
            all_test_text_feats.extend(sentence_embeddings)
        all_test_text_feats = np.array(all_test_text_feats)
    all_test_text_feats = np.array(all_test_text_feats)

    # CLEAR MEMORY
    del ds_tr, ds_te
    del embed_dataloader_tr, embed_dataloader_te
    del model, tokenizer
    del model_output, sentence_embeddings, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    # RETURN EMBEDDINGS
    return all_train_text_feats, all_test_text_feats


# EMBEDDINGS TO LOAD/COMPUTE
# PARAMETERS = (MODEL_NAME, MAX_LENGTH, BATCH_SIZE)
# CHOOSE LARGEST BATCH SIZE WITHOUT MEMORY ERROR

models = [
    ('microsoft/deberta-base', 1024, 32),
    ('microsoft/deberta-large', 1024, 8),
    ('microsoft/deberta-v3-large', 1024, 8),
    ('allenai/longformer-base-4096', 1024, 32),
    ('google/bigbird-roberta-base', 1024, 32),
    ('google/bigbird-roberta-large', 1024, 8),
]

path = "/kaggle/input/essay-embeddings-v1/"
all_train_embeds = []
all_test_embeds = []

for (model, max_length, batch_size) in models:
    name = path + model.replace("/", "_") + ".npy"
    if os.path.exists(name):
        _, test_embed = get_embeddings(model_name=model, max_length=max_length, batch_size=batch_size,
                                       compute_train=False)
        train_embed = np.load(name)
        print(f"Loading train embeddings for {name}")
    else:
        print(f"Computing train embeddings for {name}")
        train_embed, test_embed = get_embeddings(model_name=model, max_length=max_length, batch_size=batch_size,
                                                 compute_train=True)
        np.save(name, train_embed)
    all_train_embeds.append(train_embed)
    all_test_embeds.append(test_embed)

del train_embed, test_embed

"""Combine Feature Embeddings"""
all_train_embeds = np.concatenate(all_train_embeds, axis=1)
all_test_embeds = np.concatenate(all_test_embeds, axis=1)

gc.collect()
print('Our concatenated train embeddings have shape', all_train_embeds.shape)

"""Train RAPIDS cuML SVR"""
from cuml.svm import SVR
import cuml

print('RAPIDS version', cuml.__version__)

from sklearn.metrics import cohen_kappa_score

oof = np.zeros(len(train), dtype='float32')
test_preds = np.zeros((len(test), FOLDS), dtype='float32')


def comp_score(y_true, y_pred):
    p = y_pred.clip(1, 6).round(0)
    m = cohen_kappa_score(y_true, p, weights='quadratic')
    return m


for fold in range(FOLDS):
    print('#' * 25)
    print('### Fold', fold + 1)
    print('#' * 25)

    train_index = train["fold"] != fold
    valid_index = train["fold"] == fold

    X_train = all_train_embeds[train_index,]
    y_train = train.loc[train_index, 'score'].values
    X_valid = all_train_embeds[valid_index,]
    y_valid = train.loc[valid_index, 'score'].values
    X_test = all_test_embeds

    model = SVR(C=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    test_preds[:, fold] = model.predict(X_test)
    oof[valid_index] = preds

    score = comp_score(y_valid, preds)
    print(f"=> QWK score: {score}")
    print()

print('#' * 25)
score = comp_score(train.score.values, oof)
print('Overall CV QWK score =', score)

"""Find QWK Thresholds"""


def find_thresholds(true, pred, steps=50):
    # SAVE TRIALS FOR PLOTTING
    xs = [[], [], [], [], []]
    ys = [[], [], [], [], []]

    # COMPUTE BASELINE METRIC
    threshold = [1.5, 2.5, 3.5, 4.5, 5.5]
    pred2 = pd.cut(pred, [-np.inf] + threshold + [np.inf],
                   labels=[1, 2, 3, 4, 5, 6]).astype('int32')
    best = cohen_kappa_score(true, pred2, weights="quadratic")

    # FIND FIVE OPTIMAL THRESHOLDS
    for k in range(5):
        for sign in [1, -1]:
            v = threshold[k]
            threshold2 = threshold.copy()
            stop = 0
            while stop < steps:

                # TRY NEW THRESHOLD
                v += sign * 0.001
                threshold2[k] = v
                pred2 = pd.cut(pred, [-np.inf] + threshold2 + [np.inf],
                               labels=[1, 2, 3, 4, 5, 6]).astype('int32')
                metric = cohen_kappa_score(true, pred2, weights="quadratic")

                # SAVE TRIALS FOR PLOTTING
                xs[k].append(v)
                ys[k].append(metric)

                # EARLY STOPPING
                if metric <= best:
                    stop += 1
                else:
                    stop = 0
                    best = metric
                    threshold = threshold2.copy()

    # COMPUTE FINAL METRIC
    pred2 = pd.cut(pred, [-np.inf] + threshold + [np.inf],
                   labels=[1, 2, 3, 4, 5, 6]).astype('int32')
    best = cohen_kappa_score(true, pred2, weights="quadratic")

    # RETURN RESULTS
    threshold = [np.round(t, 3) for t in threshold]
    return best, threshold, xs, ys


best, thresholds, xs, ys = find_thresholds(train.score.values, oof, steps=500)
print('Best thresholds are:', thresholds)
print('=> achieve Overall CV QWK score =', best)

"""Display Threshold Trials"""
import matplotlib.pyplot as plt

diff = 0.5
for k in range(5):
    plt.figure(figsize=(10, 3))
    plt.scatter(xs[k], ys[k], s=3)
    m = k + 1.5
    plt.xlim((m - diff, m + diff))
    i = np.where((np.array(xs[k]) > m - diff) & (np.array(xs[k]) < m + diff))[0]
    mn = np.min(np.array(ys[k])[i])
    mx = np.max(np.array(ys[k])[i])
    plt.ylim((mn, mx))

    plt.plot([thresholds[k], thresholds[k]], [mn, mx], '--',
             color='black', label='optimal threshold')

    plt.title(f"Threshold between {k + 1} and {k + 2}", size=16)
    plt.xlabel('Threshold value', size=10)
    plt.ylabel('QWK CV score', size=10)
    plt.legend()
    plt.show()

"""Create Submission CSV"""
test_preds = np.mean(test_preds, axis=1)
print('Test preds shape:', test_preds.shape)
print('First 3 test preds:', test_preds[:3])

test_preds_pp = pd.cut(test_preds, [-np.inf] + thresholds + [np.inf],
                       labels=[1, 2, 3, 4, 5, 6]).astype('int32')
print('First 3 test preds after PP:', test_preds_pp[:3])

sub = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
sub["score"] = test_preds_pp
sub.score = sub.score.astype('int32')
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
sub.head()
