from sklearn.metrics import cohen_kappa_score
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import log_evaluation, early_stopping
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from Competition_1.code.Extract_Features.main_feature_engineering import extract_features
from Competition_1.code.Light_Gradient_Boosting_Machine_LGBM.infer_lgbm import infer

a = 2.998
b = 1.092
VER = 1
path_vectorizer_models = "/home/oryza/Desktop/KK/Competition_1/models/vectorizer_models/"


def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return 'QWK', qwk, True


def qwk_obj(y_true, y_pred):
    """
    qwk_obj cung cấp các thông tin cần thiết để LightGBM hoặc các mô hình GBDT khác có thể sử dụng trong quá trình huấn luyện
    để tối ưu hóa hàm mục tiêu tùy chỉnh, từ đó cải thiện chất lượng dự đoán của mô hình.

    Một hàm mục tiêu tùy chỉnh phải tuân thủ một số quy tắc nhất định và thường bao gồm hai thành phần chính:

    Giá trị hàm mục tiêu (Objective Value): Đánh giá mức độ lỗi hiện tại.
    Đạo hàm bậc nhất và bậc hai (Gradient and Hessian): Sử dụng để cập nhật các tham số mô hình.

    :param y_true: Các giá trị mục tiêu thực tế.
    :param y_pred: Các giá trị dự đoán từ mô hình.
    :return:
    """
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    # Tổng bình phương sai số giữa dự đoán và nhãn điều chỉnh, nhân với 1/2.
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    # Đạo hàm của f theo preds.
    df = preds - labels
    # Đạo hàm của g theo preds.
    dg = preds - a

    # Gradient của hàm mục tiêu tùy chỉnh.
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))

    """
    Gradient (grad): Cho biết hướng và độ lớn của sự thay đổi cần thiết để giảm thiểu hàm mục tiêu.
    Hessian (hess): Ma trận Hessian (ở đây đơn giản hóa thành vector đơn vị) cho biết thông tin về độ cong của hàm mục tiêu,
     giúp cải thiện quá trình tối ưu hóa.
    """
    return grad, hess


def train(train_feats):
    """Add meta features"""
    # deberta_oof = joblib.load('/kaggle/input/aes2-400-20240419134941/oof.pkl')
    # print(deberta_oof.shape, train_feats.shape)
    # for i in range(6):
    #     train_feats[f'deberta_oof_{i}'] = deberta_oof[:, i]

    feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
    print('Features Number: ', len(feature_names))  # 31418

    print(train_feats.shape)

    X = train_feats[feature_names].astype(np.float32).values

    y_split = train_feats['score'].astype(int).values
    y = train_feats['score'].astype(np.float32).values - a
    oof = train_feats['score'].astype(int).values

    # Train
    n_splits = 18

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    f1_scores = []
    kappa_scores = []
    models = []
    predictions = []
    callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75, first_metric_only=True)]

    i = 1
    for train_index, test_index in skf.split(X, y_split):
        print('fold', i)
        X_train_fold, X_test_fold = X[train_index], X[test_index]

        y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]

        # LGBMRegressor thực chất là một triển khai của mô hình Gradient Boosting Decision Trees
        # với nhiều cải tiến và tối ưu hóa đặc trưng của LightGBM
        model = lgb.LGBMRegressor(
            # qwk_obj là một hàm mục tiêu tùy chỉnh.
            # các giá trị mặc định cho objective trong hồi quy là 'regression' hoặc 'regression_l2'
            # hàm mục tiêu tùy chỉnh cho phép người dùng tùy biến cách thức mà mô hình tính toán lỗi và
            # cập nhật các tham số của mình
            objective=qwk_obj,
            # 'None' có nghĩa là không có metric đánh giá nào được sử dụng trong quá trình huấn luyện
            # Điều này có thể được điều chỉnh nếu bạn sử dụng một hàm metric tùy chỉnh bên ngoài.
            metrics='None',
            # Giá trị cao hơn làm cho mô hình học nhanh hơn nhưng có thể dẫn đến overfitting.
            learning_rate=0.15,
            # Độ sâu tối đa của mỗi cây trong mô hình. Giới hạn độ sâu để tránh overfitting.
            max_depth=5,
            # Số lượng lá tối đa trên mỗi cây. Số lá càng nhiều thì mô hình càng phức tạp và dễ overfit hơn.
            num_leaves=12,
            # Tỷ lệ các đặc trưng được lấy mẫu ngẫu nhiên cho mỗi cây.
            # Giá trị này là 0.6, tức là 60% đặc trưng sẽ được chọn ngẫu nhiên cho mỗi cây.
            colsample_bytree=0.6,
            # Tham số điều chuẩn L1. Giúp tránh overfitting bằng cách thêm penalty dựa trên giá trị tuyệt đối của trọng số.
            reg_alpha=0.13,
            # Tham số điều chuẩn L2. Giúp tránh overfitting bằng cách thêm penalty dựa trên bình phương của trọng số.
            reg_lambda=0.82,
            # Số lượng cây trong mô hình. Giá trị cao hơn có thể giúp mô hình học tốt hơn
            # nhưng cũng có thể làm tăng thời gian huấn luyện và nguy cơ overfitting.
            n_estimators=1107,
            random_state=42,
            # Khi True, các cây sẽ được xây dựng thêm ngẫu nhiên bằng cách chia nhỏ ngẫu nhiên hơn là chỉ dựa vào mức giảm độ lỗi tốt nhất.
            extra_trees=True,
            # Tự động điều chỉnh trọng số lớp để xử lý dữ liệu mất cân bằng.
            # Tuy nhiên, tham số này thường được sử dụng cho các bài toán phân loại.
            class_weight='balanced',
            verbosity=- 1)

        predictor = model.fit(X_train_fold,
                              y_train_fold,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                              eval_metric=quadratic_weighted_kappa,
                              callbacks=callbacks, )
        models.append(predictor)
        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = predictions_fold + a
        oof[test_index] = predictions_fold
        predictions_fold = predictions_fold.clip(1, 6).round()
        predictions.append(predictions_fold)
        f1_fold = f1_score(y_test_fold_int, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        kappa_fold = cohen_kappa_score(y_test_fold_int, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)

        # cm = confusion_matrix(y_test_fold_int, predictions_fold, labels=[x for x in range(1, 7)])
        #
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                               display_labels=[x for x in range(1, 7)])
        # disp.plot()
        # plt.show()
        print(f'F1 score across fold: {f1_fold}')
        print(f'Cohen kappa score across fold: {kappa_fold}')
        i += 1

    mean_f1_score = np.mean(f1_scores)
    mean_kappa_score = np.mean(kappa_scores)

    print(f'Mean F1 score across {n_splits} folds: {mean_f1_score}')
    print(f'Mean Cohen kappa score across {n_splits} folds: {mean_kappa_score}')
    with open(f'models_lgbm_no_para_tf_id_counter_v{VER}.pkl', 'wb') as f:
        pickle.dump(models, f)

    return models


if __name__ == '__main__':
    train_feats = extract_features()
    models = train(train_feats)
