import pickle
import pandas as pd
import numpy as np
from Competition_1.code.Extract_Features.A_Paragraph_Features import Paragraph_Preprocess, Paragraph_Eng
from Competition_1.code.Extract_Features.B_Sentence_Features import Sentence_Preprocess, Sentence_Eng
from Competition_1.code.Extract_Features.C_Word_Features import Word_Preprocess, Word_Eng
# from Competition_1.code.Extract_Features.D_CountVectorizer_Features import vectorizer_cnt
# from Competition_1.code.Extract_Features.E_Tf_idf_features import vectorizer
from Competition_1.code.Extract_Features.main_feature_engineering import test

a = 2.998
b = 1.092
VER = 1

path_vectorizer_models = "/home/oryza/Desktop/KK/Competition_1/models/vectorizer_models/"

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


def infer(models, vectorizer_cnt):
    # Paragraph
    # tmp = Paragraph_Preprocess(test)
    # test_feats = Paragraph_Eng(tmp)

    # Sentence
    tmp = Sentence_Preprocess(test)
    test_feats = Sentence_Eng(tmp)
    # test_feats = test_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')
    # Word
    tmp = Word_Preprocess(test)
    test_feats = test_feats.merge(Word_Eng(tmp), on='essay_id', how='left')

    # Tfidf
    # test_tfid = vectorizer.fit_transform([i for i in test['full_text']])
    # dense_matrix = test_tfid.toarray()
    # df = pd.DataFrame(dense_matrix)
    # tfid_columns = [f'tfid_{i}' for i in range(len(df.columns))]
    # df.columns = tfid_columns
    # df['essay_id'] = test_feats['essay_id']
    # test_feats = test_feats.merge(df, on='essay_id', how='left')

    # CountVectorizer
    test_tfid = vectorizer_cnt.transform([i for i in test['full_text']])
    dense_matrix = test_tfid.toarray()
    df = pd.DataFrame(dense_matrix)
    tfid_columns = [f'tfid_cnt_{i}' for i in range(len(df.columns))]
    df.columns = tfid_columns
    df['essay_id'] = test_feats['essay_id']
    test_feats = test_feats.merge(df, on='essay_id', how='left')

    # for i in range(6):
    #     test_feats[f'deberta_oof_{i}'] = predicted_score[:, i]

    # Features number
    feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], test_feats.columns))
    print('Features number: ', len(feature_names))
    test_feats.head(3)

    probabilities = []
    for model in models:
        proba = model.predict(test_feats[feature_names]) + a
        probabilities.append(proba)

    predictions = np.mean(probabilities, axis=0)

    predictions = np.round(predictions.clip(1, 6))

    print(predictions)


if __name__ == '__main__':
    with open(f'models_lgbm_no_para_tf_id_counter_v{VER}.pkl', 'rb') as f:
        models = pickle.load(f)
    with open(f'{path_vectorizer_models}vectorizer_cnt.pkl', 'rb') as f:
        vectorizer_cnt = pickle.load(f)
    infer(models, vectorizer_cnt)