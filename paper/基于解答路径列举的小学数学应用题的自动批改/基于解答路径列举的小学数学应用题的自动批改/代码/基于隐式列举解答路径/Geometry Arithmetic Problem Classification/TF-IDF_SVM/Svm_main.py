import joblib
import torch
import numpy as np
from transformers import BertModel, BertTokenizer

# 加载 BERT Tokenizer 和预训练模型
tokenizer = BertTokenizer.from_pretrained("chinese-roberta-wwm-ext")
bert_model = BertModel.from_pretrained("chinese-roberta-wwm-ext")

# 加载 SVM 分类器
classifier_layer_1 = joblib.load("svm_classifier_layer_1.joblib")
classifier_layer_2 = joblib.load("svm_classifier_layer_2.joblib")
classifier_layer_3 = joblib.load("svm_classifier_layer_3.joblib")

# 加载 TF-IDF Vectorizers
tfidf_layer_1 = joblib.load("tfidf_vectorizer_layer_1.joblib")
tfidf_layer_2 = joblib.load("tfidf_vectorizer_layer_2.joblib")
tfidf_layer_3 = joblib.load("tfidf_vectorizer_layer_3.joblib")

def bert_vectorize(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def predict(text):
    # 向量化文本
    tfidf_vector_1 = tfidf_layer_1.transform([text]).toarray()
    tfidf_vector_2 = tfidf_layer_2.transform([text]).toarray()
    tfidf_vector_3 = tfidf_layer_3.transform([text]).toarray()

    bert_vector = bert_vectorize([text])

    # 层级预测
    combined_vector_1 = np.hstack((tfidf_vector_1, bert_vector))
    combined_vector_2 = np.hstack((tfidf_vector_2, bert_vector))
    combined_vector_3 = np.hstack((tfidf_vector_3, bert_vector))

    layer_1_pred = classifier_layer_1.predict(combined_vector_1)
    layer_2_pred = classifier_layer_2.predict(combined_vector_2)
    layer_3_pred = classifier_layer_3.predict(combined_vector_3)

    return layer_1_pred[0], layer_2_pred[0], layer_3_pred[0]

if __name__ == '__main__':
    # 假设有一个新的文本需要分类
    new_text = "用篱笆围一块边长分别为6米和3米的平行四边形花圃，需要篱笆多少米？"

    # 预测
    layer_1_prediction, layer_2_prediction, layer_3_prediction = predict(new_text)

    print(f"Layer 1 Prediction: {layer_1_prediction}")
    print(f"Layer 2 Prediction: {layer_2_prediction}")
    print(f"Layer 3 Prediction: {layer_3_prediction}")
