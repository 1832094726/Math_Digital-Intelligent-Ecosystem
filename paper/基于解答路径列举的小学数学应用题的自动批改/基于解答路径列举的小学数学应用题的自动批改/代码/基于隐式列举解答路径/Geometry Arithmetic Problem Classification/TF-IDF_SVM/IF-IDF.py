import json
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer

# 数据路径
data_path = "original_add_train_layer_1_layer_2_layer_3.json"
validation_data_path = "original_add_val_layer_1_layer_2_layer_3.json"

# 加载数据
def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def prepare_data(data, layer):
    texts = []
    labels = []
    for item in data:
        # 检查第三层标签是否存在
        if layer not in item or item[layer] == '':
            continue

        # 将文本和标签添加到相应的列表
        texts.append(item["original_text"])
        labels.append(int(item[layer]))

    return texts, labels

train_data = load_data(data_path)
val_data = load_data(validation_data_path)

# BERT 模型和 Tokenizer
model_path = "chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_path)
bert = BertModel.from_pretrained(model_path)

# BERT 向量化函数
def bert_vectorize(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

# 创建和训练 SVM 分类器
def train_svm_classifier(train_texts, train_labels, val_texts, val_labels):
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_val_tfidf = tfidf.transform(val_texts)

    # 保存 TF-IDF Vectorizer
    joblib.dump(tfidf, f"tfidf_vectorizer_{layer}.joblib")

    # BERT 向量化
    X_train_bert = bert_vectorize(train_texts)
    X_val_bert = bert_vectorize(val_texts)

    # 结合 TF-IDF 和 BERT 向量
    X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_bert))
    X_val_combined = np.hstack((X_val_tfidf.toarray(), X_val_bert))

    # SVM 分类器
    svm_classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm_classifier.fit(X_train_combined, train_labels)

    # 验证
    val_accuracy = svm_classifier.score(X_val_combined, val_labels)
    return svm_classifier, val_accuracy

# 对每个层级训练和验证分类器，并保存
for layer in ["layer_1", "layer_2", "layer_3"]:
    train_texts, train_labels = prepare_data(train_data, layer)
    val_texts, val_labels = prepare_data(val_data, layer)
    classifier, accuracy = train_svm_classifier(train_texts, train_labels, val_texts, val_labels)
    print(f"Validation Accuracy for {layer}: {accuracy}")

    # 保存模型
    joblib.dump(classifier, f"svm_classifier_{layer}.joblib")
