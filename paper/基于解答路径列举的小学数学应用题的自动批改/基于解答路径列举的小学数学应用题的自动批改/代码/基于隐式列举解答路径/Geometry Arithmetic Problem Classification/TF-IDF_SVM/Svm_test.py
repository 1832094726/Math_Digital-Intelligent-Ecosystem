import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertModel, BertTokenizer
import joblib


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


def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def prepare_data(data, layer):
    texts = []
    labels = []
    for item in data:
        if layer not in item or item[layer] == '':
            continue
        texts.append(item["original_text"])
        labels.append(int(item[layer]))
    return texts, labels

# 加载测试数据集
test_data_path = "original_add_test_layer_1_layer_2_layer_3.json"
test_data = load_data(test_data_path)

def bert_vectorize(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def predict(texts, classifier, tfidf_vectorizer):
    tfidf_vectors = tfidf_vectorizer.transform(texts).toarray()
    bert_vectors = bert_vectorize(texts)
    combined_vectors = np.hstack((tfidf_vectors, bert_vectors))
    predictions = classifier.predict(combined_vectors)
    return predictions

def evaluate_model(layer, classifier, tfidf_vectorizer, test_data):
    test_texts, test_labels = prepare_data(test_data, layer)
    predictions = predict(test_texts, classifier, tfidf_vectorizer)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    print(f"Layer {layer} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

if __name__ == '__main__':
    evaluate_model("layer_1", classifier_layer_1, tfidf_layer_1, test_data)
    evaluate_model("layer_2", classifier_layer_2, tfidf_layer_2, test_data)
    evaluate_model("layer_3", classifier_layer_3, tfidf_layer_3, test_data)
