import torch
import json
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import joblib

# 模型定义：BertClassifier 类的定义。
# 数据加载和预处理：加载 Math_23k 数据集，并进行必要的预处理。
# 模型加载：加载已训练的模型。
# 预测和提取：使用模型进行预测，并提取几何题目。
# 保存结果：将提取出的几何题目保存到一个新的数据集中。

# 预训练的BERT模型的名称
model_path = "../../chinese-roberta-wwm-ext"
# 定义路径和配置
data_path = "../../Datasets/Math_23K.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型定义
class BertClassifier(nn.Module):
    def __init__(self, model_path, labels_num, tfidf_features):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + tfidf_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, labels_num)

    def forward(self, input_ids, attention_mask, tfidf_features):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        combined_input = torch.cat((pooled_output, tfidf_features), dim=1)
        x = self.fc1(combined_input)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 加载数据集
def load_math_23k_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 数据预处理
def preprocess_data(data):
    return [item["original_text"] for item in data]

# 加载模型
def load_model(model_path, BertClassifier, labels_num, tfidf_features):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertClassifier(model_path, labels_num, tfidf_features)
    model.load_state_dict(torch.load(f"../Train Model_Change/layer1_classifier_if-idf_change.pth"))
    model.to(device)
    model.eval()
    return model, tokenizer

# 预测数据
def predict_data(model, tokenizer, tfidf_vectorizer, data, max_length):
    predictions = []
    for text in data:
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        tfidf_features = torch.tensor(tfidf_vectorizer.transform([text]).toarray(), dtype=torch.float32).to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, tfidf_features)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
    return predictions

# 主程序
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    labels_num = 4  # 根据您的模型实际情况调整
    tfidf_features = 1000  # 您之前定义的TF-IDF特征数量
    max_length = 256  # BERT处理的最大文本长度
    tfidf_vectorizer = joblib.load(f"../Train Model_Change/tfidf_vectorizer_bert_change.joblib")

    # 加载数据集
    math_23k_data = load_math_23k_dataset(data_path)
    processed_data = preprocess_data(math_23k_data)

    # 加载模型
    model, tokenizer = load_model(model_path, BertClassifier, labels_num, tfidf_features)

    # 预测
    predictions = predict_data(model, tokenizer, tfidf_vectorizer, processed_data, max_length)


    # 提取几何题目
    geometry_questions = [item for pred, item in zip(predictions, math_23k_data) if pred in [0, 1, 2, 3]]

    with open('../../Datasets/geometry_questions.json', 'w', encoding='utf-8') as f:
        json.dump(geometry_questions, f, ensure_ascii=False, indent=4)