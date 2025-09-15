import torch
import joblib
import json
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
# 从Math_23K分出的题目利用分类器进行分类


# 模型和数据集路径
model_path = "../../chinese-roberta-wwm-ext"
data_path = "../../Datasets/geometry_questions.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertClassifier(nn.Module):
    def __init__(self, model_path, labels_num, tfidf_features):
        super(BertClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + tfidf_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, labels_num)

    def forward(self, input_ids, attention_mask, tfidf_features):
        # 通过BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']

        combined_input = torch.cat((pooled_output, tfidf_features), dim=1)
        x = self.fc1(combined_input)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_model_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)

    # 调整状态字典中的键名
    new_state_dict = {}
    for key in state_dict.keys():
        if key == "fc.weight":
            new_state_dict["fc1.weight"] = state_dict[key]
        elif key == "fc.bias":
            new_state_dict["fc1.bias"] = state_dict[key]
        # 其他键保持不变
        else:
            new_state_dict[key] = state_dict[key]

    # 加载调整后的状态字典
    model.load_state_dict(new_state_dict, strict=False)

# 加载数据集
def load_math_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 数据预处理
def preprocess_data(data):
    return data  # 返回完整的数据项，而不仅仅是 original_text


def load_saved_models():
    layer1_classifier = BertClassifier(model_path, labels_num=4, tfidf_features=1000)
    load_model_state_dict(layer1_classifier, "../Train Model_Change/layer1_classifier_if-idf_change.pth")  # Use the custom function
    layer1_classifier.to(device).eval()
    return layer1_classifier

# 新增的函数：加载已保存的 TF-IDF Vectorizer
def load_tfidf_vectorizer():
    # 使用 joblib 加载保存的 TF-IDF Vectorizer
    vectorizer = joblib.load("../Train Model_Change/tfidf_vectorizer_bert_change.joblib")
    return vectorizer


# 新增的函数：将文本转换为 BERT 和 TF-IDF 向量
def text_to_input_tensors(text, tokenizer, vectorizer):
    # BERT 向量化
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    input_ids = inputs['input_ids'].squeeze(1).to(device)
    attention_mask = inputs['attention_mask'].squeeze(1).to(device)

    # TF-IDF 向量化
    tfidf_features = torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32).to(device)

    return input_ids, attention_mask, tfidf_features

def predict(input_text, layer1_classifier, tokenizer, vectorizer):
    input_ids, attention_mask, tfidf_features = text_to_input_tensors(input_text, tokenizer, vectorizer)

    with torch.no_grad():
        outputs_1 = layer1_classifier(input_ids, attention_mask, tfidf_features)
        pred_1 = torch.argmax(outputs_1, dim=1).item()

    return pred_1

def predict_and_save(dataset, classifier, tokenizer, vectorizer, category_labels):
    # 为每个类别创建一个空列表
    category_questions = {label: [] for label in category_labels}

    for item in dataset:
        input_text = item["original_text"]
        pred = predict(input_text, classifier, tokenizer, vectorizer)
        # 将整个数据项添加到相应类别的列表中
        if pred in category_labels:
            category_questions[pred].append(item)

    # 分别保存每个类别的问题
    for label, questions in category_questions.items():
        file_name = f'../../Datasets/geometry_questions_category_{label}.json'
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(model_path)
    vectorizer = load_tfidf_vectorizer()  # 加载已保存的 TF-IDF Vectorizer
    layer1_classifier = load_saved_models()

    # 加载数据集
    math_data = load_math_dataset(data_path)
    processed_data = preprocess_data(math_data)
    # 几何题目的标签列表
    category_labels = [0, 1, 2, 3]  # 包含所有几何题目标签

    # 对整个数据集进行预测并保存符合条件的文本
    predict_and_save(processed_data, layer1_classifier, tokenizer, vectorizer, category_labels)

