import torch
import torch.nn as nn
import json
import joblib

from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 路径配置
model_path = "../../chinese-roberta-wwm-ext"
test_data_path = "../../Datasets/change_2_add_test_layer_1_layer_2_layer_3.json"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class CustomDataset(Dataset):
    def __init__(self, data_path, layer_idx):
        self.raw_data = self.load_data(data_path)
        self.layer_idx = layer_idx
        self.data = self.filter_data()

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def filter_data(self):
        if self.layer_idx == 1:
            return self.raw_data
        else:
            raise ValueError(f"Invalid layer_idx value: {self.layer_idx}. Expected value is 1.")

    def __getitem__(self, idx):
        text = self.data[idx]["original_text"]
        label = self.get_labels(idx)
        return text, torch.tensor(label).to(device)

    def __len__(self):
        return len(self.data)

    def get_labels(self, index):
        return int(self.data[index]["layer_1"])

def create_test_dataloader(data_path, layer_idx, batch_size):
    test_data = CustomDataset(data_path, layer_idx)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_dataloader

def evaluate_model(model, dataloader, tokenizer, tfidf_vectorizer):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for texts, labels in dataloader:
            inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            tfidf_features = torch.tensor(tfidf_vectorizer.transform(texts).toarray(), dtype=torch.float32).to(device)
            outputs = model(input_ids, attention_mask, tfidf_features)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2, 3], zero_division=1)

    return accuracy, precision, recall, f1

def test_model_per_layer(model, layer_idx, batch_size):
    test_dataloader = create_test_dataloader(test_data_path, layer_idx, batch_size)
    model_file_name = f"../Train Model_Change/layer1_classifier_if-idf_change_2.pth"
    model.load_state_dict(torch.load(model_file_name))
    accuracy, precision, recall, f1 = evaluate_model(model, test_dataloader, tokenizer, tfidf_vectorizer)

    print(f"Layer {layer_idx}, Test Accuracy: {accuracy:.2f}")
    for idx in range(4):
        print(f"Class {idx}: Precision: {precision[idx]:.2f}, Recall: {recall[idx]:.2f}, F1: {f1[idx]:.2f}")

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # with open(test_data_path, "r", encoding="utf-8") as file:
    #     all_train_data = json.load(file)
    #
    # all_train_texts = [item["original_text"] for item in all_train_data]
    # tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    # tfidf_vectorizer.fit(all_train_texts)

    # 已训练好，直接加载
    tfidf_vectorizer = joblib.load("../Train Model_Change/tfidf_vectorizer_bert_change_2.joblib")

    layer1_classifier = BertClassifier(model_path, labels_num=4, tfidf_features=1000).to(device)
    # layer1_model_file = "layer1_classifier_if-idf.pth"

    test_model_per_layer(layer1_classifier, 1, batch_size=16)
