import torch
import torch.nn as nn
import json
import joblib


from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support


# 路径配置
model_path = "../../chinese-roberta-wwm-ext"  # 预训练的BERT模型名称
test_data_path = "../../Datasets/change_add_test_layer_1_layer_2_layer_3.json"  # 测试集路径

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
tokenizer = BertTokenizer.from_pretrained(model_path)

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

class CustomDataset(Dataset):
    def __init__(self, data_path, layer_idx, classifier_key=None, parent_label=None):
        self.raw_data = self.load_data(data_path)
        self.layer_idx = layer_idx
        self.classifier_key = classifier_key
        self.parent_label = parent_label
        self.data = self.filter_data()

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def filter_data(self):
        if self.layer_idx == 1:
            return self.raw_data
        elif self.layer_idx == 2:
            return [item for item in self.raw_data if item['layer_1'] == str(self.classifier_key)]
        elif self.layer_idx == 3:
            return [item for item in self.raw_data if
                    item['layer_1'] == str(self.parent_label[0]) and item['layer_2'] == str(self.parent_label[1])]
        else:
            raise ValueError(f"Invalid layer_idx value: {self.layer_idx}. Expected values are 1, 2, or 3.")

    def __getitem__(self, idx):
        # item = self.data[idx]
        text = self.data[idx]["original_text"]
        label = self.get_labels(idx)

        return text, torch.tensor(label).to(device)



    def __len__(self):
        return len(self.data)

    def get_labels(self, index):
        if self.layer_idx == 1:
            return int(self.data[index]["layer_1"])
        elif self.layer_idx == 2:
            return int(self.data[index]["layer_2"])
        elif self.layer_idx == 3:
            return int(self.data[index]["layer_3"])
        else:
            raise ValueError(f"Invalid layer value: {self.layer}. Expected values are 1, 2, or 3.")

# 测试数据集的 DataLoader 创建函数
def create_test_dataloader(data_path, layer_idx, batch_size, key=None, parent_label=None):
    test_data = CustomDataset(data_path, layer_idx=layer_idx, classifier_key=key, parent_label=parent_label)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_dataloader

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for texts, labels in dataloader:
            inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            # 计算TF-IDF特征
            tfidf_features = torch.tensor(tfidf_vectorizer.transform(texts).toarray(), dtype=torch.float32).to(device)

            # 将BERT和TF-IDF特征结合进行预测
            outputs = model(input_ids, attention_mask, tfidf_features)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    # precision = precision_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1


def test_model_per_layer(model, layer_idx, batch_size, key=None, parent_label=None):
    test_dataloader = create_test_dataloader(test_data_path, layer_idx, batch_size, key, parent_label)

    # 根据层级和关键字构建模型文件名
    if layer_idx == 2:
        model_file_name = f"../Train Model_Change/layer2_classifier_{key}_if-idf_change.pth"
    elif layer_idx == 3:
        model_file_name = f"../Train Model_Change/layer3_classifier_{parent_label[0]}_{parent_label[1]}_if-idf_change.pth"
    else:
        model_file_name = f"../Train Model_Change/layer{layer_idx}_classifier_if-idf_change.pth"

    # 加载模型状态
    model.load_state_dict(torch.load(model_file_name))

    # 评估模型
    accuracy, precision, recall, f1 = evaluate_model(model, test_dataloader)
    print(f"Layer {layer_idx}, Test Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")



if __name__ == '__main__':

    # # 加载全部文本数据用于训练TF-IDF Vectorizer
    # with open(test_data_path, "r", encoding="utf-8") as file:
    #     all_train_data = json.load(file)
    # # 以训练集文本初始化和训练TF-IDF向量器
    # all_train_texts = [item["original_text"] for item in all_train_data]
    # tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    # tfidf_vectorizer.fit(all_train_texts)

    # 之前已经训练好了，直接加载就可以了，不用上面的代码了
    tfidf_vectorizer = joblib.load("../Train Model_Change/tfidf_vectorizer_bert_change.joblib")

    # 第一层
    layer1_classifier = BertClassifier(model_path, labels_num=4, tfidf_features=1000).to(device)

    # 第二层
    layer2_classifiers = {
        0: BertClassifier(model_path, labels_num=12, tfidf_features=1000).to(device),
        1: BertClassifier(model_path, labels_num=5, tfidf_features=1000).to(device),
        2: BertClassifier(model_path, labels_num=5, tfidf_features=1000).to(device),
        3: BertClassifier(model_path, labels_num=8, tfidf_features=1000).to(device),
    }

    # 第三层
    layer3_classifiers = {
        (0, 7): BertClassifier(model_path, labels_num=2, tfidf_features=1000).to(device),
        (0, 8): BertClassifier(model_path, labels_num=2, tfidf_features=1000).to(device),
        (0, 9): BertClassifier(model_path, labels_num=2, tfidf_features=1000).to(device),
        (0, 10): BertClassifier(model_path, labels_num=3, tfidf_features=1000).to(device),
        (2, 2): BertClassifier(model_path, labels_num=2, tfidf_features=1000).to(device),
        (3, 4): BertClassifier(model_path, labels_num=4, tfidf_features=1000).to(device),
        (3, 6): BertClassifier(model_path, labels_num=3, tfidf_features=1000).to(device),
    }


    # 进行测试
    test_model_per_layer(layer1_classifier, 1, batch_size=16)
    for key in layer2_classifiers:
        classifier = layer2_classifiers[key]
        test_model_per_layer(classifier, 2, batch_size=8, key=key)
    for (layer_1_label, layer_2_label) in layer3_classifiers:
        classifier = layer3_classifiers[(layer_1_label, layer_2_label)]
        parent_label = (layer_1_label, layer_2_label)
        test_model_per_layer(classifier, 3, batch_size=2, parent_label=parent_label)