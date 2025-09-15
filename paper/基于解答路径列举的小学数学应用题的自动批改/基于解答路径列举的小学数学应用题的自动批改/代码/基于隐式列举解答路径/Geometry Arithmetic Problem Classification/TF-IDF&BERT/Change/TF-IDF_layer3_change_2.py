import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import matplotlib.pyplot as plt
import numpy as np

# 预训练的BERT模型的名称
model_path = "../../chinese-roberta-wwm-ext"
# 数据集的路径
data_path = "../../Datasets/change_2_add_train_layer_1_layer_2_layer_3.json"
validation_data_path = "../../Datasets/change_2_add_val_layer_1_layer_2_layer_3.json"

# 训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# 评估模型
def evaluate_model(model, dataloader, tokenizer, tfidf_vectorizer, max_length):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
            tfidf_features = torch.tensor(tfidf_vectorizer.transform(texts).toarray(), dtype=torch.float32).to(device)

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, tfidf_features)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    return correct_predictions / total_predictions


# 训练模型
def train_model_per_layer(model, layer_idx, criterion, optimizer, batch_size, num_epochs, max_length, key=None, parent_label=None):
    if layer_idx == 1:
        train_data = CustomDataset(data_path, layer_idx=layer_idx)
        val_data = CustomDataset(validation_data_path, layer_idx=layer_idx)
    elif layer_idx == 2:
        train_data = CustomDataset(data_path, layer_idx=layer_idx, classifier_key=key)
        val_data = CustomDataset(validation_data_path, layer_idx=layer_idx, classifier_key=key)
    elif layer_idx == 3:
        train_data = CustomDataset(data_path, layer_idx=layer_idx, parent_label=parent_label)
        val_data = CustomDataset(validation_data_path, layer_idx=layer_idx, parent_label=parent_label)
    else:
        raise ValueError("Invalid layer index. Expected values are 1, 2, or 3.")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_dataloader:
            # 批量处理文本
            inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
            tfidf_features = torch.tensor(tfidf_vectorizer.transform(texts).toarray(), dtype=torch.float32).to(device)

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            # print(
            #     f"input_ids device: {input_ids.device}, attention_mask device: {attention_mask.device}, tfidf_features device: {tfidf_features.device}, labels device: {labels.device}")

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, tfidf_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 在每次迭代后清除不再需要的变量并释放内存
            del inputs, input_ids, attention_mask, tfidf_features, labels, outputs, loss
            torch.cuda.empty_cache()

        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # 评估训练和验证的准确率
        train_accuracy = evaluate_model(model, train_dataloader, tokenizer, tfidf_vectorizer, max_length)
        val_accuracy = evaluate_model(model, val_dataloader, tokenizer, tfidf_vectorizer, max_length)

        # 添加准确率到列表
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # 计算验证损失
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for texts, labels in val_dataloader:
                inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True,
                                   max_length=max_length)
                tfidf_features = torch.tensor(tfidf_vectorizer.transform(texts).toarray(), dtype=torch.float32).to(
                    device)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                outputs = model(input_ids, attention_mask, tfidf_features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Layer {layer_idx}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}")

    # 绘制当前累积的损失和准确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epoch + 2), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy Curve')

    plt.tight_layout()
    plt.show()

    # return train_losses, val_losses, train_accuracies, val_accuracies
# 主函数
if __name__ == '__main__':


    # # 加载全部文本数据用于训练TF-IDF Vectorizer
    # with open(data_path, "r", encoding="utf-8") as file:
    #     all_train_data = json.load(file)
    # # 训练并保存 TF-IDF Vectorizer
    # tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    # all_texts = [item["original_text"] for item in all_train_data]
    # tfidf_vectorizer.fit(all_texts)
    # joblib.dump(tfidf_vectorizer, "../Train Model_Change/tfidf_vectorizer_bert_change_2.joblib")

    # 只用训练一个整体的进行保存就行了，训练第一层之后就可以将上面训练的代码注释掉
    tfidf_vectorizer = joblib.load("../Train Model_Change/tfidf_vectorizer_bert_change_2.joblib")

    criterion = nn.CrossEntropyLoss()

    # # 第一层
    # layer1_classifier = BertClassifier(model_path, labels_num=4, tfidf_features=1000).to(device)
    # optimizer = optim.Adam(list(layer1_classifier.fc1.parameters()) + list(layer1_classifier.fc2.parameters()))
    #
    # train_model_per_layer(layer1_classifier, 1, criterion, optimizer, batch_size=16, num_epochs=5, max_length=256)
    # torch.save(layer1_classifier.state_dict(), "../Train Model_Change/layer1_classifier_if-idf_change_2.pth")

    # # 第二层 change_2
    # layer2_classifiers = {
    #     0: BertClassifier(model_path, labels_num=12, tfidf_features=1000),
    #     1: BertClassifier(model_path, labels_num=5, tfidf_features=1000),
    #     2: BertClassifier(model_path, labels_num=3, tfidf_features=1000),
    #     3: BertClassifier(model_path, labels_num=8, tfidf_features=1000),
    # }
    #
    # # 分别训练每一个第二层的分类器
    # for key, classifier in layer2_classifiers.items():
    #     classifier.to(device)  # 确保模型在GPU上
    #     # 为每个分类器设置独立的优化器
    #     optimizer = optim.Adam(list(classifier.fc1.parameters()) + list(classifier.fc2.parameters()))
    #     # 训练模型
    #     train_model_per_layer(classifier, 2, criterion, optimizer, batch_size=4, num_epochs=5, max_length=256, key=key)
    #     # 保存模型状态
    #     # change_2
    #     torch.save(classifier.state_dict(), f"../Train Model_Change/layer2_classifier_{key}_if-idf_change_2.pth")
    #     # 清理内存
    #     del classifier
    #     torch.cuda.empty_cache()

    # 第三层
    layer3_classifiers = {
        (0, 7): BertClassifier(model_path, labels_num=2, tfidf_features=1000),
        (0, 8): BertClassifier(model_path, labels_num=2, tfidf_features=1000),
        (0, 9): BertClassifier(model_path, labels_num=2, tfidf_features=1000),
        (0, 10): BertClassifier(model_path, labels_num=3, tfidf_features=1000),
        (2, 0): BertClassifier(model_path, labels_num=4, tfidf_features=1000),
        (3, 4): BertClassifier(model_path, labels_num=4, tfidf_features=1000),
        (3, 6): BertClassifier(model_path, labels_num=3, tfidf_features=1000),
    }
    for (layer_1_label, layer_2_label), classifier in layer3_classifiers.items():
        classifier.to(device)  # 确保模型在GPU上
        optimizer = optim.Adam(list(classifier.fc1.parameters()) + list(classifier.fc2.parameters()))
        parent_label = (layer_1_label, layer_2_label)
        # 在训练函数中加入验证数据加载器
        train_model_per_layer(classifier, 3, criterion, optimizer, batch_size=4, num_epochs=5,
                                max_length=256, parent_label=parent_label)
        # 保存模型状态
        torch.save(classifier.state_dict(), f"../Train Model_Change/layer3_classifier_{layer_1_label}_{layer_2_label}_if-idf_change_2.pth")
        # 清理内存
        del classifier
        torch.cuda.empty_cache()
