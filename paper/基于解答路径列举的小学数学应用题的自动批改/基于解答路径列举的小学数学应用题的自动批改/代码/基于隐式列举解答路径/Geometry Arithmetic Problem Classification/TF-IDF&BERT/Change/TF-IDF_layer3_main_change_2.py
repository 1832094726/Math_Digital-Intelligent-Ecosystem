import torch
import joblib
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn

model_path = "../../chinese-roberta-wwm-ext"
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

def load_saved_models():
    layer1_classifier = BertClassifier(model_path, labels_num=4, tfidf_features=1000)
    load_model_state_dict(layer1_classifier, "../Train Model_Change/layer1_classifier_if-idf_change_2.pth")  # Use the custom function
    layer1_classifier.to(device).eval()

    layer2_keys_to_labels = {
        0: BertClassifier(model_path, labels_num=12, tfidf_features=1000),
        1: BertClassifier(model_path, labels_num=5, tfidf_features=1000),
        2: BertClassifier(model_path, labels_num=3, tfidf_features=1000),
        3: BertClassifier(model_path, labels_num=8, tfidf_features=1000),
    }

    layer2_classifiers_loaded = {}
    for i, classifier in layer2_keys_to_labels.items():
        load_model_state_dict(classifier, f"../Train Model_Change/layer2_classifier_{i}_if-idf_change_2.pth")
        classifier.to(device).eval()
        layer2_classifiers_loaded[i] = classifier

    layer3_keys_to_labels = {
        (0, 7): BertClassifier(model_path, labels_num=2, tfidf_features=1000),
        (0, 8): BertClassifier(model_path, labels_num=2, tfidf_features=1000),
        (0, 9): BertClassifier(model_path, labels_num=2, tfidf_features=1000),
        (0, 10): BertClassifier(model_path, labels_num=3, tfidf_features=1000),
        (2, 0): BertClassifier(model_path, labels_num=4, tfidf_features=1000),
        (3, 4): BertClassifier(model_path, labels_num=4, tfidf_features=1000),
        (3, 6): BertClassifier(model_path, labels_num=3, tfidf_features=1000),
    }


    layer3_classifiers_loaded = {}
    for key, classifier in layer3_keys_to_labels.items():
        # 构造文件名，例如: "layer3_classifier_0_10.pth"
        filename = f"../Train Model_Change/layer3_classifier_{key[0]}_{key[1]}_if-idf_change_2.pth"
        load_model_state_dict(classifier, filename)
        classifier.to(device).eval()
        layer3_classifiers_loaded[key] = classifier
    #
    return layer1_classifier, layer2_classifiers_loaded, layer3_classifiers_loaded
    # return layer1_classifier

# 新增的函数：加载已保存的 TF-IDF Vectorizer
def load_tfidf_vectorizer():
    # 使用 joblib 加载保存的 TF-IDF Vectorizer
    vectorizer = joblib.load("../Train Model_Change/tfidf_vectorizer_bert_change_2.joblib")
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

def predict(input_text, layer1_classifier, layer2_classifiers, layer3_classifiers, tokenizer, vectorizer):
    input_ids, attention_mask, tfidf_features = text_to_input_tensors(input_text, tokenizer, vectorizer)

    with torch.no_grad():
        outputs_1 = layer1_classifier(input_ids, attention_mask, tfidf_features)
        pred_1 = torch.argmax(outputs_1, dim=1).item()

        outputs_2 = layer2_classifiers[pred_1](input_ids, attention_mask, tfidf_features)
        pred_2 = torch.argmax(outputs_2, dim=1).item()

        if (pred_1, pred_2) in layer3_classifiers:
            outputs_3 = layer3_classifiers[(pred_1, pred_2)](input_ids, attention_mask, tfidf_features)
            pred_3 = torch.argmax(outputs_3, dim=1).item()
        else:
            pred_3 = None

    return pred_1, pred_2, pred_3

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(model_path)
    vectorizer = load_tfidf_vectorizer()  # 加载已保存的 TF-IDF Vectorizer
    layer1_classifier, layer2_classifiers, layer3_classifiers = load_saved_models()


    input_text = "将一张长6分米，宽4分米的长方形纸片卷成一个圆柱，这个圆柱的侧面积=多少平方分米。"

    predictions = predict(input_text, layer1_classifier, layer2_classifiers, layer3_classifiers, tokenizer, vectorizer)
    print(predictions)
