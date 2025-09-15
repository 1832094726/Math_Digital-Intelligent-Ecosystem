# 检查数据集文件是否存在
import os
import json

data_path = "../../Datasets/change_add_train_layer_1_layer_2_layer_3.json"  # 数据集的路径
model_path = '../../chinese-roberta-wwm-ext'
model_path_1 = '../Train Model_Change/tfidf_vectorizer_bert_change.joblib'
# validation_data_path = "layer3_classifier_(0, 7).pth"  # 验证集的路径
if not os.path.exists(model_path):
    print(f"验证数据集文件不存在: {model_path_1}")
else:
    print(f"验证数据集文件存在: {model_path_1}")