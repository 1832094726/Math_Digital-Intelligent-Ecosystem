# 检查数据集文件是否存在
import os
import json

data_path = "original_add_train_layer_1_layer_2_layer_3.json"  # 数据集的路径
validation_data_path = "layer3_classifier_(0, 7).pth"  # 验证集的路径
if not os.path.exists(validation_data_path):
    print(f"验证数据集文件不存在: {validation_data_path}")
else:
    print(f"验证数据集文件存在: {validation_data_path}")

# # 检查文件内容
# with open(validation_data_path, "r", encoding="utf-8") as file:
#     data = json.load(file)
#     print(f"验证数据集中的样本数量: {len(data)}")
