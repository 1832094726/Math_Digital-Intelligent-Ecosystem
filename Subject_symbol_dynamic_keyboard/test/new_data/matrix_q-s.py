import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from accelerate import Accelerator
from peft import PeftModel

# 确保使用 GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 Accelerator
accelerator = Accelerator()

# 加载模型和分词器
model_name = "../Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# 加载训练好的Lora模型
model = PeftModel.from_pretrained(model, model_id="../output/Qwen2_0.5/checkpoint-17500")
model = accelerator.prepare(model)

# 读取用户和物品数据
user_data = pd.read_csv("train/user_data.csv")  # user_id, question
item_data = pd.read_csv("item_data.csv")  # item_id, symbol


# Function to get embeddings from Qwen model
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)  # 启用隐藏状态的输出
    # 使用[CLS] token的嵌入（可以根据需要修改为其他层的输出）
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    embedding = last_hidden_state[:, 0, :]  # [batch_size, embedding_dim]
    return embedding


# 计算用户问题和物品符号的嵌入
def compute_matrix(user_data, item_data):
    user_embeddings = {}
    item_embeddings = {}

    # 获取每个用户的question嵌入
    for user_id, question in zip(user_data['user_id'], user_data['question']):
        user_embeddings[user_id] = get_embedding(question)

    # 获取每个物品的symbol嵌入
    for item_id, symbol in zip(item_data['item_id'], item_data['symbol']):
        item_embeddings[item_id] = get_embedding(symbol)

    # 构建评分矩阵
    ratings_matrix = []

    # 用于按 user_id 分组评分
    user_ratings = {}

    for i, (user_id, question) in enumerate(zip(user_data['user_id'], user_data['question'])):
        user_embedding = user_embeddings[user_id]
        for j, (item_id, symbol) in enumerate(zip(item_data['item_id'], item_data['symbol'])):
            item_embedding = item_embeddings[item_id]
            # 计算用户嵌入与物品嵌入的内积，作为评分
            rating = torch.matmul(user_embedding, item_embedding.T).item()  # 标量值
            ratings_matrix.append([user_id, item_id, rating])  # 每个用户与物品的评分

    #         # 按用户ID记录评分
    #         if user_id not in user_ratings:
    #             user_ratings[user_id] = []
    #         user_ratings[user_id].append(rating)
    #
    # # 针对每个用户进行 Min-Max 标准化
    # for i in range(len(ratings_matrix)):
    #     user_id, item_id, rating = ratings_matrix[i]
    #
    #     # 获取该用户的评分列表
    #     user_ratings_list = user_ratings[user_id]
    #     min_rating = min(user_ratings_list)
    #     max_rating = max(user_ratings_list)
    #
    #     # Min-Max 标准化，将评分缩放到 [0, 1]
    #     normalized_rating = (rating - min_rating) / (max_rating - min_rating) if max_rating != min_rating else 0.0
    #     ratings_matrix[i] = [user_id, item_id, normalized_rating]  # 更新为标准化后的评分

    return ratings_matrix


# 计算评分矩阵
ratings_matrix = compute_matrix(user_data, item_data)

# 将评分矩阵转换为DataFrame
ratings_df = pd.DataFrame(ratings_matrix, columns=['user_id', 'item_id', 'rating'])

# 输出为CSV文件
ratings_df.to_csv('train/rating_data_emb.csv', index=False)

print("Ratings matrix has been saved to 'rating_data_emb.csv'")
