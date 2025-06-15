import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from accelerate import Accelerator
from peft import PeftModel
from sklearn.metrics.pairwise import cosine_similarity

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
train_user_data = pd.read_csv("train/user_data.csv")  # user_id, question
test_user_data = pd.read_csv("test/user_data.csv")  # user_id, question

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

# 计算用户之间的相似度矩阵
def compute_user_similarity(user_data):
    user_embeddings = {}

    # 获取每个用户的question嵌入
    for user_id, question in zip(user_data['user_id'], user_data['question']):
        user_embeddings[user_id] = get_embedding(question)

    # 构建用户嵌入的列表
    embedding_list = []
    user_ids = []

    for user_id in user_data['user_id']:
        embedding_list.append(user_embeddings[user_id])
        user_ids.append(user_id)

    # 将用户嵌入转换为Tensor
    user_embeddings_matrix = torch.stack(embedding_list)

    user_embeddings_matrix = user_embeddings_matrix.to(torch.float32).cpu().numpy()

    # 计算用户嵌入之间的余弦相似度
    similarity_matrix = cosine_similarity(user_embeddings_matrix)

    return similarity_matrix, user_ids

def compute_test_user_similarity(train_user_data, test_user_data):
    train_user_embeddings = {}
    test_user_embeddings = {}

    # 获取训练集用户的question嵌入
    for user_id, question in zip(train_user_data['user_id'], train_user_data['question']):
        train_user_embeddings[user_id] = get_embedding(question)

    # 获取测试集用户的question嵌入
    for user_id, question in zip(test_user_data['user_id'], test_user_data['question']):
        test_user_embeddings[user_id] = get_embedding(question)

    # 构建训练集和测试集用户嵌入的列表
    train_embedding_list = []
    test_embedding_list = []
    train_user_ids = []
    test_user_ids = []

    for user_id in train_user_data['user_id']:
        train_embedding_list.append(train_user_embeddings[user_id])
        train_user_ids.append(user_id)

    for user_id in test_user_data['user_id']:
        test_embedding_list.append(test_user_embeddings[user_id])
        test_user_ids.append(user_id)

    # 将嵌入转换为Tensor
    train_embeddings_matrix = torch.stack(train_embedding_list)
    test_embeddings_matrix = torch.stack(test_embedding_list)
    train_embeddings_matrix = train_embeddings_matrix.squeeze(dim=1)
    test_embeddings_matrix = test_embeddings_matrix.squeeze(dim=1)
    print(train_embeddings_matrix.shape)
    print(test_embeddings_matrix.shape)
    train_embeddings_matrix = train_embeddings_matrix.to(torch.float32).cpu().numpy()
    test_embeddings_matrix = test_embeddings_matrix.to(torch.float32).cpu().numpy()

    # 计算测试集用户与训练集用户之间的余弦相似度
    similarity_matrix = cosine_similarity(test_embeddings_matrix, train_embeddings_matrix)
    print(similarity_matrix.shape)
    return similarity_matrix, test_user_ids, train_user_ids

# 计算用户之间的相似度矩阵
similarity_matrix,test_user_ids, train_user_ids= compute_test_user_similarity(train_user_data, test_user_data)

# 将相似度矩阵转换为DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=test_user_ids, columns=train_user_ids)

# 输出为CSV文件
similarity_df.to_csv('test/user_similarity.csv')

print("User similarity matrix has been saved to 'user_similarity.csv'")

