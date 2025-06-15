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
item_data = pd.read_csv("train/item_data.csv")  # user_id, question

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


# 计算物品之间的相似度矩阵
def compute_item_similarity(item_data):
    item_embeddings = {}

    # 获取每个用户的question嵌入
    for item_id, symbol in zip(item_data['item_id'], item_data['symbol']):
        item_embeddings[item_id] = get_embedding(symbol)

    # 构建用户嵌入的列表
    embedding_list = []
    item_ids = []

    for item_id in item_data['item_id']:
        embedding_list.append(item_embeddings[item_id])
        item_ids.append(item_id)

    # 将用户嵌入转换为Tensor
    item_embeddings_matrix = torch.stack(embedding_list)

    # 转移到CPU并转换为float32类型，再转换为NumPy数组
    item_embeddings_matrix = item_embeddings_matrix.to(torch.float32).cpu().numpy()

    # 计算用户嵌入之间的余弦相似度
    similarity_matrix = cosine_similarity(item_embeddings_matrix)

    return similarity_matrix, item_ids

# 计算用户之间的相似度矩阵
similarity_matrix,item_ids = compute_item_similarity(item_data)

# 将相似度矩阵转换为DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=item_ids, columns=item_ids)

# 输出为CSV文件
similarity_df.to_csv('train/item_similarity.csv')

print("Item similarity matrix has been saved to 'item_similarity.csv'")

