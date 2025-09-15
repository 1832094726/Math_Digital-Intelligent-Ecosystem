import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from transformers import BertTokenizer, BertModel
import logging
from matplotlib import pyplot as plt
from calculate_metrics import evaluate_symbol_predictions, evaluate_per_symbol_predictions
import datetime
import os
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer, AutoConfig
from adapters import AutoAdapterModel, BertAdapterModel, PredictionHead

nowTime = datetime.datetime.now().strftime('%m-%d-%H-%M')

#################### 以下为日志记录 ####################
# 全参Bert
model_name = 'BERT-ncf-mlp-final(' + nowTime + ')'

folder_path = f"record/{model_name}"
os.makedirs(folder_path, exist_ok=True)
# 设置日志输出格式和日志文件路径
log_file = os.path.join(folder_path, f"{model_name}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
loss_figure = os.path.join(folder_path, f"{model_name}_loss.png")
acc_figure = os.path.join(folder_path, f"{model_name}_acc.png")
prediction_file_path = f'new_data/data_predict_record/{model_name}.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################### 以下为采样函数 #####################
def negative_sampling(user_item_matrix, other1_ratings_matrix, other2_ratings_matrix, user_dict, item_dict, question_similarity_matrix, neg_ratio=1):
    """
    对用户-物品矩阵进行负采样，返回值包含 user_dict 和 item_dict 中的值。

    参数：
    - user_item_matrix: 用户-物品矩阵 (PyTorch Tensor)，形状为 [num_users, num_items]。
    - user_dict: 用户 ID 到用户值的映射字典。
    - item_dict: 物品 ID 到物品值的映射字典。
    - similarity_matrix: 用户或物品的相似度矩阵，形状应为 [num_users, ...] 或 [num_items, ...]。
    - neg_ratio: 每个正例生成的负例数量。

    返回：
    - List of tuples: [(user_value, item_value, label), ...]
    - Adjusted question_similarity_matrix: NumPy array
    - Adjusted item_similarity_matrix: NumPy array
    """
    num_users, num_items = user_item_matrix.shape
    positive_samples = []
    negative_samples = []

    for user_id in range(num_users):
        # 获取正例的物品索引  标记为1的问题
        positive_items = torch.where(user_item_matrix[user_id] == 1)[0]
        # 获取负例的物品索引  标记为0的问题
        negative_items = torch.where(user_item_matrix[user_id] == 0)[0]

        # 不管其是正样本 还是负样本 得到其相应的相似度向量
        # 提取当前用户的相似度向量
        question_vector = question_similarity_matrix[user_id]  # 获取用户对应的相似度行

        # 添加正例
        positive_samples.extend(
            [(user_dict[user_id], item_dict[item_id.item()], 1,
              other1_ratings_matrix[user_id, item_id.item()].item(),
              other2_ratings_matrix[user_id, item_id.item()].item(),
              question_vector) for item_id in positive_items]
        )

        # 采样负例
        num_negatives = len(positive_items) * neg_ratio
        sampled_negatives_indices = torch.randint(
            low=0, high=len(negative_items), size=(num_negatives,), generator=None
        )
        negative_samples.extend(
            [(user_dict[user_id], item_dict[negative_items[idx].item()], 0,
              other1_ratings_matrix[user_id, negative_items[idx].item()].item(),
              other2_ratings_matrix[user_id, negative_items[idx].item()].item(),
              question_vector) for idx in sampled_negatives_indices]
        )
    # breakpoint()
    # 合并正例和负例    22070 + 22070
    all_samples = positive_samples + negative_samples

    np.random.shuffle(all_samples)

    return all_samples


def full_sampling(user_item_matrix, user_dict, item_dict, question_similarity_matrix):
    """
    对用户-物品矩阵进行全采样，并将 user_id 和 item_id 替换为字典中的值。

    参数：
    - user_item_matrix: 用户-物品矩阵 (PyTorch Tensor)，形状为 [num_users, num_items]。
    - user_dict: 用户 ID 到用户值的映射字典。
    - item_dict: 物品 ID 到物品值的映射字典。
    - question_similarity_matrix: 原始问题相似度矩阵。
    - item_similarity_matrix: 原始项目相似度矩阵。
    返回：
    - List of tuples: [(user_value, item_value, label), ...]
    - Reordered question similarity matrix
    - Reordered item similarity matrix
    """
    num_users, num_items = user_item_matrix.shape
    all_samples = []
    id_samples = []
    for user_id in range(num_users):
        for item_id in range(num_items):
            label = user_item_matrix[user_id, item_id].item()
            question_vector = question_similarity_matrix[user_id]  # 获取用户对应的相似度行
            user_value = user_dict[user_id]
            item_value = item_dict[item_id]
            all_samples.append((user_value, item_value, label, question_vector))
            id_samples.append((user_id, item_id))

    return all_samples, id_samples


def process_matrix_data(rating_file, user_file, item_file):
    rating_df = pd.read_csv(rating_file)
    user_df = pd.read_csv(user_file)
    item_df = pd.read_csv(item_file)

    # 用户 ID 映射到对应的“问题”（question）的字典
    user_dict = user_df.set_index('user_id')['question'].to_dict()
    item_dict = item_df.set_index('item_id')['symbol'].to_dict()

    num_users = user_df['user_id'].nunique()
    num_items = item_df['item_id'].nunique()

    ratings_matrix = torch.zeros((num_users, num_items))

    for i, row in rating_df.iterrows():
        user_idx = row['user_id']
        item_idx = row['item_id']
        rating = row['rating']
        ratings_matrix[int(user_idx), int(item_idx)] = rating

    return ratings_matrix, user_dict, item_dict, user_df['question'], item_df['symbol']

def matrix_factorization(ratings_matrix, num_latent_factors=30):
    # 使用SVD进行矩阵分解
    svd = TruncatedSVD(n_components=num_latent_factors, random_state=42)
    U = svd.fit_transform(ratings_matrix)  # 用户特征矩阵
    S = svd.singular_values_  # 奇异值
    V = svd.components_  # 物品特征矩阵

    return U, S, V

def reconstruct_matrix(U, S, V):
    # 重构评分矩阵
    S_matrix = np.diag(S)
    reconstructed_matrix = np.dot(np.dot(U, S_matrix), V)
    return reconstructed_matrix

def sigmoid_transform(data, scale=5):
    return 1 / (1 + torch.exp(-scale * (data - 0.5)))

##################### 以下为计算评估和画图函数 #####################
def calculate_recommendation_cover(predictions, test_data):
    """
    计算每个用户的推荐列表中，覆盖所有真实评分为1的项所需的推荐个数，并返回这些个数的平均值。
    """
    test_user, test_item, test_ratings = zip(*test_data)  # 解包测试数据
    test_ratings = torch.tensor(test_ratings).float()  # 转换为tensor

    total_k = 0  # 用于累加所有用户的k值
    total_users = len(set(test_user))  # 用户的总数
    total_items = len(set(test_item))  # 物品的总数

    predictions.view(total_users, total_items)
    test_ratings.view(total_users, total_items)

    for i in range(total_users):
        user_predictions = predictions[i]
        user_test_ratings = test_ratings[i]

        # 根据预测评分排序并获得排序后的真实评分
        sorted_indices = torch.argsort(user_predictions, descending=True)
        sorted_test_ratings = user_test_ratings[sorted_indices]

        # 找到最后一个1的位置，表示需要的推荐个数
        relevant_indices = (sorted_test_ratings == 1).nonzero(as_tuple=True)
        if relevant_indices[0].numel() > 0:
            last_relevant_index = relevant_indices[0][-1].item()  # 获取最后一个1的索引
            k = last_relevant_index + 1
            total_k += k
        else:
            total_k += 0  # 如果没有相关项，假设不需要推荐

    average_k = total_k / total_users
    return average_k


def calculate_top_k_accuracy(predictions, true_user, true_item, true_ratings, k=10):
    """
    计算基于排名前k的推荐符号的准确率。

    参数:
    - predictions: 模型的预测结果 (tensor)
    - true_ratings: 真实评分 (tensor)
    - k: 排名前k的推荐

    返回:
    - top_k_accuracy: top-k准确率
    """
    # 获取每个用户的前k个推荐物品的索引
    _, top_k_indices = torch.topk(predictions, k, dim=1)
    true_ratings = true_ratings.view(len(set(true_user)), len(set(true_item)))

    correct = 0
    total = 0

    for i in range(len(set(true_user))):
        # 获取当前用户的真实评分为1的物品索引
        true_positive_indices = (true_ratings[i] == 1).nonzero(as_tuple=True)[0]

        # 获取当前用户的前k推荐物品的索引
        top_k_items = top_k_indices[i]

        # 计算真实评分为1的物品是否出现在前k推荐物品中
        correct += (top_k_items.unsqueeze(0) == true_positive_indices.unsqueeze(1)).sum().item()
        total += true_positive_indices.numel()  # 当前用户真实评分为1的物品数

    return correct / total if total > 0 else 0


def plot_metrics(train_losses, train_accuracies):
    epochs = np.arange(1, len(train_losses) + 1)

    # 绘制 Loss 图
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_figure)  # 保存训练集的loss图像
    plt.show()

    # 绘制 Accuracy 图
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_figure)  # 保存训练集的accuracy图像
    plt.show()


def convert_numpy_float32(obj):
    """
    定义一个递归函数来转换所有 numpy.float32 为 Python 的 float 类型。
    """
    if isinstance(obj, dict):
        # 如果是字典，则递归处理每个键值对
        return {k: convert_numpy_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # 如果是列表，则递归处理每个元素
        return [convert_numpy_float32(item) for item in obj]
    elif isinstance(obj, np.float32):
        # 如果是 numpy.float32 类型，则转换为 Python 的 float
        return float(obj)
    else:
        # 其他类型不做处理
        return obj


def evaluate_predictions(data, model_name):
    """
    根据数据评估模型的预测性能。
    """
    for item in data:
        if len(item['answer_symbols']) == 0:
            # 删除这个item
            data.remove(item)

    result = evaluate_symbol_predictions(data)

    logging.info(f"{model_name} Prediction Data Metrics:")
    logging.info(f"Accuracy: {result['accuracy']}")
    logging.info(f"Precision: {result['precision']}")
    logging.info(f"Recall: {result['recall']}")
    logging.info(f"F1 Score: {result['f1_score']}")
    logging.info(f"NDCG: {result['ndcg']}")
    logging.info(f"Average Outlier Rate: {result['average_outlier_rate']}")
    logging.info(f"MRR: {result['mrr']}")

    symbol_accuracies, overall_symbol_accuracy = evaluate_per_symbol_predictions(data)
    logging.info("Predicted Symbols Accuracy per Entry:")
    for accuracy in symbol_accuracies:
        logging.info(f"ID: {accuracy['id']}, Accuracy: {accuracy['accuracy']:.2f}")


########################### 以下为测试函数 ############################
def evaluate_model_on_test_set(model, test_data, item_similarity_matrix,symbol_to_id, k, criterion,
                               batch_size=32):
    """
    使用训练好的模型在测试集上进行预测，并计算准确率。

    参数:
    - model: 已训练好的模型
    - test_data: 测试集数据，格式为 (user, item, rating) 的三元组列表
    - question_similarity_matrix: 用户相似度矩阵
    - item_similarity_matrix: 物品相似度矩阵
    - batch_size: 批次大小
    - k: 排名前 k 的推荐物品

    返回:
    - test_accuracy: 测试集的 top-k 准确率
    """
    model.eval()  # 设置模型为评估模式

    test_user, test_item, test_ratings, question_similarity_matrix = zip(*test_data)  # 解包测试数据


    #根据符号的相关顺序 得到相应的vector
    test_ratings = torch.tensor(test_ratings).float()  # 转换为tensor

    predictions = []
    with torch.no_grad():  # 不需要计算梯度
        for i in range(0, len(test_user), batch_size):
            user_batch = test_user[i:i + batch_size]
            item_batch = test_item[i:i + batch_size]

            # 当前训练的相似矩阵数据移到GPU
            question_similarity_vector = torch.tensor(
                np.array(question_similarity_matrix[i:i + batch_size])).float().to(device)
            # breakpoint()
            item_ids = [symbol_to_id[symbol] for symbol in item_batch]  # 将符号转为 item_id
            item_similarity_vector = item_similarity_matrix[item_ids]
            item_similarity_vector = torch.tensor(item_similarity_vector, dtype=torch.float).to(device)

            pred1,_,_ = model(user_batch, item_batch, question_similarity_vector, item_similarity_vector)
            predictions.append(pred1.cpu())

    predictions = torch.cat(predictions, dim=0)  # (num_users * num_items,)

    # 重塑 predictions 为二维张量 (num_users, num_items)
    predictions = predictions.view(-1, len(set(test_item)))

    # 计算测试损失
    test_loss = criterion(predictions.view(-1), test_ratings)

    # 计算top-k准确率
    test_accuracy = calculate_top_k_accuracy(predictions, test_user, test_item, test_ratings, k)

    return test_ratings, test_loss, test_accuracy, predictions.view(-1)


########################### 以下为模型定义 ############################
class RecommendModel(nn.Module):
    def __init__(self):
        super(RecommendModel, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
        # self.encoder = BertModel.from_pretrained('chinese-bert-wwm-ext')

        self.tokenizer = AutoTokenizer.from_pretrained('chinese-bert-wwm-ext')
        self.encoder = BertAdapterModel.from_pretrained('chinese-bert-wwm-ext')

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        ## 新增线性层来投影
        self.question_projection = nn.Linear(9348, 768)  # 将相似度矩阵调整为768维
        self.item_projection = nn.Linear(132, 768)  # 将相似度矩阵调整为768维

        # 保留备用 如果相加的形式不好的话
        # self.MLP_user_projection = nn.Linear(768 + 768, 768)
        # self.MLP_item_projection = nn.Linear(768 + 768, 768)
        # self.MF_user_projection = nn.Linear(768 + 768, 768)
        # self.MF_item_projection = nn.Linear(768 + 768, 768)

        # 输入为[64, 3072]
        self.MLP = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
        )
        # 神经协同过滤网络NeuMF用于生成最终的预测 f1+f2+f3 768 + 320 + 768 = 1856
        self.NeuMF = nn.Sequential(
            nn.Linear(1856, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Linear(32, 1)
        )

    def forward(self, user_data, item_data, question_similarity_vector, item_similarity_vector, T1 = 4, T2 = 4):
        # start_total = time.time()

        start_user = time.time()
        # 这个是得到相应的问题输入 然后将其进行分词和编码 转化成相应的token 这里转化成了512
        user_data_inputs = self.tokenizer(user_data, return_tensors='pt', padding='max_length', truncation=True, max_length=320)
        # 移动相应的token到gpu上
        user_data_inputs = {key: val.to(device) for key, val in user_data_inputs.items()}

        # 维度确认  ([64, 320, 768])  还要在得到其不平均的值
        usr_outputs =  self.encoder(**user_data_inputs, output_hidden_states=True)

        usr_outputs = usr_outputs.hidden_states[-1]

        # 512*768 爆显存 所以改成320维度
        usr_outputs_mean = usr_outputs.mean(dim=1)
        # user_time = time.time() - start_user

        # 这个是符号输入
        # start_item = time.time()
        item_data_inputs = self.tokenizer(item_data, return_tensors='pt', padding='max_length', truncation=True, max_length=320)
        item_data_inputs = {key: val.to(device) for key, val in item_data_inputs.items()}
        #[64,320, 768]
        item_outputs = self.encoder(**item_data_inputs, output_hidden_states=True)

        item_outputs = item_outputs.hidden_states[-1]

        # 同上[64, 768]
        item_outputs_mean = item_outputs.mean(dim=1)
        # item_time = time.time() - start_item


        # 调整相似度向量的维度 降维操作
        # [64, 768]
        # start_sim = time.time()
        adjusted_question_similarity_vector = self.question_projection(question_similarity_vector)
        # [64, 768]
        adjusted_item_similarity_vector = self.item_projection(item_similarity_vector)
        # sim_time = time.time() - start_sim

        # 将mean向量和调整后的相似度向量相加   1已经得到
        # start_features = time.time()
        updated_question_embedding = usr_outputs_mean + adjusted_question_similarity_vector
        # 得到更新特征后的新的item信息
        updated_item_embedding = item_outputs_mean + adjusted_item_similarity_vector


        #这里点乘  [64, 768]
        feature1 = updated_question_embedding * updated_item_embedding  # [hidden_size]

        # 对于feature2的计算，需要调整维度
        # [64, 768] -> [64, 768, 1]
        combined_item_vector = updated_item_embedding.unsqueeze(2)
        # 执行矩阵乘法  3得到
        # [64, 320, 768] @ [64, 768, 1] = [64, 320, 1]
        feature2 = torch.matmul(usr_outputs, combined_item_vector)
        # 把feature2压缩到二维 [64, 320]
        feature2 = feature2.squeeze(-1)

        # [64, 768]  串联
        # 先将两个embedding拼接[64, 768] + [64, 768] -> [64, 1536]
        concatenated_embeddings = torch.cat((updated_question_embedding, updated_item_embedding), dim=1)
        # sum_embedding = updated_question_embedding + updated_item_embedding  # [hidden_size]
        # 通过MLP降维到768
        # [64, 1536] -> [64, 768]
        feature3 = self.MLP(concatenated_embeddings)
        # features_time = time.time() - start_features


        # feature1: [64, 768]
        # feature2: [64, 320]
        # feature3: [64, 768]
        #然后进行拼接操作  64*1856
        # start_pred = time.time()
        combined_features = torch.cat((feature1, feature2, feature3), dim=1)
        logit = self.NeuMF(combined_features)
        predictions1 = torch.sigmoid(logit).squeeze()
        predictions2 = torch.sigmoid(logit / T1).squeeze()
        predictions3 = torch.sigmoid(logit / T2).squeeze()
        # pred_time = time.time() - start_pred
        # total_time = time.time() - start_total

        # print(f"用户数据处理时间: {user_time:.4f}s")
        # print(f"项目数据处理时间: {item_time:.4f}s")
        # print(f"相似度向量处理时间: {sim_time:.4f}s")
        # print(f"特征计算时间: {features_time:.4f}s")
        # print(f"预测时间: {pred_time:.4f}s")
        # print(f"总执行时间: {total_time:.4f}s")

        return predictions1, predictions2, predictions3


# 读取近似矩阵的函数 返回相似矩阵
def read_similarity_matrix(file_path):
    """
    读取相似度矩阵，并排除第一行和第一列。

    参数:
    - file_path: 相似度矩阵文件的路径

    返回:
    - similarity_matrix: 已处理的相似度矩阵（NumPy 数组）
    """
    similarity_df = pd.read_csv(file_path, header=None)

    # 排除第一行和第一列
    similarity_matrix = similarity_df.values[1:, 1:]

    return similarity_matrix


# 辅助函数 将相似矩阵的顺序和此时训练或者测试的顺序相同
def reorder_similarity_matrices(test_data, question_similarity_matrix, item_similarity_matrix):
    """
    根据测试数据的顺序调整相似度矩阵的行顺序

    参数:
        test_data (list): 测试数据，格式为 [(user_id, item_id), ...]
        question_similarity_matrix (np.ndarray): 原始问题相似度矩阵
        item_similarity_matrix (np.ndarray): 原始项目相似度矩阵

    返回:
        np.ndarray, np.ndarray: 重新排序后的问题相似度矩阵和项目相似度矩阵
    """
    # 获取用户和项目索引
    user_indices = [user_id for user_id, item_id in test_data]
    item_indices = [item_id for user_id, item_id in test_data]

    # 重新调整相似度矩阵的行顺序
    reordered_question_similarity_matrix = question_similarity_matrix[user_indices]
    reordered_item_similarity_matrix = item_similarity_matrix[item_indices]

    return reordered_question_similarity_matrix, reordered_item_similarity_matrix


if __name__ == "__main__":
    # cluster矩阵
    cluster_ratings_matrix, _, _, _, _ = process_matrix_data(
    'new_data/train/rating_data_cluster.csv',
    'new_data/train/user_data.csv',
    'new_data/item_data.csv'
    )

    # 真实01矩阵
    train_ratings_matrix, train_users_dict, train_items_dict, train_user, train_item = process_matrix_data(
        'new_data/train/rating_data.csv',
        'new_data/train/user_data.csv',
        'new_data/item_data.csv'
    )

    test_ratings_matrix, test_users_dict, test_items_dict, test_user, test_item = process_matrix_data(
        'new_data/test/rating_data.csv',
        'new_data/test/user_data.csv',
        'new_data/item_data.csv'
    )
    # SVD矩阵
    num_latent_factors = 30
    U, S, V = matrix_factorization(train_ratings_matrix, num_latent_factors)
    emb_ratings_matrix = reconstruct_matrix(U, S, V)
    emb_ratings_matrix = torch.tensor(emb_ratings_matrix, dtype=torch.float32)
    emb_ratings_matrix = sigmoid_transform(emb_ratings_matrix, scale=10)

    # 因为符号后面一直需要使用 所以引入相应的映射关系
    # 读取 item_data.csv，构建符号与 item_id 的映射
    item_data = pd.read_csv('new_data/item_data.csv')  # 确保路径正确
    symbol_to_id = dict(zip(item_data['symbol'], item_data['item_id']))

    # 引入一个相似度矩阵 包含问题和符号的相似度
    question_similarity_train_matrix = read_similarity_matrix("new_data/train/user_similarity.csv")
    item_similarity_matrix = read_similarity_matrix("new_data/train/item_similarity.csv")
    # 读取测试数据的矩阵 item_similarity_test_matrix和训练的一样
    question_similarity_test_matrix = read_similarity_matrix("new_data/test/user_similarity.csv")

    # 符号矩阵为132*132
    train_data = negative_sampling(train_ratings_matrix, emb_ratings_matrix, cluster_ratings_matrix, train_users_dict, train_items_dict, question_similarity_train_matrix, neg_ratio=1)

    test_data, id_samples = full_sampling(test_ratings_matrix, test_users_dict, test_items_dict,
                                          question_similarity_test_matrix)

    # 调整测试评分矩阵顺序
    # question_similarity_test_matrix, item_similarity_test_matrix = reorder_similarity_matrices(test_data,question_similarity_test_matrix,item_similarity_test_matrix)

    train_user, train_item, train_ratings, train_emb_ratings, train_cluster_rating, question_similarity_vectors = zip(*train_data)
    train_ratings = torch.tensor(train_ratings).float().to(device)
    train_emb_ratings = torch.tensor(train_emb_ratings).float().to(device)
    train_cluster_rating = torch.tensor(train_cluster_rating).float().to(device)

    recommend_model = RecommendModel().to(device)

    ##################### 以下为调参部分 #####################
    criterion1 = nn.BCELoss()
    criterion2 = nn.BCELoss()
    criterion3 = nn.BCELoss()
    learning_rate_bert = 1e-5
    learning_rate_others = 3e-5
    weight_decay = 0.001

    # 优化器 Adam 和 AdamW
    optimizer = optim.AdamW([
        {'params': recommend_model.encoder.parameters(), 'lr': learning_rate_bert},
        {'params':recommend_model.question_projection.parameters()},
        {'params': recommend_model.item_projection.parameters()},
        # {'params': recommend_model.MLP_user_projection.parameters()},
        # {'params': recommend_model.MLP_item_projection.parameters()},
        # {'params': recommend_model.MF_item_projection.parameters()},
        # {'params': recommend_model.MF_user_projection.parameters()},
        {'params': recommend_model.MLP.parameters()},
        {'params': recommend_model.NeuMF.parameters()},
    ], lr=learning_rate_others, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.01)

    epochs = 60  # 80
    batch_size = 64
    top_k = 10
    T1 = 4
    T2 = 4
    lambda_1 = 0.2
    lambda_2 = 0.2

    logging.info(f"Scale = 10, lamda1: {lambda_1}, lambda2: {lambda_2}, T1: {T1}, T2: {T2}, learning_rate_bert: {learning_rate_bert}, learning_rate_others: {learning_rate_others}, weight_decay: {weight_decay}, num_latent_factors: {num_latent_factors}, epochs: {epochs}, batch_size: {batch_size}, top_k: {top_k}")
    ##################### 以下为训练部分 #####################
    recommend_model.train()

    train_losses = []  # 保存每个epoch的loss
    train_accuracies = []  # 保存每个epoch的accuracy

    best_accuracy = float('-inf')  # 初始化最好的accuracy为负无穷
    best_epoch = 0  # 初始化最好的epoch为0

    for epoch in range(epochs):
        # epoch_start_time = time.time()
        epoch_loss = 0
        epoch_accuracy = 0

        # start_batch_time = time.time()
        for i in range(0, len(train_user), batch_size):
            # start_user = time.time()
            optimizer.zero_grad()

            user_batch = train_user[i:i + batch_size]
            item_batch = train_item[i:i + batch_size]
            rating_batch = train_ratings[i:i + batch_size]
            emb_ratings_batch = train_emb_ratings[i:i + batch_size]
            cluster_rating_batch = train_cluster_rating[i:i + batch_size]

            # 当前训练的相似矩阵数据移到GPU
            question_similarity_vector = torch.tensor(np.array(question_similarity_vectors[i:i + batch_size])).float().to(device)
            # 对于 item_batch，使用符号查找对应的 item_id
            item_ids = [symbol_to_id[symbol] for symbol in item_batch]  # 将符号转为 item_id
            item_similarity_vector = item_similarity_matrix[item_ids]
            item_similarity_vector = torch.tensor(item_similarity_vector, dtype=torch.float).to(device)
            # breakpoint()

            # 确保question_similarity_vector的第二维大小与usr_outputs相匹配
            # if len(question_similarity_vector.shape) == 1:
            #     question_similarity_vector = question_similarity_vector.unsqueeze(1)  # 变为二维

            predictions1, predictions2, predictions3 = recommend_model(user_batch, item_batch, question_similarity_vector, item_similarity_vector, T1=T1, T2=T2)

            loss1 = criterion1(predictions1, rating_batch)
            loss2 = criterion2(predictions2, emb_ratings_batch)
            loss3 = criterion3(predictions3, cluster_rating_batch)
            total_loss = (1 - lambda_2 - lambda_1) * loss1 + lambda_1 * loss2 + lambda_2 * loss3
            # backtime = time.time()
            total_loss.backward()
            # backend = time.time()-backtime
            # breakpoint()
            # backtime2 = time.time()
            optimizer.step()

            epoch_loss += total_loss.item()
            # endtime = time.time()

            # batchtime = endtime - start_user

        # end_batch_time = time.time()-start_batch_time

        scheduler.step()

        # 计算loss和accuracy
        train_loss = epoch_loss / (len(train_user) // batch_size)
        train_losses.append(train_loss)
        # 在每个epoch结束后用测试集评估模型的准确率
        _, test_loss, test_accuracy, _ = evaluate_model_on_test_set(recommend_model, test_data,
                                                                    item_similarity_matrix,
                                                                    symbol_to_id,
                                                                     k=top_k,
                                                                    criterion=criterion1, batch_size=32)

        train_accuracies.append(test_accuracy)
        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Train_Loss: {train_loss}, Test_Loss: {test_loss}, Test_Accuracy: {test_accuracy}")

        # 如果当前epoch的损失比最好的损失小，更新最好的损失并保存模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            # best_model_state = recommend_model.state_dict()  # 保存当前模型的状态
            # 保存最好的模型
            best_model = f"best_models/{model_name}.pt"
            # torch.save(recommend_model, best_model)
            torch.save({
                'model_state_dict': recommend_model.state_dict()
            }, best_model)
            

    logging.info(f"Best model saved at epoch {best_epoch + 1} with accuracy {best_accuracy}")

    plot_metrics(train_losses, train_accuracies)

    ########################### 以下为测试部分 ############################
    # recommend_model = torch.load(best_model)
    checkpoint = torch.load(best_model)
    
    # 恢复模型状态
    recommend_model.load_state_dict(checkpoint['model_state_dict'])


    test_ratings, test_loss, test_accuracy, predictions = evaluate_model_on_test_set(recommend_model, test_data,
                                                                                     item_similarity_matrix,
                                                                                     symbol_to_id, top_k,
                                                                                     criterion1, batch_size=32)

    ############################### 以下为评估部分 ###############################
    # # 计算覆盖所有真实评分为1的项所需的推荐个数
    # k = calculate_recommendation_cover(predictions, test_data)
    # logging.info(f"Number of recommendations needed to cover all relevant items (rating=1): {k}")

    # Get top-10 recommendations for each user
    top_n = 10
    top_10_recommendations = {}

    predictions_df = pd.DataFrame({
        'user_id': [sample[0] for sample in id_samples], 'item_id': [sample[1] for sample in id_samples],
        'predicted_rating': predictions.numpy()})
    item_df = pd.read_csv('new_data/item_data.csv')
    item_df = item_df.set_index('item_id')

    # 获取每个用户的前10个推荐
    for user_id in predictions_df['user_id'].unique():
        user_predictions = predictions_df[predictions_df['user_id'] == user_id]
        top_items = user_predictions.sort_values(by='predicted_rating', ascending=False).head(top_n)
        top_items_ids = top_items['item_id'].tolist()
        logging.info(top_items_ids)

        item_symbols = {item_id: item_df.loc[item_id, 'symbol'] for item_id in top_items_ids}
        item_ratings = {item_id: top_items[top_items['item_id'] == item_id]['predicted_rating'].values[0] for item_id in
                        top_items_ids}
        top_10_recommendations[user_id] = {
            'predicted_symbols': [
                {'symbol': item_symbols[item_id], 'probability': item_ratings[item_id]} for item_id in top_items_ids
            ]
        }

    # 遍历现有数据，并根据顺序插入推荐信息
    with open('new_data/test_data.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    for i, item in enumerate(existing_data):
        try:
            item['recommended_symbols'] = top_10_recommendations[i]['predicted_symbols']
        except:
            # 如果索引超出范围，则将 recommended_symbols 设置为空
            item['recommended_symbols'] = []

    existing_data = convert_numpy_float32(existing_data)

    # 保存更新后的数据到新文件
    with open(prediction_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    # 评估模型的预测性能
    evaluate_predictions(existing_data, model_name)
