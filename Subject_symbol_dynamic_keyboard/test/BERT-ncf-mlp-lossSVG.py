import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from transformers import BertTokenizer, BertModel
import logging
from matplotlib import pyplot as plt
from calculate_metrics import evaluate_symbol_predictions,evaluate_per_symbol_predictions
import datetime
import os
from sklearn.decomposition import TruncatedSVD,NMF

nowTime = datetime.datetime.now().strftime('%m-%d-%H-%M')

#################### 以下为日志记录 ####################
model_name = 'BERT-ncf-mlp-lossSVG(' + nowTime + ')'

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
def negative_sampling(user_item_matrix, other1_ratings_matrix, user_dict, item_dict, neg_ratio=1):
    """
    对用户-物品矩阵进行负采样，返回值包含 user_dict 和 item_dict 中的值。

    参数：
    - user_item_matrix: 用户-物品矩阵 (PyTorch Tensor)，形状为 [num_users, num_items]。
    - user_dict: 用户 ID 到用户值的映射字典。
    - item_dict: 物品 ID 到物品值的映射字典。
    - neg_ratio: 每个正例生成的负例数量。

    返回：
    - List of tuples: [(user_value, item_value, label), ...]
    """
    num_users, num_items = user_item_matrix.shape
    positive_samples = []
    negative_samples = []

    for user_id in range(num_users):
        positive_items = torch.where(user_item_matrix[user_id] == 1)[0]
        negative_items = torch.where(user_item_matrix[user_id] == 0)[0]

        positive_samples.extend(
            [(user_dict[user_id], item_dict[item_id.item()], 1,  # 正例标签
              other1_ratings_matrix[user_id, item_id.item()].item())  # 获取评分值
             for item_id in positive_items]
        )

        num_negatives = len(positive_items) * neg_ratio
        sampled_negatives = torch.randint(
            low=0, high=len(negative_items), size=(num_negatives,), generator=None
        )

        negative_samples.extend(
            [(user_dict[user_id], item_dict[negative_items[idx].item()], 0,  # 负例标签
              other1_ratings_matrix[user_id, negative_items[idx].item()].item())  # 获取评分值
             for idx in sampled_negatives]
        )

    all_samples = positive_samples + negative_samples

    np.random.shuffle(all_samples)
    return all_samples

def full_sampling(user_item_matrix, user_dict, item_dict):
    """
    对用户-物品矩阵进行全采样，并将 user_id 和 item_id 替换为字典中的值。
    
    参数：
    - user_item_matrix: 用户-物品矩阵 (PyTorch Tensor)，形状为 [num_users, num_items]。
    - user_dict: 用户 ID 到用户值的映射字典。
    - item_dict: 物品 ID 到物品值的映射字典。
    
    返回：
    - List of tuples: [(user_value, item_value, label), ...]
    """
    num_users, num_items = user_item_matrix.shape
    all_samples = []
    id_samples = []
    for user_id in range(num_users):
        for item_id in range(num_items):
            label = user_item_matrix[user_id, item_id].item() 
            user_value = user_dict[user_id]  
            item_value = item_dict[item_id]  
            all_samples.append((user_value, item_value, label))  
            id_samples.append((user_id, item_id))
    return all_samples,id_samples

def process_matrix_data(rating_file, user_file, item_file):
    rating_df = pd.read_csv(rating_file)
    user_df = pd.read_csv(user_file)
    item_df = pd.read_csv(item_file)
    
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

    return ratings_matrix, user_dict, item_dict , user_df['question'], item_df['symbol']

# 使用SVD进行矩阵分解
def matrix_factorization(ratings_matrix, num_latent_factors=10):

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

# NMF 矩阵分解
def matrix_factorization_nmf(ratings_matrix, num_latent_factors):
    # 使用 sklearn 的 NMF 进行矩阵分解
    nmf = NMF(n_components=num_latent_factors, init='random', random_state=42)
    W = nmf.fit_transform(ratings_matrix)
    H = nmf.components_
    return W, H

def reconstruct_matrix_nmf(W, H):
    return torch.tensor(W @ H, dtype=torch.float32)

def exponential_mapping(x, decay_factor=1.0):
    # 对小于0的数值使用exp(x)，让它们趋近于0
    x = torch.where(x < 0, torch.exp(x * decay_factor), x)
    # 对大于0的数值使用1 - exp(-x)，使它们趋近于1
    x = torch.where(x >= 0, 1 - torch.exp(-x * decay_factor), x)
    return x

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
def evaluate_model_on_test_set(model, test_data, k, criterion, batch_size=32):
    """
    使用训练好的模型在测试集上进行预测，并计算准确率。

    参数:
    - model: 已训练好的模型
    - test_data: 测试集数据，格式为 (user, item, rating) 的三元组列表
    - batch_size: 批次大小
    - k: 排名前 k 的推荐物品

    返回:
    - test_accuracy: 测试集的 top-k 准确率
    """
    model.eval()  # 设置模型为评估模式

    test_user, test_item, test_ratings = zip(*test_data)  # 解包测试数据
    test_ratings = torch.tensor(test_ratings).float()  # 转换为tensor

    predictions = []
    with torch.no_grad():  # 不需要计算梯度
        for i in range(0, len(test_user), batch_size):
            user_batch = test_user[i:i + batch_size]
            item_batch = test_item[i:i + batch_size]
            preds1, preds2 = model(user_batch, item_batch)
            predictions.append(preds1.cpu())

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
        self.tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
        self.encoder = BertModel.from_pretrained('chinese-bert-wwm-ext')

        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        self.MLP_user_projection = nn.Linear(768, 768)
        self.MLP_item_projection = nn.Linear(768, 768)
        self.MF_user_projection = nn.Linear(768, 768)
        self.MF_item_projection = nn.Linear(768, 768)
        
        self.MLP = nn.Sequential(
            nn.Linear(1536,1152),
            nn.ReLU(),
            nn.Linear(1152,768),
            nn.LayerNorm(768),
            nn.ReLU(),
        )
        
        self.NeuMF = nn.Sequential(
            nn.Linear(1536,768),
            nn.ReLU(),
            nn.Linear(768,256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Linear(32, 1)
        )

    def forward(self, user_data, item_data, T = 4):
        user_data_inputs = self.tokenizer(user_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        user_data_inputs = {key: val.to(device) for key, val in user_data_inputs.items()}
        usr_outputs = self.encoder(**user_data_inputs).last_hidden_state[:, 0, :]
        MLP_usr_outputs = self.MLP_user_projection(usr_outputs)
        MF_usr_outputs = self.MF_user_projection(usr_outputs)

        item_data_inputs = self.tokenizer(item_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        item_data_inputs = {key: val.to(device) for key, val in item_data_inputs.items()}
        item_outputs = self.encoder(**item_data_inputs).last_hidden_state[:, 0, :]
        MLP_item_outputs = self.MLP_item_projection(item_outputs)
        MF_item_outputs = self.MF_item_projection(item_outputs)

        ui_combined = torch.cat([MLP_usr_outputs, MLP_item_outputs], dim=1)

        ui_mu = MF_usr_outputs * MF_item_outputs

        ui_combined = self.MLP(ui_combined)

        combined = torch.cat([ui_combined,ui_mu], dim=1)

        logit = self.NeuMF(combined)

        predictions1 = torch.sigmoid(logit).squeeze()
        predictions2 = torch.sigmoid(logit / T).squeeze()

        # predictions2 = torch.softmax(logits/T,dim=1)
        # predictions2 = logits/T  # 不使用 softmax，保持为实数预测
        # .view(-1)
        return predictions1, predictions2


if __name__ == "__main__":
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

    # 矩阵分解
    num_latent_factors = 20
    # U, S, V = matrix_factorization(train_ratings_matrix, num_latent_factors)
    # emb_ratings_matrix = reconstruct_matrix(U, S, V)

    W, H = matrix_factorization_nmf(train_ratings_matrix.numpy(), num_latent_factors)
    emb_ratings_matrix = reconstruct_matrix_nmf(W, H)

    emb_ratings_matrix = torch.tensor(emb_ratings_matrix, dtype=torch.float32)

    # 打印调整前的分布和调整后的分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(emb_ratings_matrix.numpy().flatten(), bins=50, alpha=0.7, color='blue', label='Before Sigmoid')
    plt.title('Distribution Before Sigmoid')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    emb_ratings_matrix = exponential_mapping(emb_ratings_matrix)
    plt.subplot(1, 2, 2)
    plt.hist(emb_ratings_matrix.numpy().flatten(), bins=50, alpha=0.7, color='red', label='After Sigmoid')
    plt.title('Distribution After Sigmoid')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('emb_ratings_matrix_distribution.png')

    train_data = negative_sampling(train_ratings_matrix, emb_ratings_matrix, train_users_dict, train_items_dict, neg_ratio=1)

    test_data, id_samples = full_sampling(test_ratings_matrix, test_users_dict, test_items_dict)

    train_user, train_item, train_ratings, train_emb_ratings = zip(*train_data)
    train_ratings = torch.tensor(train_ratings).float().to(device)
    train_emb_ratings = torch.tensor(train_emb_ratings).float().to(device)

    recommend_model = RecommendModel().to(device)

    ##################### 以下为调参部分 #####################
    criterion1 = nn.BCELoss()
    criterion2 = nn.BCELoss()
    learning_rate_bert = 1e-5
    learning_rate_others = 3e-4
    weight_decay = 0.001

    # 优化器 Adam 和 AdamW
    optimizer = optim.AdamW([
        {'params': recommend_model.encoder.parameters(), 'lr': learning_rate_bert},
        {'params': recommend_model.MLP_user_projection.parameters()},
        {'params': recommend_model.MLP_item_projection.parameters()},
        {'params': recommend_model.MF_item_projection.parameters()},
        {'params': recommend_model.MF_user_projection.parameters()},
        {'params': recommend_model.MLP.parameters()},
        {'params': recommend_model.NeuMF.parameters()},
    ], lr=learning_rate_others, weight_decay = weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.01)

    epochs = 40 # 80
    batch_size = 64
    top_k = 10
    T = 4
    lambda_ = 0.5

    ##################### 以下为训练部分 #####################
    recommend_model.train()

    train_losses = [] # 保存每个epoch的loss
    train_accuracies = [] # 保存每个epoch的accuracy

    best_accuracy = float('-inf') # 初始化最好的accuracy为负无穷
    best_epoch = 0  # 初始化最好的epoch为0

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for i in range(0, len(train_user), batch_size):
            optimizer.zero_grad()

            user_batch = train_user[i:i + batch_size]
            item_batch = train_item[i:i + batch_size]
            rating_batch = train_ratings[i:i + batch_size]
            emb_ratings_batch = train_emb_ratings[i:i + batch_size]

            predictions1, predictions2 = recommend_model(user_batch, item_batch, T=T)

            loss1 = criterion1(predictions1, rating_batch)
            loss2 = criterion2(predictions2, emb_ratings_batch)
            total_loss = lambda_ * loss1 + (1 - lambda_) * loss2

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()

        # 计算loss和accuracy
        train_loss = epoch_loss / (len(train_user)//batch_size)
        train_losses.append(train_loss)

        # 在每个epoch结束后用测试集评估模型的准确率
        _, test_loss, test_accuracy, _ = evaluate_model_on_test_set(recommend_model, test_data, k=top_k, criterion=criterion1, batch_size=32)
        train_accuracies.append(test_accuracy)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Train_Loss: {train_loss}, Test_Loss: {test_loss}, Test_Accuracy: {test_accuracy}")

        # 如果当前epoch的损失比最好的损失小，更新最好的损失并保存模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            # best_model_state = recommend_model.state_dict()  # 保存当前模型的状态
            # 保存最好的模型
            best_model = f"best_models/{model_name}.pt"
            torch.save(recommend_model, best_model)

    logging.info(f"Best model saved at epoch {best_epoch + 1} with accuracy {best_accuracy}")

    plot_metrics(train_losses, train_accuracies)

    ########################### 以下为测试部分 ############################
    recommend_model = torch.load(best_model)

    test_ratings, test_loss, test_accuracy, predictions = evaluate_model_on_test_set(recommend_model, test_data, top_k, criterion1, batch_size=32)

    ############################### 以下为评估部分 ###############################
    # # 计算覆盖所有真实评分为1的项所需的推荐个数
    # k = calculate_recommendation_cover(predictions, test_data)
    # logging.info(f"Number of recommendations needed to cover all relevant items (rating=1): {k}")

    # Get top-10 recommendations for each user
    top_n = 10
    top_10_recommendations = {}

    predictions_df = pd.DataFrame({
        'user_id': [sample[0] for sample in id_samples],'item_id': [sample[1] for sample in id_samples],'predicted_rating': predictions.numpy()})
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
