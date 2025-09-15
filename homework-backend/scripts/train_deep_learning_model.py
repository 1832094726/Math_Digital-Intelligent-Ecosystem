# -*- coding: utf-8 -*-
"""
深度学习推荐模型训练脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import mysql.connector
from services.deep_learning_recommender import DeepLearningRecommender, FusionRecommendationModel

# 数据库配置
DB_CONFIG = {
    'host': 'rm-cn-4xl3n8fy70003mo.rwlb.rds.aliyuncs.com',
    'port': 3306,
    'user': 'k12_math_user',
    'password': 'K12math2024!',
    'database': 'k12_math_education',
    'charset': 'utf8mb4'
}


class SymbolRecommendationDataset(Dataset):
    """符号推荐数据集"""
    
    def __init__(self, interactions: List[Dict], user_id_map: Dict, symbol_id_map: Dict):
        self.interactions = interactions
        self.user_id_map = user_id_map
        self.symbol_id_map = symbol_id_map
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        
        user_id = self.user_id_map.get(str(interaction['user_id']), 0)
        symbol_id = self.symbol_id_map.get(f"symbol_{interaction['symbol_id']}", 0)
        context_text = interaction.get('context_text', '')
        rating = interaction.get('rating', 0.5)  # 0-1之间的评分
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'symbol_id': torch.tensor(symbol_id, dtype=torch.long),
            'context_text': context_text,
            'rating': torch.tensor(rating, dtype=torch.float)
        }


def collate_fn(batch):
    """数据批处理函数"""
    user_ids = torch.stack([item['user_id'] for item in batch])
    symbol_ids = torch.stack([item['symbol_id'] for item in batch])
    context_texts = [item['context_text'] for item in batch]
    ratings = torch.stack([item['rating'] for item in batch])
    
    return {
        'user_ids': user_ids,
        'symbol_ids': symbol_ids,
        'context_texts': context_texts,
        'ratings': ratings
    }


def load_training_data() -> List[Dict]:
    """从数据库加载训练数据"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 查询符号推荐历史数据
        query = """
        SELECT 
            sr.student_id as user_id,
            sr.symbol_id,
            sr.context_text,
            sr.confidence_score as rating,
            sr.created_at
        FROM symbol_recommendations sr
        WHERE sr.context_text IS NOT NULL 
        AND sr.confidence_score IS NOT NULL
        ORDER BY sr.created_at DESC
        LIMIT 10000
        """
        
        cursor.execute(query)
        recommendations = cursor.fetchall()
        
        # 查询学习行为数据作为补充
        behavior_query = """
        SELECT 
            lb.student_id as user_id,
            CAST(SUBSTRING_INDEX(lb.action_details, 'symbol_id:', -1) AS UNSIGNED) as symbol_id,
            lb.context_data as context_text,
            CASE 
                WHEN lb.action_type = 'symbol_used' THEN 0.8
                WHEN lb.action_type = 'symbol_selected' THEN 0.6
                ELSE 0.4
            END as rating,
            lb.created_at
        FROM learning_behaviors lb
        WHERE lb.action_type IN ('symbol_used', 'symbol_selected', 'symbol_viewed')
        AND lb.action_details LIKE '%symbol_id:%'
        ORDER BY lb.created_at DESC
        LIMIT 5000
        """
        
        cursor.execute(behavior_query)
        behaviors = cursor.fetchall()
        
        # 合并数据
        all_data = recommendations + behaviors
        
        # 数据清洗和预处理
        cleaned_data = []
        for item in all_data:
            if item['symbol_id'] and item['context_text']:
                # 确保评分在0-1范围内
                rating = float(item['rating'])
                rating = max(0.0, min(1.0, rating))
                
                cleaned_data.append({
                    'user_id': item['user_id'],
                    'symbol_id': item['symbol_id'],
                    'context_text': str(item['context_text'])[:500],  # 限制文本长度
                    'rating': rating
                })
        
        cursor.close()
        conn.close()
        
        print(f"加载了 {len(cleaned_data)} 条训练数据")
        return cleaned_data
        
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        return generate_synthetic_data()


def generate_synthetic_data() -> List[Dict]:
    """生成合成训练数据"""
    print("生成合成训练数据...")
    
    # 数学上下文模板
    contexts = [
        "解这个二次方程：x² + 3x + 2 = 0",
        "计算这个积分：∫x²dx",
        "求这个函数的导数：f(x) = sin(x)",
        "证明这个几何定理",
        "计算这个矩阵的行列式",
        "求解这个三角函数方程",
        "分析这个概率问题",
        "计算这个极限值",
        "求这个微分方程的解",
        "证明这个数列的收敛性"
    ]
    
    # 符号ID范围
    symbol_ids = list(range(1, 51))
    user_ids = list(range(1, 101))
    
    synthetic_data = []
    
    for _ in range(5000):
        user_id = random.choice(user_ids)
        symbol_id = random.choice(symbol_ids)
        context = random.choice(contexts)
        
        # 基于符号ID生成相关性评分
        base_rating = random.uniform(0.3, 0.9)
        if symbol_id <= 10:  # 常用符号
            rating = min(1.0, base_rating + 0.2)
        elif symbol_id <= 30:  # 中等使用频率
            rating = base_rating
        else:  # 较少使用
            rating = max(0.1, base_rating - 0.2)
        
        synthetic_data.append({
            'user_id': user_id,
            'symbol_id': symbol_id,
            'context_text': context,
            'rating': rating
        })
    
    return synthetic_data


def create_id_mappings(data: List[Dict]) -> Tuple[Dict, Dict]:
    """创建用户和符号ID映射"""
    user_ids = set(str(item['user_id']) for item in data)
    symbol_ids = set(f"symbol_{item['symbol_id']}" for item in data)
    
    user_id_map = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
    symbol_id_map = {symbol_id: idx for idx, symbol_id in enumerate(sorted(symbol_ids))}
    
    return user_id_map, symbol_id_map


def train_model():
    """训练深度学习模型"""
    print("开始训练深度学习推荐模型...")
    
    # 加载数据
    training_data = load_training_data()
    if not training_data:
        print("没有可用的训练数据")
        return
    
    # 创建ID映射
    user_id_map, symbol_id_map = create_id_mappings(training_data)
    
    # 创建数据集
    dataset = SymbolRecommendationDataset(training_data, user_id_map, symbol_id_map)
    
    # 分割训练和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 初始化模型
    num_users = len(user_id_map)
    num_symbols = len(symbol_id_map)
    model = FusionRecommendationModel(num_users, num_symbols)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # 训练循环
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            user_ids = batch['user_ids'].to(device)
            symbol_ids = batch['symbol_ids'].to(device)
            context_texts = batch['context_texts']
            ratings = batch['ratings'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(user_ids, symbol_ids, context_texts).squeeze()
            loss = criterion(predictions, ratings)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_ids'].to(device)
                symbol_ids = batch['symbol_ids'].to(device)
                context_texts = batch['context_texts']
                ratings = batch['ratings'].to(device)
                
                predictions = model(user_ids, symbol_ids, context_texts).squeeze()
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # 保存模型
            recommender = DeepLearningRecommender()
            recommender.model = model
            recommender.user_id_map = user_id_map
            recommender.symbol_id_map = symbol_id_map
            recommender.reverse_symbol_map = {v: k for k, v in symbol_id_map.items()}
            recommender.save_model()
            
            print(f"保存最佳模型 (验证损失: {best_val_loss:.4f})")
        
        scheduler.step()
    
    print("模型训练完成!")


if __name__ == "__main__":
    train_model()
