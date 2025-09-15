# -*- coding: utf-8 -*-
"""
深度学习推荐模型
集成BERT、NCF和MLP的融合推荐系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Any, Tuple, Optional
import json
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTContextEncoder(nn.Module):
    """BERT上下文编码器"""
    
    def __init__(self, model_name='bert-base-chinese', hidden_size=768):
        super(BERTContextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # 冻结BERT参数以节省计算资源
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, text_inputs: List[str]) -> torch.Tensor:
        """编码文本输入"""
        # 分词和编码
        encoded = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # BERT编码
        with torch.no_grad():
            outputs = self.bert(**encoded)
            # 使用[CLS]标记的表示作为句子表示
            context_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return context_embeddings


class NCFModel(nn.Module):
    """神经协同过滤模型"""
    
    def __init__(self, num_users: int, num_symbols: int, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64, 32]):
        super(NCFModel, self).__init__()
        self.num_users = num_users
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        
        # 用户和符号嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        
        # MLP层
        input_dim = embedding_dim * 2
        self.mlp_layers = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.symbol_embedding.weight, std=0.01)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)
        symbol_emb = self.symbol_embedding(symbol_ids)
        
        # 拼接用户和符号嵌入
        x = torch.cat([user_emb, symbol_emb], dim=-1)
        
        # 通过MLP层
        for layer in self.mlp_layers:
            x = layer(x)
        
        # 输出预测分数
        output = self.output_layer(x)
        return torch.sigmoid(output)


class FusionRecommendationModel(nn.Module):
    """融合推荐模型：BERT + NCF + MLP"""
    
    def __init__(self, num_users: int, num_symbols: int, bert_hidden_size: int = 768, 
                 ncf_embedding_dim: int = 64, fusion_hidden_dims: List[int] = [256, 128, 64]):
        super(FusionRecommendationModel, self).__init__()
        
        # BERT上下文编码器
        self.bert_encoder = BERTContextEncoder()
        
        # NCF模型
        self.ncf_model = NCFModel(num_users, num_symbols, ncf_embedding_dim)
        
        # 融合层
        fusion_input_dim = bert_hidden_size + 1  # BERT输出 + NCF分数
        self.fusion_layers = nn.ModuleList()
        
        for hidden_dim in fusion_hidden_dims:
            self.fusion_layers.append(nn.Linear(fusion_input_dim, hidden_dim))
            self.fusion_layers.append(nn.ReLU())
            self.fusion_layers.append(nn.Dropout(0.3))
            fusion_input_dim = hidden_dim
        
        # 最终输出层
        self.final_output = nn.Linear(fusion_hidden_dims[-1], 1)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=fusion_hidden_dims[0], num_heads=8)
    
    def forward(self, user_ids: torch.Tensor, symbol_ids: torch.Tensor, 
                context_texts: List[str]) -> torch.Tensor:
        """前向传播"""
        # BERT编码上下文
        context_embeddings = self.bert_encoder(context_texts)
        
        # NCF预测
        ncf_scores = self.ncf_model(user_ids, symbol_ids)
        
        # 融合BERT和NCF特征
        fusion_input = torch.cat([context_embeddings, ncf_scores], dim=-1)
        
        # 通过融合层
        x = fusion_input
        for layer in self.fusion_layers:
            x = layer(x)
        
        # 应用注意力机制
        x_unsqueezed = x.unsqueeze(0)  # 添加序列维度
        attended_x, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = attended_x.squeeze(0)  # 移除序列维度
        
        # 最终预测
        output = self.final_output(x)
        return torch.sigmoid(output)


class DeepLearningRecommender:
    """深度学习推荐器"""
    
    def __init__(self, model_path: str = 'models', device: str = 'cpu'):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.user_id_map = {}
        self.symbol_id_map = {}
        self.reverse_symbol_map = {}
        
        # 确保模型目录存在
        os.makedirs(model_path, exist_ok=True)
        
        # 加载或初始化模型
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """加载或初始化模型"""
        model_file = os.path.join(self.model_path, 'fusion_model.pth')
        mapping_file = os.path.join(self.model_path, 'id_mappings.json')
        
        if os.path.exists(model_file) and os.path.exists(mapping_file):
            try:
                # 加载ID映射
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                    self.user_id_map = mappings['user_id_map']
                    self.symbol_id_map = mappings['symbol_id_map']
                    self.reverse_symbol_map = {v: k for k, v in self.symbol_id_map.items()}
                
                # 初始化模型
                num_users = len(self.user_id_map)
                num_symbols = len(self.symbol_id_map)
                self.model = FusionRecommendationModel(num_users, num_symbols)
                
                # 加载模型权重
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"成功加载模型，用户数: {num_users}, 符号数: {num_symbols}")
                
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                self._initialize_default_model()
        else:
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """初始化默认模型"""
        # 创建默认的ID映射
        self.user_id_map = {f"user_{i}": i for i in range(100)}  # 支持100个用户
        self.symbol_id_map = {f"symbol_{i}": i for i in range(50)}  # 支持50个符号
        self.reverse_symbol_map = {v: k for k, v in self.symbol_id_map.items()}
        
        # 初始化模型
        num_users = len(self.user_id_map)
        num_symbols = len(self.symbol_id_map)
        self.model = FusionRecommendationModel(num_users, num_symbols)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"初始化默认模型，用户数: {num_users}, 符号数: {num_symbols}")
    
    def get_recommendations(self, user_id: str, context_text: str, 
                          candidate_symbols: List[Dict], top_k: int = 10) -> List[Dict]:
        """获取推荐结果"""
        if not self.model:
            return []
        
        try:
            # 映射用户ID
            if user_id not in self.user_id_map:
                # 为新用户分配ID
                new_id = len(self.user_id_map)
                self.user_id_map[user_id] = new_id
            
            user_idx = self.user_id_map[user_id]
            
            # 准备候选符号
            symbol_indices = []
            valid_symbols = []
            
            for symbol in candidate_symbols:
                symbol_key = f"symbol_{symbol.get('id', 0)}"
                if symbol_key not in self.symbol_id_map:
                    # 为新符号分配ID
                    new_id = len(self.symbol_id_map)
                    self.symbol_id_map[symbol_key] = new_id
                    self.reverse_symbol_map[new_id] = symbol_key
                
                symbol_indices.append(self.symbol_id_map[symbol_key])
                valid_symbols.append(symbol)
            
            if not symbol_indices:
                return []
            
            # 准备模型输入
            user_ids = torch.tensor([user_idx] * len(symbol_indices), dtype=torch.long).to(self.device)
            symbol_ids = torch.tensor(symbol_indices, dtype=torch.long).to(self.device)
            context_texts = [context_text] * len(symbol_indices)
            
            # 模型预测
            with torch.no_grad():
                scores = self.model(user_ids, symbol_ids, context_texts)
                scores = scores.cpu().numpy().flatten()
            
            # 组合结果
            recommendations = []
            for i, (symbol, score) in enumerate(zip(valid_symbols, scores)):
                recommendations.append({
                    **symbol,
                    'score': float(score),
                    'source': 'deep_learning',
                    'model_type': 'BERT-NCF-MLP'
                })
            
            # 按分数排序并返回top_k
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"深度学习推荐失败: {e}")
            return []
    
    def save_model(self):
        """保存模型"""
        try:
            # 保存模型权重
            model_file = os.path.join(self.model_path, 'fusion_model.pth')
            torch.save(self.model.state_dict(), model_file)
            
            # 保存ID映射
            mapping_file = os.path.join(self.model_path, 'id_mappings.json')
            mappings = {
                'user_id_map': self.user_id_map,
                'symbol_id_map': self.symbol_id_map
            }
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)
            
            logger.info("模型保存成功")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
